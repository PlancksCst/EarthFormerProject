"""SEVIRI dataset for EarthFormer CSI forecasting.

The backbone consumes the previous-day satellite image sequence. Optional
auxiliary features are exposed separately for readout/query conditioning.
When targets are requested, the loader returns the next-day CSI sequence and
clear-sky GHI needed for external GHI reconstruction.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import PureWindowsPath
from typing import Any

import numpy as np
import pandas as pd
import torch
import zarr
from torch.utils.data import Dataset


class SEVIRIImageSequenceDataset(Dataset):
    """Read SEVIRI sequences as `(T, C, H, W)` tensors.

    The dataset root is expected to contain:
    - `dualet_metadata.parquet`: legacy manifest for sample/date/zarr indexing.
    - `normalization.json`: channel mean/std computed over the SEVIRI data.

    No DualET model code or feature engineering is reused here.
    """

    CSI_TARGET_COLUMNS = (
        "target",
        "target_csi",
        "target_csi_sequence",
        "target_csi_seq",
        "target_CSI",
        "CSI_target",
        "csi_target",
        "next_day_csi",
        "next_day_csi_sequence",
        "next_day_csi_seq",
    )
    CSI_TARGET_PREFIXES = (
        "target_csi_",
        "csi_target_",
        "next_day_csi_",
    )
    CLEAR_SKY_GHI_COLUMNS = (
        "clear_sky_ghi",
        "clear_ghi",
        "clearsky_ghi",
        "target_clear_sky_ghi",
        "target_clear_ghi",
        "target_clearsky_ghi",
        "clear_sky_GHI",
        "ClearSkyGHI",
    )
    CLEAR_SKY_GHI_PREFIXES = (
        "clear_sky_ghi_",
        "clear_ghi_",
        "clearsky_ghi_",
        "target_clear_sky_ghi_",
        "target_clear_ghi_",
    )
    GHI_TARGET_COLUMNS = (
        "target_ghi",
        "target_ghi_sequence",
        "target_GHI",
        "GHI_target",
        "ghi_target",
        "next_day_ghi",
    )
    GHI_TARGET_PREFIXES = (
        "target_ghi_",
        "ghi_target_",
        "next_day_ghi_",
    )
    AUXILIARY_FEATURE_NAMES = (
        "previous_day_csi",
        "clear_sky_ghi_scaled",
        "solar_elevation_scaled",
        "latitude_scaled",
        "longitude_scaled",
        "hour_sin",
        "hour_cos",
        "dayofyear_sin",
        "dayofyear_cos",
    )

    def __init__(
        self,
        dataset_root: str,
        split: str = "train",
        sequence_length: int = 13,
        output_length: int = 13,
        image_size: int = 200,
        expected_channels: int = 7,
        normalize: bool = True,
        include_target: bool = False,
        target_channel_index: int = 0,
        metadata_filename: str | None = None,
        hourly_csv: str | None = None,
        elevation_csv: str | None = None,
        locations_csv: str | None = None,
        include_auxiliary_features: bool = False,
    ) -> None:
        self.dataset_root = os.path.abspath(dataset_root)
        self.split = split
        self.sequence_length = sequence_length
        self.output_length = output_length
        self.image_size = image_size
        self.expected_channels = expected_channels
        self.normalize = normalize
        self.include_target = include_target
        self.target_channel_index = target_channel_index
        self.hourly_csv = hourly_csv
        self.elevation_csv = elevation_csv
        self.locations_csv = locations_csv
        self.include_auxiliary_features = bool(include_auxiliary_features)

        metadata_path = self._resolve_metadata_path(metadata_filename)
        norm_path = os.path.join(self.dataset_root, "normalization.json")

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(metadata_path)
        if self.normalize and not os.path.exists(norm_path):
            raise FileNotFoundError(norm_path)

        meta = pd.read_parquet(metadata_path, engine="pyarrow")
        self.meta = meta[meta["split"] == split].reset_index(drop=True)
        if len(self.meta) == 0:
            raise ValueError(f"No samples found for split={split!r} in {metadata_path}")

        self.norm: dict[str, Any] | None = None
        if self.normalize:
            with open(norm_path, "r", encoding="utf-8") as f:
                self.norm = json.load(f)
            if len(self.norm["channel_mean"]) != self.expected_channels:
                raise ValueError(
                    "normalization.json channel count does not match "
                    f"expected_channels={self.expected_channels}"
                )

        first_zarr = self._open_zarr(self.meta.iloc[0]["input_zarr"])
        self.channels = list(first_zarr.attrs.get("channels", []))
        if len(self.channels) != self.expected_channels:
            raise ValueError(
                f"Expected {self.expected_channels} channels, found {len(self.channels)}: "
                f"{self.channels}"
            )
        if not 0 <= self.target_channel_index < self.expected_channels:
            raise ValueError(
                f"target_channel_index={self.target_channel_index} is outside "
                f"[0, {self.expected_channels})"
            )

        self._zarr_cache: dict[str, Any] = {}
        self.hourly_df: pd.DataFrame | None = None
        self.elevation_df: pd.DataFrame | None = None
        self.locations_df: pd.DataFrame | None = None
        if self.include_target or self.include_auxiliary_features:
            self.hourly_df = self._load_hourly_dataframe(hourly_csv)
            self.elevation_df = self._load_elevation_dataframe(elevation_csv)
        if self.include_auxiliary_features:
            self.locations_df = self._load_locations_dataframe(locations_csv)

    def __len__(self) -> int:
        return len(self.meta)

    def _resolve_metadata_path(self, metadata_filename: str | None) -> str:
        candidates = (
            [metadata_filename]
            if metadata_filename is not None
            else ["metadata.parquet", "dualet_metadata.parquet"]
        )
        for candidate in candidates:
            if candidate is None:
                continue
            # If the candidate exists as given (absolute or relative), prefer it.
            if os.path.exists(candidate):
                return candidate

            # Otherwise, try resolving relative to the dataset root.
            path = os.path.join(self.dataset_root, candidate)
            if os.path.exists(path):
                return path
        joined = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(
            f"No metadata file found in {self.dataset_root}. Tried: {joined}"
        )

    def _open_zarr(self, path: str) -> Any:
        return zarr.open(self._resolve_data_path(path), mode="r")

    def _resolve_data_path(self, path: str) -> str:
        if os.path.exists(path):
            return path
        root_joined = os.path.join(self.dataset_root, path)
        if os.path.exists(root_joined):
            return root_joined

        candidates: list[str] = []
        windows_path = PureWindowsPath(path)
        windows_parts = [
            part
            for part in windows_path.parts
            if part not in {windows_path.drive, windows_path.root, "\\"}
        ]
        # Metadata created on Windows may contain absolute paths such as
        # C:\...\BEST_7\2019_05\2019_05.zarr.  In Colab, keep the meaningful
        # dataset-relative suffix and resolve it below the active dataset root.
        for index in range(len(windows_parts)):
            suffix = os.path.join(*windows_parts[index:])
            candidates.append(os.path.join(self.dataset_root, suffix))
            candidates.append(os.path.join(os.path.dirname(self.dataset_root), suffix))

        basenames = {os.path.basename(path), windows_path.name}
        for basename in basenames:
            candidates.append(os.path.join(self.dataset_root, basename))

        seen: set[str] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            if os.path.exists(candidate):
                return candidate
        if windows_path.drive or "\\" in path:
            tried = "\n  ".join(seen)
            raise FileNotFoundError(
                "Could not remap Windows metadata path to this runtime. "
                f"Original path: {path}\nDataset root: {self.dataset_root}\nTried:\n  {tried}"
            )
        return path

    def _resolve_csv_path(
        self,
        path: str | None,
        filename: str,
        required: bool,
    ) -> str | None:
        """Resolve a CSV path from explicit, dataset-local, Colab, or local layouts."""
        candidates: list[str] = []
        if path:
            candidates.append(os.fspath(path))
        candidates.extend(
            [
                os.path.join(self.dataset_root, filename),
                os.path.join(os.path.dirname(self.dataset_root), filename),
                os.path.join("/content/CAMS", filename),
                os.path.join("/content/datasets", filename),
                os.path.join("/content/drive/MyDrive/EarthFormer/CAMS", filename),
                os.path.abspath(
                    os.path.join(
                        os.path.dirname(__file__),
                        "..",
                        "..",
                        "..",
                        "CAMS",
                        filename,
                    )
                ),
                os.path.abspath(
                    os.path.join(
                        os.path.dirname(__file__),
                        "..",
                        "..",
                        "..",
                        "..",
                        "CAMS",
                        filename,
                    )
                ),
            ]
        )

        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return candidate
        if required:
            tried = ", ".join(candidates)
            raise FileNotFoundError(f"Could not find {filename}. Tried: {tried}")
        return None

    def _load_hourly_dataframe(self, path: str | None) -> pd.DataFrame:
        """Load the hourly CAMS/ground CSI-GHI dataframe indexed by timestamp."""
        csv_path = self._resolve_csv_path(
            path=path,
            filename="all_locations_hourly.csv",
            required=True,
        )
        assert csv_path is not None
        df = pd.read_csv(csv_path)
        if "timestamp" not in df.columns:
            raise KeyError(f"Missing 'timestamp' column in {csv_path}")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
        if not df.index.is_unique:
            df = df[~df.index.duplicated(keep="first")]
        return df

    def _load_elevation_dataframe(self, path: str | None) -> pd.DataFrame | None:
        """Load the optional elevation CSV for future feature integration."""
        csv_path = self._resolve_csv_path(
            path=path,
            filename="all_locations_elevation.csv",
            required=False,
        )
        if csv_path is None:
            return None
        df = pd.read_csv(csv_path)
        if "timestamp" not in df.columns:
            raise KeyError(f"Missing 'timestamp' column in {csv_path}")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
        if not df.index.is_unique:
            df = df[~df.index.duplicated(keep="first")]
        return df

    def _load_locations_dataframe(self, path: str | None) -> pd.DataFrame | None:
        """Load optional station latitude/longitude metadata."""
        csv_path = self._resolve_csv_path(
            path=path,
            filename="locations.csv",
            required=False,
        )
        if csv_path is None:
            return None
        df = pd.read_csv(csv_path)
        required = {"station", "latitude", "longitude"}
        missing = sorted(required.difference(df.columns))
        if missing:
            raise KeyError(f"Missing columns in {csv_path}: {missing}")
        df["station"] = df["station"].astype(str)
        return df.set_index("station", drop=False)

    def _get_zarr(self, path: str) -> Any:
        if path not in self._zarr_cache:
            self._zarr_cache[path] = self._open_zarr(path)
        return self._zarr_cache[path]

    def _parse_sequence(self, value: Any, length: int | None = None) -> np.ndarray:
        if length is None:
            length = self.sequence_length
        if isinstance(value, str):
            text = value.strip()
            if "," in text:
                parsed = np.fromstring(text, sep=",", dtype=np.int64)
            else:
                parsed = np.array(
                    [int(item) for item in text.split() if item.strip()],
                    dtype=np.int64,
                )
        elif isinstance(value, (list, tuple, np.ndarray)):
            parsed = np.array(value, dtype=np.int64)
        else:
            parsed = np.array([], dtype=np.int64)

        if parsed.size == 0:
            parsed = np.full(length, -1, dtype=np.int64)
        elif parsed.size < length:
            parsed = np.pad(
                parsed,
                (0, length - parsed.size),
                constant_values=-1,
            )
        elif parsed.size > length:
            parsed = parsed[:length]
        return parsed.astype(np.int64)

    def _parse_mask(self, value: Any) -> np.ndarray:
        if isinstance(value, str):
            text = value.strip()
            if "," in text:
                parsed = np.fromstring(text, sep=",", dtype=np.int64)
            elif all(ch in "01" for ch in text):
                parsed = np.array([int(ch) for ch in text], dtype=np.int64)
            else:
                parsed = np.array([], dtype=np.int64)
        elif isinstance(value, (list, tuple, np.ndarray)):
            parsed = np.array(value, dtype=np.int64)
        else:
            parsed = np.array([], dtype=np.int64)

        if parsed.size == 0:
            parsed = np.ones(self.sequence_length, dtype=np.bool_)
        elif parsed.size < self.sequence_length:
            parsed = np.pad(
                parsed,
                (0, self.sequence_length - parsed.size),
                constant_values=1,
            )
        elif parsed.size > self.sequence_length:
            parsed = parsed[: self.sequence_length]
        return parsed.astype(np.bool_)

    def _indices_from_row(
        self,
        row: pd.Series,
        sequence_column: str,
        start_column: str,
        end_column: str,
        length: int,
    ) -> np.ndarray:
        """Read frame indices from either sequence or start/end metadata."""
        if sequence_column in row.index:
            return self._parse_sequence(row[sequence_column], length=length)
        if start_column in row.index and end_column in row.index:
            start = int(row[start_column])
            end = int(row[end_column])
            if end < start:
                parsed = np.array([], dtype=np.int64)
            else:
                parsed = np.arange(start, end + 1, dtype=np.int64)
            if parsed.size < length:
                parsed = np.pad(parsed, (0, length - parsed.size), constant_values=-1)
            elif parsed.size > length:
                parsed = parsed[:length]
            return parsed.astype(np.int64)
        return np.full(length, -1, dtype=np.int64)

    def _find_column(self, row: pd.Series, candidates: tuple[str, ...]) -> str | None:
        """Return the first matching column, allowing case-insensitive matches."""
        columns = list(row.index)
        exact = set(columns)
        lowered = {str(column).lower(): column for column in columns}
        for candidate in candidates:
            if candidate in exact:
                return candidate
            match = lowered.get(candidate.lower())
            if match is not None:
                return str(match)
        return None

    def _parse_float_sequence(
        self,
        value: Any,
        length: int,
        column_name: str,
    ) -> np.ndarray:
        """Parse one metadata value into a fixed-length float sequence."""
        parsed: np.ndarray
        if value is None:
            parsed = np.array([], dtype=np.float32)
        elif isinstance(value, str):
            text = value.strip()
            if text.lower() in {"", "nan", "none", "null"}:
                parsed = np.array([], dtype=np.float32)
            else:
                parsed = self._parse_float_string(text)
        elif isinstance(value, torch.Tensor):
            parsed = value.detach().cpu().numpy().astype(np.float32).reshape(-1)
        elif isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            parsed = np.asarray(value, dtype=np.float32).reshape(-1)
        else:
            try:
                parsed = np.array([float(value)], dtype=np.float32)
            except (TypeError, ValueError):
                parsed = np.array([], dtype=np.float32)

        parsed = parsed.astype(np.float32).reshape(-1)
        if parsed.size == 0:
            raise ValueError(f"Column {column_name!r} does not contain any values")
        if parsed.size < length:
            raise ValueError(
                f"Column {column_name!r} contains {parsed.size} values; "
                f"expected at least {length}"
            )
        if parsed.size > length:
            parsed = parsed[:length]
        if not np.isfinite(parsed).all():
            raise RuntimeError(f"Non-finite values found in column {column_name!r}")
        return parsed

    def _parse_float_string(self, text: str) -> np.ndarray:
        """Parse a JSON-like or delimiter-separated numeric string."""
        if text.startswith("[") and text.endswith("]"):
            try:
                loaded = json.loads(text)
                return np.asarray(loaded, dtype=np.float32).reshape(-1)
            except json.JSONDecodeError:
                pass
        cleaned = (
            text.replace("[", " ")
            .replace("]", " ")
            .replace(";", " ")
            .replace("|", " ")
            .replace("\n", " ")
        )
        cleaned = cleaned.replace(",", " ")
        return np.fromstring(cleaned, sep=" ", dtype=np.float32)

    def _parse_prefixed_float_columns(
        self,
        row: pd.Series,
        prefixes: tuple[str, ...],
        label: str,
    ) -> np.ndarray | None:
        """Parse sequences stored as one scalar column per forecast hour."""
        matches: list[tuple[int, str]] = []
        for column in row.index:
            lower = str(column).lower()
            if any(lower.startswith(prefix.lower()) for prefix in prefixes):
                numbers = re.findall(r"\d+", lower)
                order = int(numbers[-1]) if numbers else len(matches)
                matches.append((order, str(column)))
        if not matches:
            return None
        columns = [column for _order, column in sorted(matches, key=lambda item: item[0])]
        values = np.asarray([row[column] for column in columns], dtype=np.float32)
        if values.size < self.output_length:
            raise ValueError(
                f"Only found {values.size} {label} columns; expected "
                f"{self.output_length}"
            )
        values = values[: self.output_length]
        if not np.isfinite(values).all():
            raise RuntimeError(f"Non-finite values found in {label} columns")
        return values.astype(np.float32)

    def _load_required_sequence(
        self,
        row: pd.Series,
        candidates: tuple[str, ...],
        prefixes: tuple[str, ...],
        label: str,
    ) -> np.ndarray:
        """Load a required forecasting sequence from metadata."""
        column = self._find_column(row, candidates)
        if column is not None:
            return self._parse_float_sequence(row[column], self.output_length, column)

        values = self._parse_prefixed_float_columns(row, prefixes, label)
        if values is not None:
            return values

        available = ", ".join(str(column) for column in row.index)
        expected = ", ".join(candidates)
        raise KeyError(
            f"No {label} sequence found. Expected one of: {expected}, "
            f"or prefixed scalar columns. Available columns: {available}"
        )

    def _load_optional_sequence(
        self,
        row: pd.Series,
        candidates: tuple[str, ...],
        prefixes: tuple[str, ...],
    ) -> np.ndarray | None:
        """Load an optional forecasting sequence from metadata."""
        column = self._find_column(row, candidates)
        if column is not None:
            return self._parse_float_sequence(row[column], self.output_length, column)
        return self._parse_prefixed_float_columns(row, prefixes, "target GHI")

    def _metadata_contains_forecasting_targets(self, row: pd.Series) -> bool:
        """Return whether target sequences are embedded directly in metadata."""
        return (
            self._find_column(row, self.CSI_TARGET_COLUMNS) is not None
            or self._parse_prefixed_float_columns(
                row,
                self.CSI_TARGET_PREFIXES,
                "target CSI",
            )
            is not None
        )

    def _location_columns(self, location: str) -> dict[str, str]:
        """Return hourly CSV column names for one location."""
        columns = {
            "csi": f"CSI_{location}",
            "ghi": f"GHI_{location}",
            "clear": f"GHI_clear_{location}",
        }
        if self.hourly_df is None:
            raise RuntimeError("Hourly CSV dataframe is not loaded")
        missing = [column for column in columns.values() if column not in self.hourly_df.columns]
        if missing:
            available = ", ".join(self.hourly_df.columns)
            raise KeyError(
                f"Missing hourly CSV columns for location={location!r}: {missing}. "
                f"Available columns: {available}"
            )
        return columns

    def _get_hourly_row(self, timestamp: pd.Timestamp) -> pd.Series | None:
        """Return one hourly CSV row for a timestamp, or None if absent."""
        if self.hourly_df is None:
            raise RuntimeError("Hourly CSV dataframe is not loaded")
        try:
            row = self.hourly_df.loc[timestamp]
        except KeyError:
            return None
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return row

    def _get_hourly_value(
        self,
        location_columns: dict[str, str],
        timestamp: pd.Timestamp,
        key: str,
    ) -> float | None:
        """Return one finite value from the hourly CSV."""
        row = self._get_hourly_row(timestamp)
        if row is None:
            return None
        value = row[location_columns[key]]
        if pd.isna(value):
            return None
        value = float(value)
        if not np.isfinite(value):
            return None
        return value

    def _target_timestamps(self, row: pd.Series) -> list[pd.Timestamp]:
        """Return the configured target-hour timestamps for one sample."""
        target_day = pd.Timestamp(row.target_day)
        return [
            target_day + pd.Timedelta(hours=4 + pos)
            for pos in range(self.output_length)
        ]

    def _get_solar_elevation(self, location: str, timestamp: pd.Timestamp) -> float | None:
        """Return solar elevation in degrees for a location/timestamp if available."""
        if self.elevation_df is not None and location in self.elevation_df.columns:
            try:
                value = self.elevation_df.loc[timestamp, location]
            except KeyError:
                value = None
            if isinstance(value, pd.Series):
                value = value.iloc[0]
            if value is not None and not pd.isna(value):
                value_float = float(value)
                if np.isfinite(value_float):
                    return value_float

        if self.hourly_df is not None:
            row = self._get_hourly_row(timestamp)
            column = f"elevation_{location}"
            if row is not None and column in row.index and not pd.isna(row[column]):
                value_float = float(row[column])
                if np.isfinite(value_float):
                    return value_float
        return None

    def _location_coordinates(self, location: str) -> tuple[float, float]:
        """Return latitude and longitude, or zeros when station metadata is unavailable."""
        if self.locations_df is None or location not in self.locations_df.index:
            return 0.0, 0.0
        row = self.locations_df.loc[location]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        latitude = float(row["latitude"]) if not pd.isna(row["latitude"]) else 0.0
        longitude = float(row["longitude"]) if not pd.isna(row["longitude"]) else 0.0
        if not np.isfinite(latitude):
            latitude = 0.0
        if not np.isfinite(longitude):
            longitude = 0.0
        return latitude, longitude

    def _load_auxiliary_features(self, row: pd.Series) -> dict[str, Any]:
        """Build per-forecast-hour auxiliary features for readout conditioning."""
        if self.hourly_df is None:
            raise RuntimeError("Hourly CSV dataframe is required for auxiliary features")

        location = str(row.location)
        columns = self._location_columns(location)
        timestamps = self._target_timestamps(row)
        latitude, longitude = self._location_coordinates(location)
        latitude_scaled = latitude / 90.0
        longitude_scaled = longitude / 180.0

        features = np.zeros(
            (self.output_length, len(self.AUXILIARY_FEATURE_NAMES)),
            dtype=np.float32,
        )
        previous_day_csi = np.zeros(self.output_length, dtype=np.float32)
        solar_elevation = np.zeros(self.output_length, dtype=np.float32)

        for pos, timestamp in enumerate(timestamps):
            prev_csi = self._get_hourly_value(
                columns,
                timestamp - pd.Timedelta(days=1),
                "csi",
            )
            clear_value = self._get_hourly_value(columns, timestamp, "clear")
            elevation_value = self._get_solar_elevation(location, timestamp)

            hour = timestamp.hour + timestamp.minute / 60.0
            day_index = timestamp.dayofyear - 1
            previous_day_csi[pos] = 0.0 if prev_csi is None else float(prev_csi)
            solar_elevation[pos] = 0.0 if elevation_value is None else float(elevation_value)

            features[pos] = np.asarray(
                [
                    previous_day_csi[pos],
                    (0.0 if clear_value is None else float(clear_value)) / 1000.0,
                    solar_elevation[pos] / 90.0,
                    latitude_scaled,
                    longitude_scaled,
                    np.sin(2.0 * np.pi * hour / 24.0),
                    np.cos(2.0 * np.pi * hour / 24.0),
                    np.sin(2.0 * np.pi * day_index / 366.0),
                    np.cos(2.0 * np.pi * day_index / 366.0),
                ],
                dtype=np.float32,
            )

        if not np.isfinite(features).all():
            raise RuntimeError(
                f"Non-finite auxiliary features for location={location} "
                f"target_day={row.target_day}"
            )

        return {
            "auxiliary_features": torch.from_numpy(features),
            "aux_features": torch.from_numpy(features),
            "auxiliary_feature_names": list(self.AUXILIARY_FEATURE_NAMES),
            "previous_day_csi": torch.from_numpy(previous_day_csi.astype(np.float32)),
            "solar_elevation": torch.from_numpy(solar_elevation.astype(np.float32)),
        }

    def _persistence_impute_csi(
        self,
        location_columns: dict[str, str],
        timestamp: pd.Timestamp,
    ) -> float | None:
        """Use previous-day CSI when the target timestamp is missing."""
        return self._get_hourly_value(
            location_columns,
            timestamp - pd.Timedelta(days=1),
            "csi",
        )

    def _interpolate_csi(
        self,
        timestamps: list[pd.Timestamp],
        values: np.ndarray,
    ) -> np.ndarray:
        """Linearly interpolate missing CSI values within one daylight sequence."""
        values = np.asarray(values, dtype=np.float32)
        if not np.isnan(values).any():
            return values

        valid_positions = np.where(~np.isnan(values))[0]
        if valid_positions.size == 0:
            return values

        for pos in np.where(np.isnan(values))[0]:
            left = valid_positions[valid_positions < pos]
            right = valid_positions[valid_positions > pos]
            if left.size and right.size:
                left_pos = left[-1]
                right_pos = right[0]
                left_hour = float(timestamps[left_pos].hour)
                right_hour = float(timestamps[right_pos].hour)
                if right_hour == left_hour:
                    values[pos] = values[left_pos]
                else:
                    values[pos] = np.interp(
                        float(timestamps[pos].hour),
                        [left_hour, right_hour],
                        [values[left_pos], values[right_pos]],
                    )
        return values

    def _load_forecasting_targets_from_csv(self, row: pd.Series) -> dict[str, torch.Tensor]:
        """Load target CSI, target GHI, and clear-sky GHI from the hourly CSV."""
        location = str(row.location)
        columns = self._location_columns(location)
        target_mask = (
            self._parse_mask(row["target_mask"])
            if "target_mask" in row.index
            else np.zeros(self.output_length, dtype=np.bool_)
        )
        timestamps = self._target_timestamps(row)

        target_csi = np.full(self.output_length, np.nan, dtype=np.float32)
        target_ghi = np.zeros(self.output_length, dtype=np.float32)
        clear_sky_ghi = np.zeros(self.output_length, dtype=np.float32)

        for pos, timestamp in enumerate(timestamps):
            if target_mask[pos]:
                continue
            csi_value = self._get_hourly_value(columns, timestamp, "csi")
            ghi_value = self._get_hourly_value(columns, timestamp, "ghi")
            clear_value = self._get_hourly_value(columns, timestamp, "clear")
            if csi_value is not None:
                target_csi[pos] = csi_value
            if ghi_value is not None:
                target_ghi[pos] = ghi_value
            if clear_value is not None:
                clear_sky_ghi[pos] = clear_value

        for pos in np.where(np.isnan(target_csi))[0]:
            value = self._persistence_impute_csi(columns, timestamps[pos])
            if value is not None:
                target_csi[pos] = value

        target_csi = self._interpolate_csi(timestamps, target_csi)
        target_csi[np.isnan(target_csi)] = 0.0

        missing_ghi = target_ghi == 0.0
        target_ghi[missing_ghi] = target_csi[missing_ghi] * clear_sky_ghi[missing_ghi]

        if not np.isfinite(target_csi).all():
            raise RuntimeError(
                f"Non-finite CSI target for location={location} target_day={row.target_day}"
            )
        if not np.isfinite(target_ghi).all():
            raise RuntimeError(
                f"Non-finite target GHI for location={location} target_day={row.target_day}"
            )
        if not np.isfinite(clear_sky_ghi).all():
            raise RuntimeError(
                f"Non-finite clear-sky GHI for location={location} target_day={row.target_day}"
            )

        return {
            "target": torch.from_numpy(target_csi.astype(np.float32)),
            "target_csi": torch.from_numpy(target_csi.astype(np.float32)),
            "clear_sky_ghi": torch.from_numpy(clear_sky_ghi.astype(np.float32)),
            "clear_ghi": torch.from_numpy(clear_sky_ghi.astype(np.float32)),
            "target_ghi": torch.from_numpy(target_ghi.astype(np.float32)),
            "target_timestamp": [str(timestamp) for timestamp in timestamps],
            "target_mask": torch.tensor(target_mask, dtype=torch.bool),
        }

    def _load_forecasting_targets(self, row: pd.Series) -> dict[str, torch.Tensor]:
        """Return CSI target and clear-sky GHI tensors for one sample."""
        if not self._metadata_contains_forecasting_targets(row):
            return self._load_forecasting_targets_from_csv(row)

        target_csi = self._load_required_sequence(
            row=row,
            candidates=self.CSI_TARGET_COLUMNS,
            prefixes=self.CSI_TARGET_PREFIXES,
            label="target CSI",
        )
        clear_sky_ghi = self._load_required_sequence(
            row=row,
            candidates=self.CLEAR_SKY_GHI_COLUMNS,
            prefixes=self.CLEAR_SKY_GHI_PREFIXES,
            label="clear-sky GHI",
        )
        target_ghi = self._load_optional_sequence(
            row=row,
            candidates=self.GHI_TARGET_COLUMNS,
            prefixes=self.GHI_TARGET_PREFIXES,
        )
        if target_ghi is None:
            target_ghi = target_csi * clear_sky_ghi

        return {
            "target": torch.from_numpy(target_csi),
            "target_csi": torch.from_numpy(target_csi),
            "clear_sky_ghi": torch.from_numpy(clear_sky_ghi),
            "clear_ghi": torch.from_numpy(clear_sky_ghi),
            "target_ghi": torch.from_numpy(target_ghi.astype(np.float32)),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.meta.iloc[index]
        input_indices = self._indices_from_row(
            row,
            sequence_column="input_indices",
            start_column="input_start",
            end_column="input_end",
            length=self.sequence_length,
        )
        image_mask = (
            self._parse_mask(row["image_mask"])
            if "image_mask" in row.index
            else np.zeros(self.sequence_length, dtype=np.bool_)
        )

        input_zarr = self._get_zarr(row.input_zarr)
        if "X" not in input_zarr:
            raise KeyError(f"Missing 'X' array in {row.input_zarr}")

        satellite = self._load_frames(input_zarr, input_indices, self.sequence_length)

        item = {
            "satellite": torch.from_numpy(satellite),
            "image_mask": torch.tensor(image_mask, dtype=torch.bool),
            "sample_id": int(row.sample_id) if "sample_id" in row.index else int(index),
            "location": str(row.location),
            "input_day": str(row.input_day),
            "target_day": str(row.target_day),
            "channels": self.channels,
        }

        if self.include_target:
            item.update(self._load_forecasting_targets(row))
        if self.include_auxiliary_features:
            item.update(self._load_auxiliary_features(row))

        return item

    def _load_frames(
        self,
        zarr_group: Any,
        indices: np.ndarray,
        length: int,
    ) -> np.ndarray:
        images: list[np.ndarray] = []
        for input_idx in indices[:length]:
            if int(input_idx) < 0:
                frame = np.zeros(
                    (self.expected_channels, self.image_size, self.image_size),
                    dtype=np.float32,
                )
            else:
                frame = np.asarray(zarr_group["X"][int(input_idx)], dtype=np.float32)
            if frame.shape != (self.expected_channels, self.image_size, self.image_size):
                raise ValueError(
                    f"Unexpected frame shape {frame.shape}; expected "
                    f"({self.expected_channels}, {self.image_size}, {self.image_size})"
                )
            images.append(frame)

        frames = np.stack(images).astype(np.float32)

        if self.normalize and self.norm is not None:
            mean = np.array(self.norm["channel_mean"], dtype=np.float32).reshape(
                1, self.expected_channels, 1, 1
            )
            std = np.array(self.norm["channel_std"], dtype=np.float32).reshape(
                1, self.expected_channels, 1, 1
            )
            frames = (frames - mean) / (std + 1e-6)

        if np.isnan(frames).any():
            raise RuntimeError("NaN detected in satellite sequence")
        return frames
