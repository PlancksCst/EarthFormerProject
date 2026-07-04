"""SEVIRI dataset for EarthFormer CSI forecasting.

The model consumes only the previous-day satellite image sequence. When targets
are requested, the loader returns the next-day CSI sequence and clear-sky GHI
needed for external GHI reconstruction.
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
        basenames = {os.path.basename(path), PureWindowsPath(path).name}
        for basename in basenames:
            basename_joined = os.path.join(self.dataset_root, basename)
            if os.path.exists(basename_joined):
                return basename_joined
        return path

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

    def _load_forecasting_targets(self, row: pd.Series) -> dict[str, torch.Tensor]:
        """Return CSI target and clear-sky GHI tensors for one sample."""
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
        input_indices = self._parse_sequence(row.input_indices)
        image_mask = self._parse_mask(row.image_mask)

        input_zarr = self._get_zarr(row.input_zarr)
        if "X" not in input_zarr:
            raise KeyError(f"Missing 'X' array in {row.input_zarr}")

        satellite = self._load_frames(input_zarr, input_indices, self.sequence_length)

        item = {
            "satellite": torch.from_numpy(satellite),
            "image_mask": torch.tensor(image_mask, dtype=torch.bool),
            "sample_id": int(row.sample_id),
            "location": str(row.location),
            "input_day": str(row.input_day),
            "target_day": str(row.target_day),
            "channels": self.channels,
        }

        if self.include_target:
            item.update(self._load_forecasting_targets(row))

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
