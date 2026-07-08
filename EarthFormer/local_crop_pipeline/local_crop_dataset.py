"""Dataset wrapper that crops station-centered 64x64 SEVIRI windows."""

from __future__ import annotations

import sys
import inspect
import importlib.util
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREP_MODELS_ROOT = PROJECT_ROOT.parent
for path in (PROJECT_ROOT, PREP_MODELS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from local_crop_pipeline.station_crop_mapping import (  # noqa: E402
    CropBounds,
    build_station_mapping,
    mapping_by_location,
)


def _load_project_seviri_dataset_class() -> type:
    """Load the EarthFormer-local dataset adapter, avoiding duplicate package shadowing."""
    module_path = PROJECT_ROOT / "earthformer_migration" / "seviri_dataset.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Missing EarthFormer dataset adapter: {module_path}")
    spec = importlib.util.spec_from_file_location(
        "_earthformer_local_seviri_dataset",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import dataset adapter from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SEVIRIImageSequenceDataset


SEVIRIImageSequenceDataset = _load_project_seviri_dataset_class()


def normalise_location(value: Any) -> str:
    """Return normalized location key."""
    return str(value).strip().upper()


class LocalCropDataset(Dataset):
    """Wrap the existing SEVIRI dataset and replace satellite images with local crops."""

    def __init__(
        self,
        dataset_root: str | Path,
        split: str = "train",
        local_crop_size: int = 64,
        crop_padding_mode: str = "edge",
        crop_bounds: CropBounds | None = None,
        locations_csv: str | Path | None = None,
        include_target: bool = True,
        include_auxiliary_features: bool = False,
        skip_bad_samples: bool = True,
        sequence_length: int = 13,
        output_length: int = 13,
        image_size: int = 200,
        expected_channels: int = 7,
        normalize: bool = True,
        metadata_filename: str | None = None,
        hourly_csv: str | Path | None = None,
        elevation_csv: str | Path | None = None,
    ) -> None:
        if local_crop_size <= 0:
            raise ValueError("local_crop_size must be positive")
        if crop_padding_mode not in {"edge", "reflect"}:
            raise ValueError("crop_padding_mode must be 'edge' or 'reflect'")
        self.local_crop_size = int(local_crop_size)
        self.crop_padding_mode = crop_padding_mode
        self.expected_image_size = int(image_size)
        self.skip_bad_samples = bool(skip_bad_samples)
        self.bad_indices: set[int] = set()
        self.replacement_index: dict[int, int] = {}
        rows = build_station_mapping(
            locations_csv=Path(locations_csv) if locations_csv is not None else None,
            bounds=crop_bounds or CropBounds(height=image_size, width=image_size),
            local_crop_size=local_crop_size,
        )
        self.station_mapping_rows = rows
        self.station_mapping = mapping_by_location(rows)
        dataset_kwargs = {
            "dataset_root": str(dataset_root),
            "split": split,
            "sequence_length": sequence_length,
            "output_length": output_length,
            "image_size": image_size,
            "expected_channels": expected_channels,
            "normalize": normalize,
            "include_target": include_target,
            "metadata_filename": metadata_filename,
            "hourly_csv": str(hourly_csv) if hourly_csv is not None else None,
            "elevation_csv": str(elevation_csv) if elevation_csv is not None else None,
            "locations_csv": str(locations_csv) if locations_csv is not None else None,
            "include_auxiliary_features": include_auxiliary_features,
        }
        supported = set(inspect.signature(SEVIRIImageSequenceDataset.__init__).parameters)
        dataset_kwargs = {
            key: value
            for key, value in dataset_kwargs.items()
            if key in supported
        }
        self.base_dataset = SEVIRIImageSequenceDataset(**dataset_kwargs)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _crop_satellite(
        self,
        satellite: torch.Tensor,
        center_y: int,
        center_x: int,
    ) -> tuple[torch.Tensor, dict[str, int]]:
        """Crop and pad a `(T,C,200,200)` tensor around a station pixel."""
        if satellite.ndim != 4:
            raise ValueError(f"Expected satellite tensor (T,C,H,W), got {tuple(satellite.shape)}")
        _, _, height, width = satellite.shape
        if height != self.expected_image_size or width != self.expected_image_size:
            raise ValueError(
                "LocalCropDataset expects the shared 200x200 image pipeline as input; "
                f"got HxW={height}x{width}."
            )
        half = self.local_crop_size // 2
        y0_raw = int(center_y) - half
        x0_raw = int(center_x) - half
        y1_raw = y0_raw + self.local_crop_size
        x1_raw = x0_raw + self.local_crop_size
        y0 = max(0, y0_raw)
        x0 = max(0, x0_raw)
        y1 = min(height, y1_raw)
        x1 = min(width, x1_raw)
        if y0 >= y1 or x0 >= x1:
            raise RuntimeError(
                f"Invalid crop window for center=({center_y},{center_x}) "
                f"and crop_size={self.local_crop_size}"
            )
        crop = satellite[..., y0:y1, x0:x1]
        pad_left = x0 - x0_raw
        pad_right = x1_raw - x1
        pad_top = y0 - y0_raw
        pad_bottom = y1_raw - y1
        if any(value > 0 for value in (pad_left, pad_right, pad_top, pad_bottom)):
            mode = "replicate" if self.crop_padding_mode == "edge" else "reflect"
            crop = F.pad(crop, (pad_left, pad_right, pad_top, pad_bottom), mode=mode)
        if tuple(crop.shape[-2:]) != (self.local_crop_size, self.local_crop_size):
            raise RuntimeError(
                f"Local crop has wrong shape {tuple(crop.shape)}; "
                f"expected spatial {self.local_crop_size}x{self.local_crop_size}."
            )
        return crop.contiguous(), {
            "local_crop_y0": y0,
            "local_crop_y1": y1,
            "local_crop_x0": x0,
            "local_crop_x1": x1,
        }

    def _sample_metadata_for_log(self, index: int) -> str:
        """Return compact sample metadata for skip/error logs."""
        meta = getattr(self.base_dataset, "meta", None)
        if meta is None:
            return f"index={index}"
        try:
            row = meta.iloc[index]
        except Exception:
            return f"index={index}"
        fields = ["sample_id", "location", "input_day", "target_day", "input_zarr"]
        parts = [f"index={index}"]
        for field in fields:
            if field in row.index:
                parts.append(f"{field}={row[field]}")
        return ", ".join(parts)

    def _load_base_item_with_skip(self, index: int) -> tuple[dict[str, Any], int, int | None]:
        """Load a base item, optionally skipping samples that fail in the base dataset."""
        dataset_length = len(self.base_dataset)
        first_error: Exception | None = None
        cached_replacement = self.replacement_index.get(index)
        if cached_replacement is not None:
            try:
                return dict(self.base_dataset[cached_replacement]), cached_replacement, index
            except Exception as exc:
                first_error = exc
                self.bad_indices.add(cached_replacement)
                self.replacement_index.pop(index, None)

        for offset in range(dataset_length):
            candidate_index = (index + offset) % dataset_length
            if candidate_index in self.bad_indices:
                continue
            try:
                item = dict(self.base_dataset[candidate_index])
            except Exception as exc:
                if first_error is None:
                    first_error = exc
                self.bad_indices.add(candidate_index)
                if not self.skip_bad_samples:
                    raise
                continue
            skipped_from = index if candidate_index != index else None
            if skipped_from is not None:
                self.replacement_index[index] = candidate_index
            return item, candidate_index, skipped_from
        raise RuntimeError(
            "All base dataset samples failed while trying to skip bad local-crop samples."
        ) from first_error

    def __getitem__(self, index: int) -> dict[str, Any]:
        item, loaded_index, skipped_from = self._load_base_item_with_skip(index)
        location_value = item.get("location")
        if location_value is None:
            raise KeyError(f"Sample index {loaded_index} is missing location metadata.")
        location = normalise_location(location_value)
        mapping = self.station_mapping.get(location)
        if mapping is None:
            raise KeyError(f"No station pixel mapping found for location={location!r}.")
        center_y = int(mapping["pixel_y"])
        center_x = int(mapping["pixel_x"])
        satellite = item.get("satellite")
        if not isinstance(satellite, torch.Tensor):
            raise KeyError("Base dataset item does not contain a satellite tensor.")
        crop, crop_meta = self._crop_satellite(satellite, center_y=center_y, center_x=center_x)
        item["satellite"] = crop
        item["local_crop_center_y"] = torch.tensor(center_y, dtype=torch.long)
        item["local_crop_center_x"] = torch.tensor(center_x, dtype=torch.long)
        item["local_crop_size"] = torch.tensor(self.local_crop_size, dtype=torch.long)
        item["local_crop_base_index"] = torch.tensor(loaded_index, dtype=torch.long)
        item["local_crop_skipped_from_index"] = torch.tensor(
            -1 if skipped_from is None else int(skipped_from),
            dtype=torch.long,
        )
        for key, value in crop_meta.items():
            item[key] = torch.tensor(value, dtype=torch.long)
        return item


def build_local_crop_dataloader(
    dataset: LocalCropDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: str = "auto",
) -> DataLoader:
    """Build a dataloader for local crop experiments."""
    pin_memory = device == "cuda" or device == "auto"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
