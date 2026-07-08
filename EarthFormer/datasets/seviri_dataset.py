"""Dataset adapter for EarthFormer SEVIRI training."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREP_MODELS_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PREP_MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(PREP_MODELS_ROOT))

from configs.config import TrainingConfig  # noqa: E402
from earthformer_migration.seviri_dataset import SEVIRIImageSequenceDataset  # noqa: E402


class SkipBadSampleDataset(Dataset):
    """Skip and log base dataset samples that fail during frame loading."""

    def __init__(self, base_dataset: Dataset, enabled: bool = True) -> None:
        self.base_dataset = base_dataset
        self.enabled = bool(enabled)
        self.bad_indices: set[int] = set()
        self.replacement_index: dict[int, int] = {}

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _sample_metadata_for_log(self, index: int) -> str:
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

    def __getitem__(self, index: int) -> dict[str, Any]:
        dataset_length = len(self.base_dataset)
        first_error: Exception | None = None
        cached_replacement = self.replacement_index.get(index)
        if cached_replacement is not None:
            try:
                item = dict(self.base_dataset[cached_replacement])
                item["base_index"] = torch.tensor(cached_replacement, dtype=torch.long)
                item["skipped_from_index"] = torch.tensor(index, dtype=torch.long)
                return item
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
                if not self.enabled:
                    raise
                continue
            if candidate_index != index:
                self.replacement_index[index] = candidate_index
            item["base_index"] = torch.tensor(candidate_index, dtype=torch.long)
            item["skipped_from_index"] = torch.tensor(
                -1 if candidate_index == index else index,
                dtype=torch.long,
            )
            return item
        raise RuntimeError("All dataset samples failed while trying to skip bad samples.") from first_error


def build_dataset(config: TrainingConfig, split: str, include_target: bool) -> Dataset:
    """Build a SEVIRI dataset split."""
    dataset_kwargs = dict(
        dataset_root=str(config.dataset_root),
        split=split,
        sequence_length=config.input_length,
        output_length=config.output_length,
        image_size=config.image_size,
        expected_channels=config.input_channels,
        normalize=config.normalize,
        include_target=include_target,
        target_channel_index=config.target_channel_index,
        metadata_filename=config.metadata_filename,
        hourly_csv=str(config.hourly_csv),
        elevation_csv=str(config.elevation_csv),
        locations_csv=str(config.locations_csv),
        include_auxiliary_features=config.use_auxiliary_features,
    )
    supported = set(inspect.signature(SEVIRIImageSequenceDataset.__init__).parameters)
    dataset_kwargs = {key: value for key, value in dataset_kwargs.items() if key in supported}
    base_dataset = SEVIRIImageSequenceDataset(**dataset_kwargs)
    return SkipBadSampleDataset(
        base_dataset,
        enabled=bool(getattr(config, "skip_bad_samples", True)),
    )


def build_dataloader(
    config: TrainingConfig,
    split: str,
    include_target: bool,
    shuffle: bool,
) -> DataLoader:
    """Build a PyTorch DataLoader for a SEVIRI split."""
    dataset = build_dataset(config=config, split=split, include_target=include_target)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.resolved_device().startswith("cuda"),
        drop_last=False,
    )
