"""Dataset adapter for EarthFormer SEVIRI training."""

from __future__ import annotations

import sys
from pathlib import Path

from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREP_MODELS_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PREP_MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(PREP_MODELS_ROOT))

from configs.config import TrainingConfig  # noqa: E402
from earthformer_migration.seviri_dataset import SEVIRIImageSequenceDataset  # noqa: E402


def build_dataset(config: TrainingConfig, split: str, include_target: bool) -> SEVIRIImageSequenceDataset:
    """Build a SEVIRI dataset split."""
    return SEVIRIImageSequenceDataset(
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
