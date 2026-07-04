"""Compatibility layer for using official EarthFormer with SEVIRI data."""

from .model import (
    EarthFormerSEVIRIMigration,
    build_seviri_earthformer,
    ensure_sevir_pretrained_checkpoint,
    load_sevir_pretrained_weights,
)
from .seviri_dataset import SEVIRIImageSequenceDataset

__all__ = [
    "EarthFormerSEVIRIMigration",
    "SEVIRIImageSequenceDataset",
    "build_seviri_earthformer",
    "ensure_sevir_pretrained_checkpoint",
    "load_sevir_pretrained_weights",
]
