"""Validate official EarthFormer on local SEVIRI image sequences."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_PREP_MODELS_DIR = Path(__file__).resolve().parents[1]
if str(_PREP_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(_PREP_MODELS_DIR))

from earthformer_migration.model import (
    EarthFormerSEVIRIMigration,
    DEFAULT_CHECKPOINT_PATH,
    build_seviri_earthformer,
    ensure_sevir_pretrained_checkpoint,
    load_sevir_pretrained_weights,
)
from earthformer_migration.seviri_dataset import SEVIRIImageSequenceDataset


DEFAULT_DATASET_ROOT = Path(
    "/content/drive/MyDrive/EarthFormer/datasets"
)

def _device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = _device_from_arg(args.device)

    if args.skip_download:
        checkpoint_path = os.path.abspath(args.checkpoint)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)
    else:
        checkpoint_path = ensure_sevir_pretrained_checkpoint(args.checkpoint)

    dataset = SEVIRIImageSequenceDataset(
        dataset_root=args.dataset_root,
        split=args.split,
        normalize=not args.no_normalize,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    batch = next(iter(loader))

    model = build_seviri_earthformer()
    load_report = load_sevir_pretrained_weights(model, checkpoint_path=checkpoint_path)
    wrapped = EarthFormerSEVIRIMigration(model).to(device)
    wrapped.eval()

    satellite = batch["satellite"].to(device)
    with torch.no_grad():
        result = wrapped.forward_trace(satellite)

    prediction = result["prediction"]
    latent = result["pre_head_latent"]
    trace = result["trace"]

    summary = {
        "dataset_root": os.path.abspath(args.dataset_root),
        "split": args.split,
        "sample_id": int(batch["sample_id"][0]),
        "location": batch["location"][0],
        "input_day": batch["input_day"][0],
        "target_day": batch["target_day"][0],
        "channels": dataset.channels,
        "input_batch_shape_btchw": tuple(batch["satellite"].shape),
        "device": str(device),
        "trace": {key: value for key, value in trace.items()},
        "prediction_dtype": str(prediction.dtype),
        "latent_dtype": str(latent.dtype),
        "prediction_finite": bool(torch.isfinite(prediction).all().item()),
        "latent_finite": bool(torch.isfinite(latent).all().item()),
        "load_report": load_report.as_dict(),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
