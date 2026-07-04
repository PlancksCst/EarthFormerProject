"""Run EarthFormer inference on a SEVIRI dataset split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PREP_MODELS_ROOT = PROJECT_ROOT.parent
if str(PREP_MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(PREP_MODELS_ROOT))

from configs.config import build_arg_parser, config_from_args  # noqa: E402
from datasets.seviri_dataset import build_dataloader  # noqa: E402
from models.model import build_training_model  # noqa: E402
from training.checkpoint import load_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse inference arguments."""
    parser = build_arg_parser()
    parser.description = "Run EarthFormer inference on SEVIRI imagery."
    parser.add_argument("--split", default="test")
    parser.add_argument("--model-checkpoint", type=Path, default=None)
    parser.add_argument("--output-file", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """Run inference and save prediction/latent tensors."""
    args = parse_args()
    config = config_from_args(args)
    config.prepare_directories()
    device = torch.device(config.resolved_device())

    loader = build_dataloader(
        config=config,
        split=args.split,
        include_target=False,
        shuffle=False,
    )
    model = build_training_model(config).to(device)
    if args.model_checkpoint is not None:
        checkpoint = load_checkpoint(args.model_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
    model.eval()

    batch = next(iter(loader))
    inputs = batch["satellite"].to(device, non_blocking=True)
    with torch.no_grad():
        result = model(inputs, return_latent=True)

    output_file = args.output_file or (config.output_dir / "inference_sample.pt")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "prediction": result["prediction"].cpu(),
            "pre_head_latent": result["pre_head_latent"].cpu(),
            "sample_id": batch["sample_id"],
            "location": batch["location"],
            "input_day": batch["input_day"],
            "target_day": batch["target_day"],
        },
        output_file,
    )
    print(f"Saved inference output to {output_file}")


if __name__ == "__main__":
    main()
