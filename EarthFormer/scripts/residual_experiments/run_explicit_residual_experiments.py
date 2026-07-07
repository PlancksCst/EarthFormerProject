"""Run explicit residual EarthFormer SEVIRI experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
TRAIN_MODULE = "EarthFormer.training.train"
DEFAULT_LOSSES = ("masked_residual_weighted_huber",)
ALL_LOSSES = (
    "masked_residual_huber",
    "masked_residual_weighted_huber",
    "masked_residual_ramp_weighted_huber",
)
PRESETS = ("explicit_residual_head", "explicit_residual_gated")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run explicit residual CSI experiments.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--pretrained-earthformer-checkpoint", type=Path, required=True)
    parser.add_argument("--residual-scale", type=float, default=0.3)
    parser.add_argument("--huber-beta", type=float, default=0.1)
    parser.add_argument("--cloudy-weight", type=float, default=1.0)
    parser.add_argument("--ramp-weight", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true", default=True)
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--all-loss-variants", action="store_true")
    return parser.parse_args()


def validate_checkpoint(path: Path) -> None:
    """Fail before launching child runs when the checkpoint is obviously unusable."""
    if not path.exists():
        raise FileNotFoundError(f"Missing pretrained EarthFormer checkpoint: {path}")
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"Pretrained EarthFormer checkpoint is empty or not a file: {path}")


def run_one(args: argparse.Namespace, preset: str, residual_loss: str) -> None:
    run_name = f"{preset}_{residual_loss}"
    output_dir = args.output_dir / run_name
    checkpoint_dir = args.checkpoint_dir / run_name
    command = [
        sys.executable,
        "-m",
        TRAIN_MODULE,
        "--dataset-root",
        str(args.dataset_root),
        "--train-split",
        args.train_split,
        "--val-split",
        args.val_split,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--output-dir",
        str(output_dir),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--pretrained-earthformer-checkpoint",
        str(args.pretrained_earthformer_checkpoint),
        "--fix-preset",
        preset,
        "--residual-loss",
        residual_loss,
        "--residual-scale",
        str(args.residual_scale),
        "--huber-beta",
        str(args.huber_beta),
        "--cloudy-weight",
        str(args.cloudy_weight),
        "--ramp-weight",
        str(args.ramp_weight),
    ]
    if args.amp:
        command.append("--amp")
    if args.unfreeze_backbone:
        command.append("--unfreeze-backbone")
    else:
        command.append("--freeze-backbone")
    if preset == "explicit_residual_gated":
        command.append("--use-auxiliary-features")

    print("Running:", " ".join(command))
    subprocess.run(command, cwd=str(PROJECT_ROOT.parent), check=True)


def main() -> None:
    args = parse_args()
    validate_checkpoint(args.pretrained_earthformer_checkpoint)
    losses = ALL_LOSSES if args.all_loss_variants else DEFAULT_LOSSES
    for preset in PRESETS:
        for residual_loss in losses:
            run_one(args, preset=preset, residual_loss=residual_loss)


if __name__ == "__main__":
    main()
