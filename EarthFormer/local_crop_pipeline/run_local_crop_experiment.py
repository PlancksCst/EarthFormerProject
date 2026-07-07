"""Run the full station-centered local crop experiment pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full local crop experiment.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--metadata-filename", type=str, default=None)
    parser.add_argument("--hourly-csv", type=Path, default=None)
    parser.add_argument("--elevation-csv", type=Path, default=None)
    parser.add_argument("--locations-csv", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--local-crop-size", type=int, default=64)
    parser.add_argument("--crop-padding-mode", choices=("edge", "reflect"), default="edge")
    parser.add_argument("--loss", type=str, default="masked_weighted_huber")
    parser.add_argument("--huber-beta", type=float, default=0.1)
    parser.add_argument("--cloudy-weight", type=float, default=1.0)
    parser.add_argument("--crop-lat-min", type=float, default=33.0)
    parser.add_argument("--crop-lat-max", type=float, default=34.7)
    parser.add_argument("--crop-lon-min", type=float, default=35.0)
    parser.add_argument("--crop-lon-max", type=float, default=36.7)
    parser.add_argument("--earthformer-checkpoint", type=Path, default=None)
    parser.add_argument("--pretrained-earthformer-checkpoint", type=Path, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--use-auxiliary-features", action="store_true")
    return parser


def optional(args: argparse.Namespace, pairs: list[tuple[str, object | None]]) -> list[str]:
    output: list[str] = []
    for flag, value in pairs:
        if value is not None:
            output.extend([flag, str(value)])
    return output


def run(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)


def main() -> None:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    common_bounds = [
        "--local-crop-size",
        str(args.local_crop_size),
        "--crop-lat-min",
        str(args.crop_lat_min),
        "--crop-lat-max",
        str(args.crop_lat_max),
        "--crop-lon-min",
        str(args.crop_lon_min),
        "--crop-lon-max",
        str(args.crop_lon_max),
    ]
    common_files = optional(
        args,
        [
            ("--locations-csv", args.locations_csv),
            ("--metadata-filename", args.metadata_filename),
            ("--hourly-csv", args.hourly_csv),
            ("--elevation-csv", args.elevation_csv),
        ],
    )

    run(
        [
            sys.executable,
            str(SCRIPT_DIR / "station_crop_mapping.py"),
            "--output-dir",
            str(args.output_dir),
            *common_bounds,
            *optional(args, [("--locations-csv", args.locations_csv)]),
        ]
    )
    run(
        [
            sys.executable,
            str(SCRIPT_DIR / "plot_local_crop_sanity.py"),
            "--dataset-root",
            str(args.dataset_root),
            "--split",
            args.val_split,
            "--output-dir",
            str(args.output_dir / "sanity"),
            "--num-samples",
            "20",
            "--crop-padding-mode",
            args.crop_padding_mode,
            *common_bounds,
            *optional(args, [("--locations-csv", args.locations_csv)]),
        ]
    )
    train_output = args.output_dir / "local_crop_cnn_gru"
    train_checkpoint = args.checkpoint_dir / "local_crop_cnn_gru"
    train_command = [
        sys.executable,
        str(SCRIPT_DIR / "train_local_crop_model.py"),
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
        str(train_output),
        "--checkpoint-dir",
        str(train_checkpoint),
        "--crop-padding-mode",
        args.crop_padding_mode,
        "--loss",
        args.loss,
        "--huber-beta",
        str(args.huber_beta),
        "--cloudy-weight",
        str(args.cloudy_weight),
        *common_bounds,
        *common_files,
    ]
    if args.amp:
        train_command.append("--amp")
    if args.use_auxiliary_features:
        train_command.append("--use-auxiliary-features")
    run(train_command)

    eval_command = [
        sys.executable,
        str(SCRIPT_DIR / "evaluate_local_crop_against_baselines.py"),
        "--dataset-root",
        str(args.dataset_root),
        "--checkpoint",
        str(train_checkpoint / "best.pt"),
        "--train-split",
        args.train_split,
        "--eval-split",
        args.val_split,
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--output-dir",
        str(args.output_dir / "evaluation"),
        "--crop-padding-mode",
        args.crop_padding_mode,
        *common_bounds,
        *common_files,
        *optional(
            args,
            [
                ("--earthformer-checkpoint", args.earthformer_checkpoint),
                ("--pretrained-earthformer-checkpoint", args.pretrained_earthformer_checkpoint),
            ],
        ),
    ]
    if args.use_auxiliary_features:
        eval_command.append("--use-auxiliary-features")
    run(eval_command)


if __name__ == "__main__":
    main()
