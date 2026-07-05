"""Configuration objects for EarthFormer SEVIRI training."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path


def project_root() -> Path:
    """Return the root directory of the training repository."""
    return Path(__file__).resolve().parents[1]


METADATA_FILENAMES = ("metadata.parquet", "dualet_metadata.parquet")


def has_metadata(path: Path) -> bool:
    """Return whether a directory looks like a prepared SEVIRI dataset root."""
    return path.is_dir() and any((path / name).exists() for name in METADATA_FILENAMES)


def discover_kaggle_dataset_root() -> Path | None:
    """Return a Kaggle dataset root when exactly one mounted dataset matches."""
    kaggle_input = Path("/kaggle/input")
    if not kaggle_input.exists():
        return None
    candidates = [child for child in kaggle_input.iterdir() if has_metadata(child)]
    if len(candidates) == 1:
        return candidates[0]
    return None


def discover_dataset_root() -> Path:
    """Discover a dataset root without hardcoding machine-specific paths."""
    env_value = os.environ.get("EARTHFORMER_DATASET_ROOT")
    if env_value:
        return Path(env_value)

    root = project_root()
    candidates = [
        Path("/content/datasets"),
        Path("/content/drive/MyDrive/EarthFormer/datasets"),
        root / "data",
        root.parent / "data",
        root.parent.parent / "verification_datasets" / "BEST_7_3months",
        root.parent.parent / "verification_datasets" / "BEST_7_full_year",
        root.parent.parent.parent / "verification_datasets" / "BEST_7_3months",
        root.parent.parent.parent / "verification_datasets" / "BEST_7_full_year",
    ]
    kaggle_root = discover_kaggle_dataset_root()
    if kaggle_root is not None:
        candidates.insert(0, kaggle_root)

    for candidate in candidates:
        if has_metadata(candidate):
            return candidate
    return root / "data"


def discover_checkpoint_dir() -> Path:
    """Discover a portable checkpoint directory."""
    env_value = os.environ.get("EARTHFORMER_CHECKPOINT_DIR")
    if env_value:
        return Path(env_value)
    if Path("/content/datasets").exists() or Path("/content").exists():
        return Path("/content/checkpoints")
    return project_root() / "checkpoints"


def discover_output_dir() -> Path:
    """Discover a portable output directory."""
    env_value = os.environ.get("EARTHFORMER_OUTPUT_DIR")
    if env_value:
        return Path(env_value)
    if Path("/content/datasets").exists() or Path("/content").exists():
        return Path("/content/outputs")
    return project_root() / "outputs"


def discover_cams_csv(filename: str, env_name: str) -> Path:
    """Discover a CAMS CSV file across local, Colab, and dataset-mounted layouts."""
    env_value = os.environ.get(env_name)
    if env_value:
        return Path(env_value)

    root = project_root()
    candidates = [
        Path("/content/CAMS") / filename,
        Path("/content/datasets") / filename,
        Path("/content/drive/MyDrive/EarthFormer/CAMS") / filename,
        root.parents[2] / "CAMS" / filename,
        root.parent / "CAMS" / filename,
        root / "data" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return root.parents[2] / "CAMS" / filename


def discover_hourly_csv() -> Path:
    """Discover the hourly CAMS/ground CSI-GHI CSV."""
    return discover_cams_csv("all_locations_hourly.csv", "EARTHFORMER_HOURLY_CSV")


def discover_elevation_csv() -> Path:
    """Discover the hourly solar-elevation CSV."""
    return discover_cams_csv(
        "all_locations_elevation.csv",
        "EARTHFORMER_ELEVATION_CSV",
    )


@dataclass
class TrainingConfig:
    """Runtime configuration for backbone fine-tuning."""

    dataset_root: Path = field(default_factory=discover_dataset_root)
    metadata_filename: str | None = os.environ.get("EARTHFORMER_METADATA_FILE")
    hourly_csv: Path = field(default_factory=discover_hourly_csv)
    elevation_csv: Path = field(default_factory=discover_elevation_csv)
    batch_size: int = int(os.environ.get("EARTHFORMER_BATCH_SIZE", "8"))
    learning_rate: float = float(os.environ.get("EARTHFORMER_LR", "1e-4"))
    weight_decay: float = float(os.environ.get("EARTHFORMER_WEIGHT_DECAY", "1e-4"))
    epochs: int = int(os.environ.get("EARTHFORMER_EPOCHS", "20"))
    num_workers: int = int(os.environ.get("EARTHFORMER_NUM_WORKERS", "2"))
    device: str = os.environ.get("EARTHFORMER_DEVICE", "auto")
    checkpoint_dir: Path = field(default_factory=discover_checkpoint_dir)
    output_dir: Path = field(default_factory=discover_output_dir)
    pretrained_checkpoint: Path | None = None
    resume_checkpoint: Path | None = None
    random_seed: int = int(os.environ.get("EARTHFORMER_SEED", "42"))
    mixed_precision: bool = os.environ.get("EARTHFORMER_MIXED_PRECISION", "0") == "1"
    amp_dtype: str = os.environ.get("EARTHFORMER_AMP_DTYPE", "bf16")
    gradient_clip: float = float(os.environ.get("EARTHFORMER_GRADIENT_CLIP", "1.0"))
    scheduler_t_max: int | None = None
    scheduler_eta_min: float = float(os.environ.get("EARTHFORMER_ETA_MIN", "1e-6"))
    train_split: str = "train"
    val_split: str = "val"
    image_size: int = 200
    input_length: int = int(os.environ.get("EARTHFORMER_INPUT_LENGTH", "13"))
    output_length: int = int(os.environ.get("EARTHFORMER_OUTPUT_LENGTH", "13"))
    input_channels: int = 7
    output_channels: int = 1
    target_channel_index: int = int(os.environ.get("EARTHFORMER_TARGET_CHANNEL", "0"))
    normalize: bool = True
    log_filename: str = "training_log.csv"
    readout_type: str = os.environ.get("EARTHFORMER_READOUT_TYPE", "perceiver_io")
    readout_latent_dim: int = int(os.environ.get("EARTHFORMER_READOUT_LATENT_DIM", "16"))
    query_dimension: int = int(os.environ.get("EARTHFORMER_QUERY_DIM", "64"))
    num_output_queries: int | None = None
    num_attention_heads: int = int(os.environ.get("EARTHFORMER_READOUT_HEADS", "4"))
    readout_dropout: float = float(os.environ.get("EARTHFORMER_READOUT_DROPOUT", "0.1"))
    regression_hidden_dim: int = int(os.environ.get("EARTHFORMER_REGRESSION_HIDDEN", "32"))
    freeze_earthformer: bool = os.environ.get("EARTHFORMER_FREEZE_BACKBONE", "0") == "1"

    def resolved_device(self) -> str:
        """Resolve `auto` into a concrete torch device string."""
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def prepare_directories(self) -> None:
        """Create checkpoint and output directories."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser used by training and inference scripts."""
    parser = argparse.ArgumentParser(description="Fine-tune EarthFormer on SEVIRI imagery.")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--metadata-filename", type=str, default=None)
    parser.add_argument("--hourly-csv", type=Path, default=None)
    parser.add_argument("--elevation-csv", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--pretrained-checkpoint", type=Path, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--gradient-clip", type=float, default=None)
    parser.add_argument("--scheduler-t-max", type=int, default=None)
    parser.add_argument("--scheduler-eta-min", type=float, default=None)
    parser.add_argument("--input-length", type=int, default=None)
    parser.add_argument("--output-length", type=int, default=None)
    parser.add_argument("--target-channel-index", type=int, default=None)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--readout-type", type=str, default=None)
    parser.add_argument("--readout-latent-dim", type=int, default=None)
    parser.add_argument("--query-dimension", type=int, default=None)
    parser.add_argument("--num-output-queries", type=int, default=None)
    parser.add_argument("--num-attention-heads", type=int, default=None)
    parser.add_argument("--readout-dropout", type=float, default=None)
    parser.add_argument("--regression-hidden-dim", type=int, default=None)
    parser.add_argument("--freeze-earthformer", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace | None = None) -> TrainingConfig:
    """Build a config from defaults plus command-line overrides."""
    if args is None:
        args = build_arg_parser().parse_args()
    cfg = TrainingConfig()

    overrides = {
        "dataset_root": args.dataset_root,
        "metadata_filename": args.metadata_filename,
        "hourly_csv": args.hourly_csv,
        "elevation_csv": args.elevation_csv,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "num_workers": args.num_workers,
        "device": args.device,
        "checkpoint_dir": args.checkpoint_dir,
        "output_dir": args.output_dir,
        "pretrained_checkpoint": args.pretrained_checkpoint,
        "resume_checkpoint": args.resume_checkpoint,
        "random_seed": args.seed,
        "amp_dtype": args.amp_dtype,
        "gradient_clip": args.gradient_clip,
        "scheduler_t_max": args.scheduler_t_max,
        "scheduler_eta_min": args.scheduler_eta_min,
        "input_length": args.input_length,
        "output_length": args.output_length,
        "target_channel_index": args.target_channel_index,
        "readout_type": args.readout_type,
        "readout_latent_dim": args.readout_latent_dim,
        "query_dimension": args.query_dimension,
        "num_output_queries": args.num_output_queries,
        "num_attention_heads": args.num_attention_heads,
        "readout_dropout": args.readout_dropout,
        "regression_hidden_dim": args.regression_hidden_dim,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(cfg, key, value)
    if args.amp:
        cfg.mixed_precision = True
    if args.no_amp:
        cfg.mixed_precision = False
    if args.no_normalize:
        cfg.normalize = False
    if args.freeze_earthformer:
        cfg.freeze_earthformer = True
    return cfg
