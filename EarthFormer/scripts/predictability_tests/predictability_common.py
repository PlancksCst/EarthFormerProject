"""Shared helpers for standalone CSI predictability tests."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
EARTHFORMER_DIR = SCRIPT_DIR.parents[1]
PROJECT_ROOT = EARTHFORMER_DIR.parent
DIAGNOSTIC_DIR = SCRIPT_DIR.parent / "diagnostics"

for candidate in (PROJECT_ROOT, EARTHFORMER_DIR, DIAGNOSTIC_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from configs.config import build_arg_parser, config_from_args  # noqa: E402
from datasets.seviri_dataset import build_dataset  # noqa: E402
from models.model import build_perceiver_readout_model  # noqa: E402
from training.checkpoint import load_checkpoint, load_model_state_dict_compatible  # noqa: E402
from utils.artifacts import ArtifactMirror  # noqa: E402
from utils.precision import autocast_context, resolve_amp_dtype  # noqa: E402
from utils.seed import seed_everything  # noqa: E402
from diagnostic_common import (  # type: ignore  # noqa: E402
    batch_prediction_rows,
    clear_sky_tensor,
    diagnostic_valid_mask_tensor,
    metrics_from_rows,
    target_ghi_tensor,
    target_tensor,
)


@dataclass(frozen=True)
class PredictabilityContext:
    """Runtime context shared by predictability tests."""

    config: Any
    args: argparse.Namespace
    output_dir: Path
    device: torch.device
    artifact_mirror: ArtifactMirror


def add_predictability_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add predictability-test CLI arguments."""
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--solar-elevation-threshold", type=float, default=5.0)
    parser.add_argument("--lead-hours", default="1,2,3,4,6")
    parser.add_argument("--history-hours", type=int, default=6)
    parser.add_argument(
        "--model",
        choices=("simple_cnn_lstm", "frozen_earthformer_pool_mlp", "all"),
        default="simple_cnn_lstm",
    )
    parser.add_argument("--run-short-horizon", action="store_true")
    parser.add_argument("--short-horizon-cache-days", type=int, default=32)
    parser.add_argument("--short-horizon-shuffle", dest="short_horizon_shuffle", action="store_true", default=True)
    parser.add_argument("--no-short-horizon-shuffle", dest="short_horizon_shuffle", action="store_false")
    parser.add_argument("--prediction-std-ratio-threshold", type=float, default=0.10)
    return parser


def parse_args(description: str) -> argparse.Namespace:
    """Parse project config args plus predictability-test arguments."""
    parser = build_arg_parser()
    parser.description = description
    add_predictability_args(parser)
    return parser.parse_args()


def build_context(args: argparse.Namespace, default_subdir: str) -> PredictabilityContext:
    """Resolve config, output directory, device, and artifact mirror."""
    config = config_from_args(args)
    config.prepare_directories()
    seed_everything(config.random_seed)
    output_dir = Path(args.output_dir) if args.output_dir is not None else config.output_dir / "predictability_tests" / default_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_mirror = ArtifactMirror(
        checkpoint_dir=config.checkpoint_dir,
        output_dir=config.output_dir,
        enabled=config.mirror_artifacts,
    )
    return PredictabilityContext(
        config=config,
        args=args,
        output_dir=output_dir,
        device=torch.device(config.resolved_device()),
        artifact_mirror=artifact_mirror,
    )


def capped_dataset(config: Any, split: str, include_target: bool, max_samples: int | None) -> Any:
    """Build an existing project dataset with an optional first-N cap."""
    dataset = build_dataset(config=config, split=split, include_target=include_target)
    if max_samples is not None and max_samples > 0 and max_samples < len(dataset):
        return Subset(dataset, list(range(max_samples)))
    return dataset


def capped_loader(context: PredictabilityContext, split: str, max_samples: int | None, shuffle: bool = False) -> DataLoader:
    """Build a DataLoader over a capped split."""
    dataset = capped_dataset(context.config, split=split, include_target=True, max_samples=max_samples)
    return DataLoader(
        dataset,
        batch_size=context.config.batch_size,
        shuffle=shuffle,
        num_workers=context.config.num_workers,
        pin_memory=context.config.resolved_device().startswith("cuda"),
        drop_last=False,
    )


def parse_int_list(text: str) -> list[int]:
    """Parse comma-separated positive integers."""
    values = [int(item.strip()) for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer")
    return values


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(path, index=False)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **payload}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=json_default)


def json_default(value: Any) -> Any:
    """Serialize common non-JSON values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return str(value)


def mirror_outputs(context: PredictabilityContext) -> None:
    """Mirror outputs to Drive when configured."""
    context.artifact_mirror.mirror_output_tree(context.output_dir)


@lru_cache(maxsize=4)
def load_hourly_frame(path: Path) -> pd.DataFrame:
    """Load the hourly CAMS/ground CSV."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Hourly CSV not found: {path}")
    frame = pd.read_csv(path)
    if "timestamp" not in frame.columns:
        raise KeyError(f"Missing timestamp column in {path}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.set_index("timestamp").sort_index()
    if not frame.index.is_unique:
        frame = frame[~frame.index.duplicated(keep="first")]
    return frame


def location_columns(frame: pd.DataFrame, location: str) -> dict[str, str] | None:
    """Return CSI/GHI/clear columns for one location, or None when unavailable."""
    columns = {
        "csi": f"CSI_{location}",
        "ghi": f"GHI_{location}",
        "clear": f"GHI_clear_{location}",
    }
    return columns if all(column in frame.columns for column in columns.values()) else None


def hourly_value(frame: pd.DataFrame, columns: dict[str, str], timestamp: pd.Timestamp, key: str) -> float | None:
    """Return one finite value from an hourly dataframe."""
    try:
        row = frame.loc[timestamp]
    except KeyError:
        return None
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    value = row.get(columns[key])
    if pd.isna(value):
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def previous_day_csi_from_csv(config: Any, batch: dict[str, Any], horizon: int) -> torch.Tensor | None:
    """Return previous-day CSI from the hourly CSV, with NaN for missing values."""
    try:
        frame = load_hourly_frame(Path(config.hourly_csv))
    except (FileNotFoundError, KeyError):
        return None
    locations = batch.get("location")
    input_days = batch.get("input_day")
    if locations is None or input_days is None:
        return None
    batch_size = len(locations) if isinstance(locations, (list, tuple)) else 1
    values = np.full((batch_size, horizon), np.nan, dtype=np.float32)
    any_found = False
    for sample_index in range(batch_size):
        location = str(locations[sample_index]) if isinstance(locations, (list, tuple)) else str(locations)
        columns = location_columns(frame, location)
        if columns is None:
            continue
        day_value = input_days[sample_index] if isinstance(input_days, (list, tuple)) else input_days
        try:
            input_day = pd.Timestamp(day_value)
        except Exception:
            continue
        for hour_index in range(horizon):
            timestamp = input_day + pd.Timedelta(hours=4 + hour_index)
            value = hourly_value(frame, columns, timestamp, "csi")
            if value is not None:
                values[sample_index, hour_index] = value
                any_found = True
    return torch.from_numpy(values) if any_found else None


def prediction_rows(
    batch: dict[str, Any],
    split: str,
    sample_start: int,
    prediction: torch.Tensor,
    target: torch.Tensor,
    clear: torch.Tensor,
    valid: torch.Tensor,
    target_ghi: torch.Tensor,
    label_name: str,
    label_value: str,
) -> list[dict[str, Any]]:
    """Build long prediction rows with a method/baseline label and hour_index."""
    rows = batch_prediction_rows(
        batch=batch,
        split=split,
        sample_start=sample_start,
        pred_csi=prediction,
        target_csi=target,
        clear_sky_ghi=clear,
        valid_mask=valid,
        target_ghi=target_ghi,
    )
    for row in rows:
        row[label_name] = label_value
        row["method"] = label_value
        row["hour_index"] = int(row["forecast_hour"]) - 1
        row["date"] = row.get("target_day")
    return rows


def grouped_metrics(rows: list[dict[str, Any]], label_col: str = "method") -> dict[str, list[dict[str, Any]]]:
    """Compute overall, per-location, and per-hour metrics grouped by label."""
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {"overall": [], "per_location": [], "per_hour": []}

    overall = [
        metrics_from_rows(group.to_dict("records"), {label_col: label})
        for label, group in frame.groupby(label_col, sort=False)
    ]
    per_location = [
        metrics_from_rows(group.to_dict("records"), {label_col: label, "location": location})
        for (label, location), group in frame.groupby([label_col, "location"], sort=False)
    ] if "location" in frame.columns else []
    per_hour = [
        metrics_from_rows(group.to_dict("records"), {label_col: label, "forecast_hour": int(hour)})
        for (label, hour), group in frame.groupby([label_col, "forecast_hour"], sort=False)
    ] if "forecast_hour" in frame.columns else []
    return {"overall": overall, "per_location": per_location, "per_hour": per_hour}


def save_metric_tables(output_dir: Path, prefix: str, rows: list[dict[str, Any]], label_col: str = "method") -> dict[str, str]:
    """Save grouped metrics and return output paths."""
    tables = grouped_metrics(rows, label_col=label_col)
    paths = {
        "overall": output_dir / f"{prefix}_metrics.csv",
        "per_location": output_dir / "metrics_per_location.csv",
        "per_hour": output_dir / "metrics_per_hour.csv",
    }
    write_csv(paths["overall"], tables["overall"])
    write_csv(paths["per_location"], tables["per_location"])
    write_csv(paths["per_hour"], tables["per_hour"])
    return {key: str(value) for key, value in paths.items()}


def plot_sample_predictions(rows: list[dict[str, Any]], output_dir: Path, label_col: str = "method", limit: int = 8) -> None:
    """Save target-vs-prediction plots for a few samples."""
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for plot_index, (sample_index, group) in enumerate(frame.groupby("sample_index", sort=True), start=1):
        if plot_index > limit:
            break
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        target = group.sort_values("forecast_hour").drop_duplicates("forecast_hour")
        ax.plot(target["forecast_hour"], target["target_csi"], color="black", marker="o", linewidth=2.0, label="target")
        for label, label_group in group.groupby(label_col, sort=False):
            label_group = label_group.sort_values("forecast_hour")
            ax.plot(label_group["forecast_hour"], label_group["predicted_csi"], marker="o", label=str(label))
        ax.set_xlabel("Forecast hour")
        ax.set_ylabel("CSI")
        ax.set_title(f"Predictability sample {sample_index}")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"sample_{int(sample_index):04d}.png", dpi=180)
        plt.close(fig)


def plot_rmse_bar(metrics_rows: list[dict[str, Any]], output_path: Path, label_col: str = "method") -> None:
    """Save a CSI/GHI RMSE comparison bar plot."""
    frame = pd.DataFrame(metrics_rows)
    if frame.empty or label_col not in frame.columns:
        return
    labels = frame[label_col].astype(str).tolist()
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8.0, 0.9 * len(labels)), 4.8))
    ax.bar(x - width / 2, frame["CSI_RMSE"], width=width, label="CSI RMSE")
    ax.bar(x + width / 2, frame["GHI_RMSE"], width=width, label="GHI RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("RMSE")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def checked_checkpoint_path(args: argparse.Namespace) -> Path:
    """Resolve and validate a required checkpoint path."""
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for image-model predictability tests")
    path = Path(args.checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def load_checked_image_model(context: PredictabilityContext) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load the trained image-only model from a required checkpoint."""
    path = checked_checkpoint_path(context.args)
    print(f"Loading checkpoint: {path}")
    checkpoint = load_checkpoint(path, map_location=context.device)
    if isinstance(checkpoint, dict):
        print(
            "Checkpoint metadata: "
            f"epoch={checkpoint.get('epoch')} "
            f"best_metric={checkpoint.get('best_metric', checkpoint.get('best_loss'))}"
        )
        state = checkpoint["model"] if "model" in checkpoint else checkpoint
    else:
        state = checkpoint
        checkpoint = {}
    model = build_perceiver_readout_model(context.config).to(context.device)
    load_model_state_dict_compatible(model, state)
    model.eval()
    return model, checkpoint


def maybe_autocast(context: PredictabilityContext):
    """Return the configured autocast context."""
    use_amp = bool(context.config.mixed_precision and context.device.type == "cuda")
    amp_dtype = resolve_amp_dtype(context.config.amp_dtype, context.device) if use_amp else None
    return autocast_context(device=context.device, enabled=use_amp, dtype=amp_dtype)


def common_cli(args: argparse.Namespace) -> list[str]:
    """Return common CLI arguments for wrapper child processes."""
    pairs = [
        ("--dataset-root", args.dataset_root),
        ("--checkpoint", args.checkpoint),
        ("--train-split", args.train_split),
        ("--eval-split", args.eval_split),
        ("--batch-size", args.batch_size),
        ("--num-workers", args.num_workers),
        ("--device", args.device),
        ("--checkpoint-dir", args.checkpoint_dir),
        ("--hourly-csv", args.hourly_csv),
        ("--elevation-csv", args.elevation_csv),
        ("--clear-sky-threshold", args.clear_sky_threshold),
        ("--solar-elevation-threshold", args.solar_elevation_threshold),
        ("--max-train-samples", args.max_train_samples),
        ("--max-eval-samples", args.max_eval_samples),
        ("--lead-hours", args.lead_hours),
        ("--history-hours", args.history_hours),
        ("--model", args.model),
        ("--epochs", args.epochs),
        ("--short-horizon-cache-days", getattr(args, "short_horizon_cache_days", None)),
        ("--prediction-std-ratio-threshold", getattr(args, "prediction_std_ratio_threshold", None)),
    ]
    cli: list[str] = []
    for flag, value in pairs:
        if value is not None:
            cli.extend([flag, str(value)])
    if getattr(args, "no_artifact_mirror", False):
        cli.append("--no-artifact-mirror")
    if getattr(args, "short_horizon_shuffle", True):
        cli.append("--short-horizon-shuffle")
    else:
        cli.append("--no-short-horizon-shuffle")
    return cli


def run_child(script: Path, args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    """Run a predictability child script."""
    command = [sys.executable, str(script), *common_cli(args), "--output-dir", str(output_dir)]
    print(f"\nRunning: {script.name}", flush=True)
    print(" ".join(str(part) for part in command), flush=True)
    result = subprocess.run(command, text=True, check=False)
    return {
        "script": script.name,
        "returncode": result.returncode,
        "ok": result.returncode == 0,
    }
