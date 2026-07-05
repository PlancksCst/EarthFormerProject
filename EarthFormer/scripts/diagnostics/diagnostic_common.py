"""Shared utilities for standalone failure-analysis diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
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

for candidate in (PROJECT_ROOT, EARTHFORMER_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from configs.config import build_arg_parser, config_from_args  # noqa: E402
from datasets.seviri_dataset import build_dataset  # noqa: E402
from models.model import build_perceiver_readout_model  # noqa: E402
from training.checkpoint import load_checkpoint, load_model_state_dict_compatible  # noqa: E402
from training.losses import valid_hour_mask  # noqa: E402
from training.validate import ensure_forecast_target, reconstruct_ghi  # noqa: E402
from utils.artifacts import ArtifactMirror  # noqa: E402
from utils.precision import autocast_context, resolve_amp_dtype  # noqa: E402
from utils.seed import seed_everything  # noqa: E402

EPS = 1.0e-8


@dataclass(frozen=True)
class DiagnosticsContext:
    """Resolved runtime objects shared by diagnostics."""

    config: Any
    args: argparse.Namespace
    output_dir: Path
    device: torch.device
    artifact_mirror: ArtifactMirror


def add_common_diagnostic_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add common diagnostic CLI arguments on top of the project parser."""
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--noise-std", type=float, default=0.05)
    return parser


def parse_common_args(description: str) -> argparse.Namespace:
    """Parse project config flags plus diagnostic flags."""
    parser = build_arg_parser()
    parser.description = description
    add_common_diagnostic_args(parser)
    return parser.parse_args()


def build_context(args: argparse.Namespace, default_subdir: str | None = None) -> DiagnosticsContext:
    """Build config, output directories, device, and artifact mirror."""
    config = config_from_args(args)
    if getattr(args, "seed", None) is not None:
        config.random_seed = int(args.seed)
    if getattr(args, "clear_sky_threshold", None) is not None:
        config.clear_sky_threshold = float(args.clear_sky_threshold)
    config.prepare_directories()
    seed_everything(config.random_seed)
    output_dir = args.output_dir or (config.output_dir / "diagnostics_failure_analysis")
    if default_subdir:
        output_dir = output_dir / default_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_mirror = ArtifactMirror(
        checkpoint_dir=config.checkpoint_dir,
        output_dir=config.output_dir,
        enabled=config.mirror_artifacts,
    )
    return DiagnosticsContext(
        config=config,
        args=args,
        output_dir=output_dir,
        device=torch.device(config.resolved_device()),
        artifact_mirror=artifact_mirror,
    )


def dataset_for_split(config: Any, split: str, include_target: bool = True, max_samples: int | None = None) -> Any:
    """Build a dataset, optionally capped to the first `max_samples` items."""
    dataset = build_dataset(config=config, split=split, include_target=include_target)
    if max_samples is None or max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    return Subset(dataset, list(range(max_samples)))


def dataloader_for_split(
    config: Any,
    split: str,
    include_target: bool = True,
    shuffle: bool = False,
    max_samples: int | None = None,
) -> DataLoader:
    """Build a dataloader using the configured batch and worker settings."""
    dataset = dataset_for_split(config, split, include_target=include_target, max_samples=max_samples)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.resolved_device().startswith("cuda"),
        drop_last=False,
    )


def unwrap_dataset(dataset: Any) -> Any:
    """Return the underlying dataset if this is a Subset."""
    return dataset.dataset if isinstance(dataset, Subset) else dataset


def dataset_row(dataset: Any, index: int) -> Any | None:
    """Return the original metadata row for a dataset sample when available."""
    if isinstance(dataset, Subset):
        actual_index = int(dataset.indices[index])
        base = dataset.dataset
    else:
        actual_index = int(index)
        base = dataset
    metadata = getattr(base, "meta", None)
    if metadata is None:
        return None
    return metadata.iloc[actual_index]


def metadata_value_from_batch(batch: dict[str, Any], key: str, index: int) -> Any:
    """Return one metadata value from a collated batch."""
    value = batch.get(key)
    if isinstance(value, torch.Tensor):
        item = value[index]
        return item.detach().cpu().item() if item.numel() == 1 else item.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return value[index] if index < len(value) else None
    return value


def tensor_to_cpu_float(value: torch.Tensor) -> torch.Tensor:
    """Detach a tensor as float32 on CPU."""
    return value.detach().float().cpu()


def target_tensor(batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    """Return target CSI tensor on device."""
    target = batch.get("target", batch.get("target_csi"))
    if not isinstance(target, torch.Tensor):
        raise KeyError("Batch does not contain target or target_csi tensor")
    return ensure_forecast_target(target, "target").to(device, non_blocking=True)


def clear_sky_tensor(batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    """Return clear-sky GHI tensor on device."""
    clear = batch.get("clear_sky_ghi", batch.get("clear_ghi"))
    if not isinstance(clear, torch.Tensor):
        raise KeyError("Batch does not contain clear_sky_ghi or clear_ghi tensor")
    return ensure_forecast_target(clear, "clear_sky_ghi").to(device, non_blocking=True)


def target_ghi_tensor(batch: dict[str, Any], target_csi: torch.Tensor, clear_sky_ghi: torch.Tensor) -> torch.Tensor:
    """Return target GHI from batch or reconstruct it."""
    value = batch.get("target_ghi")
    if isinstance(value, torch.Tensor):
        return ensure_forecast_target(value, "target_ghi").to(target_csi.device, non_blocking=True)
    return reconstruct_ghi(target_csi, clear_sky_ghi)


def valid_mask_tensor(
    batch: dict[str, Any],
    target_csi: torch.Tensor,
    clear_sky_ghi: torch.Tensor,
    clear_sky_threshold: float,
) -> torch.Tensor:
    """Return the physical valid-hour mask."""
    target_mask = batch.get("target_mask")
    if isinstance(target_mask, torch.Tensor):
        target_mask = target_mask.to(target_csi.device, non_blocking=True)
    else:
        target_mask = None
    return valid_hour_mask(
        target_mask=target_mask,
        reference=target_csi,
        clear_sky_ghi=clear_sky_ghi,
        clear_sky_threshold=clear_sky_threshold,
    )


def batch_prediction_rows(
    batch: dict[str, Any],
    split: str,
    sample_start: int,
    pred_csi: torch.Tensor,
    target_csi: torch.Tensor,
    clear_sky_ghi: torch.Tensor,
    valid_mask: torch.Tensor,
    target_ghi: torch.Tensor | None = None,
    baseline_name: str | None = None,
) -> list[dict[str, Any]]:
    """Convert batched predictions into long-form rows."""
    pred = tensor_to_cpu_float(pred_csi).numpy()
    target = tensor_to_cpu_float(target_csi).numpy()
    clear = tensor_to_cpu_float(clear_sky_ghi).numpy()
    valid = valid_mask.detach().cpu().numpy().astype(bool)
    if target_ghi is None:
        target_ghi_np = target * clear
    else:
        target_ghi_np = tensor_to_cpu_float(target_ghi).numpy()
    pred_ghi = pred * clear
    rows: list[dict[str, Any]] = []
    batch_size, horizon = pred.shape
    for batch_index in range(batch_size):
        metadata = {
            "split": split,
            "sample_index": sample_start + batch_index,
            "sample_id": metadata_value_from_batch(batch, "sample_id", batch_index),
            "location": metadata_value_from_batch(batch, "location", batch_index),
            "input_day": metadata_value_from_batch(batch, "input_day", batch_index),
            "target_day": metadata_value_from_batch(batch, "target_day", batch_index),
        }
        if baseline_name is not None:
            metadata["baseline"] = baseline_name
        for hour in range(horizon):
            rows.append(
                {
                    **metadata,
                    "forecast_hour": hour + 1,
                    "valid_hour": bool(valid[batch_index, hour]),
                    "target_csi": float(target[batch_index, hour]),
                    "predicted_csi": float(pred[batch_index, hour]),
                    "clear_sky_ghi": float(clear[batch_index, hour]),
                    "target_ghi": float(target_ghi_np[batch_index, hour]),
                    "predicted_ghi": float(pred_ghi[batch_index, hour]),
                    "error_csi": float(pred[batch_index, hour] - target[batch_index, hour]),
                    "error_ghi": float(pred_ghi[batch_index, hour] - target_ghi_np[batch_index, hour]),
                }
            )
    return rows


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Return regression metrics with safe finite filtering."""
    pred = np.asarray(prediction, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    mask = np.isfinite(pred) & np.isfinite(tgt)
    pred = pred[mask]
    tgt = tgt[mask]
    if pred.size == 0:
        return {
            "count": 0.0,
            "MAE": math.nan,
            "RMSE": math.nan,
            "nRMSE": math.nan,
            "R2": math.nan,
            "MBE": math.nan,
            "Pearson": math.nan,
        }
    error = pred - tgt
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    nrmse = float(rmse / max(float(np.mean(np.abs(tgt))), EPS))
    residual = float(np.sum(error**2))
    centered = float(np.sum((tgt - np.mean(tgt)) ** 2))
    r2 = float(1.0 - residual / max(centered, EPS))
    if pred.size < 2 or float(np.std(pred)) < EPS or float(np.std(tgt)) < EPS:
        pearson = math.nan
    else:
        pearson = float(np.corrcoef(pred, tgt)[0, 1])
    return {
        "count": float(pred.size),
        "MAE": mae,
        "RMSE": rmse,
        "nRMSE": nrmse,
        "R2": r2,
        "MBE": float(np.mean(error)),
        "Pearson": pearson,
    }


def metrics_from_rows(rows: Iterable[dict[str, Any]], group: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compute CSI/GHI metrics from prediction rows where valid_hour is true."""
    frame = pd.DataFrame(list(rows))
    prefix = dict(group or {})
    if frame.empty:
        return {**prefix, "valid_count": 0}
    valid = frame[frame["valid_hour"].astype(bool)].copy()
    csi = regression_metrics(valid["predicted_csi"].to_numpy(), valid["target_csi"].to_numpy())
    ghi = regression_metrics(valid["predicted_ghi"].to_numpy(), valid["target_ghi"].to_numpy())
    result: dict[str, Any] = {**prefix, "valid_count": int(csi["count"])}
    for key, value in csi.items():
        if key != "count":
            result[f"CSI_{key}"] = value
    for key, value in ghi.items():
        if key != "count":
            result[f"GHI_{key}"] = value
    result["valid_fraction"] = len(valid) / max(len(frame), 1)
    return result


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write rows to CSV using pandas."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(path, index=False)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **payload,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=json_default)


def json_default(value: Any) -> Any:
    """Serialize common values for JSON."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    return str(value)


def mirror_outputs(context: DiagnosticsContext) -> None:
    """Mirror the diagnostics output tree to Drive when available."""
    context.artifact_mirror.mirror_output_tree(context.output_dir)


def resolve_checkpoint(context: DiagnosticsContext) -> Path:
    """Return the requested checkpoint or default best checkpoint."""
    checkpoint = context.args.checkpoint or (context.config.checkpoint_dir / "best.pt")
    return Path(checkpoint)


def load_trained_model(context: DiagnosticsContext) -> torch.nn.Module:
    """Build model, load checkpoint if present, and switch to eval mode."""
    model = build_perceiver_readout_model(context.config).to(context.device)
    checkpoint_path = resolve_checkpoint(context)
    if checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path, map_location=context.device)
        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        load_model_state_dict_compatible(model, state)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model.eval()
    return model


def maybe_autocast(context: DiagnosticsContext):
    """Return the configured autocast context for inference."""
    use_amp = bool(context.config.mixed_precision and context.device.type == "cuda")
    amp_dtype = resolve_amp_dtype(context.config.amp_dtype, context.device) if use_amp else None
    return autocast_context(device=context.device, enabled=use_amp, dtype=amp_dtype)


def sample_rows_for_loader(
    loader: DataLoader,
    split: str,
    device: torch.device,
    clear_sky_threshold: float,
    prediction_fn: Any,
    baseline_name: str | None = None,
) -> list[dict[str, Any]]:
    """Run a prediction function over a dataloader and return long-form rows."""
    rows: list[dict[str, Any]] = []
    sample_start = 0
    for batch in loader:
        target = target_tensor(batch, device)
        clear = clear_sky_tensor(batch, device)
        target_ghi = target_ghi_tensor(batch, target, clear)
        valid = valid_mask_tensor(batch, target, clear, clear_sky_threshold)
        pred = prediction_fn(batch, target, clear, valid)
        rows.extend(
            batch_prediction_rows(
                batch=batch,
                split=split,
                sample_start=sample_start,
                pred_csi=pred,
                target_csi=target,
                clear_sky_ghi=clear,
                valid_mask=valid,
                target_ghi=target_ghi,
                baseline_name=baseline_name,
            )
        )
        sample_start += target.shape[0]
    return rows


def plot_metric_bar(metrics: pd.DataFrame, output_path: Path) -> None:
    """Plot CSI/GHI RMSE bars by method/baseline."""
    if metrics.empty:
        return
    label_col = "baseline" if "baseline" in metrics.columns else "method"
    labels = metrics[label_col].astype(str).tolist()
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8.0, 0.8 * len(labels)), 4.8))
    ax.bar(x - width / 2, metrics["CSI_RMSE"], width=width, label="CSI RMSE")
    ax.bar(x + width / 2, metrics["GHI_RMSE"], width=width, label="GHI RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("RMSE")
    ax.set_title("Baseline/Model RMSE Comparison")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_series(path: Path, hours: np.ndarray, series: dict[str, np.ndarray], title: str, ylabel: str = "CSI") -> None:
    """Plot target and prediction time series."""
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for label, values in series.items():
        ax.plot(hours, values, marker="o", label=label)
    ax.set_xticks(hours)
    ax.set_xlabel("Forecast hour")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
