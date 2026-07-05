"""Comprehensive evaluation for the trained EarthFormer + Perceiver model."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PREP_MODELS_ROOT = PROJECT_ROOT.parent
if str(PREP_MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(PREP_MODELS_ROOT))

from configs.config import build_arg_parser, config_from_args  # noqa: E402
from datasets.seviri_dataset import build_dataloader  # noqa: E402
from models.model import build_perceiver_readout_model  # noqa: E402
from training.checkpoint import load_checkpoint  # noqa: E402
from training.losses import valid_mask_from_target_mask  # noqa: E402
from training.validate import ensure_forecast_target, reconstruct_ghi  # noqa: E402
from utils.precision import amp_dtype_label, autocast_context, resolve_amp_dtype  # noqa: E402
from utils.seed import seed_everything  # noqa: E402


HOURS = np.arange(1, 14)
EPS = 1.0e-8


@dataclass(frozen=True)
class EvaluationDirs:
    """Filesystem layout for evaluation artifacts."""

    root: Path
    metrics: Path
    predictions: Path
    figures: Path
    timeseries: Path
    best_cases: Path
    worst_cases: Path
    heatmaps: Path
    sample_predictions: Path
    diagnostics: Path

    @classmethod
    def create(cls, root: Path) -> "EvaluationDirs":
        """Create and return all evaluation directories."""
        dirs = cls(
            root=root,
            metrics=root / "metrics",
            predictions=root / "predictions",
            figures=root / "figures",
            timeseries=root / "figures" / "timeseries",
            best_cases=root / "figures" / "best_cases",
            worst_cases=root / "figures" / "worst_cases",
            heatmaps=root / "figures" / "heatmaps",
            sample_predictions=root / "figures" / "sample_predictions",
            diagnostics=root / "diagnostics",
        )
        for path in dirs.__dict__.values():
            path.mkdir(parents=True, exist_ok=True)
        return dirs


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = build_arg_parser()
    parser.description = "Evaluate a trained EarthFormer + Perceiver CSI forecasting model."
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-samples", "--num_samples", dest="num_samples", type=int, default=8)
    parser.add_argument("--evaluation-dir", "--output_dir", dest="evaluation_dir", type=Path, default=None)
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=None)
    parser.add_argument("--best-worst-count", type=int, default=10)
    parser.add_argument("--max-scatter-points", type=int, default=50_000)
    parser.add_argument("--input-frame-indices", default="0,6,12")
    parser.add_argument("--satellite-channel-index", type=int, default=0)
    parser.add_argument("--save-diagnostics", action="store_true")
    return parser.parse_args()


def configure_style() -> None:
    """Set a consistent publication-oriented matplotlib style."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 240,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.7,
            "lines.linewidth": 2.0,
        }
    )


def finite_numpy(values: pd.Series | np.ndarray) -> np.ndarray:
    """Return a finite float numpy array."""
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Compute scalar regression metrics with safe denominators."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.isfinite(prediction) & np.isfinite(target)
    prediction = prediction[mask]
    target = target[mask]
    if prediction.size == 0:
        return {
            "count": 0.0,
            "rmse": np.nan,
            "mae": np.nan,
            "mbe": np.nan,
            "r2": np.nan,
            "pearson_r": np.nan,
        }

    error = prediction - target
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    mbe = float(np.mean(error))
    centered = target - np.mean(target)
    sst = float(np.sum(centered**2))
    r2 = float(1.0 - np.sum(error**2) / max(sst, EPS))

    pred_std = float(np.std(prediction))
    target_std = float(np.std(target))
    if prediction.size < 2 or pred_std < EPS or target_std < EPS:
        pearson = 0.0
    else:
        pearson = float(np.corrcoef(prediction, target)[0, 1])

    return {
        "count": float(prediction.size),
        "rmse": rmse,
        "mae": mae,
        "mbe": mbe,
        "r2": r2,
        "pearson_r": pearson,
    }


def paired_metric_row(frame: pd.DataFrame, prefix: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return CSI and GHI metrics for a prediction dataframe subset."""
    row: dict[str, Any] = dict(prefix or {})
    csi = regression_metrics(frame["pred_csi"].to_numpy(), frame["target_csi"].to_numpy())
    ghi = regression_metrics(frame["pred_ghi"].to_numpy(), frame["target_ghi"].to_numpy())
    row["count"] = int(csi["count"])
    for key, value in csi.items():
        if key != "count":
            row[f"CSI_{key.upper() if key != 'pearson_r' else 'PearsonR'}"] = value
    for key, value in ghi.items():
        if key != "count":
            row[f"GHI_{key.upper() if key != 'pearson_r' else 'PearsonR'}"] = value
    return row


def save_metric_tables(predictions: pd.DataFrame, dirs: EvaluationDirs) -> dict[str, pd.DataFrame]:
    """Compute and save overall and grouped metric CSV files."""
    valid = predictions[predictions["valid"]].copy()
    valid["target_datetime"] = pd.to_datetime(valid["target_day"], errors="coerce")
    valid["month"] = valid["target_datetime"].dt.to_period("M").astype(str)
    valid.loc[valid["target_datetime"].isna(), "month"] = "unknown"

    overall = pd.DataFrame([paired_metric_row(valid, {"split": str(valid["split"].iloc[0]) if len(valid) else ""})])
    per_hour = pd.DataFrame(
        [
            paired_metric_row(group, {"forecast_hour": int(hour)})
            for hour, group in valid.groupby("forecast_hour", sort=True)
        ]
    )
    per_location = pd.DataFrame(
        [
            paired_metric_row(group, {"location": str(location)})
            for location, group in valid.groupby("location", sort=True)
        ]
    )
    per_month = pd.DataFrame(
        [
            paired_metric_row(group, {"month": str(month)})
            for month, group in valid.groupby("month", sort=True)
        ]
    )

    overall.to_csv(dirs.metrics / "overall.csv", index=False)
    per_hour.to_csv(dirs.metrics / "per_hour.csv", index=False)
    per_location.to_csv(dirs.metrics / "per_location.csv", index=False)
    per_month.to_csv(dirs.metrics / "per_month.csv", index=False)
    return {
        "overall": overall,
        "per_hour": per_hour,
        "per_location": per_location,
        "per_month": per_month,
    }


def value_from_batch(batch: dict[str, Any], key: str, index: int) -> Any:
    """Return one sample metadata value from a collated batch."""
    value = batch.get(key)
    if isinstance(value, torch.Tensor):
        item = value[index]
        return item.detach().cpu().item() if item.numel() == 1 else item.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


def parse_indices(text: str, maximum: int) -> list[int]:
    """Parse comma-separated non-negative frame indices."""
    indices: list[int] = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        index = int(raw)
        if 0 <= index < maximum and index not in indices:
            indices.append(index)
    if not indices:
        indices = [0, maximum // 2, maximum - 1]
    return indices


def select_indices(total: int, count: int, seed: int) -> set[int]:
    """Choose deterministic random sample indices."""
    if total <= 0 or count <= 0:
        return set()
    rng = np.random.default_rng(seed)
    chosen = rng.choice(total, size=min(count, total), replace=False)
    return {int(item) for item in chosen}


def sample_key(sample: pd.Series | dict[str, Any]) -> str:
    """Build a compact filename-safe sample identifier."""
    location = str(sample.get("location", "location"))
    target_day = str(sample.get("target_day", "day")).replace(":", "-")
    sample_id = str(sample.get("sample_id", sample.get("sample_index", "sample")))
    return f"{sample_id}_{location}_{target_day}".replace("/", "-").replace("\\", "-")


def load_model(config: Any, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Build the model and load a checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = build_perceiver_readout_model(config).to(device)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()
    return model


def target_ghi_tensor(batch: dict[str, Any], targets: torch.Tensor, clear_sky_ghi: torch.Tensor) -> torch.Tensor:
    """Return target GHI from the batch or reconstruct it."""
    value = batch.get("target_ghi")
    if isinstance(value, torch.Tensor):
        return ensure_forecast_target(value, "target_ghi").to(targets.device, non_blocking=True)
    return reconstruct_ghi(targets, clear_sky_ghi)


def make_prediction_rows(
    batch: dict[str, Any],
    sample_start: int,
    split: str,
    predictions_csi: torch.Tensor,
    predictions_ghi: torch.Tensor,
    targets_csi: torch.Tensor,
    targets_ghi: torch.Tensor,
    clear_sky_ghi: torch.Tensor,
    valid_mask: torch.Tensor,
) -> list[dict[str, Any]]:
    """Convert one model batch into long-form prediction rows."""
    pred_csi = predictions_csi.detach().float().cpu().numpy()
    pred_ghi = predictions_ghi.detach().float().cpu().numpy()
    target_csi = targets_csi.detach().float().cpu().numpy()
    target_ghi = targets_ghi.detach().float().cpu().numpy()
    clear = clear_sky_ghi.detach().float().cpu().numpy()
    valid = valid_mask.detach().cpu().numpy().astype(bool)

    rows: list[dict[str, Any]] = []
    batch_size, horizon = pred_csi.shape
    for batch_index in range(batch_size):
        metadata = {
            "split": split,
            "sample_index": sample_start + batch_index,
            "sample_id": value_from_batch(batch, "sample_id", batch_index),
            "location": value_from_batch(batch, "location", batch_index),
            "input_day": value_from_batch(batch, "input_day", batch_index),
            "target_day": value_from_batch(batch, "target_day", batch_index),
        }
        for hour in range(horizon):
            rows.append(
                {
                    **metadata,
                    "forecast_hour": hour + 1,
                    "valid": bool(valid[batch_index, hour]),
                    "target_csi": float(target_csi[batch_index, hour]),
                    "pred_csi": float(pred_csi[batch_index, hour]),
                    "error_csi": float(pred_csi[batch_index, hour] - target_csi[batch_index, hour]),
                    "abs_error_csi": float(abs(pred_csi[batch_index, hour] - target_csi[batch_index, hour])),
                    "clear_sky_ghi": float(clear[batch_index, hour]),
                    "target_ghi": float(target_ghi[batch_index, hour]),
                    "pred_ghi": float(pred_ghi[batch_index, hour]),
                    "error_ghi": float(pred_ghi[batch_index, hour] - target_ghi[batch_index, hour]),
                    "abs_error_ghi": float(abs(pred_ghi[batch_index, hour] - target_ghi[batch_index, hour])),
                }
            )
    return rows


def collect_sample(
    batch: dict[str, Any],
    batch_index: int,
    sample_index: int,
    satellite: torch.Tensor,
    predictions_csi: torch.Tensor,
    predictions_ghi: torch.Tensor,
    targets_csi: torch.Tensor,
    targets_ghi: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, Any]:
    """Collect one sample for plotting."""
    return {
        "sample_index": sample_index,
        "sample_id": value_from_batch(batch, "sample_id", batch_index),
        "location": value_from_batch(batch, "location", batch_index),
        "input_day": value_from_batch(batch, "input_day", batch_index),
        "target_day": value_from_batch(batch, "target_day", batch_index),
        "satellite": satellite[batch_index].detach().cpu().float().numpy(),
        "prediction_csi": predictions_csi[batch_index].detach().float().cpu().numpy(),
        "prediction_ghi": predictions_ghi[batch_index].detach().float().cpu().numpy(),
        "target_csi": targets_csi[batch_index].detach().float().cpu().numpy(),
        "target_ghi": targets_ghi[batch_index].detach().float().cpu().numpy(),
        "valid_mask": valid_mask[batch_index].detach().cpu().numpy().astype(bool),
    }


def run_inference(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    config: Any,
    args: argparse.Namespace,
    device: torch.device,
    dirs: EvaluationDirs,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Run model inference and return predictions plus selected samples."""
    selected = select_indices(
        total=len(loader.dataset),
        count=max(args.num_samples, args.best_worst_count),
        seed=config.random_seed,
    )
    rows: list[dict[str, Any]] = []
    selected_samples: list[dict[str, Any]] = []
    use_amp = config.mixed_precision and device.type == "cuda"
    amp_dtype = resolve_amp_dtype(config.amp_dtype, device) if use_amp else None
    sample_start = 0

    with torch.no_grad():
        for batch in loader:
            inputs = batch["satellite"].to(device, non_blocking=True)
            targets = ensure_forecast_target(batch["target"], "target").to(device, non_blocking=True)
            clear_sky_ghi = ensure_forecast_target(batch["clear_sky_ghi"], "clear_sky_ghi").to(
                device,
                non_blocking=True,
            )
            target_ghi = target_ghi_tensor(batch, targets, clear_sky_ghi)
            target_mask = batch.get("target_mask")
            if isinstance(target_mask, torch.Tensor):
                target_mask = target_mask.to(device, non_blocking=True)
            valid_mask = valid_mask_from_target_mask(target_mask, targets)

            with autocast_context(device=device, enabled=use_amp, dtype=amp_dtype):
                predictions_csi = model(inputs)
            if predictions_csi.shape != targets.shape:
                raise RuntimeError(
                    "Prediction and target shapes differ: "
                    f"{tuple(predictions_csi.shape)} vs {tuple(targets.shape)}"
                )
            if not torch.isfinite(predictions_csi).all():
                raise RuntimeError("Non-finite CSI predictions encountered during evaluation.")

            predictions_ghi = reconstruct_ghi(predictions_csi.float(), clear_sky_ghi)
            rows.extend(
                make_prediction_rows(
                    batch=batch,
                    sample_start=sample_start,
                    split=args.split,
                    predictions_csi=predictions_csi,
                    predictions_ghi=predictions_ghi,
                    targets_csi=targets,
                    targets_ghi=target_ghi,
                    clear_sky_ghi=clear_sky_ghi,
                    valid_mask=valid_mask,
                )
            )

            batch_size = inputs.shape[0]
            for offset in range(batch_size):
                absolute_index = sample_start + offset
                if absolute_index in selected:
                    selected_samples.append(
                        collect_sample(
                            batch=batch,
                            batch_index=offset,
                            sample_index=absolute_index,
                            satellite=inputs,
                            predictions_csi=predictions_csi,
                            predictions_ghi=predictions_ghi,
                            targets_csi=targets,
                            targets_ghi=target_ghi,
                            valid_mask=valid_mask,
                        )
                    )
                    if args.save_diagnostics:
                        save_optional_diagnostics(
                            model=model,
                            sample_input=inputs[offset : offset + 1],
                            sample=selected_samples[-1],
                            device=device,
                            use_amp=use_amp,
                            amp_dtype=amp_dtype,
                            dirs=dirs,
                        )
            sample_start += batch_size

    predictions = pd.DataFrame(rows)
    predictions.to_csv(dirs.predictions / "predictions.csv", index=False)
    return predictions, selected_samples


def save_optional_diagnostics(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    sample: dict[str, Any],
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype | None,
    dirs: EvaluationDirs,
) -> None:
    """Save available latent/readout diagnostics for one selected sample."""
    try:
        with torch.no_grad(), autocast_context(device=device, enabled=use_amp, dtype=amp_dtype):
            debug = model(sample_input, return_debug=True)
        payload = {
            "prediction": debug.get("prediction", None).detach().cpu() if isinstance(debug.get("prediction"), torch.Tensor) else None,
            "pre_head_latent": debug.get("pre_head_latent", None).detach().cpu()
            if isinstance(debug.get("pre_head_latent"), torch.Tensor)
            else None,
            "readout": {
                key: value.detach().cpu()
                for key, value in debug.get("readout", {}).items()
                if isinstance(value, torch.Tensor)
            },
            "earthformer_trace": debug.get("earthformer_trace", {}),
            "note": "Perceiver attention weights are not saved because the readout uses need_weights=False.",
        }
        torch.save(payload, dirs.diagnostics / f"{sample_key(sample)}_diagnostics.pt")
    except Exception as exc:
        note = pd.DataFrame([{"sample": sample_key(sample), "diagnostic_error": str(exc)}])
        path = dirs.diagnostics / "skipped_diagnostics.csv"
        note.to_csv(path, mode="a", header=not path.exists(), index=False)


def maybe_downsample(frame: pd.DataFrame, max_points: int, seed: int) -> pd.DataFrame:
    """Downsample a dataframe for scatter plots while preserving reproducibility."""
    if len(frame) <= max_points:
        return frame
    return frame.sample(n=max_points, random_state=seed)


def plot_scatter(
    frame: pd.DataFrame,
    target_col: str,
    pred_col: str,
    label: str,
    path: Path,
    max_points: int,
    seed: int,
) -> None:
    """Plot prediction-target scatter with identity and regression lines."""
    data = maybe_downsample(frame[[target_col, pred_col]].dropna(), max_points, seed)
    if data.empty:
        return
    target = data[target_col].to_numpy(dtype=np.float64)
    pred = data[pred_col].to_numpy(dtype=np.float64)
    metrics = regression_metrics(pred, target)
    lower = float(np.nanmin([target.min(), pred.min()]))
    upper = float(np.nanmax([target.max(), pred.max()]))
    padding = 0.04 * max(upper - lower, EPS)
    lower -= padding
    upper += padding

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.scatter(target, pred, s=8, alpha=0.22, edgecolors="none", rasterized=True)
    ax.plot([lower, upper], [lower, upper], color="black", linestyle="--", linewidth=1.2, label="1:1")
    if len(target) >= 2:
        slope, intercept = np.polyfit(target, pred, deg=1)
        ax.plot([lower, upper], [slope * lower + intercept, slope * upper + intercept], color="#c43b3b", label="Fit")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel(f"Target {label}")
    ax.set_ylabel(f"Predicted {label}")
    ax.set_title(f"{label} Prediction vs Target")
    ax.text(
        0.04,
        0.96,
        f"R2 = {metrics['r2']:.3f}\nPearson r = {metrics['pearson_r']:.3f}\nRMSE = {metrics['rmse']:.3f}",
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#cccccc"},
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_error_histogram(frame: pd.DataFrame, error_col: str, label: str, path: Path) -> None:
    """Plot prediction error histogram."""
    error = finite_numpy(frame[error_col])
    if error.size == 0:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.hist(error, bins=60, color="#4c78a8", alpha=0.82)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
    ax.axvline(float(np.mean(error)), color="#c43b3b", linewidth=1.5, label="Mean error")
    ax.set_xlabel(f"Prediction error ({label}: pred - target)")
    ax.set_ylabel("Count")
    ax.set_title(f"{label} Error Distribution")
    ax.text(
        0.98,
        0.94,
        f"mean = {np.mean(error):.4f}\nstd = {np.std(error):.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#cccccc"},
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_prediction_distribution(
    frame: pd.DataFrame,
    target_col: str,
    pred_col: str,
    label: str,
    path: Path,
) -> None:
    """Overlay predicted and target value distributions."""
    target = finite_numpy(frame[target_col])
    pred = finite_numpy(frame[pred_col])
    if target.size == 0 or pred.size == 0:
        return
    combined = np.concatenate([target, pred])
    lower = float(np.min(combined))
    upper = float(np.max(combined))
    if upper <= lower:
        upper = lower + EPS
    bins = np.linspace(lower, upper, 65)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.hist(target, bins=bins, alpha=0.55, density=True, label=f"Target {label}", color="#4c78a8")
    ax.hist(pred, bins=bins, alpha=0.55, density=True, label=f"Predicted {label}", color="#f58518")
    ax.set_xlabel(label)
    ax.set_ylabel("Density")
    ax.set_title(f"{label} Prediction Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_residuals(
    frame: pd.DataFrame,
    target_col: str,
    pred_col: str,
    error_col: str,
    label: str,
    path: Path,
    max_points: int,
    seed: int,
) -> None:
    """Plot residuals against targets and predictions."""
    data = maybe_downsample(frame[[target_col, pred_col, error_col]].dropna(), max_points, seed)
    if data.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    axes[0].scatter(data[target_col], data[error_col], s=8, alpha=0.22, edgecolors="none", rasterized=True)
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1.1)
    axes[0].set_xlabel(f"Target {label}")
    axes[0].set_ylabel(f"Residual {label} (pred - target)")
    axes[0].set_title("Residual vs Target")

    axes[1].scatter(data[pred_col], data[error_col], s=8, alpha=0.22, edgecolors="none", rasterized=True)
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.1)
    axes[1].set_xlabel(f"Predicted {label}")
    axes[1].set_title("Residual vs Prediction")
    fig.suptitle(f"{label} Residual Analysis")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_hourly_rmse(per_hour: pd.DataFrame, path: Path) -> None:
    """Plot RMSE against forecast hour for CSI and GHI."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    axes[0].plot(per_hour["forecast_hour"], per_hour["CSI_RMSE"], marker="o", color="#4c78a8")
    axes[0].set_xlabel("Forecast hour (sunrise to sunset)")
    axes[0].set_ylabel("CSI RMSE")
    axes[0].set_title("CSI Error by Forecast Hour")
    axes[0].set_xticks(HOURS)

    axes[1].plot(per_hour["forecast_hour"], per_hour["GHI_RMSE"], marker="o", color="#f58518")
    axes[1].set_xlabel("Forecast hour (sunrise to sunset)")
    axes[1].set_ylabel("GHI RMSE")
    axes[1].set_title("GHI Error by Forecast Hour")
    axes[1].set_xticks(HOURS)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_heatmap(
    frame: pd.DataFrame,
    value_col: str,
    title: str,
    path: Path,
) -> None:
    """Plot a forecast-hour by location heatmap."""
    pivot = frame.pivot(index="location", columns="forecast_hour", values=value_col).sort_index()
    if pivot.empty:
        return
    height = max(4.0, 0.35 * len(pivot.index) + 1.5)
    fig, ax = plt.subplots(figsize=(10.5, height))
    image = ax.imshow(pivot.to_numpy(dtype=np.float64), aspect="auto", cmap="magma")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(int(col)) for col in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Forecast hour")
    ax.set_ylabel("Location")
    ax.set_title(title)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("RMSE")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def per_location_hour_metrics(valid: pd.DataFrame) -> pd.DataFrame:
    """Compute location-hour RMSE values for heatmaps."""
    rows: list[dict[str, Any]] = []
    for (location, hour), group in valid.groupby(["location", "forecast_hour"], sort=True):
        csi = regression_metrics(group["pred_csi"].to_numpy(), group["target_csi"].to_numpy())
        ghi = regression_metrics(group["pred_ghi"].to_numpy(), group["target_ghi"].to_numpy())
        rows.append(
            {
                "location": location,
                "forecast_hour": int(hour),
                "CSI_RMSE": csi["rmse"],
                "GHI_RMSE": ghi["rmse"],
            }
        )
    return pd.DataFrame(rows)


def plot_timeseries_sample(sample: dict[str, Any], kind: str, path: Path) -> None:
    """Plot one CSI or GHI sample time series."""
    if kind == "csi":
        target = sample["target_csi"]
        prediction = sample["prediction_csi"]
        ylabel = "CSI"
    else:
        target = sample["target_ghi"]
        prediction = sample["prediction_ghi"]
        ylabel = "GHI"
    valid = sample["valid_mask"]
    hours = np.arange(1, len(target) + 1)

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.plot(hours[valid], target[valid], marker="o", label=f"Target {ylabel}", color="#4c78a8")
    ax.plot(hours[valid], prediction[valid], marker="o", label=f"Predicted {ylabel}", color="#f58518")
    ax.set_xticks(hours)
    ax.set_xlabel("Forecast hour (sunrise to sunset)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} forecast | {sample.get('location')} | {sample.get('target_day')}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def robust_image(image: np.ndarray) -> np.ndarray:
    """Scale a satellite frame for display using robust percentiles."""
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image)
    low, high = np.percentile(finite, [2.0, 98.0])
    if high <= low:
        high = low + EPS
    return np.clip((image - low) / (high - low), 0.0, 1.0)


def plot_sample_prediction(
    sample: dict[str, Any],
    frame_indices: list[int],
    channel_index: int,
    path: Path,
) -> None:
    """Plot representative satellite frames and CSI prediction errors."""
    satellite = sample["satellite"]
    if satellite.ndim != 4:
        return
    channel_index = min(max(channel_index, 0), satellite.shape[1] - 1)
    frame_indices = [idx for idx in frame_indices if idx < satellite.shape[0]]
    if not frame_indices:
        frame_indices = [0, satellite.shape[0] // 2, satellite.shape[0] - 1]

    target = sample["target_csi"]
    prediction = sample["prediction_csi"]
    error = np.abs(prediction - target)
    valid = sample["valid_mask"]
    hours = np.arange(1, len(target) + 1)

    width = max(10.0, 3.3 * len(frame_indices))
    fig = plt.figure(figsize=(width, 8.6))
    grid = fig.add_gridspec(3, len(frame_indices), height_ratios=[1.25, 1.0, 0.8])

    for column, frame_index in enumerate(frame_indices):
        ax = fig.add_subplot(grid[0, column])
        ax.imshow(robust_image(satellite[frame_index, channel_index]), cmap="gray")
        ax.set_title(f"Input frame {frame_index + 1}")
        ax.set_xticks([])
        ax.set_yticks([])

    ax_series = fig.add_subplot(grid[1, :])
    ax_series.plot(hours[valid], target[valid], marker="o", label="Target CSI", color="#4c78a8")
    ax_series.plot(hours[valid], prediction[valid], marker="o", label="Predicted CSI", color="#f58518")
    ax_series.set_xticks(hours)
    ax_series.set_xlabel("Forecast hour (sunrise to sunset)")
    ax_series.set_ylabel("CSI")
    ax_series.legend()

    ax_error = fig.add_subplot(grid[2, :])
    ax_error.bar(hours[valid], error[valid], color="#54a24b", alpha=0.85)
    ax_error.set_xticks(hours)
    ax_error.set_xlabel("Forecast hour")
    ax_error.set_ylabel("|CSI error|")

    fig.suptitle(f"Sample {sample_key(sample)}")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def sample_level_metrics(valid: pd.DataFrame) -> pd.DataFrame:
    """Compute per-sample RMSE for best/worst case selection."""
    rows: list[dict[str, Any]] = []
    for sample_index, group in valid.groupby("sample_index", sort=True):
        csi = regression_metrics(group["pred_csi"].to_numpy(), group["target_csi"].to_numpy())
        ghi = regression_metrics(group["pred_ghi"].to_numpy(), group["target_ghi"].to_numpy())
        first = group.iloc[0]
        rows.append(
            {
                "sample_index": int(sample_index),
                "sample_id": first["sample_id"],
                "location": first["location"],
                "input_day": first["input_day"],
                "target_day": first["target_day"],
                "CSI_RMSE": csi["rmse"],
                "GHI_RMSE": ghi["rmse"],
                "count": int(csi["count"]),
            }
        )
    return pd.DataFrame(rows).sort_values("CSI_RMSE")


def plot_case_from_rows(group: pd.DataFrame, path: Path, title: str) -> None:
    """Plot CSI and GHI time series for one case from prediction rows."""
    group = group.sort_values("forecast_hour")
    valid = group["valid"].to_numpy(dtype=bool)
    hours = group["forecast_hour"].to_numpy(dtype=int)
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.0), sharex=True)

    axes[0].plot(hours[valid], group.loc[valid, "target_csi"], marker="o", label="Target CSI")
    axes[0].plot(hours[valid], group.loc[valid, "pred_csi"], marker="o", label="Predicted CSI")
    axes[0].set_ylabel("CSI")
    axes[0].legend()

    axes[1].plot(hours[valid], group.loc[valid, "target_ghi"], marker="o", label="Target GHI")
    axes[1].plot(hours[valid], group.loc[valid, "pred_ghi"], marker="o", label="Predicted GHI")
    axes[1].set_xlabel("Forecast hour (sunrise to sunset)")
    axes[1].set_ylabel("GHI")
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_best_and_worst_cases(
    predictions: pd.DataFrame,
    sample_metrics: pd.DataFrame,
    dirs: EvaluationDirs,
    count: int,
) -> None:
    """Generate plots for the best and worst sample-level predictions."""
    best = sample_metrics.head(count)
    worst = sample_metrics.tail(count).sort_values("CSI_RMSE", ascending=False)
    best.to_csv(dirs.metrics / "best_cases.csv", index=False)
    worst.to_csv(dirs.metrics / "worst_cases.csv", index=False)

    for rank, row in enumerate(best.itertuples(index=False), start=1):
        group = predictions[predictions["sample_index"] == row.sample_index]
        title = f"Best case {rank}: {row.location} {row.target_day} | CSI RMSE={row.CSI_RMSE:.4f}"
        plot_case_from_rows(group, dirs.best_cases / f"best_{rank:02d}_{sample_key(row._asdict())}.png", title)

    for rank, row in enumerate(worst.itertuples(index=False), start=1):
        group = predictions[predictions["sample_index"] == row.sample_index]
        title = f"Worst case {rank}: {row.location} {row.target_day} | CSI RMSE={row.CSI_RMSE:.4f}"
        plot_case_from_rows(group, dirs.worst_cases / f"worst_{rank:02d}_{sample_key(row._asdict())}.png", title)


def generate_figures(
    predictions: pd.DataFrame,
    metric_tables: dict[str, pd.DataFrame],
    selected_samples: list[dict[str, Any]],
    dirs: EvaluationDirs,
    args: argparse.Namespace,
    seed: int,
) -> None:
    """Generate all evaluation figures."""
    valid = predictions[predictions["valid"]].copy()
    plot_scatter(valid, "target_csi", "pred_csi", "CSI", dirs.figures / "scatter_csi.png", args.max_scatter_points, seed)
    plot_scatter(valid, "target_ghi", "pred_ghi", "GHI", dirs.figures / "scatter_ghi.png", args.max_scatter_points, seed)

    plot_error_histogram(valid, "error_csi", "CSI", dirs.figures / "error_histogram_csi.png")
    plot_error_histogram(valid, "error_ghi", "GHI", dirs.figures / "error_histogram_ghi.png")
    plot_prediction_distribution(valid, "target_csi", "pred_csi", "CSI", dirs.figures / "histogram_csi.png")
    plot_prediction_distribution(valid, "target_ghi", "pred_ghi", "GHI", dirs.figures / "histogram_ghi.png")

    plot_residuals(valid, "target_csi", "pred_csi", "error_csi", "CSI", dirs.figures / "residual_csi.png", args.max_scatter_points, seed)
    plot_residuals(valid, "target_ghi", "pred_ghi", "error_ghi", "GHI", dirs.figures / "residual_ghi.png", args.max_scatter_points, seed)
    plot_hourly_rmse(metric_tables["per_hour"], dirs.figures / "hourly_rmse.png")

    location_hour = per_location_hour_metrics(valid)
    location_hour.to_csv(dirs.metrics / "per_location_hour.csv", index=False)
    plot_heatmap(location_hour, "CSI_RMSE", "CSI RMSE by Location and Forecast Hour", dirs.heatmaps / "csi_rmse_heatmap.png")
    plot_heatmap(location_hour, "GHI_RMSE", "GHI RMSE by Location and Forecast Hour", dirs.heatmaps / "ghi_rmse_heatmap.png")

    frame_indices = parse_indices(args.input_frame_indices, maximum=13)
    for rank, sample in enumerate(selected_samples[: args.num_samples], start=1):
        key = sample_key(sample)
        plot_timeseries_sample(sample, "csi", dirs.timeseries / f"sample_{rank:02d}_{key}_csi.png")
        plot_timeseries_sample(sample, "ghi", dirs.timeseries / f"sample_{rank:02d}_{key}_ghi.png")
        plot_sample_prediction(
            sample,
            frame_indices=frame_indices,
            channel_index=args.satellite_channel_index,
            path=dirs.sample_predictions / f"sample_{rank:02d}_{key}_input_prediction.png",
        )

    sample_metrics = sample_level_metrics(valid)
    sample_metrics.to_csv(dirs.metrics / "per_sample.csv", index=False)
    plot_best_and_worst_cases(predictions, sample_metrics, dirs, count=args.best_worst_count)


def main() -> None:
    """Run full evaluation."""
    args = parse_args()
    config = config_from_args(args)
    if args.batch_size is not None:
        config.batch_size = int(args.batch_size)
    config.prepare_directories()
    seed_everything(config.random_seed)
    configure_style()

    evaluation_root = args.evaluation_dir or (PROJECT_ROOT / "evaluation")
    dirs = EvaluationDirs.create(evaluation_root)
    checkpoint_path = args.checkpoint or (config.checkpoint_dir / "best.pt")
    device = torch.device(config.resolved_device())

    loader = build_dataloader(config=config, split=args.split, include_target=True, shuffle=False)
    model = load_model(config=config, checkpoint_path=checkpoint_path, device=device)

    predictions, selected_samples = run_inference(
        model=model,
        loader=loader,
        config=config,
        args=args,
        device=device,
        dirs=dirs,
    )
    metric_tables = save_metric_tables(predictions, dirs)
    generate_figures(
        predictions=predictions,
        metric_tables=metric_tables,
        selected_samples=selected_samples,
        dirs=dirs,
        args=args,
        seed=config.random_seed,
    )

    summary = metric_tables["overall"].iloc[0].to_dict()
    print("Evaluation complete")
    print(f"split={args.split}")
    print(f"checkpoint={checkpoint_path}")
    print(f"predictions={dirs.predictions / 'predictions.csv'}")
    print(f"metrics={dirs.metrics}")
    print(f"figures={dirs.figures}")
    print(
        "overall: "
        f"CSI_RMSE={summary['CSI_RMSE']:.6f}, "
        f"CSI_MAE={summary['CSI_MAE']:.6f}, "
        f"GHI_RMSE={summary['GHI_RMSE']:.6f}, "
        f"GHI_MAE={summary['GHI_MAE']:.6f}"
    )
    if config.mixed_precision:
        amp_dtype = resolve_amp_dtype(config.amp_dtype, device) if device.type == "cuda" else None
        print(f"amp={config.mixed_precision}, amp_dtype={amp_dtype_label(amp_dtype)}")


if __name__ == "__main__":
    main()
