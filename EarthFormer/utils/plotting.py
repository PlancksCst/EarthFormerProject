"""Plotting helpers for CSI/GHI forecasting training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import math  # noqa: E402
import torch  # noqa: E402


def plots_dir(output_dir: str | Path) -> Path:
    """Return and create the plots directory."""
    path = Path(output_dir) / "plots"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_list(value: Any) -> list[float]:
    if isinstance(value, torch.Tensor):
        return [float(item) for item in value.detach().cpu().reshape(-1)]
    return [float(item) for item in value]


def _history_values(history: list[dict[str, float]], key: str) -> list[float]:
    return [float(row[key]) for row in history if key in row]


def plot_loss_curves(history: list[dict[str, float]], output_dir: str | Path) -> Path:
    """Plot training and validation loss over epochs."""
    path = plots_dir(output_dir) / "loss_curves.png"
    epochs = _history_values(history, "epoch")
    train_loss = _history_values(history, "train_loss")
    val_loss = _history_values(history, "val_loss")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, train_loss, marker="o", label="Training loss")
    ax.plot(epochs, val_loss, marker="o", label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_metric_curves(history: list[dict[str, float]], output_dir: str | Path) -> Path:
    """Plot CSI and reconstructed GHI learning curves."""
    path = plots_dir(output_dir) / "forecast_metric_curves.png"
    epochs = _history_values(history, "epoch")
    metric_names = ("MAE", "RMSE", "nRMSE", "R2")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for axis, metric in zip(axes.reshape(-1), metric_names):
        csi_key = f"CSI_{metric}"
        ghi_key = f"GHI_{metric}"
        axis.plot(epochs, _history_values(history, csi_key), marker="o", label=f"CSI {metric}")
        axis.plot(epochs, _history_values(history, ghi_key), marker="o", label=f"GHI {metric}")
        axis.set_title(metric)
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_forecast_sample(
    sample: dict[str, Any],
    output_dir: str | Path,
    epoch: int,
    kind: str,
) -> Path:
    """Plot one validation sample for CSI or GHI."""
    if kind not in {"csi", "ghi"}:
        raise ValueError(f"Unsupported sample plot kind: {kind}")

    target_key = f"target_{kind}"
    prediction_key = f"prediction_{kind}"
    target = _to_list(sample[target_key])
    prediction = _to_list(sample[prediction_key])
    hours = list(range(1, len(target) + 1))

    path = plots_dir(output_dir) / f"{kind}_prediction_epoch_{epoch:03d}.png"
    label = kind.upper()
    title_parts = [f"{label} validation sample"]
    if sample.get("location") is not None:
        title_parts.append(str(sample["location"]))
    if sample.get("target_day") is not None:
        title_parts.append(str(sample["target_day"]))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(hours, target, marker="o", label=f"Ground truth {label}")
    ax.plot(hours, prediction, marker="o", label=f"Predicted {label}")
    ax.set_xlabel("Forecast hour")
    ax.set_ylabel(label)
    ax.set_title(" | ".join(title_parts))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_training_plots(
    history: list[dict[str, float]],
    sample: dict[str, Any] | None,
    output_dir: str | Path,
    epoch: int,
) -> dict[str, Path]:
    """Save all standard training plots for the current epoch."""
    paths = {
        "loss": plot_loss_curves(history, output_dir),
        "metrics": plot_metric_curves(history, output_dir),
    }
    if sample is not None:
        paths["csi_sample"] = plot_forecast_sample(sample, output_dir, epoch, "csi")
        paths["ghi_sample"] = plot_forecast_sample(sample, output_dir, epoch, "ghi")
    return paths


def _valid_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if bool(row.get("valid", True))]


def _row_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            values.append(number)
    return values


def _paired_values(
    rows: list[dict[str, Any]],
    target_key: str,
    prediction_key: str,
) -> tuple[list[float], list[float]]:
    target: list[float] = []
    prediction: list[float] = []
    for row in rows:
        target_value = row.get(target_key)
        prediction_value = row.get(prediction_key)
        if target_value is None or prediction_value is None:
            continue
        target_number = float(target_value)
        prediction_number = float(prediction_value)
        if math.isfinite(target_number) and math.isfinite(prediction_number):
            target.append(target_number)
            prediction.append(prediction_number)
    return target, prediction


def _plot_distribution(
    rows: list[dict[str, Any]],
    target_key: str,
    prediction_key: str,
    label: str,
    path: Path,
) -> None:
    target, prediction = _paired_values(rows, target_key, prediction_key)
    if not target or not prediction:
        return
    lower = min(min(target), min(prediction))
    upper = max(max(target), max(prediction))
    if upper <= lower:
        upper = lower + 1.0e-8
    bins = [lower + (upper - lower) * index / 60 for index in range(61)]

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.hist(target, bins=bins, alpha=0.55, density=True, label=f"Ground truth {label}")
    ax.hist(prediction, bins=bins, alpha=0.55, density=True, label=f"Predicted {label}")
    ax.set_xlim(lower, upper)
    ax.set_xlabel(label)
    ax.set_ylabel("Density")
    ax.set_title(f"{label} Prediction Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_scatter(
    rows: list[dict[str, Any]],
    target_key: str,
    prediction_key: str,
    label: str,
    path: Path,
) -> None:
    target, prediction = _paired_values(rows, target_key, prediction_key)
    if not target or not prediction:
        return
    lower = min(min(target), min(prediction))
    upper = max(max(target), max(prediction))
    padding = 0.04 * max(upper - lower, 1.0e-8)
    lower -= padding
    upper += padding

    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    ax.scatter(target, prediction, s=9, alpha=0.25, edgecolors="none", rasterized=True)
    ax.plot([lower, upper], [lower, upper], color="black", linestyle="--", linewidth=1.2)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel(f"Ground truth {label}")
    ax.set_ylabel(f"Predicted {label}")
    ax.set_title(f"{label} Prediction vs Target")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_residual_histogram(
    rows: list[dict[str, Any]],
    target_key: str,
    prediction_key: str,
    label: str,
    path: Path,
) -> None:
    target, prediction = _paired_values(rows, target_key, prediction_key)
    residual = [pred - tgt for tgt, pred in zip(target, prediction)]
    if not residual:
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.hist(residual, bins=60, alpha=0.82)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
    ax.axvline(sum(residual) / len(residual), color="#c43b3b", linewidth=1.4, label="Mean residual")
    ax.set_xlabel(f"{label} residual (prediction - target)")
    ax.set_ylabel("Count")
    ax.set_title(f"{label} Residual Histogram")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_residual_vs_prediction(
    rows: list[dict[str, Any]],
    target_key: str,
    prediction_key: str,
    label: str,
    path: Path,
) -> None:
    target, prediction = _paired_values(rows, target_key, prediction_key)
    residual = [pred - tgt for tgt, pred in zip(target, prediction)]
    if not residual:
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(prediction, residual, s=9, alpha=0.25, edgecolors="none", rasterized=True)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel(f"Predicted {label}")
    ax.set_ylabel(f"{label} residual")
    ax.set_title(f"{label} Residual vs Prediction")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_validation_diagnostic_plots(
    prediction_rows: list[dict[str, Any]],
    output_dir: str | Path,
    epoch: int,
    plot_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Save distribution, scatter, and residual diagnostics for validation."""
    rows = _valid_rows(prediction_rows)
    target_dir = Path(plot_dir) if plot_dir is not None else plots_dir(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"epoch_{epoch:03d}"
    paths = {
        "histogram_csi": target_dir / f"histogram_csi_{suffix}.png",
        "histogram_ghi": target_dir / f"histogram_ghi_{suffix}.png",
        "scatter_csi": target_dir / f"scatter_csi_{suffix}.png",
        "scatter_ghi": target_dir / f"scatter_ghi_{suffix}.png",
        "residual_histogram_csi": target_dir / f"residual_histogram_csi_{suffix}.png",
        "residual_histogram_ghi": target_dir / f"residual_histogram_ghi_{suffix}.png",
        "residual_vs_prediction_csi": target_dir / f"residual_vs_prediction_csi_{suffix}.png",
        "residual_vs_prediction_ghi": target_dir / f"residual_vs_prediction_ghi_{suffix}.png",
    }
    _plot_distribution(rows, "target_csi", "predicted_csi", "CSI", paths["histogram_csi"])
    _plot_distribution(rows, "target_ghi", "predicted_ghi", "GHI", paths["histogram_ghi"])
    _plot_scatter(rows, "target_csi", "predicted_csi", "CSI", paths["scatter_csi"])
    _plot_scatter(rows, "target_ghi", "predicted_ghi", "GHI", paths["scatter_ghi"])
    _plot_residual_histogram(rows, "target_csi", "predicted_csi", "CSI", paths["residual_histogram_csi"])
    _plot_residual_histogram(rows, "target_ghi", "predicted_ghi", "GHI", paths["residual_histogram_ghi"])
    _plot_residual_vs_prediction(rows, "target_csi", "predicted_csi", "CSI", paths["residual_vs_prediction_csi"])
    _plot_residual_vs_prediction(rows, "target_ghi", "predicted_ghi", "GHI", paths["residual_vs_prediction_ghi"])
    return {key: path for key, path in paths.items() if path.exists()}
