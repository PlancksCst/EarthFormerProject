"""Plotting helpers for CSI/GHI forecasting training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
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
