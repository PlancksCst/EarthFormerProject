"""Validation utilities for CSI forecasting."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from utils.metrics import forecast_metrics
except ImportError:
    from EarthFormer.utils.metrics import forecast_metrics  # type: ignore


def ensure_forecast_target(target: torch.Tensor, name: str = "target") -> torch.Tensor:
    """Validate and return a `(B, T)` forecasting tensor."""
    if target.ndim != 2:
        raise ValueError(f"Expected {name} with shape (B,T), got {tuple(target.shape)}")
    return target.float()


def reconstruct_ghi(csi: torch.Tensor, clear_sky_ghi: torch.Tensor) -> torch.Tensor:
    """Reconstruct GHI from predicted or observed CSI and clear-sky GHI."""
    return csi * clear_sky_ghi


def _first_metadata_value(batch: dict[str, Any], key: str) -> Any:
    value = batch.get(key)
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        first = value[0]
        return first.item() if first.numel() == 1 else first.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    use_amp: bool = False,
) -> dict[str, Any]:
    """Return validation loss, CSI metrics, GHI metrics, and one plot sample."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    amp_enabled = use_amp and device.type == "cuda"
    csi_predictions: list[torch.Tensor] = []
    csi_targets: list[torch.Tensor] = []
    ghi_predictions: list[torch.Tensor] = []
    ghi_targets: list[torch.Tensor] = []
    sample: dict[str, Any] | None = None

    for batch in dataloader:
        inputs = batch["satellite"].to(device, non_blocking=True)
        targets = ensure_forecast_target(batch["target"], "target").to(
            device,
            non_blocking=True,
        )
        clear_sky_ghi = ensure_forecast_target(
            batch["clear_sky_ghi"],
            "clear_sky_ghi",
        ).to(device, non_blocking=True)
        target_ghi = batch.get("target_ghi")
        if target_ghi is None:
            target_ghi_tensor = reconstruct_ghi(targets, clear_sky_ghi)
        else:
            target_ghi_tensor = ensure_forecast_target(target_ghi, "target_ghi").to(
                device,
                non_blocking=True,
            )

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            predictions = model(inputs)
            if predictions.shape != targets.shape:
                raise ValueError(
                    "Prediction and target shapes differ: "
                    f"{tuple(predictions.shape)} vs {tuple(targets.shape)}"
                )
            loss = criterion(predictions, targets)

        prediction_ghi = reconstruct_ghi(predictions, clear_sky_ghi)
        batch_size = inputs.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

        csi_predictions.append(predictions.detach().cpu())
        csi_targets.append(targets.detach().cpu())
        ghi_predictions.append(prediction_ghi.detach().cpu())
        ghi_targets.append(target_ghi_tensor.detach().cpu())

        if sample is None:
            sample = {
                "prediction_csi": predictions[0].detach().cpu(),
                "target_csi": targets[0].detach().cpu(),
                "prediction_ghi": prediction_ghi[0].detach().cpu(),
                "target_ghi": target_ghi_tensor[0].detach().cpu(),
                "clear_sky_ghi": clear_sky_ghi[0].detach().cpu(),
                "sample_id": _first_metadata_value(batch, "sample_id"),
                "location": _first_metadata_value(batch, "location"),
                "input_day": _first_metadata_value(batch, "input_day"),
                "target_day": _first_metadata_value(batch, "target_day"),
            }

    if total_samples == 0:
        raise ValueError("Validation dataloader produced no samples")

    all_csi_predictions = torch.cat(csi_predictions, dim=0)
    all_csi_targets = torch.cat(csi_targets, dim=0)
    all_ghi_predictions = torch.cat(ghi_predictions, dim=0)
    all_ghi_targets = torch.cat(ghi_targets, dim=0)

    metrics: dict[str, Any] = {"val_loss": total_loss / total_samples}
    metrics.update(forecast_metrics(all_csi_predictions, all_csi_targets, prefix="CSI"))
    metrics.update(forecast_metrics(all_ghi_predictions, all_ghi_targets, prefix="GHI"))
    metrics["sample"] = sample
    return metrics
