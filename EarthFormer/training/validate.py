"""Validation utilities for CSI forecasting."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from training.debugging import assert_finite, assert_scalar_finite
    from training.losses import valid_mask_from_target_mask
except ImportError:
    from EarthFormer.training.debugging import assert_finite, assert_scalar_finite  # type: ignore
    from EarthFormer.training.losses import valid_mask_from_target_mask  # type: ignore

try:
    from utils.metrics import forecast_metrics
    from utils.precision import autocast_context
except ImportError:
    from EarthFormer.utils.metrics import forecast_metrics  # type: ignore
    from EarthFormer.utils.precision import autocast_context  # type: ignore


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


def _metadata_value(batch: dict[str, Any], key: str, index: int) -> Any:
    value = batch.get(key)
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        item = value[index]
        return item.item() if item.numel() == 1 else item.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return value[index] if index < len(value) else None
    return value


def _prediction_rows(
    batch: dict[str, Any],
    predictions_csi: torch.Tensor,
    predictions_ghi: torch.Tensor,
    targets_csi: torch.Tensor,
    targets_ghi: torch.Tensor,
    clear_sky_ghi: torch.Tensor,
    valid_mask: torch.Tensor,
) -> list[dict[str, Any]]:
    pred_csi = predictions_csi.detach().float().cpu()
    pred_ghi = predictions_ghi.detach().float().cpu()
    target_csi = targets_csi.detach().float().cpu()
    target_ghi = targets_ghi.detach().float().cpu()
    clear = clear_sky_ghi.detach().float().cpu()
    valid = valid_mask.detach().cpu().bool()

    rows: list[dict[str, Any]] = []
    batch_size, horizon = pred_csi.shape
    for sample_index in range(batch_size):
        metadata = {
            "sample_id": _metadata_value(batch, "sample_id", sample_index),
            "location": _metadata_value(batch, "location", sample_index),
            "input_day": _metadata_value(batch, "input_day", sample_index),
            "day": _metadata_value(batch, "target_day", sample_index),
            "target_day": _metadata_value(batch, "target_day", sample_index),
        }
        for hour_index in range(horizon):
            rows.append(
                {
                    **metadata,
                    "hour": hour_index + 1,
                    "forecast_hour": hour_index + 1,
                    "valid": bool(valid[sample_index, hour_index]),
                    "target_csi": float(target_csi[sample_index, hour_index]),
                    "predicted_csi": float(pred_csi[sample_index, hour_index]),
                    "target_ghi": float(target_ghi[sample_index, hour_index]),
                    "predicted_ghi": float(pred_ghi[sample_index, hour_index]),
                    "clear_sky_ghi": float(clear[sample_index, hour_index]),
                }
            )
    return rows


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Callable[..., torch.Tensor],
    device: torch.device,
    use_amp: bool = False,
    amp_dtype: torch.dtype | None = None,
    collect_predictions: bool = False,
) -> dict[str, Any]:
    """Return validation loss, CSI metrics, GHI metrics, and one plot sample."""
    model.eval()
    total_loss = 0.0
    total_valid_positions = 0
    amp_enabled = use_amp and device.type == "cuda"
    csi_predictions: list[torch.Tensor] = []
    csi_targets: list[torch.Tensor] = []
    ghi_predictions: list[torch.Tensor] = []
    ghi_targets: list[torch.Tensor] = []
    prediction_rows: list[dict[str, Any]] = []
    sample: dict[str, Any] | None = None

    for batch_index, batch in enumerate(dataloader):
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
        target_mask = batch.get("target_mask")
        if isinstance(target_mask, torch.Tensor):
            target_mask = target_mask.to(device, non_blocking=True)
            assert_finite(
                "target_mask",
                target_mask.float(),
                batch=batch,
                batch_index=batch_index,
            )
        valid_mask = valid_mask_from_target_mask(target_mask, targets)
        valid_count = int(valid_mask.sum().detach().cpu())
        if valid_count == 0:
            raise RuntimeError(
                "No valid target positions in validation batch. "
                "Mask convention is target_mask=0 valid, target_mask=1 invalid."
            )

        assert_finite("inputs", inputs, batch=batch, batch_index=batch_index)
        assert_finite("targets", targets, batch=batch, batch_index=batch_index)
        assert_finite(
            "clear_sky_ghi",
            clear_sky_ghi,
            batch=batch,
            batch_index=batch_index,
        )
        assert_finite(
            "target_ghi",
            target_ghi_tensor,
            batch=batch,
            batch_index=batch_index,
        )

        with autocast_context(device=device, enabled=amp_enabled, dtype=amp_dtype):
            predictions = model(inputs)
            if predictions.shape != targets.shape:
                raise ValueError(
                    "Prediction and target shapes differ: "
                    f"{tuple(predictions.shape)} vs {tuple(targets.shape)}"
                )
            assert_finite(
                "predictions",
                predictions,
                batch=batch,
                batch_index=batch_index,
            )
            loss = criterion(predictions, targets, valid_mask=valid_mask)
            assert_scalar_finite("loss", loss, batch=batch, batch_index=batch_index)

        prediction_ghi = reconstruct_ghi(predictions, clear_sky_ghi)
        assert_finite(
            "predicted_ghi",
            prediction_ghi,
            batch=batch,
            batch_index=batch_index,
        )
        total_loss += float(loss.item()) * valid_count
        total_valid_positions += valid_count

        valid_mask_cpu = valid_mask.detach().cpu()
        csi_predictions.append(predictions.detach().cpu()[valid_mask_cpu])
        csi_targets.append(targets.detach().cpu()[valid_mask_cpu])
        ghi_predictions.append(prediction_ghi.detach().cpu()[valid_mask_cpu])
        ghi_targets.append(target_ghi_tensor.detach().cpu()[valid_mask_cpu])
        if collect_predictions:
            prediction_rows.extend(
                _prediction_rows(
                    batch=batch,
                    predictions_csi=predictions,
                    predictions_ghi=prediction_ghi,
                    targets_csi=targets,
                    targets_ghi=target_ghi_tensor,
                    clear_sky_ghi=clear_sky_ghi,
                    valid_mask=valid_mask,
                )
            )

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

    if total_valid_positions == 0:
        raise ValueError("Validation dataloader produced no samples")

    all_csi_predictions = torch.cat(csi_predictions, dim=0)
    all_csi_targets = torch.cat(csi_targets, dim=0)
    all_ghi_predictions = torch.cat(ghi_predictions, dim=0)
    all_ghi_targets = torch.cat(ghi_targets, dim=0)

    metrics: dict[str, Any] = {"val_loss": total_loss / total_valid_positions}
    metrics.update(forecast_metrics(all_csi_predictions, all_csi_targets, prefix="CSI"))
    metrics.update(forecast_metrics(all_ghi_predictions, all_ghi_targets, prefix="GHI"))
    metrics["sample"] = sample
    if collect_predictions:
        metrics["predictions"] = prediction_rows
    return metrics
