"""Residual-space losses for explicit CSI residual experiments."""

from __future__ import annotations

import torch
from torch.nn import functional as F


RESIDUAL_LOSS_CHOICES = (
    "masked_residual_huber",
    "masked_residual_weighted_huber",
    "masked_residual_ramp_weighted_huber",
)


def _check_shapes(prediction: torch.Tensor, target: torch.Tensor) -> None:
    if prediction.shape != target.shape:
        raise ValueError(
            "Prediction and target residual shapes differ: "
            f"{tuple(prediction.shape)} vs {tuple(target.shape)}"
        )


def _valid(valid_mask: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor:
    if valid_mask is None:
        return torch.ones_like(reference, dtype=torch.bool)
    if valid_mask.shape != reference.shape:
        raise ValueError(
            "valid_mask and residual target shapes differ: "
            f"{tuple(valid_mask.shape)} vs {tuple(reference.shape)}"
        )
    return valid_mask.to(device=reference.device, dtype=torch.bool)


def _masked_reduce(
    elementwise_loss: torch.Tensor,
    weights: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    weights = weights.to(device=elementwise_loss.device, dtype=elementwise_loss.dtype)
    masked_weights = weights.masked_fill(~valid_mask, 0.0)
    normalizer = masked_weights.sum()
    if int(valid_mask.sum().detach().cpu()) == 0 or float(normalizer.detach().cpu()) <= 0.0:
        raise RuntimeError("No valid target positions available for residual loss.")
    return (elementwise_loss * masked_weights).sum() / normalizer.clamp_min(1.0e-12)


def cloudy_residual_weights(target_csi: torch.Tensor, cloudy_weight: float) -> torch.Tensor:
    """Return ``1 + cloudy_weight * (1 - target_csi)``."""
    if cloudy_weight <= 0.0:
        return torch.ones_like(target_csi)
    return (1.0 + float(cloudy_weight) * (1.0 - target_csi)).clamp_min(1.0e-6)


def ramp_residual_weights(
    target_csi: torch.Tensor,
    ramp_weight: float,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ramp weights from consecutive target-CSI changes."""
    if ramp_weight <= 0.0:
        return torch.ones_like(target_csi)
    delta = torch.zeros_like(target_csi)
    if target_csi.shape[-1] > 1:
        pair_delta = (target_csi[..., 1:] - target_csi[..., :-1]).abs()
        if valid_mask is not None:
            valid = valid_mask.to(device=target_csi.device, dtype=torch.bool)
            pair_delta = pair_delta.masked_fill(~(valid[..., 1:] & valid[..., :-1]), 0.0)
        delta[..., 1:] = pair_delta
    return 1.0 + float(ramp_weight) * delta


def masked_residual_huber(
    pred_residual: torch.Tensor,
    target_residual: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    target_csi: torch.Tensor | None = None,
    beta: float = 0.1,
    cloudy_weight: float = 1.0,
    ramp_weight: float = 1.0,
) -> torch.Tensor:
    """Huber loss on residual CSI targets over valid forecast hours."""
    del target_csi, cloudy_weight, ramp_weight
    _check_shapes(pred_residual, target_residual)
    target = target_residual.to(device=pred_residual.device, dtype=pred_residual.dtype)
    valid = _valid(valid_mask, target)
    loss = F.smooth_l1_loss(pred_residual, target, reduction="none", beta=float(beta))
    return _masked_reduce(loss, torch.ones_like(target), valid)


def masked_residual_weighted_huber(
    pred_residual: torch.Tensor,
    target_residual: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    target_csi: torch.Tensor | None = None,
    beta: float = 0.1,
    cloudy_weight: float = 1.0,
    ramp_weight: float = 1.0,
) -> torch.Tensor:
    """Cloudy-weighted Huber loss on residual CSI targets."""
    del ramp_weight
    _check_shapes(pred_residual, target_residual)
    target = target_residual.to(device=pred_residual.device, dtype=pred_residual.dtype)
    target_csi_tensor = target if target_csi is None else target_csi.to(
        device=pred_residual.device,
        dtype=pred_residual.dtype,
    )
    valid = _valid(valid_mask, target)
    loss = F.smooth_l1_loss(pred_residual, target, reduction="none", beta=float(beta))
    weights = cloudy_residual_weights(target_csi_tensor, cloudy_weight)
    return _masked_reduce(loss, weights, valid)


def masked_residual_ramp_weighted_huber(
    pred_residual: torch.Tensor,
    target_residual: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    target_csi: torch.Tensor | None = None,
    beta: float = 0.1,
    cloudy_weight: float = 1.0,
    ramp_weight: float = 1.0,
) -> torch.Tensor:
    """Cloudy- and ramp-weighted Huber loss on residual CSI targets."""
    _check_shapes(pred_residual, target_residual)
    target = target_residual.to(device=pred_residual.device, dtype=pred_residual.dtype)
    target_csi_tensor = target if target_csi is None else target_csi.to(
        device=pred_residual.device,
        dtype=pred_residual.dtype,
    )
    valid = _valid(valid_mask, target)
    loss = F.smooth_l1_loss(pred_residual, target, reduction="none", beta=float(beta))
    weights = cloudy_residual_weights(target_csi_tensor, cloudy_weight)
    weights = weights * ramp_residual_weights(target_csi_tensor, ramp_weight, valid_mask=valid)
    return _masked_reduce(loss, weights, valid)


class ResidualLoss:
    """Configurable residual loss callable for explicit residual presets."""

    def __init__(
        self,
        loss_name: str = "masked_residual_weighted_huber",
        beta: float = 0.1,
        cloudy_weight: float = 1.0,
        ramp_weight: float = 1.0,
    ) -> None:
        if loss_name not in RESIDUAL_LOSS_CHOICES:
            raise ValueError(
                f"Unsupported residual loss '{loss_name}'. "
                f"Expected one of: {', '.join(RESIDUAL_LOSS_CHOICES)}"
            )
        self.loss_name = loss_name
        self.beta = float(beta)
        self.cloudy_weight = float(cloudy_weight)
        self.ramp_weight = float(ramp_weight)

    def __call__(
        self,
        pred_residual: torch.Tensor,
        target_residual: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        target_csi: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fn = globals()[self.loss_name]
        return fn(
            pred_residual=pred_residual,
            target_residual=target_residual,
            valid_mask=valid_mask,
            target_csi=target_csi,
            beta=self.beta,
            cloudy_weight=self.cloudy_weight,
            ramp_weight=self.ramp_weight,
        )
