"""Loss functions for EarthFormer training."""

from __future__ import annotations

import torch
from torch import nn


def valid_mask_from_target_mask(
    target_mask: torch.Tensor | None,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Return `True` where targets are valid.

    Dataset masks use the original preprocessing convention:
    `0` means valid and `1` means invalid, padded, or missing.
    """
    if target_mask is None:
        return torch.ones_like(reference, dtype=torch.bool)
    mask = target_mask.to(device=reference.device)
    if mask.shape != reference.shape:
        raise ValueError(
            "target_mask and target shapes differ: "
            f"{tuple(mask.shape)} vs {tuple(reference.shape)}"
        )
    return ~mask.bool()


def masked_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute MSE over valid positions, with invalid positions contributing zero."""
    if prediction.shape != target.shape:
        raise ValueError(
            "Prediction and target shapes differ: "
            f"{tuple(prediction.shape)} vs {tuple(target.shape)}"
        )
    if valid_mask is None:
        return torch.mean((prediction - target) ** 2)
    if valid_mask.shape != target.shape:
        raise ValueError(
            "valid_mask and target shapes differ: "
            f"{tuple(valid_mask.shape)} vs {tuple(target.shape)}"
        )
    valid_mask = valid_mask.to(device=prediction.device, dtype=torch.bool)
    valid_count = valid_mask.sum()
    if int(valid_count.detach().cpu()) == 0:
        raise RuntimeError(
            "No valid target positions available for loss. "
            "Remember: target_mask=0 means valid and target_mask=1 means invalid."
        )
    squared_error = (prediction - target) ** 2
    masked_error = squared_error.masked_fill(~valid_mask, 0.0)
    return masked_error.sum() / valid_count.clamp_min(1)


class MSELoss:
    """Small callable wrapper around MSE, optionally using a valid-position mask."""

    def __init__(self) -> None:
        self.loss = nn.MSELoss()

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute mean squared error."""
        if valid_mask is not None:
            return masked_mse_loss(prediction, target, valid_mask)
        return self.loss(prediction, target)
