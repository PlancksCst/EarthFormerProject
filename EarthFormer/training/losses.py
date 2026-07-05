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


def valid_hour_mask(
    target_mask: torch.Tensor | None,
    reference: torch.Tensor,
    clear_sky_ghi: torch.Tensor | None = None,
    clear_sky_threshold: float = 20.0,
) -> torch.Tensor:
    """Return physically valid forecast hours.

    Validity combines the dataset mask convention with a clear-sky GHI
    threshold. Synthetic diagnostic targets may omit ``clear_sky_ghi``; in
    that case this falls back to the original target-mask behavior.
    """
    valid_mask = valid_mask_from_target_mask(target_mask, reference)
    if clear_sky_ghi is None:
        return valid_mask

    clear = clear_sky_ghi.to(device=reference.device)
    if clear.shape != reference.shape:
        raise ValueError(
            "clear_sky_ghi and target shapes differ: "
            f"{tuple(clear.shape)} vs {tuple(reference.shape)}"
        )
    return valid_mask & torch.isfinite(clear) & (clear > float(clear_sky_threshold))


def _check_prediction_target_shapes(prediction: torch.Tensor, target: torch.Tensor) -> None:
    if prediction.shape != target.shape:
        raise ValueError(
            "Prediction and target shapes differ: "
            f"{tuple(prediction.shape)} vs {tuple(target.shape)}"
        )


def masked_weighted_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute weighted MSE over valid positions."""
    _check_prediction_target_shapes(prediction, target)
    if valid_mask is None:
        valid_mask = torch.ones_like(target, dtype=torch.bool)
    elif valid_mask.shape != target.shape:
        raise ValueError(
            "valid_mask and target shapes differ: "
            f"{tuple(valid_mask.shape)} vs {tuple(target.shape)}"
        )
    valid_mask = valid_mask.to(device=prediction.device, dtype=torch.bool)

    if weights is None:
        weights = torch.ones_like(target, dtype=prediction.dtype, device=prediction.device)
    else:
        if weights.shape != target.shape:
            raise ValueError(
                "weights and target shapes differ: "
                f"{tuple(weights.shape)} vs {tuple(target.shape)}"
            )
        weights = weights.to(device=prediction.device, dtype=prediction.dtype)

    weighted_mask = weights.masked_fill(~valid_mask, 0.0)
    normalizer = weighted_mask.sum()
    if int(valid_mask.sum().detach().cpu()) == 0 or float(normalizer.detach().cpu()) <= 0.0:
        raise RuntimeError(
            "No valid target positions available for loss. "
            "Remember: target_mask=0 means valid and target_mask=1 means invalid."
        )
    squared_error = (prediction - target) ** 2
    return (squared_error * weighted_mask).sum() / normalizer.clamp_min(1.0e-12)


def masked_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute MSE over valid positions, with invalid positions contributing zero."""
    if valid_mask is None:
        _check_prediction_target_shapes(prediction, target)
        return torch.mean((prediction - target) ** 2)
    return masked_weighted_mse_loss(prediction, target, valid_mask=valid_mask)


class MSELoss:
    """Forecasting MSE with optional valid masking, CSI weighting, and GHI loss."""

    def __init__(
        self,
        low_csi_weight: float = 0.0,
        low_csi_threshold: float = 0.7,
        ghi_loss_weight: float = 0.0,
    ) -> None:
        self.loss = nn.MSELoss()
        self.low_csi_weight = float(low_csi_weight)
        self.low_csi_threshold = float(low_csi_threshold)
        self.ghi_loss_weight = float(ghi_loss_weight)

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        clear_sky_ghi: torch.Tensor | None = None,
        target_ghi: torch.Tensor | None = None,
        return_components: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Compute mean squared error."""
        _check_prediction_target_shapes(prediction, target)
        if valid_mask is None and self.low_csi_weight <= 0.0 and self.ghi_loss_weight <= 0.0:
            csi_loss = self.loss(prediction, target)
        else:
            weights = None
            if self.low_csi_weight > 0.0:
                low_csi = (target < self.low_csi_threshold).to(
                    device=prediction.device,
                    dtype=prediction.dtype,
                )
                weights = 1.0 + self.low_csi_weight * low_csi
            csi_loss = masked_weighted_mse_loss(
                prediction,
                target,
                valid_mask=valid_mask,
                weights=weights,
            )

        ghi_loss = prediction.new_zeros(())
        if self.ghi_loss_weight > 0.0:
            if clear_sky_ghi is None:
                raise ValueError("clear_sky_ghi is required when ghi_loss_weight > 0")
            clear = clear_sky_ghi.to(device=prediction.device, dtype=prediction.dtype)
            if clear.shape != target.shape:
                raise ValueError(
                    "clear_sky_ghi and target shapes differ: "
                    f"{tuple(clear.shape)} vs {tuple(target.shape)}"
                )
            if target_ghi is None:
                target_ghi_tensor = target * clear
            else:
                target_ghi_tensor = target_ghi.to(device=prediction.device, dtype=prediction.dtype)
                if target_ghi_tensor.shape != target.shape:
                    raise ValueError(
                        "target_ghi and target shapes differ: "
                        f"{tuple(target_ghi_tensor.shape)} vs {tuple(target.shape)}"
                    )
            pred_ghi = prediction * clear
            ghi_loss = masked_weighted_mse_loss(
                pred_ghi,
                target_ghi_tensor,
                valid_mask=valid_mask,
            )

        total_loss = csi_loss + self.ghi_loss_weight * ghi_loss
        if return_components:
            return {
                "loss": total_loss,
                "csi_loss": csi_loss,
                "ghi_loss": ghi_loss,
            }
        return total_loss
