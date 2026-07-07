"""Loss functions for EarthFormer training."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


SUPPORTED_LOSSES = (
    "masked_mse",
    "masked_mae",
    "masked_huber",
    "masked_weighted_mse",
    "masked_weighted_huber",
    "masked_ramp_weighted_mse",
    "masked_hybrid_mse_correlation",
    "masked_hybrid_huber_correlation",
)


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


def _resolve_valid_mask(
    valid_mask: torch.Tensor | None,
    target: torch.Tensor,
    prediction: torch.Tensor,
) -> torch.Tensor:
    """Return a boolean valid mask on the prediction device."""
    if valid_mask is None:
        return torch.ones_like(target, dtype=torch.bool, device=prediction.device)
    if valid_mask.shape != target.shape:
        raise ValueError(
            "valid_mask and target shapes differ: "
            f"{tuple(valid_mask.shape)} vs {tuple(target.shape)}"
        )
    return valid_mask.to(device=prediction.device, dtype=torch.bool)


def _resolve_weights(
    weights: torch.Tensor | None,
    target: torch.Tensor,
    prediction: torch.Tensor,
) -> torch.Tensor:
    """Return non-negative weights on the prediction device."""
    if weights is None:
        return torch.ones_like(target, dtype=prediction.dtype, device=prediction.device)
    if weights.shape != target.shape:
        raise ValueError(
            "weights and target shapes differ: "
            f"{tuple(weights.shape)} vs {tuple(target.shape)}"
        )
    return weights.to(device=prediction.device, dtype=prediction.dtype).clamp_min(0.0)


def _masked_weighted_reduce(
    elementwise_loss: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reduce an elementwise loss over physically valid forecast hours."""
    if elementwise_loss.shape != target.shape:
        raise ValueError(
            "elementwise_loss and target shapes differ: "
            f"{tuple(elementwise_loss.shape)} vs {tuple(target.shape)}"
        )
    valid = _resolve_valid_mask(valid_mask, target, prediction)
    resolved_weights = _resolve_weights(weights, target, prediction)
    weighted_mask = resolved_weights.masked_fill(~valid, 0.0)
    normalizer = weighted_mask.sum()
    if int(valid.sum().detach().cpu()) == 0 or float(normalizer.detach().cpu()) <= 0.0:
        raise RuntimeError(
            "No valid target positions available for loss. "
            "Remember: target_mask=0 means valid and target_mask=1 means invalid."
        )
    return (elementwise_loss * weighted_mask).sum() / normalizer.clamp_min(1.0e-12)


def masked_weighted_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute weighted MSE over valid positions."""
    _check_prediction_target_shapes(prediction, target)
    return _masked_weighted_reduce(
        (prediction - target) ** 2,
        target=target,
        prediction=prediction,
        valid_mask=valid_mask,
        weights=weights,
    )


def masked_weighted_mae_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute weighted MAE over valid positions."""
    _check_prediction_target_shapes(prediction, target)
    return _masked_weighted_reduce(
        (prediction - target).abs(),
        target=target,
        prediction=prediction,
        valid_mask=valid_mask,
        weights=weights,
    )


def masked_weighted_huber_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    beta: float = 0.1,
) -> torch.Tensor:
    """Compute weighted SmoothL1/Huber loss over valid positions."""
    _check_prediction_target_shapes(prediction, target)
    elementwise = F.smooth_l1_loss(
        prediction,
        target,
        reduction="none",
        beta=float(beta),
    )
    return _masked_weighted_reduce(
        elementwise,
        target=target,
        prediction=prediction,
        valid_mask=valid_mask,
        weights=weights,
    )


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


def masked_mae_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute MAE over valid positions, with invalid positions contributing zero."""
    if valid_mask is None:
        _check_prediction_target_shapes(prediction, target)
        return torch.mean((prediction - target).abs())
    return masked_weighted_mae_loss(prediction, target, valid_mask=valid_mask)


def masked_huber_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    beta: float = 0.1,
) -> torch.Tensor:
    """Compute SmoothL1/Huber loss over valid positions."""
    if valid_mask is None:
        _check_prediction_target_shapes(prediction, target)
        return F.smooth_l1_loss(
            prediction,
            target,
            reduction="mean",
            beta=float(beta),
        )
    return masked_weighted_huber_loss(
        prediction,
        target,
        valid_mask=valid_mask,
        beta=beta,
    )


def cloudy_csi_weights(target: torch.Tensor, cloudy_weight: float) -> torch.Tensor:
    """Return weights that emphasize cloudy or low-CSI hours."""
    if cloudy_weight <= 0.0:
        return torch.ones_like(target)
    return (1.0 + float(cloudy_weight) * (1.0 - target)).clamp_min(1.0e-6)


def ramp_csi_weights(
    target: torch.Tensor,
    ramp_weight: float,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return weights that emphasize large temporal target changes."""
    if ramp_weight <= 0.0:
        return torch.ones_like(target)
    delta = torch.zeros_like(target)
    if target.shape[-1] > 1:
        pair_delta = (target[..., 1:] - target[..., :-1]).abs()
        if valid_mask is not None:
            valid = valid_mask.to(device=target.device, dtype=torch.bool)
            pair_valid = valid[..., 1:] & valid[..., :-1]
            pair_delta = pair_delta.masked_fill(~pair_valid, 0.0)
        delta[..., 1:] = pair_delta
    return 1.0 + float(ramp_weight) * delta


def masked_pearson_correlation(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Compute Pearson correlation over valid positions without producing NaNs."""
    _check_prediction_target_shapes(prediction, target)
    valid = _resolve_valid_mask(valid_mask, target, prediction)
    if int(valid.sum().detach().cpu()) < 2:
        return prediction.new_zeros(())
    pred = prediction[valid]
    tgt = target.to(device=prediction.device, dtype=prediction.dtype)[valid]
    pred_centered = pred - pred.mean()
    tgt_centered = tgt - tgt.mean()
    numerator = (pred_centered * tgt_centered).sum()
    denominator = torch.sqrt(
        pred_centered.pow(2).sum().clamp_min(eps)
        * tgt_centered.pow(2).sum().clamp_min(eps)
    )
    corr = numerator / denominator.clamp_min(eps)
    finite_corr = torch.where(torch.isfinite(corr), corr, prediction.new_zeros(()))
    return finite_corr.clamp(-1.0, 1.0)


class MSELoss:
    """Configurable masked forecasting loss for CSI fine-tuning.

    The class name is kept for backwards compatibility with existing training
    imports, but the implementation now supports the loss sweep used for
    regression-to-the-mean diagnostics.
    """

    def __init__(
        self,
        loss_name: str = "masked_mse",
        low_csi_weight: float = 0.0,
        low_csi_threshold: float = 0.7,
        ghi_loss_weight: float = 0.0,
        huber_beta: float = 0.1,
        cloudy_weight: float = 1.0,
        ramp_weight: float = 1.0,
        lambda_corr: float = 0.1,
    ) -> None:
        if loss_name not in SUPPORTED_LOSSES:
            raise ValueError(
                f"Unsupported loss '{loss_name}'. "
                f"Expected one of: {', '.join(SUPPORTED_LOSSES)}"
            )
        self.loss = nn.MSELoss()
        self.loss_name = loss_name
        self.low_csi_weight = float(low_csi_weight)
        self.low_csi_threshold = float(low_csi_threshold)
        self.ghi_loss_weight = float(ghi_loss_weight)
        self.huber_beta = float(huber_beta)
        self.cloudy_weight = float(cloudy_weight)
        self.ramp_weight = float(ramp_weight)
        self.lambda_corr = float(lambda_corr)

    def _csi_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return final CSI loss, base loss, and correlation penalty."""
        zero = prediction.new_zeros(())
        if self.loss_name == "masked_mse":
            base_loss = masked_mse_loss(prediction, target, valid_mask=valid_mask)
            return base_loss, base_loss, zero
        if self.loss_name == "masked_mae":
            base_loss = masked_mae_loss(prediction, target, valid_mask=valid_mask)
            return base_loss, base_loss, zero
        if self.loss_name == "masked_huber":
            base_loss = masked_huber_loss(
                prediction,
                target,
                valid_mask=valid_mask,
                beta=self.huber_beta,
            )
            return base_loss, base_loss, zero
        if self.loss_name == "masked_weighted_mse":
            weights = cloudy_csi_weights(target, self.cloudy_weight)
            base_loss = masked_weighted_mse_loss(
                prediction,
                target,
                valid_mask=valid_mask,
                weights=weights,
            )
            return base_loss, base_loss, zero
        if self.loss_name == "masked_weighted_huber":
            weights = cloudy_csi_weights(target, self.cloudy_weight)
            base_loss = masked_weighted_huber_loss(
                prediction,
                target,
                valid_mask=valid_mask,
                weights=weights,
                beta=self.huber_beta,
            )
            return base_loss, base_loss, zero
        if self.loss_name == "masked_ramp_weighted_mse":
            weights = ramp_csi_weights(target, self.ramp_weight, valid_mask=valid_mask)
            base_loss = masked_weighted_mse_loss(
                prediction,
                target,
                valid_mask=valid_mask,
                weights=weights,
            )
            return base_loss, base_loss, zero
        if self.loss_name == "masked_hybrid_mse_correlation":
            base_loss = masked_mse_loss(prediction, target, valid_mask=valid_mask)
            corr_loss = 1.0 - masked_pearson_correlation(
                prediction,
                target,
                valid_mask=valid_mask,
            )
            return base_loss + self.lambda_corr * corr_loss, base_loss, corr_loss
        if self.loss_name == "masked_hybrid_huber_correlation":
            base_loss = masked_huber_loss(
                prediction,
                target,
                valid_mask=valid_mask,
                beta=self.huber_beta,
            )
            corr_loss = 1.0 - masked_pearson_correlation(
                prediction,
                target,
                valid_mask=valid_mask,
            )
            return base_loss + self.lambda_corr * corr_loss, base_loss, corr_loss
        raise RuntimeError(f"Unhandled loss '{self.loss_name}'")

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        clear_sky_ghi: torch.Tensor | None = None,
        target_ghi: torch.Tensor | None = None,
        return_components: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Compute the configured masked forecasting loss."""
        _check_prediction_target_shapes(prediction, target)
        target = target.to(device=prediction.device, dtype=prediction.dtype)
        csi_loss, base_loss, corr_loss = self._csi_loss(
            prediction,
            target,
            valid_mask=valid_mask,
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
                "base_loss": base_loss,
                "correlation_loss": corr_loss,
            }
        return total_loss
