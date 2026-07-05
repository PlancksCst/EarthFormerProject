"""Basic tensor metrics."""

from __future__ import annotations

import torch


def mae(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return mean absolute error."""
    return torch.mean(torch.abs(prediction - target))


def rmse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return root mean squared error."""
    return torch.sqrt(torch.mean((prediction - target) ** 2))


def mbe(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return mean bias error."""
    return torch.mean(prediction - target)


def nrmse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Return RMSE normalized by mean absolute target value."""
    denominator = torch.mean(torch.abs(target)).clamp_min(eps)
    return rmse(prediction, target) / denominator


def r2_score(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Return coefficient of determination."""
    residual = torch.sum((target - prediction) ** 2)
    centered = torch.sum((target - torch.mean(target)) ** 2)
    return 1.0 - residual / centered.clamp_min(eps)


def forecast_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    prefix: str,
) -> dict[str, float]:
    """Return scalar MAE, RMSE, nRMSE, R2, and MBE metrics with a prefix."""
    prediction = prediction.detach().float().reshape(-1)
    target = target.detach().float().reshape(-1)
    finite = torch.isfinite(prediction) & torch.isfinite(target)
    prediction = prediction[finite]
    target = target[finite]
    if prediction.numel() == 0:
        return {
            f"{prefix}_MAE": float("nan"),
            f"{prefix}_RMSE": float("nan"),
            f"{prefix}_nRMSE": float("nan"),
            f"{prefix}_R2": float("nan"),
            f"{prefix}_MBE": float("nan"),
        }
    return {
        f"{prefix}_MAE": float(mae(prediction, target).cpu()),
        f"{prefix}_RMSE": float(rmse(prediction, target).cpu()),
        f"{prefix}_nRMSE": float(nrmse(prediction, target).cpu()),
        f"{prefix}_R2": float(r2_score(prediction, target).cpu()),
        f"{prefix}_MBE": float(mbe(prediction, target).cpu()),
    }
