"""Basic tensor metrics."""

from __future__ import annotations

import torch


def mae(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return mean absolute error."""
    return torch.mean(torch.abs(prediction - target))


def rmse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return root mean squared error."""
    return torch.sqrt(torch.mean((prediction - target) ** 2))
