"""Loss functions for EarthFormer training."""

from __future__ import annotations

import torch
from torch import nn


class MSELoss:
    """Small callable wrapper around `torch.nn.MSELoss`."""

    def __init__(self) -> None:
        self.loss = nn.MSELoss()

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute mean squared error."""
        return self.loss(prediction, target)
