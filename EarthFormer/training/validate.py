"""Validation utilities."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader


def target_to_nthwc(target: torch.Tensor) -> torch.Tensor:
    """Convert dataset target `(B,T,C,H,W)` to EarthFormer `(B,T,H,W,C)`."""
    if target.ndim != 5:
        raise ValueError(f"Expected target with 5 dims, got {tuple(target.shape)}")
    return target.permute(0, 1, 3, 4, 2).contiguous()


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    use_amp: bool = False,
) -> float:
    """Return the average validation loss."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    amp_enabled = use_amp and device.type == "cuda"

    for batch in dataloader:
        inputs = batch["satellite"].to(device, non_blocking=True)
        targets = target_to_nthwc(batch["target"]).to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            predictions = model(inputs)
            loss = criterion(predictions, targets)
        batch_size = inputs.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Validation dataloader produced no samples")
    return total_loss / total_samples
