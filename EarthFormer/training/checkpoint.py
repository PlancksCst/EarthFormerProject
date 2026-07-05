"""Checkpoint helpers for EarthFormer training."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def _serialize_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _serialize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]
    return value


def serialize_config(config: Any | None) -> dict[str, Any] | None:
    """Convert a config object into a checkpoint-safe dictionary."""
    if config is None:
        return None
    if is_dataclass(config):
        return _serialize_value(asdict(config))
    if hasattr(config, "__dict__"):
        return _serialize_value(vars(config))
    if isinstance(config, dict):
        return _serialize_value(config)
    return {"repr": repr(config)}


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    best_loss: float,
    config: Any | None = None,
    best_metric_name: str = "val_loss",
    best_metric: float | None = None,
    extra_state: dict[str, Any] | None = None,
) -> None:
    """Save a full training checkpoint."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metric_value = best_loss if best_metric is None else best_metric
    payload: dict[str, Any] = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_loss": best_loss,
        "best_metric_name": best_metric_name,
        "best_metric": metric_value,
        "config": serialize_config(config),
        "extra_state": extra_state or {},
    }
    torch.save(payload, checkpoint_path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint dictionary."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def resume_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    scaler: torch.amp.GradScaler | None,
    map_location: str | torch.device = "cpu",
) -> tuple[int, float]:
    """Resume training state and return `(next_epoch, best_loss)`."""
    checkpoint = load_checkpoint(path, map_location=map_location)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    next_epoch = int(checkpoint["epoch"]) + 1
    best_loss = float(checkpoint.get("best_loss", float("inf")))
    return next_epoch, best_loss
