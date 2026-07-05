"""Checkpoint helpers for EarthFormer training."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

COMPATIBLE_MISSING_MODEL_KEY_PREFIXES = (
    "readout.hour_embeddings",
    "readout.hour_embedding_projection.",
)


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


def load_model_state_dict_compatible(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> torch.nn.modules.module._IncompatibleKeys:
    """Load model weights while tolerating newly added query-hour parameters.

    Older checkpoints do not contain the explicit hour-query embedding added to
    the Perceiver readout. Missing keys are allowed only for that narrow
    compatibility surface; all other missing or unexpected keys still raise.
    """
    result = model.load_state_dict(state_dict, strict=False)
    missing = [
        key
        for key in result.missing_keys
        if not key.startswith(COMPATIBLE_MISSING_MODEL_KEY_PREFIXES)
    ]
    unexpected = list(result.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(
            "Model checkpoint is incompatible. "
            f"Missing keys: {missing}. Unexpected keys: {unexpected}."
        )
    if result.missing_keys:
        print(
            "Loaded checkpoint with newly initialized hour-query readout "
            f"parameters: {list(result.missing_keys)}"
        )
    return result


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
    load_model_state_dict_compatible(model, checkpoint["model"])
    optimizer_restored = True
    try:
        optimizer.load_state_dict(checkpoint["optimizer"])
    except ValueError as error:
        optimizer_restored = False
        print(
            "WARNING: optimizer state could not be restored exactly, likely "
            "because this checkpoint predates hour-query readout parameters. "
            f"Continuing with a freshly initialized optimizer. Details: {error}"
        )
    if scheduler is not None and checkpoint.get("scheduler") is not None and optimizer_restored:
        try:
            scheduler.load_state_dict(checkpoint["scheduler"])
        except Exception as error:
            print(
                "WARNING: scheduler state could not be restored exactly. "
                f"Continuing with a freshly initialized scheduler. Details: {error}"
            )
    elif scheduler is not None and checkpoint.get("scheduler") is not None and not optimizer_restored:
        print("WARNING: scheduler state was not restored because optimizer state was reset.")
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    next_epoch = int(checkpoint["epoch"]) + 1
    best_loss = float(checkpoint.get("best_loss", float("inf")))
    return next_epoch, best_loss
