"""Numerical debugging helpers for forecasting training."""

from __future__ import annotations

from typing import Any

import torch


def _metadata_value(batch: dict[str, Any] | None, key: str, index: int) -> Any:
    if batch is None or key not in batch:
        return None
    value = batch[key]
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.detach().cpu().item()
        if index >= value.shape[0]:
            return None
        item = value[index]
        return item.detach().cpu().item() if item.numel() == 1 else item.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return value[index] if index < len(value) else None
    return value


def batch_context(
    batch: dict[str, Any] | None,
    batch_index: int | None = None,
    sample_limit: int = 4,
) -> str:
    """Return compact sample metadata for error messages."""
    parts: list[str] = []
    if batch_index is not None:
        parts.append(f"batch_index={batch_index}")
    if batch is None:
        return ", ".join(parts)

    batch_size = 0
    satellite = batch.get("satellite")
    if isinstance(satellite, torch.Tensor) and satellite.ndim > 0:
        batch_size = int(satellite.shape[0])
    else:
        sample_id = batch.get("sample_id")
        if isinstance(sample_id, torch.Tensor) and sample_id.ndim > 0:
            batch_size = int(sample_id.shape[0])
        elif isinstance(sample_id, (list, tuple)):
            batch_size = len(sample_id)

    samples: list[str] = []
    for index in range(min(sample_limit, batch_size)):
        sample_parts = []
        for key in ("sample_id", "location", "target_day"):
            value = _metadata_value(batch, key, index)
            if value is not None:
                sample_parts.append(f"{key}={value}")
        if sample_parts:
            samples.append("{" + ", ".join(sample_parts) + "}")
    if samples:
        parts.append("samples=[" + ", ".join(samples) + "]")
    return ", ".join(parts)


def tensor_stats(tensor: torch.Tensor) -> str:
    """Return shape, dtype, finite flag, and numeric summary for a tensor."""
    detached = tensor.detach()
    stats = [
        f"shape={tuple(detached.shape)}",
        f"dtype={detached.dtype}",
        f"device={detached.device}",
    ]
    if detached.numel() == 0:
        stats.append("numel=0")
        return ", ".join(stats)

    finite = torch.isfinite(detached)
    stats.append(f"finite={bool(finite.all().cpu())}")
    values = detached.float()
    finite_values = values[finite]
    if finite_values.numel() == 0:
        stats.extend(["min=nan", "max=nan", "mean=nan", "std=nan"])
    else:
        stats.extend(
            [
                f"min={float(finite_values.min().cpu()):.8g}",
                f"max={float(finite_values.max().cpu()):.8g}",
                f"mean={float(finite_values.mean().cpu()):.8g}",
                f"std={float(finite_values.std(unbiased=False).cpu()):.8g}",
            ]
        )
    return ", ".join(stats)


def assert_finite(
    name: str,
    tensor: torch.Tensor,
    batch: dict[str, Any] | None = None,
    batch_index: int | None = None,
) -> None:
    """Raise a detailed error if a tensor contains NaN or Inf."""
    if torch.isfinite(tensor).all():
        return
    context = batch_context(batch=batch, batch_index=batch_index)
    message = f"Non-finite tensor detected: {name}. {tensor_stats(tensor)}"
    if context:
        message = f"{message}. {context}"
    raise RuntimeError(message)


def assert_scalar_finite(
    name: str,
    value: torch.Tensor,
    batch: dict[str, Any] | None = None,
    batch_index: int | None = None,
) -> None:
    """Raise a detailed error if a scalar tensor is non-finite."""
    if value.ndim != 0:
        raise ValueError(f"Expected scalar tensor for {name}, got {tuple(value.shape)}")
    assert_finite(name=name, tensor=value.reshape(1), batch=batch, batch_index=batch_index)


def gradient_summary(model: torch.nn.Module) -> dict[str, float | int | str | None]:
    """Return aggregate gradient statistics for trainable parameters."""
    total_sq = 0.0
    largest = 0.0
    smallest_nonzero: float | None = None
    finite = True
    nonfinite_parameter: str | None = None
    parameter_count = 0
    grad_parameter_count = 0

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        parameter_count += 1
        if parameter.grad is None:
            continue
        grad_parameter_count += 1
        grad = parameter.grad.detach()
        if not torch.isfinite(grad).all():
            finite = False
            nonfinite_parameter = name
            break
        norm = float(grad.float().norm().cpu())
        total_sq += norm * norm
        largest = max(largest, float(grad.float().abs().max().cpu()))
        nonzero = grad.float().abs()
        nonzero = nonzero[nonzero > 0]
        if nonzero.numel() > 0:
            current_min = float(nonzero.min().cpu())
            if smallest_nonzero is None or current_min < smallest_nonzero:
                smallest_nonzero = current_min

    return {
        "finite": finite,
        "nonfinite_parameter": nonfinite_parameter,
        "total_norm": total_sq ** 0.5,
        "largest_abs": largest,
        "smallest_nonzero_abs": smallest_nonzero,
        "parameter_count": parameter_count,
        "grad_parameter_count": grad_parameter_count,
    }


def assert_gradients_finite(
    model: torch.nn.Module,
    batch: dict[str, Any] | None = None,
    batch_index: int | None = None,
) -> dict[str, float | int | str | None]:
    """Raise if any gradient is NaN or Inf, otherwise return gradient stats."""
    summary = gradient_summary(model)
    if summary["finite"]:
        return summary
    context = batch_context(batch=batch, batch_index=batch_index)
    parameter = summary["nonfinite_parameter"]
    message = f"Non-finite gradient detected in parameter {parameter}"
    if context:
        message = f"{message}. {context}"
    raise RuntimeError(message)
