"""Precision helpers for EarthFormer training and diagnostics."""

from __future__ import annotations

from contextlib import AbstractContextManager

import torch


def resolve_amp_dtype(dtype_name: str, device: torch.device) -> torch.dtype | None:
    """Return the torch dtype used for CUDA autocast."""
    if device.type != "cuda":
        return None
    normalized = dtype_name.strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        checker = getattr(torch.cuda, "is_bf16_supported", None)
        if checker is not None and not checker():
            raise RuntimeError(
                "BF16 AMP was requested but this CUDA device does not support BF16. "
                "Use full precision or pass --amp-dtype fp16 explicitly."
            )
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError(f"Unsupported AMP dtype: {dtype_name!r}. Use 'bf16' or 'fp16'.")


def amp_dtype_label(dtype: torch.dtype | None) -> str | None:
    """Return a compact serializable AMP dtype label."""
    if dtype is None:
        return None
    if dtype is torch.bfloat16:
        return "bf16"
    if dtype is torch.float16:
        return "fp16"
    return str(dtype)


def autocast_context(
    device: torch.device,
    enabled: bool,
    dtype: torch.dtype | None,
) -> AbstractContextManager[None]:
    """Return a torch autocast context with an optional dtype."""
    kwargs: dict[str, object] = {
        "device_type": device.type,
        "enabled": enabled,
    }
    if dtype is not None:
        kwargs["dtype"] = dtype
    return torch.amp.autocast(**kwargs)


def build_grad_scaler(
    enabled: bool,
    dtype: torch.dtype | None,
) -> torch.amp.GradScaler:
    """Build GradScaler only for FP16 AMP; BF16 and FP32 do not need scaling."""
    scaler_enabled = bool(enabled and dtype is torch.float16)
    try:
        return torch.amp.GradScaler(enabled=scaler_enabled, init_scale=2.0**8)
    except TypeError:
        return torch.amp.GradScaler(enabled=scaler_enabled)
