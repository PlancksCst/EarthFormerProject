"""Shared helpers for Perceiver forecasting diagnostics."""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.utils.data import DataLoader, Subset

SCRIPT_DIR = Path(__file__).resolve().parent
EARTHFORMER_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = EARTHFORMER_DIR.parent

for candidate in (PROJECT_ROOT, EARTHFORMER_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from configs.config import TrainingConfig  # noqa: E402
from datasets.seviri_dataset import build_dataloader, build_dataset  # noqa: E402
from models.model import build_perceiver_readout_model  # noqa: E402
from training.losses import MSELoss  # noqa: E402
from utils.seed import seed_everything  # noqa: E402
from utils.precision import (  # noqa: E402
    autocast_context,
    build_grad_scaler as build_precision_grad_scaler,
    resolve_amp_dtype,
)


def now_utc() -> str:
    """Return an ISO UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def diagnostics_dir(config: TrainingConfig) -> Path:
    """Return and create the diagnostics output directory."""
    path = Path(config.output_dir) / "diagnostics"
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_default(value: Any) -> Any:
    """JSON serializer for common diagnostic values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    return str(value)


def save_json_report(config: TrainingConfig, name: str, payload: dict[str, Any]) -> Path:
    """Save a JSON report under `outputs/diagnostics`."""
    path = diagnostics_dir(config) / f"{name}.json"
    report = {
        "generated_at": now_utc(),
        "name": name,
        **payload,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=json_default)
    return path


def append_csv_row(path: Path, row: dict[str, Any], fieldnames: Iterable[str] | None = None) -> None:
    """Append one row to a CSV file, creating a header when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames or row.keys())
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field) for field in fields})


def print_json(payload: dict[str, Any]) -> None:
    """Print a JSON payload to stdout."""
    print(json.dumps(payload, indent=2, default=json_default))


def resolve_device(config: TrainingConfig) -> torch.device:
    """Resolve and return the configured torch device."""
    return torch.device(config.resolved_device())


def use_amp(config: TrainingConfig, device: torch.device) -> bool:
    """Return whether CUDA AMP should be used."""
    return bool(config.mixed_precision and device.type == "cuda")


def autocast_dtype(config: TrainingConfig, device: torch.device) -> torch.dtype | None:
    """Return the configured autocast dtype when AMP is enabled."""
    return resolve_amp_dtype(config.amp_dtype, device) if use_amp(config, device) else None


def prepare_config(config: TrainingConfig) -> TrainingConfig:
    """Seed and create output directories."""
    config.prepare_directories()
    diagnostics_dir(config)
    seed_everything(config.random_seed)
    return config


def load_batch(
    config: TrainingConfig,
    split: str,
    device: torch.device,
    include_target: bool = False,
    shuffle: bool = False,
) -> dict[str, Any]:
    """Load a single batch from the configured dataset."""
    loader = build_dataloader(
        config=config,
        split=split,
        include_target=include_target,
        shuffle=shuffle,
    )
    batch = next(iter(loader))
    batch["satellite"] = batch["satellite"].to(device, non_blocking=True)
    return batch


def build_model(config: TrainingConfig, device: torch.device) -> nn.Module:
    """Build the complete EarthFormer + Perceiver readout model."""
    return build_perceiver_readout_model(config).to(device)


def tensor_finite(value: torch.Tensor) -> bool:
    """Return whether a tensor contains only finite values."""
    return bool(torch.isfinite(value).all().detach().cpu().item())


def tensor_stats(value: torch.Tensor) -> dict[str, Any]:
    """Return shape, dtype, finite flag, and numeric summary statistics."""
    detached = value.detach()
    finite = tensor_finite(detached)
    stats_tensor = detached.float()
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "finite": finite,
        "mean": float(stats_tensor.mean().detach().cpu()),
        "std": float(stats_tensor.std(unbiased=False).detach().cpu()),
        "min": float(stats_tensor.min().detach().cpu()),
        "max": float(stats_tensor.max().detach().cpu()),
    }


def forward_debug(
    model: nn.Module,
    inputs: torch.Tensor,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    """Run a debug forward pass through the complete forecasting model."""
    with autocast_context(device=device, enabled=amp_enabled, dtype=amp_dtype):
        return model(inputs, return_debug=True)


def make_sanity_target(
    inputs: torch.Tensor,
    output_length: int,
    mode: str = "satellite_mean",
) -> torch.Tensor:
    """Create a deterministic scalar target from satellite input only.

    This target is for optimization sanity tests only. It does not use CSI,
    GHI, location, time, or any auxiliary metadata.
    """
    batch_size = inputs.shape[0]
    if mode == "zeros":
        return inputs.new_zeros(batch_size, output_length)
    if mode != "satellite_mean":
        raise ValueError(f"Unsupported target mode: {mode}")

    sequence = inputs[:, :output_length, 0, :, :]
    if sequence.shape[1] < output_length:
        pad = inputs.new_zeros(
            batch_size,
            output_length - sequence.shape[1],
            inputs.shape[-2],
            inputs.shape[-1],
        )
        sequence = torch.cat([sequence, pad], dim=1)
    return sequence.mean(dim=(-1, -2))


def attention_tensors(model: nn.Module, pre_head_latent: torch.Tensor) -> dict[str, torch.Tensor]:
    """Recompute Perceiver readout attention weights for inspection only."""
    readout = model.readout
    spatial_tokens = readout.flatten_spatial_tokens(pre_head_latent)
    batch_size, steps, num_tokens, _channels = spatial_tokens.shape
    queries = readout.timestep_queries(batch_size=batch_size, steps=steps)
    normalized_tokens = readout.token_norm(spatial_tokens)
    normalized_queries = readout.query_norm(queries)

    keys = normalized_tokens.reshape(batch_size * steps, num_tokens, readout.latent_dim)
    values = keys
    query = normalized_queries.reshape(batch_size * steps, 1, readout.query_dim)
    attention_output, attention_weights = readout.cross_attention(
        query=query,
        key=keys,
        value=values,
        need_weights=True,
        average_attn_weights=False,
    )
    return {
        "spatial_tokens": spatial_tokens,
        "queries": queries,
        "query": query,
        "key": keys,
        "value": values,
        "attention_output": attention_output.reshape(batch_size, steps, readout.query_dim),
        "attention_weights": attention_weights,
    }


def attention_summary(tensors: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Summarize readout attention weights and Q/K/V norms."""
    weights = tensors["attention_weights"].detach().float()
    probabilities = weights.clamp_min(1.0e-12)
    entropy = -(probabilities * probabilities.log()).sum(dim=-1)

    query_norm = tensors["query"].detach().float().norm(dim=-1)
    key_norm = tensors["key"].detach().float().norm(dim=-1)
    value_norm = tensors["value"].detach().float().norm(dim=-1)

    return {
        "attention_entropy": tensor_stats(entropy),
        "mean_attention_weight": float(weights.mean().detach().cpu()),
        "max_attention_weight": float(weights.max().detach().cpu()),
        "min_attention_weight": float(weights.min().detach().cpu()),
        "query_norm": tensor_stats(query_norm),
        "key_norm": tensor_stats(key_norm),
        "value_norm": tensor_stats(value_norm),
        "finite": all(
            tensor_finite(tensor)
            for tensor in (
                weights,
                entropy,
                query_norm,
                key_norm,
                value_norm,
            )
        ),
    }


def build_optimizer(config: TrainingConfig, model: nn.Module) -> Optimizer:
    """Build the optimizer used by sanity tests."""
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters available for optimizer")
    return AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def build_scheduler(
    config: TrainingConfig,
    optimizer: Optimizer,
    epochs: int,
) -> LRScheduler:
    """Build the cosine scheduler used by sanity tests."""
    t_max = config.scheduler_t_max or epochs
    return CosineAnnealingLR(
        optimizer,
        T_max=max(1, t_max),
        eta_min=config.scheduler_eta_min,
    )


def build_scaler(
    enabled: bool,
    amp_dtype: torch.dtype | None = None,
) -> torch.amp.GradScaler:
    """Build an AMP GradScaler across PyTorch versions."""
    return build_precision_grad_scaler(enabled=enabled, dtype=amp_dtype)


def gradient_summary(model: nn.Module) -> dict[str, Any]:
    """Return finite checks and norms for all trainable gradients."""
    total_sq = 0.0
    grad_entries: list[dict[str, Any]] = []
    missing: list[str] = []
    nonfinite: list[str] = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name)
            continue
        grad = parameter.grad.detach()
        finite = tensor_finite(grad)
        if not finite:
            nonfinite.append(name)
        norm = float(grad.float().norm().detach().cpu()) if finite else float("nan")
        if math.isfinite(norm):
            total_sq += norm * norm
        grad_entries.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "finite": finite,
                "norm": norm,
            }
        )

    grad_entries.sort(key=lambda item: item["norm"] if math.isfinite(item["norm"]) else -1.0, reverse=True)
    return {
        "all_finite": len(nonfinite) == 0,
        "total_norm": math.sqrt(total_sq),
        "max_norm": grad_entries[0]["norm"] if grad_entries else 0.0,
        "parameters_with_grad": len(grad_entries),
        "missing_gradients": len(missing),
        "missing_gradient_names": missing[:20],
        "nonfinite_gradient_names": nonfinite,
        "top_gradient_norms": grad_entries[:20],
    }


def capture_trainable_parameters(model: nn.Module) -> dict[str, torch.Tensor]:
    """Clone trainable parameters before an optimizer step."""
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def count_updated_parameters(
    before: dict[str, torch.Tensor],
    model: nn.Module,
) -> dict[str, Any]:
    """Count trainable tensors changed by an optimizer step."""
    updated_names: list[str] = []
    for name, parameter in model.named_parameters():
        if name not in before:
            continue
        if not torch.equal(before[name], parameter.detach()):
            updated_names.append(name)
    return {
        "updated_parameter_tensors": len(updated_names),
        "checked_parameter_tensors": len(before),
        "updated_parameter_names": updated_names[:20],
    }


def train_one_batch(
    model: nn.Module,
    inputs: torch.Tensor,
    target: torch.Tensor,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    config: TrainingConfig,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    """Run one train step and return diagnostics."""
    model.train()
    criterion = MSELoss()
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(device=device, enabled=amp_enabled, dtype=amp_dtype):
        prediction = model(inputs)
        loss = criterion(prediction, target)

    scaler.scale(loss).backward()
    if amp_enabled:
        scaler.unscale_(optimizer)
    grad_report = gradient_summary(model)
    if config.gradient_clip > 0:
        nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
    scaler.step(optimizer)
    scaler.update()

    return {
        "loss": float(loss.detach().cpu()),
        "loss_finite": bool(torch.isfinite(loss).detach().cpu().item()),
        "prediction": prediction.detach(),
        "prediction_finite": tensor_finite(prediction),
        "prediction_variance": float(prediction.detach().float().var(unbiased=False).cpu()),
        "gradient_summary": grad_report,
    }


def tiny_dataloader(
    config: TrainingConfig,
    split: str,
    samples: int,
) -> DataLoader:
    """Build a small deterministic dataloader for overfit/resume checks."""
    dataset = build_dataset(config=config, split=split, include_target=False)
    sample_count = min(samples, len(dataset))
    subset = Subset(dataset, list(range(sample_count)))
    return DataLoader(
        subset,
        batch_size=min(config.batch_size, sample_count),
        shuffle=False,
        num_workers=0,
        pin_memory=config.resolved_device().startswith("cuda"),
        drop_last=False,
    )


class Timer:
    """Simple elapsed-time timer."""

    def __init__(self) -> None:
        self.start = time.perf_counter()

    def elapsed(self) -> float:
        """Return elapsed seconds."""
        return time.perf_counter() - self.start
