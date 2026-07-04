"""Verify EarthFormer-to-Perceiver readout tensor flow."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREP_MODELS_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PREP_MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(PREP_MODELS_ROOT))

from configs.config import build_arg_parser, config_from_args  # noqa: E402
from datasets.seviri_dataset import build_dataloader  # noqa: E402
from models.model import build_perceiver_readout_model  # noqa: E402
from utils.seed import seed_everything  # noqa: E402


def tensor_shape(value: torch.Tensor) -> list[int]:
    """Return a JSON-serializable tensor shape."""
    return list(value.shape)


def tensor_memory_mb(value: torch.Tensor) -> float:
    """Return tensor storage size in MiB."""
    return float(value.numel() * value.element_size() / (1024**2))


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """Count module parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if (parameter.requires_grad or not trainable_only)
    )


def has_finite_gradient(parameters: Any) -> bool:
    """Return whether any parameter has a finite, nonzero gradient."""
    for parameter in parameters:
        if parameter.grad is None:
            continue
        if torch.isfinite(parameter.grad).all() and bool((parameter.grad != 0).any()):
            return True
    return False


def verify_checkpoint_roundtrip(model: torch.nn.Module, config: Any, device: torch.device) -> dict[str, Any]:
    """Save and strictly reload the combined EarthFormer-readout state dict."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "earthformer_perceiver_readout.pt"
        torch.save(model.state_dict(), checkpoint_path)
        fresh_model = build_perceiver_readout_model(config).to(device)
        try:
            state_dict = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=True,
            )
        except TypeError:
            state_dict = torch.load(checkpoint_path, map_location=device)
        load_result = fresh_model.load_state_dict(
            state_dict,
            strict=True,
        )
    return {
        "strict": True,
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
    }


def main() -> None:
    """Run one forward/backward pass and print readout diagnostics."""
    parser = build_arg_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    seed_everything(config.random_seed)

    device = torch.device(config.resolved_device())
    loader = build_dataloader(
        config=config,
        split=config.train_split,
        include_target=False,
        shuffle=False,
    )
    batch = next(iter(loader))
    inputs = batch["satellite"].to(device)

    model = build_perceiver_readout_model(config).to(device)
    model.train()
    model.zero_grad(set_to_none=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    result = model(inputs, return_debug=True)
    prediction = result["prediction"]
    loss = prediction.square().mean()
    loss.backward()

    readout_debug = result["readout"]
    diagnostics: dict[str, Any] = {
        "dataset_root": str(config.dataset_root),
        "input": tensor_shape(inputs),
        "pre_head_latent": tensor_shape(result["pre_head_latent"]),
        "flattened_tokens": tensor_shape(readout_debug["flattened_tokens"]),
        "queries": tensor_shape(readout_debug["queries"]),
        "attention_output": tensor_shape(readout_debug["attention_output"]),
        "regression_output": tensor_shape(readout_debug["regression_output"]),
        "prediction": tensor_shape(prediction),
        "loss": float(loss.detach().cpu()),
        "earthformer_grad_ok": has_finite_gradient(model.earthformer_parameters()),
        "readout_grad_ok": has_finite_gradient(model.readout_parameters()),
        "earthformer_parameters": count_parameters(model.earthformer),
        "readout_parameters": count_parameters(model.readout),
        "readout_trainable_parameters": count_parameters(model.readout, trainable_only=True),
        "earthformer_frozen": config.freeze_earthformer,
        "tensor_memory_mb": {
            "pre_head_latent": tensor_memory_mb(result["pre_head_latent"]),
            "flattened_tokens": tensor_memory_mb(readout_debug["flattened_tokens"]),
            "queries": tensor_memory_mb(readout_debug["queries"]),
            "attention_output": tensor_memory_mb(readout_debug["attention_output"]),
            "regression_output": tensor_memory_mb(readout_debug["regression_output"]),
        },
        "checkpoint_roundtrip": verify_checkpoint_roundtrip(model, config, device),
    }
    if device.type == "cuda":
        diagnostics["cuda_memory_mb"] = {
            "allocated": torch.cuda.memory_allocated(device) / (1024**2),
            "peak_allocated": torch.cuda.max_memory_allocated(device) / (1024**2),
        }

    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
