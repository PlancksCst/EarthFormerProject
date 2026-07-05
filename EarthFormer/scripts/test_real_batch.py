"""Run one real CSI forecasting training batch with finite diagnostics."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PREP_MODELS_ROOT = PROJECT_ROOT.parent
if str(PREP_MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(PREP_MODELS_ROOT))

from configs.config import build_arg_parser, config_from_args  # noqa: E402
from datasets.seviri_dataset import build_dataloader  # noqa: E402
from models.model import build_perceiver_readout_model  # noqa: E402
from training.debugging import (  # noqa: E402
    assert_finite,
    assert_gradients_finite,
    assert_scalar_finite,
    tensor_stats,
)
from training.losses import MSELoss, valid_mask_from_target_mask  # noqa: E402
from training.validate import ensure_forecast_target, reconstruct_ghi  # noqa: E402
from utils.artifacts import ArtifactMirror  # noqa: E402
from utils.precision import (  # noqa: E402
    amp_dtype_label,
    autocast_context,
    build_grad_scaler,
    resolve_amp_dtype,
)
from utils.seed import seed_everything  # noqa: E402


def json_default(value: Any) -> Any:
    """Serialize common diagnostic values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    return str(value)


def diagnostics_dir(output_dir: Path) -> Path:
    """Return and create the diagnostics directory."""
    path = output_dir / "diagnostics"
    path.mkdir(parents=True, exist_ok=True)
    return path


def first_values(batch: dict[str, Any]) -> dict[str, Any]:
    """Return compact metadata for the first sample in the batch."""
    result: dict[str, Any] = {}
    for key in ("sample_id", "location", "input_day", "target_day"):
        value = batch.get(key)
        if isinstance(value, torch.Tensor):
            result[key] = value[0].detach().cpu().item()
        elif isinstance(value, (list, tuple)):
            result[key] = value[0] if value else None
        else:
            result[key] = value
    return result


def main() -> None:
    """Run one real train batch and print/save diagnostics."""
    parser = build_arg_parser()
    parser.description = "Test one real EarthFormer + Perceiver CSI training batch."
    parser.add_argument("--split", default="train")
    parser.add_argument("--report-name", default="test_real_batch")
    args = parser.parse_args()

    config = config_from_args(args)
    config.prepare_directories()
    seed_everything(config.random_seed)

    device = torch.device(config.resolved_device())
    use_amp = config.mixed_precision and device.type == "cuda"
    amp_dtype = resolve_amp_dtype(config.amp_dtype, device) if use_amp else None
    loader = build_dataloader(
        config=config,
        split=args.split,
        include_target=True,
        shuffle=False,
    )
    model = build_perceiver_readout_model(config).to(device)
    criterion = MSELoss()
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = build_grad_scaler(enabled=use_amp, dtype=amp_dtype)

    model.train()
    batch = next(iter(loader))
    inputs = batch["satellite"].to(device, non_blocking=True)
    targets = ensure_forecast_target(batch["target"]).to(device, non_blocking=True)
    clear_sky_ghi = ensure_forecast_target(batch["clear_sky_ghi"], "clear_sky_ghi").to(
        device,
        non_blocking=True,
    )
    target_mask = batch.get("target_mask")
    if isinstance(target_mask, torch.Tensor):
        target_mask = target_mask.to(device, non_blocking=True)
        assert_finite("target_mask", target_mask.float(), batch=batch, batch_index=0)
    valid_mask = valid_mask_from_target_mask(target_mask, targets)
    valid_count = int(valid_mask.sum().detach().cpu())
    if valid_count == 0:
        raise RuntimeError(
            "No valid target positions in real batch. "
            "Mask convention is target_mask=0 valid, target_mask=1 invalid."
        )

    assert_finite("inputs", inputs, batch=batch, batch_index=0)
    assert_finite("targets", targets, batch=batch, batch_index=0)
    assert_finite("clear_sky_ghi", clear_sky_ghi, batch=batch, batch_index=0)

    optimizer.zero_grad(set_to_none=True)
    with autocast_context(device=device, enabled=use_amp, dtype=amp_dtype):
        predictions = model(inputs)
        assert_finite("predictions", predictions, batch=batch, batch_index=0)
        predicted_ghi = reconstruct_ghi(predictions, clear_sky_ghi)
        assert_finite("predicted_ghi", predicted_ghi, batch=batch, batch_index=0)
        loss = criterion(predictions, targets, valid_mask=valid_mask)
        assert_scalar_finite("loss", loss, batch=batch, batch_index=0)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradient_stats = assert_gradients_finite(model, batch=batch, batch_index=0)
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
    if not isinstance(grad_norm, torch.Tensor):
        grad_norm = torch.tensor(float(grad_norm), device=device)
    assert_scalar_finite("gradient_norm", grad_norm.detach().reshape(()), batch=batch, batch_index=0)
    scaler.step(optimizer)
    scaler.update()

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "PASS",
        "dataset_root": str(config.dataset_root),
        "hourly_csv": str(config.hourly_csv),
        "split": args.split,
        "device": str(device),
        "amp_enabled": use_amp,
        "amp_dtype": amp_dtype_label(amp_dtype),
        "sample": first_values(batch),
        "valid_count": valid_count,
        "input": tensor_stats(inputs),
        "target": tensor_stats(targets),
        "clear_sky_ghi": tensor_stats(clear_sky_ghi),
        "prediction": tensor_stats(predictions),
        "predicted_ghi": tensor_stats(predicted_ghi),
        "loss": float(loss.detach().cpu()),
        "gradient_stats": gradient_stats,
        "optimizer_step": "completed",
    }
    path = diagnostics_dir(config.output_dir) / f"{args.report_name}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=json_default)
    ArtifactMirror(
        checkpoint_dir=config.checkpoint_dir,
        output_dir=config.output_dir,
        enabled=config.mirror_artifacts,
    ).mirror_output_file(path)
    report["report_path"] = str(path)
    print(json.dumps(report, indent=2, default=json_default))


if __name__ == "__main__":
    main()
