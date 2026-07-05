"""Run EarthFormer inference on a SEVIRI dataset split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PREP_MODELS_ROOT = PROJECT_ROOT.parent
if str(PREP_MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(PREP_MODELS_ROOT))

from configs.config import build_arg_parser, config_from_args  # noqa: E402
from datasets.seviri_dataset import build_dataloader  # noqa: E402
from models.model import build_perceiver_readout_model  # noqa: E402
from training.checkpoint import load_checkpoint, load_model_state_dict_compatible  # noqa: E402
from training.losses import valid_hour_mask  # noqa: E402
from training.validate import ensure_forecast_target, reconstruct_ghi  # noqa: E402
from utils.artifacts import ArtifactMirror  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse inference arguments."""
    parser = build_arg_parser()
    parser.description = "Run EarthFormer inference on SEVIRI imagery."
    parser.add_argument("--split", default="test")
    parser.add_argument("--model-checkpoint", type=Path, default=None)
    parser.add_argument("--output-file", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """Run inference and save prediction/latent tensors."""
    args = parse_args()
    config = config_from_args(args)
    config.prepare_directories()
    device = torch.device(config.resolved_device())

    loader = build_dataloader(
        config=config,
        split=args.split,
        include_target=True,
        shuffle=False,
    )
    model = build_perceiver_readout_model(config).to(device)
    if args.model_checkpoint is not None:
        checkpoint = load_checkpoint(args.model_checkpoint, map_location=device)
        load_model_state_dict_compatible(model, checkpoint["model"])
    model.eval()

    batch = next(iter(loader))
    inputs = batch["satellite"].to(device, non_blocking=True)
    clear_sky_ghi = ensure_forecast_target(batch["clear_sky_ghi"]).to(
        device,
        non_blocking=True,
    )
    with torch.no_grad():
        result = model(inputs, return_debug=True)
        prediction_csi = result["prediction"]
        prediction_ghi = reconstruct_ghi(prediction_csi, clear_sky_ghi)

    target_csi = batch.get("target")
    target_ghi = batch.get("target_ghi")
    target_mask = batch.get("target_mask")
    if isinstance(target_csi, torch.Tensor):
        target_csi_device = ensure_forecast_target(target_csi).to(device, non_blocking=True)
        if isinstance(target_mask, torch.Tensor):
            target_mask = target_mask.to(device, non_blocking=True)
        valid_hour = valid_hour_mask(
            target_mask=target_mask,
            reference=target_csi_device,
            clear_sky_ghi=clear_sky_ghi,
            clear_sky_threshold=config.clear_sky_threshold,
        ).cpu()
    else:
        valid_hour = None

    output_file = args.output_file or (config.output_dir / "inference_sample.pt")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "prediction_csi": prediction_csi.cpu(),
            "prediction_ghi": prediction_ghi.cpu(),
            "target_csi": target_csi.cpu() if isinstance(target_csi, torch.Tensor) else None,
            "target_ghi": target_ghi.cpu() if isinstance(target_ghi, torch.Tensor) else None,
            "clear_sky_ghi": clear_sky_ghi.cpu(),
            "valid_hour": valid_hour,
            "pre_head_latent": result["pre_head_latent"].cpu(),
            "sample_id": batch["sample_id"],
            "location": batch["location"],
            "input_day": batch["input_day"],
            "target_day": batch["target_day"],
        },
        output_file,
    )
    ArtifactMirror(
        checkpoint_dir=config.checkpoint_dir,
        output_dir=config.output_dir,
        enabled=config.mirror_artifacts,
    ).mirror_output_file(output_file)
    print(f"Saved inference output to {output_file}")


if __name__ == "__main__":
    main()
