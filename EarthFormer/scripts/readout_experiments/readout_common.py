"""Shared helpers for experimental readout fix tests."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
EARTHFORMER_DIR = SCRIPT_DIR.parents[1]
PROJECT_ROOT = EARTHFORMER_DIR.parent
DIAGNOSTIC_DIR = SCRIPT_DIR.parent / "diagnostics"

for candidate in (PROJECT_ROOT, EARTHFORMER_DIR, DIAGNOSTIC_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from configs.config import build_arg_parser, config_from_args  # noqa: E402
from datasets.seviri_dataset import build_dataset  # noqa: E402
from models.model import build_perceiver_readout_model  # noqa: E402
from training.checkpoint import load_checkpoint, load_model_state_dict_compatible  # noqa: E402
from utils.artifacts import ArtifactMirror  # noqa: E402
from utils.precision import autocast_context, resolve_amp_dtype  # noqa: E402
from utils.seed import seed_everything  # noqa: E402
from diagnostic_common import (  # type: ignore  # noqa: E402
    batch_prediction_rows,
    clear_sky_tensor,
    diagnostic_valid_mask_tensor,
    metrics_from_rows,
    regression_metrics,
    target_ghi_tensor,
    target_tensor,
    write_csv,
    write_json,
)


@dataclass(frozen=True)
class ExperimentContext:
    """Runtime context for readout experiments."""

    config: Any
    args: argparse.Namespace
    output_dir: Path
    device: torch.device
    artifact_mirror: ArtifactMirror


def add_readout_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add shared readout experiment arguments."""
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--solar-elevation-threshold", type=float, default=5.0)
    parser.add_argument("--readout-types", default="temporal_pool_mlp,temporal_attention_pool,latent_summary_plus_query")
    parser.add_argument("--experiment-epochs", type=int, default=None)
    parser.add_argument("--latent-token-stride", type=int, default=1)
    parser.add_argument("--image-dependence-weight", type=float, default=0.05)
    parser.add_argument("--image-dependence-margin", type=float, default=0.05)
    parser.add_argument("--run-image-dependence-penalty", action="store_true")
    return parser


def readout_epoch_count(args: argparse.Namespace, default: int = 20) -> int:
    """Return the experiment epoch count without shadowing the project parser."""
    value = getattr(args, "experiment_epochs", None)
    if value is None:
        value = getattr(args, "epochs", None)
    return int(value if value is not None else default)


def parse_readout_args(description: str) -> argparse.Namespace:
    """Parse project config args plus readout experiment args."""
    parser = build_arg_parser()
    parser.description = description
    add_readout_args(parser)
    return parser.parse_args()


def build_experiment_context(args: argparse.Namespace, subdir: str | None = None) -> ExperimentContext:
    """Build config, output dir, and artifact mirror."""
    config = config_from_args(args)
    config.prepare_directories()
    seed_everything(config.random_seed)
    root = args.output_dir or (config.output_dir / "readout_experiments")
    output_dir = root / subdir if subdir else root
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_mirror = ArtifactMirror(
        checkpoint_dir=config.checkpoint_dir,
        output_dir=config.output_dir,
        enabled=config.mirror_artifacts,
    )
    return ExperimentContext(
        config=config,
        args=args,
        output_dir=output_dir,
        device=torch.device(config.resolved_device()),
        artifact_mirror=artifact_mirror,
    )


def build_loader(context: ExperimentContext, split: str, include_target: bool = True, shuffle: bool = False) -> DataLoader:
    """Build a capped dataloader for experiments."""
    dataset = build_dataset(context.config, split=split, include_target=include_target)
    if context.args.max_samples is not None and context.args.max_samples > 0:
        indices = list(range(min(context.args.max_samples, len(dataset))))
        dataset = torch.utils.data.Subset(dataset, indices)
    return DataLoader(
        dataset,
        batch_size=context.config.batch_size,
        shuffle=shuffle,
        num_workers=context.config.num_workers,
        pin_memory=context.config.resolved_device().startswith("cuda"),
        drop_last=False,
    )


def checkpoint_path(context: ExperimentContext) -> Path:
    """Return checkpoint path."""
    return Path(context.args.checkpoint or (context.config.checkpoint_dir / "best.pt"))


def load_model(context: ExperimentContext) -> nn.Module:
    """Build and load the current EarthFormer + Perceiver checkpoint."""
    model = build_perceiver_readout_model(context.config).to(context.device)
    path = checkpoint_path(context)
    if path.exists():
        checkpoint = load_checkpoint(path, map_location=context.device)
        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        load_model_state_dict_compatible(model, state)
    else:
        print(f"WARNING: checkpoint not found, using initialized model: {path}")
    model.eval()
    return model


def amp_context(context: ExperimentContext):
    """Return configured autocast context."""
    use_amp = bool(context.config.mixed_precision and context.device.type == "cuda")
    amp_dtype = resolve_amp_dtype(context.config.amp_dtype, context.device) if use_amp else None
    return autocast_context(context.device, enabled=use_amp, dtype=amp_dtype)


def extract_latent_split(context: ExperimentContext, model: nn.Module, split: str) -> dict[str, Any]:
    """Extract detached pre-head latents, targets, masks, and Perceiver predictions."""
    loader = build_loader(context, split=split, include_target=True, shuffle=False)
    latents: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    clear_values: list[torch.Tensor] = []
    target_ghi_values: list[torch.Tensor] = []
    valid_values: list[torch.Tensor] = []
    perceiver_predictions: list[torch.Tensor] = []
    metadata_batches: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["satellite"].to(context.device, non_blocking=True)
            target = target_tensor(batch, context.device)
            clear = clear_sky_tensor(batch, context.device)
            target_ghi = target_ghi_tensor(batch, target, clear)
            valid, solar = diagnostic_valid_mask_tensor(
                context.config,
                batch,
                target,
                clear,
                context.config.clear_sky_threshold,
                float(context.args.solar_elevation_threshold),
            )
            with amp_context(context):
                debug = model(inputs, return_debug=True)
            latents.append(debug["pre_head_latent"].detach().float().cpu())
            perceiver_predictions.append(debug["prediction"].detach().float().cpu())
            targets.append(target.detach().float().cpu())
            clear_values.append(clear.detach().float().cpu())
            target_ghi_values.append(target_ghi.detach().float().cpu())
            valid_values.append(valid.detach().cpu())
            metadata = {key: value for key, value in batch.items() if key != "satellite"}
            if solar is not None:
                metadata["solar_elevation"] = solar.detach().cpu()
            metadata_batches.append(metadata)

    return {
        "latents": torch.cat(latents, dim=0),
        "target": torch.cat(targets, dim=0),
        "clear_sky_ghi": torch.cat(clear_values, dim=0),
        "target_ghi": torch.cat(target_ghi_values, dim=0),
        "valid": torch.cat(valid_values, dim=0).bool(),
        "perceiver_prediction": torch.cat(perceiver_predictions, dim=0),
        "metadata_batches": metadata_batches,
    }


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Masked MSE for detached readout experiments."""
    return ((prediction - target) ** 2).masked_fill(~valid, 0.0).sum() / valid.sum().clamp_min(1)


def prediction_rows_from_split(data: dict[str, Any], split: str, method: str, prediction: torch.Tensor) -> list[dict[str, Any]]:
    """Build long prediction rows from tensor predictions and stored metadata."""
    rows: list[dict[str, Any]] = []
    offset = 0
    sample_start = 0
    for batch in data["metadata_batches"]:
        target_batch = batch.get("target", batch.get("target_csi"))
        batch_size = int(target_batch.shape[0]) if isinstance(target_batch, torch.Tensor) else data["target"].shape[0] - offset
        batch_size = min(batch_size, prediction.shape[0] - offset)
        method_rows = batch_prediction_rows(
            batch=batch,
            split=split,
            sample_start=sample_start,
            pred_csi=prediction[offset : offset + batch_size],
            target_csi=data["target"][offset : offset + batch_size],
            clear_sky_ghi=data["clear_sky_ghi"][offset : offset + batch_size],
            valid_mask=data["valid"][offset : offset + batch_size],
            target_ghi=data["target_ghi"][offset : offset + batch_size],
        )
        for row in method_rows:
            row["method"] = method
        rows.extend(method_rows)
        offset += batch_size
        sample_start += batch_size
    return rows


def metric_rows(rows: list[dict[str, Any]], method_key: str = "method") -> list[dict[str, Any]]:
    """Compute metrics grouped by method."""
    frame = pd.DataFrame(rows)
    if frame.empty or method_key not in frame.columns:
        return []
    return [
        metrics_from_rows(group.to_dict("records"), {method_key: method})
        for method, group in frame.groupby(method_key, sort=False)
    ]


def mirror(context: ExperimentContext) -> None:
    """Mirror experiment output tree."""
    context.artifact_mirror.mirror_output_tree(context.output_dir)


def run_script(path: Path, args: argparse.Namespace, extra: list[str] | None = None) -> dict[str, Any]:
    """Run a child experiment script."""
    command = [sys.executable, str(path), *common_cli(args), *(extra or [])]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return {
        "script": path.name,
        "returncode": result.returncode,
        "ok": result.returncode == 0,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def common_cli(args: argparse.Namespace) -> list[str]:
    """Return common CLI flags for child scripts."""
    pairs = [
        ("--dataset-root", args.dataset_root),
        ("--checkpoint", args.checkpoint),
        ("--split", args.split),
        ("--batch-size", args.batch_size),
        ("--num-workers", args.num_workers),
        ("--device", args.device),
        ("--output-dir", args.output_dir),
        ("--clear-sky-threshold", args.clear_sky_threshold),
        ("--solar-elevation-threshold", args.solar_elevation_threshold),
        ("--max-samples", args.max_samples),
        ("--checkpoint-dir", args.checkpoint_dir),
        ("--hourly-csv", args.hourly_csv),
        ("--elevation-csv", args.elevation_csv),
    ]
    cli: list[str] = []
    for flag, value in pairs:
        if value is not None:
            cli.extend([flag, str(value)])
    if getattr(args, "no_artifact_mirror", False):
        cli.append("--no-artifact-mirror")
    return cli


def plot_comparison(frame: pd.DataFrame, output_dir: Path, limit: int = 12) -> None:
    """Plot target, Perceiver, and experiment predictions for several samples."""
    if frame.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, (sample_index, group) in enumerate(frame.groupby("sample_index", sort=True), start=1):
        if index > limit:
            break
        hours = sorted(group["forecast_hour"].unique())
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        target_group = group.sort_values("forecast_hour").drop_duplicates("forecast_hour")
        ax.plot(
            target_group["forecast_hour"],
            target_group["target_csi"],
            color="black",
            marker="o",
            linewidth=2.0,
            label="target",
        )
        for method, method_group in group.groupby("method", sort=False):
            method_group = method_group.sort_values("forecast_hour")
            ax.plot(
                method_group["forecast_hour"],
                method_group["predicted_csi"],
                marker="o",
                label=str(method),
            )
        ax.set_xticks(hours)
        ax.set_xlabel("Forecast hour")
        ax.set_ylabel("CSI")
        ax.set_title(f"Readout experiment sample {sample_index}")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"sample_{int(sample_index):04d}.png", dpi=180)
        plt.close(fig)


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON summary."""
    write_json(path, payload)


def save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write CSV rows."""
    write_csv(path, rows)
