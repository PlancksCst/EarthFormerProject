"""Train tiny probes on detached EarthFormer latent features."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from .diagnostic_common import (
        batch_prediction_rows,
        build_context,
        clear_sky_tensor,
        dataloader_for_split,
        maybe_autocast,
        metrics_from_rows,
        mirror_outputs,
        parse_common_args,
        plot_series,
        resolve_checkpoint,
        target_ghi_tensor,
        target_tensor,
        valid_mask_tensor,
        write_csv,
        write_json,
    )
except ImportError:
    from diagnostic_common import (  # type: ignore
        batch_prediction_rows,
        build_context,
        clear_sky_tensor,
        dataloader_for_split,
        maybe_autocast,
        metrics_from_rows,
        mirror_outputs,
        parse_common_args,
        plot_series,
        resolve_checkpoint,
        target_ghi_tensor,
        target_tensor,
        valid_mask_tensor,
        write_csv,
        write_json,
    )

from models.model import build_perceiver_readout_model  # noqa: E402
from training.checkpoint import load_checkpoint, load_model_state_dict_compatible  # noqa: E402


def build_model_for_latents(context: Any) -> tuple[nn.Module, bool]:
    """Build the forecasting model and load checkpoint when available."""
    model = build_perceiver_readout_model(context.config).to(context.device)
    checkpoint_path = resolve_checkpoint(context)
    checkpoint_loaded = False
    if checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path, map_location=context.device)
        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        load_model_state_dict_compatible(model, state)
        checkpoint_loaded = True
    else:
        print(f"WARNING: checkpoint not found for Perceiver comparison: {checkpoint_path}")
    model.eval()
    return model, checkpoint_loaded


def pooled_features(pre_head_latent: torch.Tensor) -> torch.Tensor:
    """Pool `(B,T,H,W,C)` latents into `(B,T,F)` features."""
    latent = pre_head_latent.detach().float().cpu()
    mean_pool = latent.mean(dim=(2, 3))
    max_pool = latent.amax(dim=(2, 3))
    std_pool = latent.std(dim=(2, 3), unbiased=False)
    center_pool = latent[:, :, latent.shape[2] // 2, latent.shape[3] // 2, :]
    return torch.cat([mean_pool, max_pool, std_pool, center_pool], dim=-1)


def extract_split(context: Any, model: nn.Module, split: str) -> dict[str, Any]:
    """Extract detached latent features and targets for a split."""
    loader = dataloader_for_split(
        context.config,
        split=split,
        include_target=True,
        shuffle=False,
        max_samples=context.args.max_samples,
    )
    features: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    clear_values: list[torch.Tensor] = []
    target_ghi_values: list[torch.Tensor] = []
    valid_values: list[torch.Tensor] = []
    perceiver_predictions: list[torch.Tensor] = []
    batches_for_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            inputs = batch["satellite"].to(context.device, non_blocking=True)
            target = target_tensor(batch, context.device)
            clear = clear_sky_tensor(batch, context.device)
            target_ghi = target_ghi_tensor(batch, target, clear)
            valid = valid_mask_tensor(batch, target, clear, context.config.clear_sky_threshold)
            with maybe_autocast(context):
                debug = model(inputs, return_debug=True)
            features.append(pooled_features(debug["pre_head_latent"]))
            perceiver_predictions.append(debug["prediction"].detach().float().cpu())
            targets.append(target.detach().float().cpu())
            clear_values.append(clear.detach().float().cpu())
            target_ghi_values.append(target_ghi.detach().float().cpu())
            valid_values.append(valid.detach().cpu())
            batches_for_rows.append({key: value for key, value in batch.items() if key != "satellite"})
    return {
        "features": torch.cat(features, dim=0),
        "target": torch.cat(targets, dim=0),
        "clear_sky_ghi": torch.cat(clear_values, dim=0),
        "target_ghi": torch.cat(target_ghi_values, dim=0),
        "valid": torch.cat(valid_values, dim=0),
        "perceiver_prediction": torch.cat(perceiver_predictions, dim=0),
        "batches": batches_for_rows,
    }


class SmallMLP(nn.Module):
    """Small detached-latent probe."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        hidden = min(128, max(32, input_dim * 2))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_probe(
    name: str,
    train: dict[str, Any],
    val: dict[str, Any],
    epochs: int = 120,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    """Train a linear or small MLP scalar probe over per-hour latent features."""
    x_train = train["features"].reshape(-1, train["features"].shape[-1])
    y_train = train["target"].reshape(-1)
    valid_train = train["valid"].reshape(-1).bool()
    x_val = val["features"].reshape(-1, val["features"].shape[-1])

    x_mean = x_train[valid_train].mean(dim=0, keepdim=True)
    x_std = x_train[valid_train].std(dim=0, keepdim=True, unbiased=False).clamp_min(1.0e-6)
    x_train = (x_train - x_mean) / x_std
    x_val = (x_val - x_mean) / x_std

    if name == "latent_linear":
        probe: nn.Module = nn.Linear(x_train.shape[-1], 1)
    elif name == "latent_mlp":
        probe = SmallMLP(x_train.shape[-1])
    else:
        raise ValueError(name)

    optimizer = torch.optim.AdamW(probe.parameters(), lr=1.0e-3, weight_decay=1.0e-3)
    dataset = TensorDataset(x_train[valid_train], y_train[valid_train])
    loader = DataLoader(dataset, batch_size=min(512, len(dataset)), shuffle=True)
    log: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        total = 0.0
        count = 0
        for batch_x, batch_y in loader:
            pred = probe(batch_x)
            loss = torch.mean((pred - batch_y) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * batch_x.shape[0]
            count += batch_x.shape[0]
        log.append({"probe": name, "epoch": epoch, "loss": total / max(count, 1)})
    with torch.no_grad():
        pred_val = probe(x_val).reshape_as(val["target"])
    return pred_val, log


def rows_from_tensor_prediction(split_data: dict[str, Any], split: str, method: str, prediction: torch.Tensor) -> list[dict[str, Any]]:
    """Convert tensor predictions back to long-form rows."""
    rows: list[dict[str, Any]] = []
    sample_start = 0
    offset = 0
    for batch in split_data["batches"]:
        batch_target = batch.get("target", batch.get("target_csi"))
        if isinstance(batch_target, torch.Tensor):
            batch_size = int(batch_target.shape[0])
        elif isinstance(batch.get("sample_id"), (list, tuple)):
            batch_size = len(batch["sample_id"])
        else:
            batch_size = prediction.shape[0] - offset
        batch_size = min(batch_size, prediction.shape[0] - offset)
        pred = prediction[offset : offset + batch_size]
        target = split_data["target"][offset : offset + batch_size]
        clear = split_data["clear_sky_ghi"][offset : offset + batch_size]
        valid = split_data["valid"][offset : offset + batch_size]
        target_ghi = split_data["target_ghi"][offset : offset + batch_size]
        method_rows = batch_prediction_rows(
            batch=batch,
            split=split,
            sample_start=sample_start,
            pred_csi=pred,
            target_csi=target,
            clear_sky_ghi=clear,
            valid_mask=valid,
            target_ghi=target_ghi,
        )
        for row in method_rows:
            row["method"] = method
        rows.extend(method_rows)
        offset += batch_size
        sample_start += batch_size
    return rows


def main() -> None:
    """Run latent probe diagnostics."""
    args = parse_common_args("Train tiny probes on detached EarthFormer latent features.")
    context = build_context(args, default_subdir="latent_probe")
    model, checkpoint_loaded = build_model_for_latents(context)
    train_data = extract_split(context, model, "train")
    val_data = extract_split(context, model, args.split)
    all_rows: list[dict[str, Any]] = []
    training_log: list[dict[str, float]] = []

    for probe_name in ("latent_linear", "latent_mlp"):
        pred, log = train_probe(probe_name, train_data, val_data)
        all_rows.extend(rows_from_tensor_prediction(val_data, args.split, probe_name, pred))
        training_log.extend(log)

    if checkpoint_loaded:
        all_rows.extend(
            rows_from_tensor_prediction(
                val_data,
                args.split,
                "perceiver_checkpoint",
                val_data["perceiver_prediction"],
            )
        )

    metrics = [
        metrics_from_rows(group.to_dict("records"), {"method": method})
        for method, group in pd.DataFrame(all_rows).groupby("method", sort=False)
    ]
    predictions_path = context.output_dir / "latent_probe_predictions.csv"
    metrics_path = context.output_dir / "latent_probe_metrics.csv"
    log_path = context.output_dir / "latent_probe_training_log.csv"
    write_csv(predictions_path, all_rows)
    write_csv(metrics_path, metrics)
    write_csv(log_path, training_log)
    plot_examples(pd.DataFrame(all_rows), context.output_dir / "plots")
    write_json(
        context.output_dir / "latent_probe_summary.json",
        {
            "dataset_root": str(context.config.dataset_root),
            "checkpoint_loaded": checkpoint_loaded,
            "split": args.split,
            "max_samples": args.max_samples,
            "metrics_csv": str(metrics_path),
            "predictions_csv": str(predictions_path),
            "training_log_csv": str(log_path),
        },
    )
    mirror_outputs(context)
    print(pd.DataFrame(metrics).to_string(index=False))


def plot_examples(frame: pd.DataFrame, output_dir: Path, limit: int = 12) -> None:
    """Plot target vs Perceiver/probe predictions for a few samples."""
    if frame.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for count, (sample_index, group) in enumerate(frame.groupby("sample_index", sort=True), start=1):
        if count > limit:
            break
        hours = np.arange(1, int(group["forecast_hour"].max()) + 1)
        series: dict[str, np.ndarray] = {}
        first_method = group[group["method"] == group["method"].iloc[0]].sort_values("forecast_hour")
        series["target"] = first_method["target_csi"].to_numpy()
        for method, method_group in group.groupby("method", sort=False):
            method_group = method_group.sort_values("forecast_hour")
            series[str(method)] = method_group["predicted_csi"].to_numpy()
        plot_series(
            output_dir / f"latent_probe_sample_{int(sample_index):04d}.png",
            hours,
            series,
            title=f"Latent probe comparison | sample {sample_index}",
            ylabel="CSI",
        )


if __name__ == "__main__":
    main()
