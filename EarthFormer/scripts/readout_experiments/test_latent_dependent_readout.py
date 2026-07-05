"""Train small frozen-latent readouts to test whether the Perceiver head is the bottleneck."""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from readout_common import (  # type: ignore
    amp_context,
    batch_prediction_rows,
    build_experiment_context,
    build_loader,
    clear_sky_tensor,
    diagnostic_valid_mask_tensor,
    extract_latent_split,
    load_model,
    masked_mse,
    metric_rows,
    mirror,
    parse_readout_args,
    plot_comparison,
    prediction_rows_from_split,
    readout_epoch_count,
    save_rows,
    target_ghi_tensor,
    target_tensor,
    write_summary,
)


class TemporalPoolMLP(nn.Module):
    """Predict CSI from per-hour spatial mean and standard deviation of latents."""

    def __init__(self, latent_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.regression = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """Return `(B,T)` predictions from `(B,T,H,W,C)` latents."""
        mean = latents.mean(dim=(2, 3))
        std = latents.std(dim=(2, 3), unbiased=False)
        features = torch.cat([mean, std], dim=-1)
        return self.regression(features).squeeze(-1)


class TemporalAttentionPool(nn.Module):
    """Per-hour query attention where regression uses only attended latent content."""

    def __init__(self, latent_dim: int, query_dim: int = 64, heads: int = 4, horizon: int = 13) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.query_dim = int(query_dim)
        self.horizon = int(horizon)
        self.queries = nn.Parameter(torch.empty(self.horizon, self.query_dim))
        self.token_norm = nn.LayerNorm(self.latent_dim)
        self.query_norm = nn.LayerNorm(self.query_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.query_dim,
            num_heads=heads,
            kdim=self.latent_dim,
            vdim=self.latent_dim,
            batch_first=True,
        )
        self.regression = nn.Sequential(
            nn.Linear(self.query_dim, max(32, self.query_dim // 2)),
            nn.GELU(),
            nn.Linear(max(32, self.query_dim // 2), 1),
        )
        nn.init.normal_(self.queries, mean=0.0, std=0.02)

    def attended(self, latents: torch.Tensor) -> torch.Tensor:
        """Return `(B,T,D)` attended latent embeddings."""
        bsz, steps, height, width, channels = latents.shape
        if channels != self.latent_dim:
            raise ValueError(f"Expected latent_dim={self.latent_dim}, got {channels}")
        if steps > self.horizon:
            raise ValueError(f"Input steps={steps} exceeds horizon={self.horizon}")
        tokens = latents.reshape(bsz, steps, height * width, channels)
        tokens = self.token_norm(tokens).reshape(bsz * steps, height * width, channels)
        queries = self.queries[:steps].unsqueeze(0).expand(bsz, steps, self.query_dim)
        queries = self.query_norm(queries).reshape(bsz * steps, 1, self.query_dim)
        output, _weights = self.attention(query=queries, key=tokens, value=tokens, need_weights=False)
        return output.reshape(bsz, steps, self.query_dim)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """Return `(B,T)` predictions from attended latent content."""
        return self.regression(self.attended(latents)).squeeze(-1)


class LatentSummaryPlusQuery(TemporalAttentionPool):
    """Attention output plus mean/std latent summary for each forecast hour."""

    def __init__(self, latent_dim: int, query_dim: int = 64, heads: int = 4, horizon: int = 13) -> None:
        super().__init__(latent_dim=latent_dim, query_dim=query_dim, heads=heads, horizon=horizon)
        hidden_dim = max(32, query_dim // 2)
        self.regression = nn.Sequential(
            nn.Linear(query_dim + 2 * latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """Return `(B,T)` predictions from attended tokens plus pooled summaries."""
        attended = self.attended(latents)
        mean = latents.mean(dim=(2, 3))
        std = latents.std(dim=(2, 3), unbiased=False)
        features = torch.cat([attended, mean, std], dim=-1)
        return self.regression(features).squeeze(-1)


def add_local_args(args: argparse.Namespace) -> None:
    """Attach defaults for script-local arguments if the parser did not define them."""
    if not hasattr(args, "latent_token_stride"):
        args.latent_token_stride = 1


def selected_readouts(args: argparse.Namespace) -> list[str]:
    """Return requested readout names."""
    names = [item.strip() for item in str(args.readout_types).split(",") if item.strip()]
    return names or ["temporal_pool_mlp", "temporal_attention_pool", "latent_summary_plus_query"]


def make_readout(name: str, latent_dim: int, context: Any) -> nn.Module:
    """Construct one experimental readout."""
    query_dim = int(context.config.query_dimension)
    heads = int(context.config.num_attention_heads)
    horizon = int(context.config.output_length)
    if name == "temporal_pool_mlp":
        return TemporalPoolMLP(latent_dim=latent_dim, hidden_dim=max(32, 4 * latent_dim))
    if name == "temporal_attention_pool":
        return TemporalAttentionPool(latent_dim=latent_dim, query_dim=query_dim, heads=heads, horizon=horizon)
    if name == "latent_summary_plus_query":
        return LatentSummaryPlusQuery(latent_dim=latent_dim, query_dim=query_dim, heads=heads, horizon=horizon)
    raise ValueError(f"Unknown readout type: {name}")


def maybe_stride_latents(latents: torch.Tensor, stride: int) -> torch.Tensor:
    """Optionally spatially subsample cached latents for experimental attention tests."""
    if stride <= 1:
        return latents
    return latents[:, :, ::stride, ::stride, :].contiguous()


def train_readout(
    readout: nn.Module,
    name: str,
    train_data: dict[str, Any],
    val_data: dict[str, Any],
    context: Any,
    epochs: int,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Train one detached-latent readout and return validation predictions."""
    readout = readout.to(context.device)
    dataset = TensorDataset(train_data["latents"], train_data["target"], train_data["valid"])
    loader = DataLoader(dataset, batch_size=context.config.batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.AdamW(
        readout.parameters(),
        lr=float(context.config.head_learning_rate),
        weight_decay=float(context.config.weight_decay),
    )
    log_rows: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        readout.train()
        total_loss = 0.0
        total_valid = 0
        for latent, target, valid in loader:
            latent = latent.to(context.device, non_blocking=True)
            target = target.to(context.device, non_blocking=True)
            valid = valid.to(context.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = readout(latent)
            loss = masked_mse(prediction, target, valid)
            loss.backward()
            optimizer.step()
            valid_count = int(valid.sum().detach().cpu())
            total_loss += float(loss.detach().cpu()) * valid_count
            total_valid += valid_count
        readout.eval()
        with torch.no_grad():
            val_prediction = readout(val_data["latents"].to(context.device)).detach().cpu()
            val_loss = float(masked_mse(val_prediction, val_data["target"], val_data["valid"]).detach().cpu())
        train_loss = total_loss / max(total_valid, 1)
        log_rows.append(
            {
                "readout": name,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )
        print(f"{name} epoch {epoch:03d}/{epochs:03d} train={train_loss:.6f} val={val_loss:.6f}")
    with torch.no_grad():
        prediction = readout(val_data["latents"].to(context.device)).detach().cpu()
    return prediction, log_rows


def finite_delta_rows(
    method: str,
    perturbation: str,
    pred_real: torch.Tensor,
    pred_other: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, Any]:
    """Return aggregate image-sensitivity deltas."""
    mask = valid.detach().cpu().bool()
    pred_real_cpu = pred_real.detach().cpu()
    pred_other_cpu = pred_other.detach().cpu()
    target_cpu = target.detach().cpu()
    delta = (pred_other_cpu - pred_real_cpu)[mask]
    if delta.numel() == 0:
        return {
            "method": method,
            "perturbation": perturbation,
            "valid_count": 0,
            "mean_abs_delta": float("nan"),
            "rmse_delta": float("nan"),
            "max_abs_delta": float("nan"),
            "real_forecast_rmse": float("nan"),
            "perturbed_forecast_rmse": float("nan"),
        }
    array = delta.float().numpy()
    real_error = (pred_real_cpu - target_cpu)[mask].float().numpy()
    perturbed_error = (pred_other_cpu - target_cpu)[mask].float().numpy()
    return {
        "method": method,
        "perturbation": perturbation,
        "valid_count": int(delta.numel()),
        "mean_abs_delta": float(np.mean(np.abs(array))),
        "rmse_delta": float(np.sqrt(np.mean(array**2))),
        "max_abs_delta": float(np.max(np.abs(array))),
        "real_forecast_rmse": float(np.sqrt(np.mean(real_error**2))),
        "perturbed_forecast_rmse": float(np.sqrt(np.mean(perturbed_error**2))),
    }


def image_sensitivity(
    context: Any,
    source_model: nn.Module,
    readouts: dict[str, nn.Module],
    split: str,
    token_stride: int,
) -> list[dict[str, Any]]:
    """Run image-vs-zero/random sensitivity for experimental readouts."""
    loader = build_loader(context, split=split, include_target=True, shuffle=False)
    rows: list[dict[str, Any]] = []
    source_model.eval()
    for readout in readouts.values():
        readout.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            inputs = batch["satellite"].to(context.device, non_blocking=True)
            target = target_tensor(batch, context.device)
            clear = clear_sky_tensor(batch, context.device)
            valid, _solar = diagnostic_valid_mask_tensor(
                context.config,
                batch,
                target,
                clear,
                context.config.clear_sky_threshold,
                float(context.args.solar_elevation_threshold),
            )
            random_inputs = torch.roll(inputs, shifts=1, dims=0) if inputs.shape[0] > 1 else torch.zeros_like(inputs)
            variants = {
                "real_image": inputs,
                "zero_image": torch.zeros_like(inputs),
                "random_other_image": random_inputs,
            }
            latent_by_variant: dict[str, torch.Tensor] = {}
            for variant_name, variant_inputs in variants.items():
                with amp_context(context):
                    latent = source_model(variant_inputs, return_debug=True)["pre_head_latent"]
                latent_by_variant[variant_name] = maybe_stride_latents(latent.detach().float(), token_stride)
            for method, readout in readouts.items():
                pred_real = readout(latent_by_variant["real_image"].to(context.device)).detach().cpu()
                for perturbation in ("zero_image", "random_other_image"):
                    pred_other = readout(latent_by_variant[perturbation].to(context.device)).detach().cpu()
                    row = finite_delta_rows(method, perturbation, pred_real, pred_other, target, valid)
                    row["batch_index"] = batch_index
                    rows.append(row)
    return rows


def main() -> None:
    """Run frozen-latent readout experiments."""
    args = parse_readout_args("Frozen EarthFormer latent readout experiments.")
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--latent-token-stride", type=int, default=1)
    local_args, _unknown = parser.parse_known_args()
    args.latent_token_stride = int(local_args.latent_token_stride)
    add_local_args(args)

    context = build_experiment_context(args, subdir="latent_dependent_readout")
    source_model = load_model(context)
    source_model.eval()
    for parameter in source_model.parameters():
        parameter.requires_grad = False

    train_data = extract_latent_split(context, source_model, split=context.config.train_split)
    val_data = extract_latent_split(context, source_model, split=args.split)
    train_data["latents"] = maybe_stride_latents(train_data["latents"], args.latent_token_stride)
    val_data["latents"] = maybe_stride_latents(val_data["latents"], args.latent_token_stride)
    latent_dim = int(train_data["latents"].shape[-1])
    epochs = readout_epoch_count(args, default=20)

    prediction_rows = prediction_rows_from_split(
        val_data,
        split=args.split,
        method="current_perceiver",
        prediction=val_data["perceiver_prediction"],
    )
    training_log: list[dict[str, Any]] = []
    trained_readouts: dict[str, nn.Module] = {}

    for name in selected_readouts(args):
        readout = make_readout(name, latent_dim=latent_dim, context=context)
        prediction, rows = train_readout(readout, name, train_data, val_data, context, epochs)
        trained_readouts[name] = readout
        training_log.extend(rows)
        prediction_rows.extend(
            prediction_rows_from_split(
                val_data,
                split=args.split,
                method=name,
                prediction=prediction,
            )
        )

    metrics = metric_rows(prediction_rows)
    sensitivity_rows = image_sensitivity(
        context,
        source_model=source_model,
        readouts=trained_readouts,
        split=args.split,
        token_stride=args.latent_token_stride,
    )

    metrics_path = context.output_dir / "readout_experiment_metrics.csv"
    predictions_path = context.output_dir / "readout_experiment_predictions.csv"
    training_log_path = context.output_dir / "readout_experiment_training_log.csv"
    sensitivity_path = context.output_dir / "readout_image_sensitivity.csv"
    save_rows(metrics_path, metrics)
    save_rows(predictions_path, prediction_rows)
    save_rows(training_log_path, training_log)
    save_rows(sensitivity_path, sensitivity_rows)
    plot_comparison(pd.DataFrame(prediction_rows), context.output_dir / "plots", limit=12)

    metrics_frame = pd.DataFrame(metrics)
    perceiver_rmse = float(
        metrics_frame.loc[metrics_frame["method"] == "current_perceiver", "CSI_RMSE"].iloc[0]
    ) if not metrics_frame.empty and (metrics_frame["method"] == "current_perceiver").any() else float("nan")
    best_row = metrics_frame.sort_values("CSI_RMSE").iloc[0].to_dict() if not metrics_frame.empty else {}
    summary = {
        "split": args.split,
        "max_samples": args.max_samples,
        "epochs": epochs,
        "latent_token_stride": args.latent_token_stride,
        "current_perceiver_CSI_RMSE": perceiver_rmse,
        "best_method": best_row.get("method"),
        "best_CSI_RMSE": best_row.get("CSI_RMSE"),
        "experimental_readout_beats_perceiver": bool(
            np.isfinite(perceiver_rmse)
            and best_row.get("method") != "current_perceiver"
            and float(best_row.get("CSI_RMSE", np.inf)) < perceiver_rmse
        ),
        "metrics_csv": str(metrics_path),
        "predictions_csv": str(predictions_path),
        "sensitivity_csv": str(sensitivity_path),
    }
    write_summary(context.output_dir / "readout_experiment_summary.json", summary)
    print(json.dumps(summary, indent=2))
    mirror(context)


if __name__ == "__main__":
    main()
