"""Test whether the trained Perceiver readout can ignore EarthFormer latents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from readout_common import (  # type: ignore
    amp_context,
    batch_prediction_rows,
    build_experiment_context,
    build_loader,
    clear_sky_tensor,
    diagnostic_valid_mask_tensor,
    load_model,
    metric_rows,
    mirror,
    parse_readout_args,
    plot_comparison,
    save_rows,
    target_ghi_tensor,
    target_tensor,
    write_summary,
)


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Return a safe Pearson correlation for flattened arrays."""
    if a.size < 2 or float(np.std(a)) < 1.0e-8 or float(np.std(b)) < 1.0e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def delta_summary(
    normal: torch.Tensor,
    variant: torch.Tensor,
    valid: torch.Tensor,
    perturbation: str,
    batch_index: int,
) -> dict[str, Any]:
    """Summarize the prediction change caused by a latent replacement."""
    valid_cpu = valid.detach().cpu().bool()
    normal_cpu = normal.detach().float().cpu()
    variant_cpu = variant.detach().float().cpu()
    delta = variant_cpu - normal_cpu
    valid_delta = delta[valid_cpu]
    valid_normal = normal_cpu[valid_cpu]
    valid_variant = variant_cpu[valid_cpu]
    if valid_delta.numel() == 0:
        return {
            "batch_index": batch_index,
            "perturbation": perturbation,
            "valid_count": 0,
            "mean_abs_delta": float("nan"),
            "rmse_delta": float("nan"),
            "max_abs_delta": float("nan"),
            "prediction_correlation": float("nan"),
        }
    delta_np = valid_delta.numpy()
    return {
        "batch_index": batch_index,
        "perturbation": perturbation,
        "valid_count": int(valid_delta.numel()),
        "mean_abs_delta": float(np.mean(np.abs(delta_np))),
        "rmse_delta": float(np.sqrt(np.mean(delta_np**2))),
        "max_abs_delta": float(np.max(np.abs(delta_np))),
        "prediction_correlation": correlation(valid_normal.numpy(), valid_variant.numpy()),
    }


def replace_latents(latent: torch.Tensor) -> dict[str, torch.Tensor]:
    """Create latent-token replacements for the shortcut diagnostic."""
    channel_mean = latent.mean(dim=(0, 1, 2, 3), keepdim=True).expand_as(latent)
    if latent.shape[0] > 1:
        random_other = torch.roll(latent, shifts=1, dims=0)
    else:
        random_other = channel_mean
    return {
        "normal": latent,
        "latent_zero": torch.zeros_like(latent),
        "latent_mean_vector": channel_mean,
        "latent_random_other": random_other,
    }


def main() -> None:
    """Run the latent replacement shortcut diagnostic."""
    args = parse_readout_args("Query-only shortcut diagnostic for the Perceiver readout.")
    context = build_experiment_context(args, subdir="query_shortcut")
    model = load_model(context)
    loader = build_loader(context, split=args.split, include_target=True, shuffle=False)

    prediction_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    sample_start = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            inputs = batch["satellite"].to(context.device, non_blocking=True)
            target = target_tensor(batch, context.device)
            clear = clear_sky_tensor(batch, context.device)
            target_ghi = target_ghi_tensor(batch, target, clear)
            valid, _solar = diagnostic_valid_mask_tensor(
                context.config,
                batch,
                target,
                clear,
                context.config.clear_sky_threshold,
                float(args.solar_elevation_threshold),
            )

            with amp_context(context):
                debug = model(inputs, return_debug=True)
            latent = debug["pre_head_latent"].detach().float()
            variants = replace_latents(latent)
            predictions: dict[str, torch.Tensor] = {
                "normal": debug["prediction"].detach().float().cpu()
            }
            for name, variant_latent in variants.items():
                if name == "normal":
                    continue
                with amp_context(context):
                    prediction = model.readout(variant_latent.to(context.device), return_debug=False)
                predictions[name] = prediction.detach().float().cpu()
                delta_rows.append(
                    delta_summary(
                        predictions["normal"],
                        predictions[name],
                        valid,
                        perturbation=name,
                        batch_index=batch_index,
                    )
                )

            for method, prediction in predictions.items():
                rows = batch_prediction_rows(
                    batch=batch,
                    split=args.split,
                    sample_start=sample_start,
                    pred_csi=prediction,
                    target_csi=target.detach().cpu(),
                    clear_sky_ghi=clear.detach().cpu(),
                    valid_mask=valid.detach().cpu(),
                    target_ghi=target_ghi.detach().cpu(),
                )
                for row in rows:
                    row["method"] = method
                prediction_rows.extend(rows)
            sample_start += target.shape[0]

    metrics = metric_rows(prediction_rows)
    predictions_path = context.output_dir / "query_shortcut_predictions.csv"
    metrics_path = context.output_dir / "query_shortcut_metrics.csv"
    delta_path = context.output_dir / "query_shortcut_delta_metrics.csv"
    save_rows(predictions_path, prediction_rows)
    save_rows(metrics_path, metrics)
    save_rows(delta_path, delta_rows)

    frame = pd.DataFrame(prediction_rows)
    plot_comparison(frame, context.output_dir / "plots", limit=12)

    delta_frame = pd.DataFrame(delta_rows)
    max_delta = float(delta_frame["mean_abs_delta"].max()) if not delta_frame.empty else float("nan")
    mean_delta = float(delta_frame["mean_abs_delta"].mean()) if not delta_frame.empty else float("nan")
    summary = {
        "checkpoint": str(context.args.checkpoint or (context.config.checkpoint_dir / "best.pt")),
        "split": args.split,
        "max_samples": args.max_samples,
        "mean_abs_delta": mean_delta,
        "max_mean_abs_delta": max_delta,
        "current_readout_ignores_latents": bool(np.isfinite(max_delta) and max_delta < 0.02),
        "metrics_csv": str(metrics_path),
        "predictions_csv": str(predictions_path),
        "delta_csv": str(delta_path),
    }
    write_summary(context.output_dir / "prediction_delta_summary.json", summary)
    print(json.dumps(summary, indent=2))
    mirror(context)


if __name__ == "__main__":
    main()
