"""Experimental image-dependence penalty test for the current model copy."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from readout_common import (  # type: ignore
    amp_context,
    batch_prediction_rows,
    build_experiment_context,
    build_loader,
    clear_sky_tensor,
    diagnostic_valid_mask_tensor,
    load_model,
    masked_mse,
    metric_rows,
    mirror,
    parse_readout_args,
    plot_comparison,
    readout_epoch_count,
    save_rows,
    target_ghi_tensor,
    target_tensor,
    write_summary,
)


def valid_mean_abs_delta(pred_real: torch.Tensor, pred_zero: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Return mean absolute prediction delta over valid hours."""
    if int(valid.sum().detach().cpu()) == 0:
        return pred_real.new_zeros(())
    return torch.abs(pred_real - pred_zero)[valid].mean()


def evaluate_model(
    model: nn.Module,
    context: Any,
    split: str,
    method: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run inference and return prediction rows plus real-vs-zero delta rows."""
    model.eval()
    loader = build_loader(context, split=split, include_target=True, shuffle=False)
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
                float(context.args.solar_elevation_threshold),
            )
            with amp_context(context):
                pred_real = model(inputs)
                pred_zero = model(torch.zeros_like(inputs))
            rows = batch_prediction_rows(
                batch=batch,
                split=split,
                sample_start=sample_start,
                pred_csi=pred_real.detach().cpu(),
                target_csi=target.detach().cpu(),
                clear_sky_ghi=clear.detach().cpu(),
                valid_mask=valid.detach().cpu(),
                target_ghi=target_ghi.detach().cpu(),
            )
            for row in rows:
                row["method"] = method
            prediction_rows.extend(rows)
            delta = (pred_zero - pred_real).detach().cpu()[valid.detach().cpu().bool()]
            if delta.numel() == 0:
                mean_abs = rmse = max_abs = float("nan")
            else:
                array = delta.float().numpy()
                mean_abs = float(np.mean(np.abs(array)))
                rmse = float(np.sqrt(np.mean(array**2)))
                max_abs = float(np.max(np.abs(array)))
            delta_rows.append(
                {
                    "method": method,
                    "batch_index": batch_index,
                    "perturbation": "zero_image",
                    "valid_count": int(valid.sum().detach().cpu()),
                    "mean_abs_delta": mean_abs,
                    "rmse_delta": rmse,
                    "max_abs_delta": max_abs,
                }
            )
            sample_start += target.shape[0]
    return prediction_rows, delta_rows


def train_with_penalty(model: nn.Module, context: Any, epochs: int) -> list[dict[str, Any]]:
    """Train a copy of the current readout with an image-dependence penalty."""
    if hasattr(model, "earthformer_parameters"):
        for parameter in model.earthformer_parameters():
            parameter.requires_grad = False
    trainable = list(model.readout_parameters()) if hasattr(model, "readout_parameters") else list(model.parameters())
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(context.config.head_learning_rate),
        weight_decay=float(context.config.weight_decay),
    )
    loader = build_loader(context, split=context.config.train_split, include_target=True, shuffle=True)
    log_rows: list[dict[str, Any]] = []
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_forecast = 0.0
        total_penalty = 0.0
        total_valid = 0
        for batch in loader:
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
            if int(valid.sum().detach().cpu()) == 0:
                continue
            optimizer.zero_grad(set_to_none=True)
            with amp_context(context):
                pred_real = model(inputs)
                pred_zero = model(torch.zeros_like(inputs))
                forecast_loss = masked_mse(pred_real, target, valid)
                delta = valid_mean_abs_delta(pred_real, pred_zero, valid)
                dependence_loss = torch.relu(
                    pred_real.new_tensor(float(context.args.image_dependence_margin)) - delta
                )
                loss = forecast_loss + float(context.args.image_dependence_weight) * dependence_loss
            loss.backward()
            optimizer.step()
            valid_count = int(valid.sum().detach().cpu())
            total_loss += float(loss.detach().cpu()) * valid_count
            total_forecast += float(forecast_loss.detach().cpu()) * valid_count
            total_penalty += float(dependence_loss.detach().cpu()) * valid_count
            total_valid += valid_count
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_valid, 1),
            "forecast_loss": total_forecast / max(total_valid, 1),
            "image_dependence_loss": total_penalty / max(total_valid, 1),
        }
        log_rows.append(row)
        print(
            f"image-dependence epoch {epoch:03d}/{epochs:03d} "
            f"loss={row['train_loss']:.6f} forecast={row['forecast_loss']:.6f} "
            f"penalty={row['image_dependence_loss']:.6f}"
        )
    return log_rows


def main() -> None:
    """Run the experimental image-dependence penalty test."""
    args = parse_readout_args("Experimental image-dependence penalty test.")
    context = build_experiment_context(args, subdir="image_dependence_penalty")
    epochs = readout_epoch_count(args, default=5)

    model = load_model(context)
    before_rows, before_delta = evaluate_model(model, context, split=args.split, method="before_penalty")
    log_rows = train_with_penalty(model, context, epochs=epochs)
    after_rows, after_delta = evaluate_model(model, context, split=args.split, method="after_penalty")

    prediction_rows = before_rows + after_rows
    delta_rows = before_delta + after_delta
    metrics = metric_rows(prediction_rows)

    metrics_path = context.output_dir / "image_dependence_penalty_metrics.csv"
    predictions_path = context.output_dir / "image_dependence_penalty_predictions.csv"
    delta_path = context.output_dir / "image_dependence_penalty_deltas.csv"
    log_path = context.output_dir / "image_dependence_penalty_training_log.csv"
    save_rows(metrics_path, metrics)
    save_rows(predictions_path, prediction_rows)
    save_rows(delta_path, delta_rows)
    save_rows(log_path, log_rows)
    plot_comparison(pd.DataFrame(prediction_rows), context.output_dir / "plots", limit=12)

    metrics_frame = pd.DataFrame(metrics)
    before_rmse = float(
        metrics_frame.loc[metrics_frame["method"] == "before_penalty", "CSI_RMSE"].iloc[0]
    ) if not metrics_frame.empty and (metrics_frame["method"] == "before_penalty").any() else float("nan")
    after_rmse = float(
        metrics_frame.loc[metrics_frame["method"] == "after_penalty", "CSI_RMSE"].iloc[0]
    ) if not metrics_frame.empty and (metrics_frame["method"] == "after_penalty").any() else float("nan")
    summary = {
        "split": args.split,
        "max_samples": args.max_samples,
        "epochs": epochs,
        "image_dependence_weight": float(args.image_dependence_weight),
        "image_dependence_margin": float(args.image_dependence_margin),
        "before_CSI_RMSE": before_rmse,
        "after_CSI_RMSE": after_rmse,
        "image_dependence_penalty_helped": bool(
            np.isfinite(before_rmse) and np.isfinite(after_rmse) and after_rmse < before_rmse
        ),
        "metrics_csv": str(metrics_path),
        "predictions_csv": str(predictions_path),
        "deltas_csv": str(delta_path),
    }
    write_summary(context.output_dir / "image_dependence_penalty_summary.json", summary)
    print(json.dumps(summary, indent=2))
    mirror(context)


if __name__ == "__main__":
    main()
