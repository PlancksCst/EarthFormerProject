"""Evaluate non-image CSI/GHI climatology baselines on project splits."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import torch

from predictability_common import (  # type: ignore
    build_context,
    capped_loader,
    clear_sky_tensor,
    diagnostic_valid_mask_tensor,
    mirror_outputs,
    parse_args,
    plot_rmse_bar,
    plot_sample_predictions,
    prediction_rows,
    previous_day_csi_from_csv,
    save_metric_tables,
    target_ghi_tensor,
    target_tensor,
    write_csv,
    write_json,
)


def train_climatology(context: Any) -> dict[str, Any]:
    """Compute global, hourly, and location-hour CSI means from the training split."""
    horizon = int(context.config.output_length)
    hourly_sum = np.zeros(horizon, dtype=np.float64)
    hourly_count = np.zeros(horizon, dtype=np.float64)
    global_sum = 0.0
    global_count = 0.0
    loc_hour_sum: dict[tuple[str, int], float] = {}
    loc_hour_count: dict[tuple[str, int], float] = {}
    solar_available = False

    loader = capped_loader(context, context.args.train_split, context.args.max_train_samples, shuffle=False)
    for batch in loader:
        target = target_tensor(batch, context.device)
        clear = clear_sky_tensor(batch, context.device)
        valid, solar = diagnostic_valid_mask_tensor(
            context.config,
            batch,
            target,
            clear,
            context.config.clear_sky_threshold,
            float(context.args.solar_elevation_threshold),
        )
        solar_available = solar_available or solar is not None
        target_np = target.detach().cpu().numpy()
        valid_np = valid.detach().cpu().numpy().astype(bool)
        locations = batch.get("location", ["unknown"] * target_np.shape[0])
        for sample_index in range(target_np.shape[0]):
            location = str(locations[sample_index]) if isinstance(locations, (list, tuple)) else str(locations)
            for hour_index in range(horizon):
                if not valid_np[sample_index, hour_index]:
                    continue
                value = float(target_np[sample_index, hour_index])
                if not np.isfinite(value):
                    continue
                hourly_sum[hour_index] += value
                hourly_count[hour_index] += 1.0
                global_sum += value
                global_count += 1.0
                key = (location, hour_index)
                loc_hour_sum[key] = loc_hour_sum.get(key, 0.0) + value
                loc_hour_count[key] = loc_hour_count.get(key, 0.0) + 1.0

    if global_count == 0:
        raise RuntimeError("Training split produced no valid CSI values for climatology")
    global_mean = global_sum / global_count
    hourly_mean = np.divide(hourly_sum, hourly_count, out=np.full(horizon, global_mean), where=hourly_count > 0)
    loc_hour_mean = {
        f"{location}::{hour_index}": loc_hour_sum[(location, hour_index)] / loc_hour_count[(location, hour_index)]
        for location, hour_index in loc_hour_sum
    }
    return {
        "global_mean": float(global_mean),
        "hourly_mean": hourly_mean.astype(np.float32),
        "location_hour_mean": loc_hour_mean,
        "solar_elevation_available": solar_available,
        "train_valid_count": int(global_count),
    }


def location_hour_prediction(batch: dict[str, Any], stats: dict[str, Any], horizon: int) -> torch.Tensor:
    """Return location-hour climatology predictions for a batch."""
    hourly = np.asarray(stats["hourly_mean"], dtype=np.float32)
    values = np.tile(hourly.reshape(1, horizon), (len(batch.get("target", batch["target_csi"])), 1))
    locations = batch.get("location")
    if locations is None:
        return torch.from_numpy(values)
    for sample_index in range(values.shape[0]):
        location = str(locations[sample_index]) if isinstance(locations, (list, tuple)) else str(locations)
        for hour_index in range(horizon):
            key = f"{location}::{hour_index}"
            if key in stats["location_hour_mean"]:
                values[sample_index, hour_index] = float(stats["location_hour_mean"][key])
    return torch.from_numpy(values.astype(np.float32))


def evaluate_baselines(context: Any, stats: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate all available climatology baselines on the eval split."""
    horizon = int(context.config.output_length)
    loader = capped_loader(context, context.args.eval_split, context.args.max_eval_samples, shuffle=False)
    all_rows: list[dict[str, Any]] = []
    sample_start = 0
    previous_available_count = 0
    previous_valid_count = 0
    solar_available = False

    for batch in loader:
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
        solar_available = solar_available or solar is not None
        batch_size = target.shape[0]
        predictions = {
            "global_constant_mean": torch.full_like(target, float(stats["global_mean"])),
            "hourly_climatology": torch.as_tensor(stats["hourly_mean"], device=context.device).view(1, horizon).expand_as(target),
            "location_hour_climatology": location_hour_prediction(batch, stats, horizon).to(context.device),
        }
        previous = previous_day_csi_from_csv(context.config, batch, horizon)
        if previous is not None:
            previous = previous.to(context.device)
            finite_previous = torch.isfinite(previous)
            if bool((finite_previous & valid).any().detach().cpu()):
                predictions["previous_day_csi_persistence"] = previous
                previous_available_count += batch_size
                previous_valid_count += int((finite_previous & valid).sum().detach().cpu())

        for name, prediction in predictions.items():
            all_rows.extend(
                prediction_rows(
                    batch=batch,
                    split=context.args.eval_split,
                    sample_start=sample_start,
                    prediction=prediction.detach().cpu(),
                    target=target.detach().cpu(),
                    clear=clear.detach().cpu(),
                    valid=valid.detach().cpu(),
                    target_ghi=target_ghi.detach().cpu(),
                    label_name="baseline",
                    label_value=name,
                )
            )
        sample_start += batch_size

    summary = {
        "previous_day_csi_persistence_available": previous_available_count > 0,
        "previous_day_csi_persistence_samples": previous_available_count,
        "previous_day_csi_persistence_valid_points": previous_valid_count,
        "solar_elevation_available": solar_available or bool(stats.get("solar_elevation_available")),
    }
    return all_rows, summary


def main() -> None:
    """Run climatology baseline evaluation."""
    args = parse_args("Evaluate non-image climatology CSI baselines.")
    context = build_context(args, default_subdir="climatology_baselines")
    stats = train_climatology(context)
    rows, eval_summary = evaluate_baselines(context, stats)

    predictions_path = context.output_dir / "climatology_baseline_predictions.csv"
    write_csv(predictions_path, rows)
    metric_paths = save_metric_tables(context.output_dir, "climatology_baseline", rows, label_col="baseline")
    metrics_frame = pd.read_csv(metric_paths["overall"])
    plot_rmse_bar(metrics_frame.to_dict("records"), context.output_dir / "climatology_baseline_rmse.png", label_col="baseline")
    plot_sample_predictions(rows, context.output_dir / "sample_plots", label_col="baseline", limit=8)

    summary = {
        "dataset_root": str(context.config.dataset_root),
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "clear_sky_threshold": context.config.clear_sky_threshold,
        "solar_elevation_threshold": float(args.solar_elevation_threshold),
        "train_valid_count": stats["train_valid_count"],
        "global_mean": stats["global_mean"],
        **eval_summary,
        "predictions_csv": str(predictions_path),
        "metrics": metric_paths,
    }
    summary_path = context.output_dir / "climatology_baseline_summary.json"
    write_json(summary_path, summary)
    mirror_outputs(context)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
