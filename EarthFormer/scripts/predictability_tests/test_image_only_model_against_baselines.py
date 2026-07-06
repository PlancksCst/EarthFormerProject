"""Compare the trained image-only EarthFormer model against climatology baselines."""

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
    load_checked_image_model,
    maybe_autocast,
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
from test_climatology_baselines import location_hour_prediction, train_climatology  # type: ignore


def run_model_and_baselines(context: Any, stats: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run image model and baselines over identical eval batches."""
    model, checkpoint = load_checked_image_model(context)
    horizon = int(context.config.output_length)
    loader = capped_loader(context, context.args.eval_split, context.args.max_eval_samples, shuffle=False)
    all_rows: list[dict[str, Any]] = []
    sample_start = 0
    previous_available = False

    with torch.no_grad():
        for batch in loader:
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
            with maybe_autocast(context):
                image_prediction = model(inputs)

            predictions = {
                "image_only_earthformer_perceiver": image_prediction,
                "global_constant_mean": torch.full_like(target, float(stats["global_mean"])),
                "hourly_climatology": torch.as_tensor(stats["hourly_mean"], device=context.device).view(1, horizon).expand_as(target),
                "location_hour_climatology": location_hour_prediction(batch, stats, horizon).to(context.device),
            }
            previous = previous_day_csi_from_csv(context.config, batch, horizon)
            if previous is not None:
                previous = previous.to(context.device)
                if bool((torch.isfinite(previous) & valid).any().detach().cpu()):
                    predictions["previous_day_csi_persistence"] = previous
                    previous_available = True

            for method, prediction in predictions.items():
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
                        label_name="method",
                        label_value=method,
                    )
                )
            sample_start += target.shape[0]

    return all_rows, {
        "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "checkpoint_best_metric": checkpoint.get("best_metric", checkpoint.get("best_loss")) if isinstance(checkpoint, dict) else None,
        "previous_day_csi_persistence_available": previous_available,
    }


def interpretation(metrics: pd.DataFrame, previous_available: bool) -> dict[str, Any]:
    """Return comparison flags and a rule-based interpretation."""
    if metrics.empty:
        return {"recommended_interpretation": "no_metrics_available"}
    image_rows = metrics[metrics["method"] == "image_only_earthformer_perceiver"]
    if image_rows.empty:
        return {"recommended_interpretation": "image_model_missing_from_metrics"}
    image_rmse = float(image_rows["CSI_RMSE"].iloc[0])
    baselines = metrics[metrics["method"] != "image_only_earthformer_perceiver"].copy()
    best_baseline = baselines.sort_values("CSI_RMSE").iloc[0].to_dict() if not baselines.empty else {}

    def beats(name: str) -> bool | None:
        rows = metrics[metrics["method"] == name]
        if rows.empty:
            return None
        return bool(image_rmse < float(rows["CSI_RMSE"].iloc[0]))

    beats_hourly = beats("hourly_climatology")
    beats_location = beats("location_hour_climatology")
    beats_persistence = beats("previous_day_csi_persistence") if previous_available else None
    if beats_hourly and (beats_location is True or beats_location is None) and (beats_persistence is True or beats_persistence is None):
        recommendation = "image_model_beats_available_baselines_next_day"
    elif beats_hourly is False or beats_location is False or beats_persistence is False:
        recommendation = "image_model_does_not_beat_simple_next_day_baselines"
    else:
        recommendation = "baseline_comparison_inconclusive"
    return {
        "best_baseline_by_CSI_RMSE": best_baseline.get("method"),
        "best_baseline_CSI_RMSE": best_baseline.get("CSI_RMSE"),
        "image_model_CSI_RMSE": image_rmse,
        "image_model_beats_hourly_climatology": beats_hourly,
        "image_model_beats_location_hour_climatology": beats_location,
        "image_model_beats_previous_day_persistence": beats_persistence,
        "recommended_interpretation": recommendation,
    }


def main() -> None:
    """Run the image-only-vs-baselines comparison."""
    args = parse_args("Compare trained image-only model against climatology baselines.")
    context = build_context(args, default_subdir="image_only_vs_baselines")
    stats = train_climatology(context)
    rows, run_summary = run_model_and_baselines(context, stats)

    predictions_path = context.output_dir / "image_only_vs_baseline_predictions.csv"
    write_csv(predictions_path, rows)
    metric_paths = save_metric_tables(context.output_dir, "image_only_vs_baseline", rows, label_col="method")
    metrics_frame = pd.read_csv(metric_paths["overall"])
    plot_rmse_bar(metrics_frame.to_dict("records"), context.output_dir / "image_only_vs_baseline_rmse.png", label_col="method")
    plot_sample_predictions(rows, context.output_dir / "sample_plots", label_col="method", limit=8)

    summary = {
        "dataset_root": str(context.config.dataset_root),
        "checkpoint": str(args.checkpoint),
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        **run_summary,
        **interpretation(metrics_frame, bool(run_summary["previous_day_csi_persistence_available"])),
        "predictions_csv": str(predictions_path),
        "metrics": metric_paths,
    }
    write_json(context.output_dir / "image_only_vs_baseline_summary.json", summary)
    mirror_outputs(context)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
