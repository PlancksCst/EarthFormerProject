"""Evaluate simple CSI/GHI baselines against the forecasting split."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

try:
    from .diagnostic_common import (
        build_context,
        dataloader_for_split,
        metrics_from_rows,
        mirror_outputs,
        parse_common_args,
        plot_metric_bar,
        sample_rows_for_loader,
        write_csv,
        write_json,
    )
except ImportError:
    from diagnostic_common import (  # type: ignore
        build_context,
        dataloader_for_split,
        metrics_from_rows,
        mirror_outputs,
        parse_common_args,
        plot_metric_bar,
        sample_rows_for_loader,
        write_csv,
        write_json,
    )


class RunningMean:
    """Tiny running-mean accumulator."""

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, values: np.ndarray) -> None:
        finite = values[np.isfinite(values)]
        self.total += float(finite.sum())
        self.count += int(finite.size)

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else float("nan")


def valid_numpy(batch: dict[str, Any], clear_sky_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Return target CSI and valid mask on CPU numpy."""
    target = batch["target"].detach().float().cpu().numpy()
    clear = batch["clear_sky_ghi"].detach().float().cpu().numpy()
    mask = batch.get("target_mask")
    if isinstance(mask, torch.Tensor):
        valid = ~mask.detach().cpu().numpy().astype(bool)
    else:
        valid = np.ones_like(target, dtype=bool)
    valid &= np.isfinite(clear) & (clear > clear_sky_threshold)
    return target, valid


def compute_climatologies(context: Any) -> dict[str, Any]:
    """Compute train mean, hourly mean, and location-hour mean CSI."""
    loader = dataloader_for_split(
        config=context.config,
        split="train",
        include_target=True,
        shuffle=False,
        max_samples=context.args.max_samples,
    )
    global_mean = RunningMean()
    hourly = [RunningMean() for _ in range(context.config.output_length)]
    location_hour: dict[tuple[str, int], RunningMean] = defaultdict(RunningMean)

    for batch in loader:
        target, valid = valid_numpy(batch, context.config.clear_sky_threshold)
        batch_size, horizon = target.shape
        global_mean.update(target[valid])
        locations = batch.get("location", ["unknown"] * batch_size)
        for hour in range(horizon):
            hourly[hour].update(target[:, hour][valid[:, hour]])
        for index in range(batch_size):
            location = str(locations[index]) if isinstance(locations, (list, tuple)) else str(locations)
            for hour in range(horizon):
                if valid[index, hour]:
                    location_hour[(location, hour)].update(np.asarray([target[index, hour]], dtype=np.float64))

    hourly_values = np.asarray(
        [
            item.mean if item.count else global_mean.mean
            for item in hourly
        ],
        dtype=np.float32,
    )
    location_hour_values = {
        key: value.mean
        for key, value in location_hour.items()
        if value.count > 0 and np.isfinite(value.mean)
    }
    return {
        "global_mean": float(global_mean.mean),
        "hourly": hourly_values,
        "location_hour": location_hour_values,
        "train_valid_count": int(global_mean.count),
    }


def previous_day_csi(batch: dict[str, Any], device: torch.device) -> torch.Tensor | None:
    """Return previous-day/input CSI when the dataset item exposes it."""
    for key in (
        "input_csi",
        "previous_day_csi",
        "previous_csi",
        "prev_csi",
        "input_day_csi",
    ):
        value = batch.get(key)
        if isinstance(value, torch.Tensor) and value.ndim == 2:
            return value.float().to(device, non_blocking=True)
    return None


def evaluate_baseline(context: Any, name: str, climatology: dict[str, Any]) -> list[dict[str, Any]]:
    """Evaluate one baseline and return long prediction rows."""
    loader = dataloader_for_split(
        config=context.config,
        split=context.args.split,
        include_target=True,
        shuffle=False,
        max_samples=context.args.max_samples,
    )
    horizon = context.config.output_length
    hourly = torch.as_tensor(climatology["hourly"], dtype=torch.float32, device=context.device)
    global_mean = float(climatology["global_mean"])
    location_hour = climatology["location_hour"]

    def predict(batch: dict[str, Any], target: torch.Tensor, clear: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        batch_size = target.shape[0]
        if name == "constant_train_mean":
            return torch.full_like(target, global_mean)
        if name == "hourly_climatology":
            return hourly[:horizon].unsqueeze(0).expand(batch_size, horizon).clone()
        if name == "location_hour_climatology":
            result = hourly[:horizon].unsqueeze(0).expand(batch_size, horizon).clone()
            locations = batch.get("location", ["unknown"] * batch_size)
            for index in range(batch_size):
                location = str(locations[index]) if isinstance(locations, (list, tuple)) else str(locations)
                for hour in range(horizon):
                    value = location_hour.get((location, hour))
                    if value is not None and np.isfinite(value):
                        result[index, hour] = float(value)
            return result
        if name == "previous_day_csi_persistence":
            previous = previous_day_csi(batch, context.device)
            if previous is None:
                raise KeyError("previous-day CSI is unavailable in dataset items")
            if previous.shape != target.shape:
                raise KeyError(
                    f"previous-day CSI shape {tuple(previous.shape)} does not match target shape {tuple(target.shape)}"
                )
            return previous
        if name == "clear_sky_csi_1":
            return torch.ones_like(target)
        raise ValueError(f"Unknown baseline: {name}")

    return sample_rows_for_loader(
        loader=loader,
        split=context.args.split,
        device=context.device,
        clear_sky_threshold=context.config.clear_sky_threshold,
        prediction_fn=predict,
        baseline_name=name,
    )


def main() -> None:
    """Run baseline diagnostics."""
    args = parse_common_args("Evaluate non-neural CSI/GHI baselines.")
    context = build_context(args, default_subdir="baselines")
    climatology = compute_climatologies(context)
    baseline_names = [
        "constant_train_mean",
        "hourly_climatology",
        "location_hour_climatology",
        "previous_day_csi_persistence",
        "clear_sky_csi_1",
    ]
    all_rows: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    unavailable: dict[str, str] = {}

    for name in baseline_names:
        try:
            rows = evaluate_baseline(context, name, climatology)
        except KeyError as error:
            unavailable[name] = str(error)
            continue
        all_rows.extend(rows)
        metrics.append(metrics_from_rows(rows, {"baseline": name}))

    metrics_frame = pd.DataFrame(metrics)
    predictions_path = context.output_dir / "baseline_predictions.csv"
    metrics_path = context.output_dir / "baseline_metrics.csv"
    summary_path = context.output_dir / "baseline_summary.json"
    plot_path = context.output_dir / "baseline_rmse_comparison.png"

    write_csv(predictions_path, all_rows)
    write_csv(metrics_path, metrics)
    plot_metric_bar(metrics_frame, plot_path)
    write_json(
        summary_path,
        {
            "dataset_root": str(context.config.dataset_root),
            "split": context.args.split,
            "max_samples": context.args.max_samples,
            "clear_sky_threshold": context.config.clear_sky_threshold,
            "train_valid_count": climatology["train_valid_count"],
            "global_train_mean_csi": climatology["global_mean"],
            "available_baselines": [row["baseline"] for row in metrics],
            "unavailable_baselines": unavailable,
            "metrics_csv": str(metrics_path),
            "predictions_csv": str(predictions_path),
            "plot": str(plot_path),
        },
    )
    mirror_outputs(context)
    print(metrics_frame.to_string(index=False) if not metrics_frame.empty else "No baselines evaluated.")
    if unavailable:
        print(f"Unavailable baselines: {unavailable}")


if __name__ == "__main__":
    main()
