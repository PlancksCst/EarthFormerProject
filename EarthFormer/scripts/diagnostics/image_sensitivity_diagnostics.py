"""Diagnose whether forecasts change when SEVIRI image inputs are perturbed."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd
import torch

try:
    from .diagnostic_common import (
        batch_prediction_rows,
        build_context,
        clear_sky_tensor,
        dataloader_for_split,
        load_trained_model,
        maybe_autocast,
        metrics_from_rows,
        mirror_outputs,
        parse_common_args,
        plot_series,
        regression_metrics,
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
        load_trained_model,
        maybe_autocast,
        metrics_from_rows,
        mirror_outputs,
        parse_common_args,
        plot_series,
        regression_metrics,
        target_ghi_tensor,
        target_tensor,
        valid_mask_tensor,
        write_csv,
        write_json,
    )


PERTURBATIONS = (
    "real",
    "zero",
    "random_other_sample",
    "time_reversed",
    "channel_shuffled",
    "noisy",
)


def next_replacement(iterator: Any, loader: Any, device: torch.device, target_shape: torch.Size) -> tuple[Any, torch.Tensor]:
    """Return a replacement image batch, cycling when needed."""
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    images = batch["satellite"].to(device, non_blocking=True)
    if images.shape[0] < target_shape[0]:
        repeats = int(np.ceil(target_shape[0] / max(images.shape[0], 1)))
        images = images.repeat((repeats, 1, 1, 1, 1))
    return iterator, images[: target_shape[0]]


def perturb_inputs(
    inputs: torch.Tensor,
    replacement: torch.Tensor,
    name: str,
    noise_std: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Return perturbed image input."""
    if name == "real":
        return inputs
    if name == "zero":
        return torch.zeros_like(inputs)
    if name == "random_other_sample":
        return replacement.to(device=inputs.device, dtype=inputs.dtype)
    if name == "time_reversed":
        return torch.flip(inputs, dims=[1])
    if name == "channel_shuffled":
        permutation = torch.randperm(inputs.shape[2], generator=generator, device=inputs.device)
        return inputs[:, :, permutation, :, :]
    if name == "noisy":
        noise = torch.randn(
            inputs.shape,
            generator=generator,
            device=inputs.device,
            dtype=inputs.dtype,
        )
        return inputs + float(noise_std) * noise
    raise ValueError(f"Unknown perturbation: {name}")


def prediction_delta_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute prediction-change and forecast-error metrics by perturbation."""
    frame = pd.DataFrame(rows)
    metrics: list[dict[str, Any]] = []
    if frame.empty:
        return metrics
    for name, group in frame.groupby("perturbation", sort=False):
        valid = group[group["valid_hour"].astype(bool)]
        delta = valid["predicted_csi"].to_numpy() - valid["real_predicted_csi"].to_numpy()
        forecast = metrics_from_rows(valid.to_dict("records"), {"perturbation": str(name)})
        delta_metrics = regression_metrics(valid["predicted_csi"].to_numpy(), valid["real_predicted_csi"].to_numpy())
        forecast.update(
            {
                "delta_MAE_vs_real": delta_metrics["MAE"],
                "delta_RMSE_vs_real": delta_metrics["RMSE"],
                "delta_mean_abs": float(np.mean(np.abs(delta))) if delta.size else np.nan,
                "delta_Pearson_vs_real": delta_metrics["Pearson"],
            }
        )
        metrics.append(forecast)
    return metrics


def main() -> None:
    """Run image sensitivity diagnostics."""
    args = parse_common_args("Evaluate model sensitivity to image perturbations.")
    parser_max_plots = getattr(args, "max_plots", None)
    if parser_max_plots is None:
        setattr(args, "max_plots", 20)
    context = build_context(args, default_subdir="image_sensitivity")
    model = load_trained_model(context)
    loader = dataloader_for_split(
        context.config,
        split=args.split,
        include_target=True,
        shuffle=False,
        max_samples=args.max_samples,
    )
    replacement_loader = dataloader_for_split(
        context.config,
        split=args.split,
        include_target=True,
        shuffle=True,
        max_samples=args.max_samples,
    )
    replacement_iterator = iter(replacement_loader)
    generator = torch.Generator(device=context.device)
    generator.manual_seed(int(context.config.random_seed))

    rows: list[dict[str, Any]] = []
    plot_count = 0
    sample_start = 0
    with torch.no_grad():
        for batch in loader:
            inputs = batch["satellite"].to(context.device, non_blocking=True)
            replacement_iterator, replacement = next_replacement(
                replacement_iterator,
                replacement_loader,
                context.device,
                inputs.shape,
            )
            target = target_tensor(batch, context.device)
            clear = clear_sky_tensor(batch, context.device)
            target_ghi = target_ghi_tensor(batch, target, clear)
            valid = valid_mask_tensor(
                batch,
                target,
                clear,
                context.config.clear_sky_threshold,
            )
            predictions: dict[str, torch.Tensor] = {}
            for name in PERTURBATIONS:
                perturbed = perturb_inputs(
                    inputs,
                    replacement,
                    name,
                    args.noise_std,
                    generator,
                )
                with maybe_autocast(context):
                    predictions[name] = model(perturbed).detach().float()
            real_prediction = predictions["real"]
            for name, pred in predictions.items():
                method_rows = batch_prediction_rows(
                    batch=batch,
                    split=args.split,
                    sample_start=sample_start,
                    pred_csi=pred,
                    target_csi=target,
                    clear_sky_ghi=clear,
                    valid_mask=valid,
                    target_ghi=target_ghi,
                )
                real_cpu = real_prediction.detach().float().cpu().numpy()
                for row in method_rows:
                    hour = int(row["forecast_hour"]) - 1
                    local_index = int(row["sample_index"]) - sample_start
                    row["perturbation"] = name
                    row["real_predicted_csi"] = float(real_cpu[local_index, hour])
                    row["delta_vs_real"] = float(row["predicted_csi"] - row["real_predicted_csi"])
                rows.extend(method_rows)

            batch_size = inputs.shape[0]
            for local_index in range(batch_size):
                if plot_count >= int(args.max_plots):
                    break
                hours = np.arange(1, target.shape[1] + 1)
                series = {"target": target[local_index].detach().cpu().numpy()}
                for name in PERTURBATIONS:
                    series[name] = predictions[name][local_index].detach().cpu().numpy()
                sample_id = row_id(batch, local_index, sample_start)
                plot_series(
                    context.output_dir / "plots" / f"{sample_id}_sensitivity.png",
                    hours,
                    series,
                    title=f"Image sensitivity | {sample_id}",
                    ylabel="CSI",
                )
                plot_count += 1
            sample_start += batch_size

    metrics = prediction_delta_metrics(rows)
    predictions_path = context.output_dir / "image_sensitivity_predictions.csv"
    metrics_path = context.output_dir / "image_sensitivity_metrics.csv"
    summary_path = context.output_dir / "image_sensitivity_summary.json"
    write_csv(predictions_path, rows)
    write_csv(metrics_path, metrics)
    plot_delta_bars(pd.DataFrame(metrics), context.output_dir / "image_sensitivity_delta_bars.png")
    write_json(
        summary_path,
        {
            "dataset_root": str(context.config.dataset_root),
            "checkpoint": str(args.checkpoint or context.config.checkpoint_dir / "best.pt"),
            "split": args.split,
            "max_samples": args.max_samples,
            "noise_std": args.noise_std,
            "metrics_csv": str(metrics_path),
            "predictions_csv": str(predictions_path),
            "plot_count": plot_count,
        },
    )
    mirror_outputs(context)
    print(pd.DataFrame(metrics).to_string(index=False) if metrics else "No sensitivity rows produced.")


def row_id(batch: dict[str, Any], local_index: int, sample_start: int) -> str:
    """Return a compact sample id for filenames."""
    sample_id = batch.get("sample_id")
    location = batch.get("location")
    target_day = batch.get("target_day")
    sid = sample_id[local_index] if isinstance(sample_id, (list, tuple)) else sample_start + local_index
    loc = location[local_index] if isinstance(location, (list, tuple)) else "location"
    day = target_day[local_index] if isinstance(target_day, (list, tuple)) else "day"
    return f"{sid}_{loc}_{day}".replace("/", "-").replace("\\", "-").replace(":", "-")


def plot_delta_bars(frame: pd.DataFrame, path: Any) -> None:
    """Plot average absolute prediction deltas by perturbation."""
    if frame.empty or "delta_mean_abs" not in frame.columns:
        return
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.bar(frame["perturbation"].astype(str), frame["delta_mean_abs"].astype(float))
    ax.set_ylabel("Mean |prediction delta| vs real input")
    ax.set_title("Image Perturbation Sensitivity")
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
