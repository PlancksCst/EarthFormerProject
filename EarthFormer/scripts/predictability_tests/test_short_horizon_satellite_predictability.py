"""Short-horizon satellite predictability tests for CSI forecasting."""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from predictability_common import (  # type: ignore
    build_context,
    capped_dataset,
    grouped_metrics,
    load_checked_image_model,
    load_hourly_frame,
    location_columns,
    maybe_autocast,
    mirror_outputs,
    parse_args,
    parse_int_list,
    plot_rmse_bar,
    save_metric_tables,
    write_csv,
    write_json,
)
from diagnostic_common import load_elevation_frame, solar_column  # type: ignore


@dataclass(frozen=True)
class ShortHorizonIndex:
    """One short-horizon sample index."""

    day_index: int
    start_index: int
    end_index: int
    target_hour_index: int
    lead_hours: int


def base_metadata_row(dataset: Any, index: int) -> Any:
    """Return metadata row from a base dataset or Subset."""
    if isinstance(dataset, Subset):
        actual_index = int(dataset.indices[index])
        base = dataset.dataset
    else:
        actual_index = int(index)
        base = dataset
    return getattr(base, "meta").iloc[actual_index]


class ShortHorizonDataset(Dataset):
    """Diagnostic same-day sequence windows built from the existing daily dataset."""

    def __init__(
        self,
        config: Any,
        split: str,
        lead_hours: int,
        history_hours: int,
        max_day_samples: int | None,
        clear_sky_threshold: float,
        solar_elevation_threshold: float,
        include_satellite: bool = True,
        cache_days: int = 32,
    ) -> None:
        self.config = config
        self.split = split
        self.lead_hours = int(lead_hours)
        self.history_hours = int(history_hours)
        self.clear_sky_threshold = float(clear_sky_threshold)
        self.solar_elevation_threshold = float(solar_elevation_threshold)
        self.include_satellite = bool(include_satellite)
        self.cache_days = max(0, int(cache_days))
        self._item_cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self.base = capped_dataset(config, split=split, include_target=True, max_samples=max_day_samples)
        self.hourly = load_hourly_frame(Path(config.hourly_csv))
        self.elevation = load_elevation_frame(str(config.elevation_csv))
        self.indices: list[ShortHorizonIndex] = []
        horizon = int(config.input_length)
        for day_index in range(len(self.base)):
            for target_hour_index in range(horizon):
                end_index = target_hour_index - self.lead_hours
                start_index = end_index - self.history_hours + 1
                if start_index < 0 or end_index < 0 or target_hour_index >= horizon:
                    continue
                self.indices.append(
                    ShortHorizonIndex(
                        day_index=day_index,
                        start_index=start_index,
                        end_index=end_index,
                        target_hour_index=target_hour_index,
                        lead_hours=self.lead_hours,
                    )
                )
        if not self.indices:
            raise ValueError(
                f"No short-horizon windows for split={split}, "
                f"lead_hours={lead_hours}, history_hours={history_hours}"
            )

    def __len__(self) -> int:
        return len(self.indices)

    def _hourly_values(self, location: str, timestamp: pd.Timestamp) -> tuple[float, float, float]:
        columns = location_columns(self.hourly, location)
        if columns is None:
            return np.nan, np.nan, np.nan
        row = self.hourly.loc[timestamp] if timestamp in self.hourly.index else None
        if row is None:
            return np.nan, np.nan, np.nan
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        csi = float(row[columns["csi"]]) if pd.notna(row[columns["csi"]]) else np.nan
        ghi = float(row[columns["ghi"]]) if pd.notna(row[columns["ghi"]]) else np.nan
        clear = float(row[columns["clear"]]) if pd.notna(row[columns["clear"]]) else np.nan
        return csi, ghi, clear

    def _solar_elevation(self, location: str, timestamp: pd.Timestamp) -> float | None:
        if self.elevation is None:
            return None
        column = solar_column(self.elevation, location)
        if column is None or timestamp not in self.elevation.index:
            return None
        value = self.elevation.loc[timestamp, column]
        if isinstance(value, pd.Series):
            value = value.iloc[0]
        if pd.isna(value):
            return None
        value = float(value)
        return value if np.isfinite(value) else None

    def _base_item(self, day_index: int) -> dict[str, Any]:
        """Return one existing dataset item, with a small per-worker LRU cache."""
        if self.cache_days <= 0:
            return self.base[day_index]
        if day_index in self._item_cache:
            item = self._item_cache.pop(day_index)
            self._item_cache[day_index] = item
            return item
        item = self.base[day_index]
        self._item_cache[day_index] = item
        while len(self._item_cache) > self.cache_days:
            self._item_cache.popitem(last=False)
        return item

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.indices[index]
        row = base_metadata_row(self.base, entry.day_index)
        if self.include_satellite:
            item = self._base_item(entry.day_index)
            location = str(item["location"])
            input_day_value = item["input_day"]
            sample_id = item.get("sample_id", entry.day_index)
        else:
            location = str(row.location)
            input_day_value = row.input_day
            sample_id = int(row.sample_id) if "sample_id" in row.index else entry.day_index
        input_day = pd.Timestamp(input_day_value)
        target_timestamp = input_day + pd.Timedelta(hours=4 + entry.target_hour_index)
        current_timestamp = input_day + pd.Timedelta(hours=4 + entry.end_index)
        target_csi, target_ghi, clear = self._hourly_values(location, target_timestamp)
        current_csi, _current_ghi, _current_clear = self._hourly_values(location, current_timestamp)
        solar = self._solar_elevation(location, target_timestamp)
        solar_available = solar is not None
        target_finite = np.isfinite(target_csi) and np.isfinite(clear)
        if target_finite and not np.isfinite(target_ghi):
            target_ghi = target_csi * clear
        daylight_valid = target_finite and clear > self.clear_sky_threshold
        if solar_available:
            daylight_valid = daylight_valid and solar >= self.solar_elevation_threshold
        result = {
            "target": torch.tensor(0.0 if not np.isfinite(target_csi) else target_csi, dtype=torch.float32),
            "target_csi": torch.tensor(0.0 if not np.isfinite(target_csi) else target_csi, dtype=torch.float32),
            "target_ghi": torch.tensor(0.0 if not np.isfinite(target_ghi) else target_ghi, dtype=torch.float32),
            "clear_sky_ghi": torch.tensor(0.0 if not np.isfinite(clear) else clear, dtype=torch.float32),
            "valid": torch.tensor(bool(daylight_valid), dtype=torch.bool),
            "current_csi": torch.tensor(np.nan if not np.isfinite(current_csi) else current_csi, dtype=torch.float32),
            "sample_id": int(sample_id) if not isinstance(sample_id, torch.Tensor) else int(sample_id.item()),
            "location": location,
            "input_day": str(input_day_value),
            "target_day": str(input_day.date()),
            "target_timestamp": str(target_timestamp),
            "forecast_hour": int(entry.target_hour_index + 1),
            "hour_index": int(entry.target_hour_index),
            "lead_hours": int(entry.lead_hours),
            "history_hours": int(self.history_hours),
            "solar_elevation": float("nan") if solar is None else float(solar),
            "solar_elevation_available": bool(solar_available),
            "source_metadata_sample_id": int(row.sample_id) if "sample_id" in row.index else int(entry.day_index),
        }
        if self.include_satellite:
            satellite = item["satellite"][entry.start_index : entry.end_index + 1]
            if satellite.shape[0] != self.history_hours:
                raise RuntimeError(f"Bad history tensor length: {satellite.shape[0]} != {self.history_hours}")
            result["satellite"] = satellite
        return result


class SimpleCNNLSTM(nn.Module):
    """Small diagnostic CNN-GRU model for short-horizon scalar CSI prediction."""

    def __init__(self, input_channels: int = 7, hidden_dim: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=5, stride=4, padding=2),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=4, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.temporal = nn.GRU(input_size=32, hidden_size=hidden_dim, batch_first=True)
        self.regression = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return scalar CSI predictions for `(B,T,C,H,W)` inputs."""
        bsz, steps, channels, height, width = x.shape
        features = self.encoder(x.reshape(bsz * steps, channels, height, width)).flatten(1)
        features = features.reshape(bsz, steps, -1)
        _sequence, hidden = self.temporal(features)
        return self.regression(hidden[-1]).squeeze(-1)


class FrozenEarthFormerPoolMLP(nn.Module):
    """Diagnostic scalar readout over frozen EarthFormer latents."""

    def __init__(self, context: Any) -> None:
        super().__init__()
        image_model, _checkpoint = load_checked_image_model(context)
        self.earthformer = image_model.earthformer
        for parameter in self.earthformer.parameters():
            parameter.requires_grad = False
        latent_dim = int(context.config.readout_latent_dim)
        self.input_length = int(context.config.input_length)
        self.regression = nn.Sequential(
            nn.Linear(2 * latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pad history to EarthFormer length, pool latents, and predict scalar CSI."""
        if x.shape[1] < self.input_length:
            pad_shape = (x.shape[0], self.input_length - x.shape[1], *x.shape[2:])
            x = torch.cat([torch.zeros(pad_shape, dtype=x.dtype, device=x.device), x], dim=1)
        elif x.shape[1] > self.input_length:
            x = x[:, -self.input_length :]
        with torch.no_grad():
            latent = self.earthformer.forward_latent(x, return_trace=False)
        mean = latent.mean(dim=(1, 2, 3))
        std = latent.std(dim=(1, 2, 3), unbiased=False)
        return self.regression(torch.cat([mean, std], dim=-1)).squeeze(-1)


def short_loader(
    context: Any,
    split: str,
    lead: int,
    max_day_samples: int | None,
    shuffle: bool,
    include_satellite: bool = True,
) -> DataLoader:
    """Build a short-horizon dataloader for one lead time."""
    dataset = ShortHorizonDataset(
        config=context.config,
        split=split,
        lead_hours=lead,
        history_hours=int(context.args.history_hours),
        max_day_samples=max_day_samples,
        clear_sky_threshold=float(context.config.clear_sky_threshold),
        solar_elevation_threshold=float(context.args.solar_elevation_threshold),
        include_satellite=include_satellite,
        cache_days=int(context.args.short_horizon_cache_days),
    )
    return DataLoader(
        dataset,
        batch_size=context.config.batch_size,
        shuffle=shuffle,
        num_workers=context.config.num_workers,
        pin_memory=context.config.resolved_device().startswith("cuda"),
        drop_last=False,
        persistent_workers=context.config.num_workers > 0,
    )


def make_model(context: Any) -> nn.Module:
    """Construct the requested diagnostic model."""
    if context.args.model == "simple_cnn_lstm":
        return SimpleCNNLSTM(input_channels=int(context.config.input_channels)).to(context.device)
    if context.args.model == "frozen_earthformer_pool_mlp":
        return FrozenEarthFormerPoolMLP(context).to(context.device)
    raise ValueError(f"Unknown short-horizon model: {context.args.model}")


def masked_scalar_mse(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """MSE over valid scalar targets."""
    if int(valid.sum().detach().cpu()) == 0:
        return prediction.new_zeros(())
    return ((prediction - target) ** 2)[valid].mean()


def train_model(context: Any, lead: int) -> nn.Module:
    """Train one diagnostic short-horizon model for one lead time."""
    model = make_model(context)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(context.config.learning_rate), weight_decay=float(context.config.weight_decay))
    loader = short_loader(
        context,
        context.args.train_split,
        lead,
        context.args.max_train_samples,
        shuffle=bool(context.args.short_horizon_shuffle),
        include_satellite=True,
    )
    epochs = int(context.config.epochs)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_valid = 0
        for batch in loader:
            inputs = batch["satellite"].to(context.device, non_blocking=True)
            target = batch["target"].to(context.device, non_blocking=True)
            valid = batch["valid"].to(context.device, non_blocking=True).bool()
            if int(valid.sum().detach().cpu()) == 0:
                continue
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)
            loss = masked_scalar_mse(prediction, target, valid)
            loss.backward()
            optimizer.step()
            valid_count = int(valid.sum().detach().cpu())
            total_loss += float(loss.detach().cpu()) * valid_count
            total_valid += valid_count
        print(f"lead={lead} epoch={epoch:03d}/{epochs:03d} loss={total_loss / max(total_valid, 1):.6f}")
    return model


def train_short_climatology(loader: DataLoader) -> dict[str, Any]:
    """Compute hourly and location-hour means from short-horizon training rows."""
    hour_sum: dict[int, float] = {}
    hour_count: dict[int, float] = {}
    loc_hour_sum: dict[tuple[str, int], float] = {}
    loc_hour_count: dict[tuple[str, int], float] = {}
    global_sum = 0.0
    global_count = 0.0
    for batch in loader:
        target = batch["target"].float()
        valid = batch["valid"].bool()
        hours = batch["hour_index"]
        locations = batch["location"]
        for index in range(target.shape[0]):
            if not bool(valid[index]) or not torch.isfinite(target[index]):
                continue
            value = float(target[index])
            hour = int(hours[index])
            location = str(locations[index]) if isinstance(locations, (list, tuple)) else str(locations)
            hour_sum[hour] = hour_sum.get(hour, 0.0) + value
            hour_count[hour] = hour_count.get(hour, 0.0) + 1.0
            key = (location, hour)
            loc_hour_sum[key] = loc_hour_sum.get(key, 0.0) + value
            loc_hour_count[key] = loc_hour_count.get(key, 0.0) + 1.0
            global_sum += value
            global_count += 1.0
    if global_count == 0:
        raise RuntimeError("No valid short-horizon training targets")
    global_mean = global_sum / global_count
    return {
        "global_mean": global_mean,
        "hour_mean": {str(hour): hour_sum[hour] / hour_count[hour] for hour in hour_sum},
        "loc_hour_mean": {f"{loc}::{hour}": loc_hour_sum[(loc, hour)] / loc_hour_count[(loc, hour)] for loc, hour in loc_hour_sum},
    }


def scalar_rows(batch: dict[str, Any], prediction: torch.Tensor, method: str, sample_start: int, split: str) -> list[dict[str, Any]]:
    """Create long rows for scalar short-horizon predictions."""
    pred = prediction.detach().float().cpu()
    target = batch["target"].detach().float().cpu()
    clear = batch["clear_sky_ghi"].detach().float().cpu()
    target_ghi = batch["target_ghi"].detach().float().cpu()
    valid = batch["valid"].detach().cpu().bool()
    rows: list[dict[str, Any]] = []
    for index in range(pred.shape[0]):
        rows.append(
            {
                "split": split,
                "sample_index": sample_start + index,
                "sample_id": int(batch["sample_id"][index]),
                "location": batch["location"][index],
                "input_day": batch["input_day"][index],
                "target_day": batch["target_day"][index],
                "date": batch["target_day"][index],
                "target_timestamp": batch["target_timestamp"][index],
                "forecast_hour": int(batch["forecast_hour"][index]),
                "hour_index": int(batch["hour_index"][index]),
                "lead_hours": int(batch["lead_hours"][index]),
                "history_hours": int(batch["history_hours"][index]),
                "method": method,
                "valid_hour": bool(valid[index]),
                "target_csi": float(target[index]),
                "predicted_csi": float(pred[index]),
                "clear_sky_ghi": float(clear[index]),
                "target_ghi": float(target_ghi[index]),
                "predicted_ghi": float(pred[index] * clear[index]),
                "error_csi": float(pred[index] - target[index]),
                "error_ghi": float(pred[index] * clear[index] - target_ghi[index]),
            }
        )
    return rows


def baseline_predictions(batch: dict[str, Any], stats: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Return short-horizon baseline predictions for a batch."""
    target = batch["target"].float()
    hourly = []
    loc_hour = []
    locations = batch["location"]
    for index in range(target.shape[0]):
        hour = int(batch["hour_index"][index])
        location = str(locations[index]) if isinstance(locations, (list, tuple)) else str(locations)
        hourly_value = float(stats["hour_mean"].get(str(hour), stats["global_mean"]))
        loc_value = float(stats["loc_hour_mean"].get(f"{location}::{hour}", hourly_value))
        hourly.append(hourly_value)
        loc_hour.append(loc_value)
    predictions = {
        "hourly_climatology": torch.tensor(hourly, dtype=torch.float32),
        "location_hour_climatology": torch.tensor(loc_hour, dtype=torch.float32),
    }
    current = batch["current_csi"].float()
    if bool((torch.isfinite(current) & batch["valid"].bool()).any()):
        predictions["current_hour_csi_persistence"] = current
    return predictions


def evaluate_lead(context: Any, model: nn.Module, lead: int, stats: dict[str, Any]) -> list[dict[str, Any]]:
    """Evaluate model and baselines for one lead."""
    loader = short_loader(context, context.args.eval_split, lead, context.args.max_eval_samples, shuffle=False, include_satellite=True)
    rows: list[dict[str, Any]] = []
    sample_start = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs = batch["satellite"].to(context.device, non_blocking=True)
            with maybe_autocast(context):
                model_prediction = model(inputs).detach().cpu()
            rows.extend(scalar_rows(batch, model_prediction, context.args.model, sample_start, context.args.eval_split))
            for method, prediction in baseline_predictions(batch, stats).items():
                rows.extend(scalar_rows(batch, prediction, method, sample_start, context.args.eval_split))
            sample_start += model_prediction.shape[0]
    return rows


def save_short_plots(rows: list[dict[str, Any]], output_dir: Path, limit_per_lead: int = 4) -> None:
    """Save target-vs-prediction scatter-ish sequence plots per lead."""
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for lead, lead_group in frame.groupby("lead_hours", sort=True):
        for plot_index, (sample_index, group) in enumerate(lead_group.groupby("sample_index", sort=True), start=1):
            if plot_index > limit_per_lead:
                break
            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            target = group["target_csi"].iloc[0]
            ax.axhline(target, color="black", linewidth=2.0, label="target CSI")
            labels = []
            values = []
            for method, method_group in group.groupby("method", sort=False):
                labels.append(str(method))
                values.append(float(method_group["predicted_csi"].iloc[0]))
            ax.scatter(labels, values)
            ax.set_ylabel("CSI")
            ax.set_title(f"Lead {lead}h sample {sample_index}")
            ax.tick_params(axis="x", rotation=25)
            ax.grid(True, axis="y", alpha=0.25)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(output_dir / f"lead_{int(lead)}h_sample_{int(sample_index):04d}.png", dpi=180)
            plt.close(fig)


def per_lead_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute metrics by method and lead time."""
    frame = pd.DataFrame(rows)
    if frame.empty:
        return []
    metrics = []
    for (lead, method), group in frame.groupby(["lead_hours", "method"], sort=True):
        row = grouped_metrics(group.to_dict("records"), label_col="method")["overall"][0]
        row["lead_hours"] = int(lead)
        metrics.append(row)
    return metrics


def short_summary(metric_rows: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
    """Return lead-wise flags and interpretation."""
    frame = pd.DataFrame(metric_rows)
    best_model_per_lead: dict[str, str] = {}
    beats_climatology = {}
    beats_persistence = {}
    for lead, group in frame.groupby("lead_hours", sort=True):
        best = group.sort_values("CSI_RMSE").iloc[0]
        best_model_per_lead[str(int(lead))] = str(best["method"])
        model_rows = group[group["method"] == model_name]
        if model_rows.empty:
            continue
        model_rmse = float(model_rows["CSI_RMSE"].iloc[0])
        clim = group[group["method"].isin(["hourly_climatology", "location_hour_climatology"])]
        pers = group[group["method"] == "current_hour_csi_persistence"]
        beats_climatology[str(int(lead))] = bool(not clim.empty and model_rmse < float(clim["CSI_RMSE"].min()))
        beats_persistence[str(int(lead))] = None if pers.empty else bool(model_rmse < float(pers["CSI_RMSE"].iloc[0]))
    any_satellite_beats = any(value is True for value in beats_climatology.values())
    all_fail = all(value is False for value in beats_climatology.values()) if beats_climatology else True
    if any_satellite_beats:
        interpretation = "satellite_signal_detected_at_short_horizon"
    elif all_fail:
        interpretation = "short_horizon_satellite_model_does_not_beat_climatology_check_data_pipeline_or_image_signal"
    else:
        interpretation = "short_horizon_result_inconclusive"
    return {
        "best_model_per_lead": best_model_per_lead,
        "satellite_model_beats_climatology": beats_climatology,
        "satellite_model_beats_persistence": beats_persistence,
        "interpretation": interpretation,
    }


def main() -> None:
    """Run the short-horizon satellite predictability experiment."""
    args = parse_args("Short-horizon satellite predictability test.")
    context = build_context(args, default_subdir="short_horizon")
    if args.epochs is None:
        context.config.epochs = 10
    leads = parse_int_list(args.lead_hours)
    all_rows: list[dict[str, Any]] = []
    for lead in leads:
        train_loader = short_loader(
            context,
            args.train_split,
            lead,
            args.max_train_samples,
            shuffle=False,
            include_satellite=False,
        )
        stats = train_short_climatology(train_loader)
        train_windows = len(short_loader(context, args.train_split, lead, args.max_train_samples, shuffle=False, include_satellite=False).dataset)
        eval_windows = len(short_loader(context, args.eval_split, lead, args.max_eval_samples, shuffle=False, include_satellite=False).dataset)
        print(
            f"lead={lead}h windows: train={train_windows}, eval={eval_windows}, "
            f"epochs={int(context.config.epochs)}, batch_size={context.config.batch_size}, "
            f"cache_days={int(context.args.short_horizon_cache_days)}, "
            f"shuffle={bool(context.args.short_horizon_shuffle)}"
        )
        model = train_model(context, lead)
        all_rows.extend(evaluate_lead(context, model, lead, stats))

    predictions_path = context.output_dir / "short_horizon_predictions.csv"
    write_csv(predictions_path, all_rows)
    metric_paths = save_metric_tables(context.output_dir, "short_horizon", all_rows, label_col="method")
    lead_metrics = per_lead_metrics(all_rows)
    per_lead_path = context.output_dir / "per_lead_time_metrics.csv"
    write_csv(per_lead_path, lead_metrics)
    metrics_frame = pd.read_csv(metric_paths["overall"])
    plot_rmse_bar(metrics_frame.to_dict("records"), context.output_dir / "short_horizon_rmse.png", label_col="method")
    save_short_plots(all_rows, context.output_dir / "sample_plots")

    summary = {
        "dataset_root": str(context.config.dataset_root),
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "lead_hours_tested": leads,
        "history_hours": int(args.history_hours),
        "model": args.model,
        "epochs": int(context.config.epochs),
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        **short_summary(lead_metrics, args.model),
        "predictions_csv": str(predictions_path),
        "metrics": metric_paths,
        "per_lead_time_metrics_csv": str(per_lead_path),
    }
    write_json(context.output_dir / "short_horizon_summary.json", summary)
    mirror_outputs(context)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
