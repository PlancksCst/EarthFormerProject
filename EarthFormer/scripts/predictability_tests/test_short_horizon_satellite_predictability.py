"""Short-horizon satellite predictability tests for CSI forecasting."""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt 

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
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
from diagnostic_common import metrics_from_rows  # type: ignore
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


def requested_model_names(model_name: str) -> list[str]:
    """Return diagnostic model names to run."""
    if model_name == "all":
        return ["simple_cnn_lstm", "frozen_earthformer_pool_mlp"]
    return [model_name]


def make_model(context: Any, model_name: str) -> nn.Module:
    """Construct the requested diagnostic model."""
    if model_name == "simple_cnn_lstm":
        return SimpleCNNLSTM(input_channels=int(context.config.input_channels)).to(context.device)
    if model_name == "frozen_earthformer_pool_mlp":
        return FrozenEarthFormerPoolMLP(context).to(context.device)
    raise ValueError(f"Unknown short-horizon model: {model_name}")


def masked_scalar_mse(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """MSE over valid scalar targets."""
    if int(valid.sum().detach().cpu()) == 0:
        return prediction.new_zeros(())
    return ((prediction - target) ** 2)[valid].mean()


def train_model(context: Any, lead: int, model_name: str) -> nn.Module:
    """Train one diagnostic short-horizon model for one lead time."""
    model = make_model(context, model_name)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(context.config.learning_rate),
        weight_decay=float(context.config.weight_decay),
    )
    loader = short_loader(
        context,
        context.args.train_split,
        lead,
        context.args.max_train_samples,
        shuffle=True,
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
        print(
            f"model={model_name} lead={lead} epoch={epoch:03d}/{epochs:03d} "
            f"loss={total_loss / max(total_valid, 1):.6f}"
        )
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


def evaluate_lead(
    context: Any,
    model: nn.Module,
    model_name: str,
    lead: int,
    stats: dict[str, Any],
    split: str,
    max_day_samples: int | None,
    include_baselines: bool = True,
) -> list[dict[str, Any]]:
    """Evaluate one model and optional baselines for one lead/split."""
    loader = short_loader(context, split, lead, max_day_samples, shuffle=False, include_satellite=True)
    rows: list[dict[str, Any]] = []
    sample_start = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs = batch["satellite"].to(context.device, non_blocking=True)
            with maybe_autocast(context):
                model_prediction = model(inputs).detach().cpu()
            rows.extend(scalar_rows(batch, model_prediction, model_name, sample_start, split))
            if include_baselines:
                for method, prediction in baseline_predictions(batch, stats).items():
                    rows.extend(scalar_rows(batch, prediction, method, sample_start, split))
            sample_start += model_prediction.shape[0]
    return rows


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Return finite Pearson correlation or NaN."""
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or float(np.std(x)) < 1.0e-8 or float(np.std(y)) < 1.0e-8:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def persistence_sanity_rows(context: Any, lead: int, split: str, max_day_samples: int | None) -> list[dict[str, Any]]:
    """Check correlation between current CSI and target CSI for short horizons."""
    loader = short_loader(context, split, lead, max_day_samples, shuffle=False, include_satellite=False)
    current_values: list[float] = []
    target_values: list[float] = []
    for batch in loader:
        valid = batch["valid"].bool()
        current = batch["current_csi"].float()
        target = batch["target"].float()
        mask = valid & torch.isfinite(current) & torch.isfinite(target)
        if int(mask.sum()) == 0:
            continue
        current_values.extend(current[mask].tolist())
        target_values.extend(target[mask].tolist())
    current_np = np.asarray(current_values, dtype=np.float64)
    target_np = np.asarray(target_values, dtype=np.float64)
    if current_np.size == 0:
        return [{
            "split": split,
            "lead_hours": lead,
            "valid_count": 0,
            "current_target_pearson": float("nan"),
            "current_target_rmse": float("nan"),
            "current_std": float("nan"),
            "target_std": float("nan"),
        }]
    error = current_np - target_np
    return [{
        "split": split,
        "lead_hours": lead,
        "valid_count": int(current_np.size),
        "current_target_pearson": safe_corr(current_np, target_np),
        "current_target_rmse": float(np.sqrt(np.mean(error**2))),
        "current_std": float(np.std(current_np)),
        "target_std": float(np.std(target_np)),
    }]


def perturb_inputs(inputs: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return image perturbations for short-horizon image-dependence checks."""
    shuffled = torch.roll(inputs, shifts=1, dims=0) if inputs.shape[0] > 1 else torch.zeros_like(inputs)
    return {
        "zero": torch.zeros_like(inputs),
        "shuffled": shuffled,
        "time_reversed": torch.flip(inputs, dims=[1]),
    }


def perturbation_rows(
    context: Any,
    model: nn.Module,
    model_name: str,
    lead: int,
    split: str,
    max_day_samples: int | None,
) -> list[dict[str, Any]]:
    """Compare real predictions with zero/shuffled/time-reversed image inputs."""
    loader = short_loader(context, split, lead, max_day_samples, shuffle=False, include_satellite=True)
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            inputs = batch["satellite"].to(context.device, non_blocking=True)
            target = batch["target"].to(context.device, non_blocking=True)
            valid = batch["valid"].to(context.device, non_blocking=True).bool()
            if int(valid.sum().detach().cpu()) == 0:
                continue
            with maybe_autocast(context):
                pred_real = model(inputs)
            for perturbation, perturbed in perturb_inputs(inputs).items():
                with maybe_autocast(context):
                    pred_other = model(perturbed)
                real = pred_real.detach().float().cpu()
                other = pred_other.detach().float().cpu()
                tgt = target.detach().float().cpu()
                mask = valid.detach().cpu().bool()
                delta = (other - real)[mask]
                real_error = (real - tgt)[mask]
                other_error = (other - tgt)[mask]
                rows.append(
                    {
                        "split": split,
                        "lead_hours": lead,
                        "model": model_name,
                        "batch_index": batch_index,
                        "perturbation": perturbation,
                        "valid_count": int(mask.sum()),
                        "mean_abs_delta": float(delta.abs().mean()) if delta.numel() else float("nan"),
                        "rmse_delta": float(torch.sqrt(torch.mean(delta**2))) if delta.numel() else float("nan"),
                        "real_forecast_rmse": float(torch.sqrt(torch.mean(real_error**2))) if real_error.numel() else float("nan"),
                        "perturbed_forecast_rmse": float(torch.sqrt(torch.mean(other_error**2))) if other_error.numel() else float("nan"),
                        "real_prediction_std": float(real[mask].std(unbiased=False)) if int(mask.sum()) > 1 else float("nan"),
                        "perturbed_prediction_std": float(other[mask].std(unbiased=False)) if int(mask.sum()) > 1 else float("nan"),
                    }
                )
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


def split_lead_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute metrics by split, lead, and method."""
    frame = pd.DataFrame(rows)
    if frame.empty:
        return []
    metrics = []
    for (split, lead, method), group in frame.groupby(["split", "lead_hours", "method"], sort=True):
        row = metrics_from_rows(group.to_dict("records"), {"method": method})
        row["split"] = str(split)
        row["lead_hours"] = int(lead)
        metrics.append(row)
    return metrics


def save_per_lead_scatter_and_histograms(rows: list[dict[str, Any]], output_dir: Path) -> None:
    """Save target-vs-prediction scatter plots and prediction histograms per lead/method."""
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    valid = frame[frame["valid_hour"].astype(bool)].copy()
    if valid.empty:
        return
    scatter_dir = output_dir / "scatter"
    histogram_dir = output_dir / "histograms"
    scatter_dir.mkdir(parents=True, exist_ok=True)
    histogram_dir.mkdir(parents=True, exist_ok=True)
    for (lead, method), group in valid.groupby(["lead_hours", "method"], sort=True):
        target = group["target_csi"].to_numpy(dtype=np.float64)
        prediction = group["predicted_csi"].to_numpy(dtype=np.float64)
        finite = np.isfinite(target) & np.isfinite(prediction)
        if finite.sum() == 0:
            continue
        target = target[finite]
        prediction = prediction[finite]
        low = float(min(target.min(), prediction.min()))
        high = float(max(target.max(), prediction.max()))
        if low == high:
            low -= 0.05
            high += 0.05

        fig, ax = plt.subplots(figsize=(5.8, 5.4))
        ax.scatter(target, prediction, s=14, alpha=0.55)
        ax.plot([low, high], [low, high], color="black", linewidth=1.5)
        ax.set_xlabel("Target CSI")
        ax.set_ylabel("Predicted CSI")
        ax.set_title(f"Lead {lead}h - {method}")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(scatter_dir / f"lead_{int(lead)}h_{method}.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.6, 4.8))
        ax.hist(target, bins=30, alpha=0.55, label="target")
        ax.hist(prediction, bins=30, alpha=0.55, label="prediction")
        ax.set_xlabel("CSI")
        ax.set_ylabel("Count")
        ax.set_title(f"Lead {lead}h CSI Distribution - {method}")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(histogram_dir / f"lead_{int(lead)}h_{method}.png", dpi=180)
        plt.close(fig)


def prediction_std_checks(
    rows: list[dict[str, Any]],
    model_names: list[str],
    threshold: float,
) -> list[dict[str, Any]]:
    """Return collapse warnings where prediction std is far below target std."""
    frame = pd.DataFrame(rows)
    if frame.empty:
        return []
    valid = frame[frame["valid_hour"].astype(bool)].copy()
    issues: list[dict[str, Any]] = []
    for (split, lead, method), group in valid.groupby(["split", "lead_hours", "method"], sort=True):
        if method not in model_names or split != "val":
            continue
        prediction = group["predicted_csi"].to_numpy(dtype=np.float64)
        target = group["target_csi"].to_numpy(dtype=np.float64)
        finite = np.isfinite(prediction) & np.isfinite(target)
        if finite.sum() < 2:
            continue
        prediction_std = float(np.std(prediction[finite]))
        target_std = float(np.std(target[finite]))
        ratio = prediction_std / max(target_std, 1.0e-8)
        if ratio < float(threshold):
            issues.append(
                {
                    "split": split,
                    "lead_hours": int(lead),
                    "method": method,
                    "prediction_std": prediction_std,
                    "target_std": target_std,
                    "std_ratio": ratio,
                    "threshold": float(threshold),
                }
            )
    return issues


def short_summary(metric_rows: list[dict[str, Any]], model_names: list[str]) -> dict[str, Any]:
    """Return lead-wise flags and interpretation."""
    frame = pd.DataFrame(metric_rows)
    best_model_per_lead: dict[str, str] = {}
    beats_climatology = {}
    beats_persistence = {}
    for lead, group in frame.groupby("lead_hours", sort=True):
        best = group.sort_values("CSI_RMSE").iloc[0]
        best_model_per_lead[str(int(lead))] = str(best["method"])
        model_rows = group[group["method"].isin(model_names)].copy()
        if model_rows.empty:
            continue
        best_satellite = model_rows.sort_values("CSI_RMSE").iloc[0]
        model_rmse = float(best_satellite["CSI_RMSE"])
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
    model_names = requested_model_names(args.model)
    all_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    perturb_rows: list[dict[str, Any]] = []
    persistence_rows: list[dict[str, Any]] = []
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
            f"shuffle=True"
        )
        persistence_rows.extend(persistence_sanity_rows(context, lead, args.train_split, args.max_train_samples))
        persistence_rows.extend(persistence_sanity_rows(context, lead, args.eval_split, args.max_eval_samples))
        for model_index, model_name in enumerate(model_names):
            model = train_model(context, lead, model_name)
            train_rows = evaluate_lead(
                context,
                model,
                model_name,
                lead,
                stats,
                split=args.train_split,
                max_day_samples=args.max_train_samples,
                include_baselines=False,
            )
            validation_rows = evaluate_lead(
                context,
                model,
                model_name,
                lead,
                stats,
                split=args.eval_split,
                max_day_samples=args.max_eval_samples,
                include_baselines=model_index == 0,
            )
            all_rows.extend(train_rows)
            all_rows.extend(validation_rows)
            eval_rows.extend(validation_rows)
            perturb_rows.extend(
                perturbation_rows(
                    context,
                    model,
                    model_name,
                    lead,
                    split=args.eval_split,
                    max_day_samples=args.max_eval_samples,
                )
            )

    predictions_path = context.output_dir / "short_horizon_predictions.csv"
    write_csv(predictions_path, all_rows)
    metric_paths = save_metric_tables(context.output_dir, "short_horizon", eval_rows, label_col="method")
    lead_metrics = per_lead_metrics(eval_rows)
    per_lead_path = context.output_dir / "per_lead_time_metrics.csv"
    write_csv(per_lead_path, lead_metrics)
    train_val_metrics = split_lead_metrics(all_rows)
    train_val_metrics_path = context.output_dir / "train_val_metrics.csv"
    write_csv(train_val_metrics_path, train_val_metrics)
    perturbation_path = context.output_dir / "image_perturbation_metrics.csv"
    write_csv(perturbation_path, perturb_rows)
    persistence_path = context.output_dir / "persistence_sanity.csv"
    write_csv(persistence_path, persistence_rows)
    metrics_frame = pd.read_csv(metric_paths["overall"])
    plot_rmse_bar(metrics_frame.to_dict("records"), context.output_dir / "short_horizon_rmse.png", label_col="method")
    save_short_plots(eval_rows, context.output_dir / "sample_plots")
    save_per_lead_scatter_and_histograms(eval_rows, context.output_dir / "per_lead_plots")
    collapse_issues = prediction_std_checks(
        eval_rows,
        model_names=model_names,
        threshold=float(args.prediction_std_ratio_threshold),
    )

    summary = {
        "dataset_root": str(context.config.dataset_root),
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "lead_hours_tested": leads,
        "history_hours": int(args.history_hours),
        "requested_model": args.model,
        "models_run": model_names,
        "epochs": int(context.config.epochs),
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "train_shuffle": True,
        "prediction_std_ratio_threshold": float(args.prediction_std_ratio_threshold),
        "prediction_collapse_issues": collapse_issues,
        **short_summary(lead_metrics, model_names),
        "predictions_csv": str(predictions_path),
        "metrics": metric_paths,
        "per_lead_time_metrics_csv": str(per_lead_path),
        "train_val_metrics_csv": str(train_val_metrics_path),
        "image_perturbation_metrics_csv": str(perturbation_path),
        "persistence_sanity_csv": str(persistence_path),
    }
    write_json(context.output_dir / "short_horizon_summary.json", summary)
    mirror_outputs(context)
    print(json.dumps(summary, indent=2))
    if collapse_issues:
        raise RuntimeError(
            "Satellite model prediction collapse detected. "
            f"Prediction std is below {float(args.prediction_std_ratio_threshold):.3f} "
            f"of target std for: {collapse_issues}"
        )


if __name__ == "__main__":
    main()
