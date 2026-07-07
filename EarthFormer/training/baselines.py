"""Training-time statistical baselines for residual CSI forecasting."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch

from training.losses import valid_hour_mask


def _as_float_sequence(value: Any, name: str) -> torch.Tensor:
    """Return a one-dimensional float tensor from a dataset field."""
    if value is None:
        raise KeyError(f"Missing required sequence field: {name}")
    tensor = value.detach().clone() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.ndim != 1:
        raise ValueError(f"Expected {name} with shape (T,), got {tuple(tensor.shape)}")
    return tensor.float()


def _batch_value(batch: dict[str, Any], key: str, index: int) -> Any:
    """Return a per-sample metadata value from a collated batch."""
    value = batch.get(key)
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        item = value[index]
        return item.item() if item.numel() == 1 else item.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return value[index] if index < len(value) else None
    return value


@dataclass
class ClimatologyBaseline:
    """Hourly and location-hour CSI climatology for residual prediction."""

    mode: str
    hour_mean: torch.Tensor
    global_mean: float
    location_hour_mean: dict[str, torch.Tensor]
    location_hour_count: dict[str, torch.Tensor]

    @classmethod
    def from_dataset(
        cls,
        dataset: Any,
        output_length: int,
        mode: str = "location_hour_climatology",
        clear_sky_threshold: float = 20.0,
    ) -> "ClimatologyBaseline":
        """Estimate climatology from the training dataset split."""
        if mode not in {"global_mean", "hourly_climatology", "location_hour_climatology"}:
            raise ValueError(
                "residual_baseline must be one of: "
                "global_mean, hourly_climatology, location_hour_climatology"
            )

        hour_sum = torch.zeros(output_length, dtype=torch.float64)
        hour_count = torch.zeros(output_length, dtype=torch.float64)
        location_sum: dict[str, torch.Tensor] = defaultdict(
            lambda: torch.zeros(output_length, dtype=torch.float64)
        )
        location_count: dict[str, torch.Tensor] = defaultdict(
            lambda: torch.zeros(output_length, dtype=torch.float64)
        )

        for index in range(len(dataset)):
            item = dataset[index]
            target_value = item.get("target", item.get("target_csi"))
            target = _as_float_sequence(target_value, "target")[:output_length]
            if target.numel() != output_length:
                raise ValueError(
                    f"Expected target length {output_length}, got {target.numel()} at index {index}"
                )
            clear_value = item.get("clear_sky_ghi", item.get("clear_ghi"))
            clear_sky = (
                _as_float_sequence(clear_value, "clear_sky_ghi")[:output_length]
                if clear_value is not None
                else None
            )
            target_mask_value = item.get("target_mask")
            target_mask = (
                _as_float_sequence(target_mask_value, "target_mask")[:output_length]
                if target_mask_value is not None
                else None
            )
            valid = valid_hour_mask(
                target_mask=target_mask,
                reference=target,
                clear_sky_ghi=clear_sky,
                clear_sky_threshold=clear_sky_threshold,
            ).cpu()
            finite = torch.isfinite(target)
            valid = valid & finite
            if not bool(valid.any()):
                continue

            hour_sum[valid] += target.double()[valid]
            hour_count[valid] += 1.0

            location = str(item.get("location", "unknown"))
            location_sum[location][valid] += target.double()[valid]
            location_count[location][valid] += 1.0

        total_count = hour_count.sum()
        if float(total_count) <= 0.0:
            raise RuntimeError("Cannot build residual baseline: no valid training CSI targets found.")

        global_mean = float(hour_sum.sum() / total_count)
        hour_mean = torch.full((output_length,), global_mean, dtype=torch.float32)
        observed_hours = hour_count > 0
        hour_mean[observed_hours] = (hour_sum[observed_hours] / hour_count[observed_hours]).float()

        location_hour_mean: dict[str, torch.Tensor] = {}
        location_hour_count: dict[str, torch.Tensor] = {}
        for location, sums in location_sum.items():
            counts = location_count[location]
            means = hour_mean.clone()
            observed = counts > 0
            means[observed] = (sums[observed] / counts[observed]).float()
            location_hour_mean[location] = means
            location_hour_count[location] = counts.float()

        return cls(
            mode=mode,
            hour_mean=hour_mean,
            global_mean=global_mean,
            location_hour_mean=location_hour_mean,
            location_hour_count=location_hour_count,
        )

    def predict(
        self,
        batch: dict[str, Any],
        horizon: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return a `(B,T)` climatology forecast for a collated batch."""
        satellite = batch.get("satellite")
        if not isinstance(satellite, torch.Tensor):
            raise KeyError("Batch must contain a satellite tensor to infer batch size")
        batch_size = int(satellite.shape[0])

        if horizon > self.hour_mean.numel():
            raise ValueError(
                f"Requested horizon {horizon}, but baseline has {self.hour_mean.numel()} hours"
            )

        if self.mode == "global_mean":
            base = torch.full((horizon,), self.global_mean, dtype=torch.float32)
        else:
            base = self.hour_mean[:horizon].float()

        prediction = base.unsqueeze(0).repeat(batch_size, 1)
        if self.mode == "location_hour_climatology":
            for sample_index in range(batch_size):
                location_value = _batch_value(batch, "location", sample_index)
                location = str(location_value) if location_value is not None else "unknown"
                location_mean = self.location_hour_mean.get(location)
                if location_mean is not None:
                    prediction[sample_index] = location_mean[:horizon].float()

        return prediction.to(device=device, dtype=dtype)
