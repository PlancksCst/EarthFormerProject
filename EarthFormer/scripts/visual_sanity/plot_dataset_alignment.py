"""Visual sanity plots for SEVIRI image sequences and CSI/GHI targets."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
EARTHFORMER_DIR = SCRIPT_DIR.parents[1]
PROJECT_ROOT = EARTHFORMER_DIR.parent
for candidate in (PROJECT_ROOT, EARTHFORMER_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from configs.config import build_arg_parser, config_from_args  # noqa: E402
from datasets.seviri_dataset import build_dataset  # noqa: E402
from utils.artifacts import ArtifactMirror  # noqa: E402

FRAME_DIFF_THRESHOLD = 1.0e-3
CHANNEL_STD_THRESHOLD = 1.0e-6
GHI_EPS = 1.0


def add_visual_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add visual-sanity CLI arguments."""
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--channel-index", type=int, default=0)
    parser.add_argument("--mode", choices=("next_day", "short_horizon"), default="next_day")
    parser.add_argument("--plot-all-channels", action="store_true")
    parser.add_argument("--history-hours", type=int, default=6)
    parser.add_argument("--lead-hours", default="1,2,3")
    parser.add_argument("--solar-elevation-threshold", type=float, default=5.0)
    return parser


def parse_args() -> argparse.Namespace:
    """Parse project and visual-sanity arguments."""
    parser = build_arg_parser()
    parser.description = "Plot visual/data sanity checks for SEVIRI CSI samples."
    add_visual_args(parser)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write CSV rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **payload}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=json_default)


def json_default(value: Any) -> Any:
    """Serialize common non-JSON values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return str(value)


def parse_int_list(text: str) -> list[int]:
    """Parse comma-separated integers."""
    values = [int(item.strip()) for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one lead hour")
    return values


def tensor_to_numpy(value: Any) -> np.ndarray | None:
    """Convert tensor/list/array values to numpy."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)):
        return np.asarray(value)
    return None


def scalar(value: Any) -> Any:
    """Return scalar-friendly metadata."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def metadata_row(dataset: Any, index: int) -> Any | None:
    """Return the metadata row for a dataset index when available."""
    meta = getattr(dataset, "meta", None)
    if meta is None:
        return None
    if index < 0 or index >= len(meta):
        return None
    return meta.iloc[index]


def parse_sequence(value: Any, length: int) -> list[int]:
    """Parse a metadata sequence field into fixed length integers."""
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if "," in text:
            values = np.fromstring(text, sep=",", dtype=np.int64)
        elif all(ch in "01" for ch in text) and len(text) == length:
            values = np.asarray([int(ch) for ch in text], dtype=np.int64)
        else:
            values = np.asarray([int(part) for part in text.split() if part], dtype=np.int64)
    elif isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        values = np.asarray(value, dtype=np.int64).reshape(-1)
    else:
        return []
    return values[:length].astype(int).tolist()


def row_value(row: Any | None, key: str) -> Any:
    """Return one metadata-row value."""
    if row is None:
        return None
    try:
        if key in row.index:
            return row[key]
    except Exception:
        return None
    return None


def load_hourly_frame(path: Path) -> pd.DataFrame | None:
    """Load hourly CAMS/ground CSV if available."""
    if not Path(path).exists():
        return None
    frame = pd.read_csv(path)
    if "timestamp" not in frame.columns:
        return None
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.set_index("timestamp").sort_index()
    if not frame.index.is_unique:
        frame = frame[~frame.index.duplicated(keep="first")]
    return frame


def load_elevation_frame(path: Path) -> pd.DataFrame | None:
    """Load solar elevation CSV if available."""
    return load_hourly_frame(path)


def location_columns(frame: pd.DataFrame, location: str) -> dict[str, str] | None:
    """Return hourly CSV columns for one location."""
    columns = {
        "csi": f"CSI_{location}",
        "ghi": f"GHI_{location}",
        "clear": f"GHI_clear_{location}",
    }
    return columns if all(column in frame.columns for column in columns.values()) else None


def solar_column(frame: pd.DataFrame, location: str) -> str | None:
    """Return best solar-elevation column for a location."""
    candidates = (
        f"solar_elevation_{location}",
        f"solar_elev_{location}",
        f"elevation_{location}",
        f"{location}_solar_elevation",
        f"{location}_solar_elev",
        f"{location}_elevation",
        "solar_elevation",
        "solar_elev",
        "elevation",
    )
    lowered = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        match = lowered.get(candidate.lower())
        if match is not None:
            return match
    return None


def hourly_value(frame: pd.DataFrame, column: str, timestamp: pd.Timestamp) -> float | None:
    """Return one finite hourly value."""
    try:
        value = frame.loc[timestamp, column]
    except KeyError:
        return None
    if isinstance(value, pd.Series):
        value = value.iloc[0]
    if pd.isna(value):
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def target_tensor(item: dict[str, Any]) -> np.ndarray:
    """Return target CSI as numpy or raise."""
    value = tensor_to_numpy(item.get("target", item.get("target_csi")))
    if value is None:
        raise KeyError("Dataset item does not contain target or target_csi")
    return np.asarray(value, dtype=np.float32).reshape(-1)


def clear_tensor(item: dict[str, Any], length: int) -> np.ndarray:
    """Return clear-sky GHI as numpy."""
    value = tensor_to_numpy(item.get("clear_sky_ghi", item.get("clear_ghi")))
    if value is None:
        return np.full(length, np.nan, dtype=np.float32)
    return np.asarray(value, dtype=np.float32).reshape(-1)[:length]


def target_ghi_tensor(item: dict[str, Any], target: np.ndarray, clear: np.ndarray) -> np.ndarray:
    """Return target GHI or reconstruct it."""
    value = tensor_to_numpy(item.get("target_ghi"))
    if value is None:
        return target * clear
    return np.asarray(value, dtype=np.float32).reshape(-1)[: target.shape[0]]


def mask_array(item: dict[str, Any], key: str, length: int) -> np.ndarray:
    """Return boolean mask where True means invalid."""
    value = tensor_to_numpy(item.get(key))
    if value is None:
        return np.zeros(length, dtype=bool)
    values = np.asarray(value).reshape(-1)[:length]
    if values.size < length:
        values = np.pad(values, (0, length - values.size), constant_values=True)
    return values.astype(bool)


def solar_sequence(
    elevation: pd.DataFrame | None,
    location: str,
    day: Any,
    length: int,
) -> tuple[np.ndarray, bool]:
    """Return hourly solar elevation for day hours 04.. if available."""
    values = np.full(length, np.nan, dtype=np.float32)
    if elevation is None:
        return values, False
    column = solar_column(elevation, location)
    if column is None:
        return values, False
    try:
        base_day = pd.Timestamp(day)
    except Exception:
        return values, False
    found = False
    for pos in range(length):
        timestamp = base_day + pd.Timedelta(hours=4 + pos)
        value = hourly_value(elevation, column, timestamp)
        if value is not None:
            values[pos] = value
            found = True
    return values, found


def valid_mask(
    target_mask: np.ndarray,
    clear: np.ndarray,
    solar: np.ndarray,
    solar_available: bool,
    clear_threshold: float,
    solar_threshold: float,
) -> np.ndarray:
    """Return physically valid target hours."""
    valid = ~target_mask & np.isfinite(clear) & (clear > float(clear_threshold))
    if solar_available:
        valid &= np.isfinite(solar) & (solar >= float(solar_threshold))
    return valid


def stats(values: np.ndarray) -> dict[str, float]:
    """Return finite min/max/mean/std."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"min": math.nan, "max": math.nan, "mean": math.nan, "std": math.nan}
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
    }


def image_scale(channel: np.ndarray) -> tuple[float, float]:
    """Return robust image scale for one sample/channel."""
    finite = channel[np.isfinite(channel)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(finite, [2, 98])
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        low, high = float(np.min(finite)), float(np.max(finite))
    if low == high:
        low -= 0.5
        high += 0.5
    return float(low), float(high)


def frame_mean_abs_diff(channel: np.ndarray) -> float:
    """Return mean absolute difference between consecutive frames."""
    if channel.shape[0] < 2:
        return math.nan
    return float(np.mean(np.abs(np.diff(channel, axis=0))))


def warning_labels(
    channel: np.ndarray,
    target: np.ndarray,
    clear: np.ndarray,
    target_ghi: np.ndarray,
    target_mask: np.ndarray,
    row: Any | None,
    actual_image_length: int,
    actual_target_length: int,
    clear_threshold: float,
) -> list[str]:
    """Return visual-sanity warning labels."""
    labels: list[str] = []
    if frame_mean_abs_diff(channel) < FRAME_DIFF_THRESHOLD:
        labels.append("frames_nearly_identical")
    if stats(channel)["std"] < CHANNEL_STD_THRESHOLD:
        labels.append("selected_channel_near_constant")
    if np.any(np.isfinite(target) & ((target < 0.0) | (target > 1.3))):
        labels.append("csi_outside_0_1p3")
    if np.any(np.isfinite(target_ghi) & np.isfinite(clear) & (clear <= float(clear_threshold)) & (np.abs(target_ghi) > GHI_EPS)):
        labels.append("ghi_nonzero_when_clear_sky_near_zero")
    if np.any(target_mask & np.isfinite(clear) & (clear > float(clear_threshold))):
        labels.append("clear_sky_positive_but_target_masked_invalid")
    input_length = row_value(row, "input_length")
    target_length = row_value(row, "target_length")
    if input_length is not None and int(input_length) != actual_image_length:
        labels.append("metadata_input_length_differs_from_tensor")
    if target_length is not None and int(target_length) != actual_target_length:
        labels.append("metadata_target_length_differs_from_tensor")
    return sorted(set(labels))


def safe_name(value: Any) -> str:
    """Return a filesystem-safe short string."""
    text = str(value) if value is not None else "unknown"
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text[:80] or "unknown"


def panel_text(lines: list[str]) -> str:
    """Format diagnostic text."""
    return "\n".join(lines)


def plot_image_sequence(
    axes: list[Any],
    frames: np.ndarray,
    image_mask: np.ndarray,
    hours: list[str],
    vmin: float,
    vmax: float,
    title_prefix: str = "",
) -> Any:
    """Plot a sequence of image frames."""
    image_handle = None
    for pos, ax in enumerate(axes):
        ax.axis("off")
        if pos >= frames.shape[0]:
            continue
        image_handle = ax.imshow(frames[pos], cmap="gray", vmin=vmin, vmax=vmax)
        invalid = bool(image_mask[pos]) if pos < len(image_mask) else False
        title = f"{title_prefix}{pos}"
        if pos < len(hours):
            title = f"{title_prefix}{hours[pos]}"
        if invalid:
            title += " [invalid]"
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("red")
                spine.set_linewidth(2.0)
        ax.set_title(title, fontsize=8)
    return image_handle


def draw_target_curves(
    ax: Any,
    target: np.ndarray,
    clear: np.ndarray,
    target_ghi: np.ndarray,
    valid: np.ndarray,
    title: str,
) -> None:
    """Draw CSI and GHI/clear-sky target curves."""
    x = np.arange(target.shape[0])
    ax.plot(x, target, marker="o", color="tab:blue", label="target CSI")
    ax.set_ylabel("CSI", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax.set_xlabel("Forecast hour index")
    ax.set_title(title)
    for idx, is_valid in enumerate(valid):
        if not is_valid:
            ax.axvspan(idx - 0.45, idx + 0.45, color="lightgray", alpha=0.35)
    ax2 = ax.twinx()
    ax2.plot(x, target_ghi, marker="s", color="tab:green", label="target GHI")
    ax2.plot(x, clear, marker="^", color="tab:orange", label="clear-sky GHI")
    ax2.set_ylabel("GHI W/m2")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)


def next_day_figure(
    item: dict[str, Any],
    row: Any | None,
    index: int,
    args: argparse.Namespace,
    output_dir: Path,
    elevation: pd.DataFrame | None,
) -> dict[str, Any]:
    """Create one next-day alignment figure and return summary row."""
    satellite = tensor_to_numpy(item.get("satellite"))
    if satellite is None:
        raise KeyError("Dataset item does not contain satellite tensor")
    satellite = np.asarray(satellite, dtype=np.float32)
    if satellite.ndim != 4:
        raise ValueError(f"Expected satellite shape (T,C,H,W), got {satellite.shape}")
    if not 0 <= args.channel_index < satellite.shape[1]:
        raise ValueError(f"channel-index={args.channel_index} outside [0,{satellite.shape[1]})")
    target = target_tensor(item)
    clear = clear_tensor(item, target.shape[0])
    target_ghi = target_ghi_tensor(item, target, clear)
    target_mask = mask_array(item, "target_mask", target.shape[0])
    image_mask = mask_array(item, "image_mask", satellite.shape[0])
    location = item.get("location", row_value(row, "location"))
    input_day = item.get("input_day", row_value(row, "input_day"))
    target_day = item.get("target_day", row_value(row, "target_day"))
    solar, solar_available = solar_sequence(elevation, str(location), target_day, target.shape[0])
    valid = valid_mask(
        target_mask,
        clear,
        solar,
        solar_available,
        args.clear_sky_threshold,
        args.solar_elevation_threshold,
    )
    channel = satellite[:, args.channel_index]
    vmin, vmax = image_scale(channel)
    channel_stats = stats(channel)
    target_stats = stats(target)
    frame_diff = frame_mean_abs_diff(channel)
    warnings = warning_labels(
        channel,
        target,
        clear,
        target_ghi,
        target_mask,
        row,
        satellite.shape[0],
        target.shape[0],
        args.clear_sky_threshold,
    )

    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    grid = fig.add_gridspec(3, 7, height_ratios=[1.0, 1.0, 1.05])
    image_axes = [fig.add_subplot(grid[0, col]) for col in range(7)]
    image_axes.extend(fig.add_subplot(grid[1, col]) for col in range(6))
    hours = [f"{4 + pos:02d}h" for pos in range(satellite.shape[0])]
    image_handle = plot_image_sequence(image_axes, channel, image_mask, hours, vmin, vmax)
    empty_ax = fig.add_subplot(grid[1, 6])
    empty_ax.axis("off")
    if image_handle is not None:
        fig.colorbar(image_handle, ax=image_axes, shrink=0.75, label=f"channel {args.channel_index}")

    curve_ax = fig.add_subplot(grid[2, :5])
    draw_target_curves(curve_ax, target, clear, target_ghi, valid, "Next-day target curves")

    text_ax = fig.add_subplot(grid[2, 5:])
    text_ax.axis("off")
    sample_id = item.get("sample_id", row_value(row, "sample_id"))
    text_lines = [
        f"sample_index: {index}",
        f"sample_id: {scalar(sample_id)}",
        f"location: {location}",
        f"input_day: {input_day}",
        f"target_day: {target_day}",
        f"valid hours: {int(valid.sum())}/{target.shape[0]}",
        f"image shape: {tuple(satellite.shape)}",
        f"target shape: {tuple(target.shape)}",
        f"ch{args.channel_index} min/max: {channel_stats['min']:.3f}/{channel_stats['max']:.3f}",
        f"ch{args.channel_index} mean/std: {channel_stats['mean']:.3f}/{channel_stats['std']:.3f}",
        f"frame diff mean abs: {frame_diff:.6f}",
        f"CSI min/max: {target_stats['min']:.3f}/{target_stats['max']:.3f}",
        f"CSI mean/std: {target_stats['mean']:.3f}/{target_stats['std']:.3f}",
        f"solar elevation: {'available' if solar_available else 'unavailable'}",
        "warnings:",
        *([f"- {label}" for label in warnings] if warnings else ["- none"]),
    ]
    text_ax.text(0.0, 1.0, panel_text(text_lines), va="top", ha="left", fontsize=9, family="monospace")
    fig.suptitle(
        f"Next-day alignment | idx={index} sample={scalar(sample_id)} "
        f"loc={location} input={input_day} target={target_day}",
        fontsize=14,
    )
    filename = output_dir / f"sample_{index:04d}_{safe_name(location)}_alignment.png"
    fig.savefig(filename, dpi=170)
    plt.close(fig)

    if args.plot_all_channels:
        plot_all_channels(satellite, image_mask, index, location, output_dir)

    return {
        "sample_index": index,
        "sample_id": scalar(sample_id),
        "location": location,
        "input_date": input_day,
        "target_date": target_day,
        "image_shape": str(tuple(satellite.shape)),
        "target_shape": str(tuple(target.shape)),
        "channel_index": int(args.channel_index),
        "image_mean": channel_stats["mean"],
        "image_std": channel_stats["std"],
        "frame_to_frame_mean_abs_diff": frame_diff,
        "target_csi_mean": target_stats["mean"],
        "target_csi_std": target_stats["std"],
        "valid_count": int(valid.sum()),
        "solar_elevation_available": bool(solar_available),
        "warnings": ";".join(warnings),
        "figure": str(filename),
    }


def plot_all_channels(
    satellite: np.ndarray,
    image_mask: np.ndarray,
    index: int,
    location: Any,
    output_dir: Path,
) -> None:
    """Plot all channels for first, middle, and last timesteps."""
    timesteps = sorted(set([0, satellite.shape[0] // 2, satellite.shape[0] - 1]))
    channels = satellite.shape[1]
    fig, axes = plt.subplots(channels, len(timesteps), figsize=(4.2 * len(timesteps), 2.5 * channels), squeeze=False)
    for channel_index in range(channels):
        channel = satellite[:, channel_index]
        vmin, vmax = image_scale(channel)
        for col, timestep in enumerate(timesteps):
            ax = axes[channel_index, col]
            ax.imshow(channel[timestep], cmap="gray", vmin=vmin, vmax=vmax)
            invalid = bool(image_mask[timestep]) if timestep < image_mask.shape[0] else False
            ax.set_title(f"ch{channel_index} t{timestep}{' invalid' if invalid else ''}", fontsize=9)
            ax.axis("off")
    fig.suptitle(f"All-channel check | sample={index} loc={location}")
    fig.tight_layout()
    fig.savefig(output_dir / f"sample_{index:04d}_{safe_name(location)}_all_channels.png", dpi=160)
    plt.close(fig)


def short_horizon_candidates(dataset: Any, args: argparse.Namespace) -> list[tuple[int, int, int, int, int]]:
    """Return `(sample_index, lead, start, end, target_index)` windows."""
    leads = parse_int_list(args.lead_hours)
    candidates: list[tuple[int, int, int, int, int]] = []
    for sample_index in range(len(dataset)):
        for lead in leads:
            for target_index in range(int(args.history_hours) + lead - 1, 13):
                end = target_index - lead
                start = end - int(args.history_hours) + 1
                if start >= 0 and end >= start:
                    candidates.append((sample_index, lead, start, end, target_index))
                    break
            if len(candidates) >= args.num_samples:
                return candidates
    return candidates


def same_day_series(
    hourly: pd.DataFrame | None,
    location: str,
    day: Any,
    length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Return same-day CSI/GHI/clear arrays from hourly CSV."""
    csi = np.full(length, np.nan, dtype=np.float32)
    ghi = np.full(length, np.nan, dtype=np.float32)
    clear = np.full(length, np.nan, dtype=np.float32)
    if hourly is None:
        return csi, ghi, clear, False
    columns = location_columns(hourly, location)
    if columns is None:
        return csi, ghi, clear, False
    try:
        base_day = pd.Timestamp(day)
    except Exception:
        return csi, ghi, clear, False
    found = False
    for pos in range(length):
        timestamp = base_day + pd.Timedelta(hours=4 + pos)
        for key, array in (("csi", csi), ("ghi", ghi), ("clear", clear)):
            value = hourly_value(hourly, columns[key], timestamp)
            if value is not None:
                array[pos] = value
                found = True
    return csi, ghi, clear, found


def short_horizon_figure(
    dataset: Any,
    sample_index: int,
    lead: int,
    start: int,
    end: int,
    target_index: int,
    args: argparse.Namespace,
    output_dir: Path,
    hourly: pd.DataFrame | None,
    elevation: pd.DataFrame | None,
) -> dict[str, Any]:
    """Create one short-horizon alignment figure."""
    item = dataset[sample_index]
    satellite = tensor_to_numpy(item.get("satellite"))
    if satellite is None:
        raise KeyError("Dataset item does not contain satellite tensor")
    satellite = np.asarray(satellite, dtype=np.float32)
    location = str(item.get("location"))
    input_day = item.get("input_day")
    sample_id = item.get("sample_id")
    same_csi, same_ghi, same_clear, same_day_available = same_day_series(hourly, location, input_day, satellite.shape[0])
    solar, solar_available = solar_sequence(elevation, location, input_day, satellite.shape[0])
    target_csi = same_csi[target_index] if target_index < same_csi.size else np.nan
    target_ghi = same_ghi[target_index] if target_index < same_ghi.size else np.nan
    clear = same_clear[target_index] if target_index < same_clear.size else np.nan
    current_csi = same_csi[end] if end < same_csi.size else np.nan
    if not np.isfinite(target_ghi) and np.isfinite(target_csi) and np.isfinite(clear):
        target_ghi = target_csi * clear
    valid = bool(np.isfinite(target_csi) and np.isfinite(clear) and clear > args.clear_sky_threshold)
    if solar_available and np.isfinite(solar[target_index]):
        valid = valid and bool(solar[target_index] >= args.solar_elevation_threshold)
    history = satellite[start : end + 1, args.channel_index]
    vmin, vmax = image_scale(history)
    image_mask = mask_array(item, "image_mask", satellite.shape[0])
    history_mask = image_mask[start : end + 1]
    history_hours = [f"{4 + pos:02d}h" for pos in range(start, end + 1)]
    image_stats = stats(history)
    persistence_error = abs(float(current_csi - target_csi)) if np.isfinite(current_csi) and np.isfinite(target_csi) else math.nan
    warnings: list[str] = []
    if frame_mean_abs_diff(history) < FRAME_DIFF_THRESHOLD:
        warnings.append("history_frames_nearly_identical")
    if image_stats["std"] < CHANNEL_STD_THRESHOLD:
        warnings.append("selected_channel_near_constant")
    if not same_day_available:
        warnings.append("same_day_csi_unavailable")
    if np.isfinite(target_csi) and (target_csi < 0 or target_csi > 1.3):
        warnings.append("target_csi_outside_0_1p3")

    fig = plt.figure(figsize=(18, 9), constrained_layout=True)
    cols = max(int(args.history_hours), 3)
    grid = fig.add_gridspec(3, cols, height_ratios=[1.0, 1.0, 1.0])
    image_axes = [fig.add_subplot(grid[0, col]) for col in range(cols)]
    image_handle = plot_image_sequence(image_axes, history, history_mask, history_hours, vmin, vmax)
    if image_handle is not None:
        fig.colorbar(image_handle, ax=image_axes, shrink=0.8, label=f"channel {args.channel_index}")

    curve_ax = fig.add_subplot(grid[1:, : max(2, cols - 2)])
    x = np.arange(satellite.shape[0])
    curve_ax.plot(x, same_csi, marker="o", color="tab:blue", label="same-day CSI")
    curve_ax.axvspan(start - 0.45, end + 0.45, color="tab:blue", alpha=0.08, label="history window")
    curve_ax.axvline(end, color="tab:purple", linestyle="--", label="current t")
    curve_ax.axvline(target_index, color="tab:red", linestyle="--", label=f"target t+{lead}")
    curve_ax.scatter([target_index], [target_csi], color="tab:red", zorder=5)
    curve_ax.set_xlabel("Same-day daylight index")
    curve_ax.set_ylabel("CSI")
    curve_ax.grid(True, alpha=0.25)
    curve_ax.legend(loc="best", fontsize=8)
    curve_ax2 = curve_ax.twinx()
    curve_ax2.plot(x, same_ghi, color="tab:green", alpha=0.5, label="GHI")
    curve_ax2.plot(x, same_clear, color="tab:orange", alpha=0.55, label="clear-sky GHI")
    curve_ax2.set_ylabel("GHI W/m2")

    text_ax = fig.add_subplot(grid[1:, max(2, cols - 2) :])
    text_ax.axis("off")
    text_lines = [
        f"sample_index: {sample_index}",
        f"sample_id: {scalar(sample_id)}",
        f"location: {location}",
        f"date: {input_day}",
        f"history_hours: {args.history_hours}",
        f"lead_hour: {lead}",
        f"history idx: {start}..{end}",
        f"target idx/hour: {target_index}/{4 + target_index:02d}h",
        f"valid target: {valid}",
        f"image min/max: {image_stats['min']:.3f}/{image_stats['max']:.3f}",
        f"image mean/std: {image_stats['mean']:.3f}/{image_stats['std']:.3f}",
        f"current CSI: {current_csi:.3f}" if np.isfinite(current_csi) else "current CSI: unavailable",
        f"target CSI: {target_csi:.3f}" if np.isfinite(target_csi) else "target CSI: unavailable",
        f"target GHI: {target_ghi:.2f}" if np.isfinite(target_ghi) else "target GHI: unavailable",
        f"clear-sky GHI: {clear:.2f}" if np.isfinite(clear) else "clear-sky GHI: unavailable",
        f"persistence abs err: {persistence_error:.3f}" if np.isfinite(persistence_error) else "persistence abs err: unavailable",
        f"solar elevation: {solar[target_index]:.2f}" if solar_available and np.isfinite(solar[target_index]) else "solar elevation: unavailable",
        "warnings:",
        *([f"- {label}" for label in warnings] if warnings else ["- none"]),
    ]
    text_ax.text(0.0, 1.0, panel_text(text_lines), va="top", ha="left", fontsize=9, family="monospace")
    fig.suptitle(
        f"Short-horizon alignment | sample={sample_index} loc={location} "
        f"lead={lead}h target={4 + target_index:02d}h",
        fontsize=14,
    )
    filename = output_dir / f"sample_{sample_index:04d}_{safe_name(location)}_lead_{lead}h.png"
    fig.savefig(filename, dpi=170)
    plt.close(fig)

    frame_diff = frame_mean_abs_diff(history)
    return {
        "sample_index": sample_index,
        "sample_id": scalar(sample_id),
        "location": location,
        "input_date": input_day,
        "target_date": input_day,
        "image_shape": str(tuple(history.shape)),
        "target_shape": "()",
        "channel_index": int(args.channel_index),
        "lead_hours": int(lead),
        "history_hours": int(args.history_hours),
        "current_index": int(end),
        "target_index": int(target_index),
        "image_mean": image_stats["mean"],
        "image_std": image_stats["std"],
        "frame_to_frame_mean_abs_diff": frame_diff,
        "target_csi_mean": float(target_csi) if np.isfinite(target_csi) else math.nan,
        "target_csi_std": 0.0 if np.isfinite(target_csi) else math.nan,
        "valid_count": int(valid),
        "current_csi": float(current_csi) if np.isfinite(current_csi) else math.nan,
        "target_csi": float(target_csi) if np.isfinite(target_csi) else math.nan,
        "target_ghi": float(target_ghi) if np.isfinite(target_ghi) else math.nan,
        "clear_sky_ghi": float(clear) if np.isfinite(clear) else math.nan,
        "persistence_abs_error": persistence_error,
        "same_day_csi_available": bool(same_day_available),
        "solar_elevation_available": bool(solar_available),
        "warnings": ";".join(warnings),
        "figure": str(filename),
    }


def run_next_day(args: argparse.Namespace, config: Any, output_root: Path) -> None:
    """Run next-day visual alignment plots."""
    output_dir = output_root / "next_day"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(config, split=args.split, include_target=True)
    elevation = load_elevation_frame(config.elevation_csv)
    rows: list[dict[str, Any]] = []
    count = min(int(args.num_samples), len(dataset))
    for index in range(count):
        row = metadata_row(dataset, index)
        rows.append(next_day_figure(dataset[index], row, index, args, output_dir, elevation))
        print(f"saved next_day sample {index + 1}/{count}")
    write_csv(output_dir / "next_day_alignment_summary.csv", rows)
    write_json(
        output_dir / "next_day_alignment_summary.json",
        {
            "dataset_root": str(config.dataset_root),
            "split": args.split,
            "num_samples": count,
            "channel_index": args.channel_index,
            "summary_csv": str(output_dir / "next_day_alignment_summary.csv"),
            "warning_counts": warning_counts(rows),
        },
    )


def run_short_horizon(args: argparse.Namespace, config: Any, output_root: Path) -> None:
    """Run short-horizon visual alignment plots."""
    output_dir = output_root / "short_horizon"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(config, split=args.split, include_target=True)
    hourly = load_hourly_frame(config.hourly_csv)
    elevation = load_elevation_frame(config.elevation_csv)
    candidates = short_horizon_candidates(dataset, args)
    rows: list[dict[str, Any]] = []
    for pos, (sample_index, lead, start, end, target_index) in enumerate(candidates, start=1):
        rows.append(
            short_horizon_figure(
                dataset,
                sample_index,
                lead,
                start,
                end,
                target_index,
                args,
                output_dir,
                hourly,
                elevation,
            )
        )
        print(f"saved short_horizon window {pos}/{len(candidates)}")
    write_csv(output_dir / "short_horizon_alignment_summary.csv", rows)
    write_json(
        output_dir / "short_horizon_alignment_summary.json",
        {
            "dataset_root": str(config.dataset_root),
            "split": args.split,
            "num_windows": len(candidates),
            "history_hours": args.history_hours,
            "lead_hours": parse_int_list(args.lead_hours),
            "channel_index": args.channel_index,
            "same_day_hourly_csv_available": hourly is not None,
            "summary_csv": str(output_dir / "short_horizon_alignment_summary.csv"),
            "warning_counts": warning_counts(rows),
        },
    )


def warning_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Count semicolon-delimited warning labels."""
    counts: dict[str, int] = {}
    for row in rows:
        for label in str(row.get("warnings") or "").split(";"):
            if label:
                counts[label] = counts.get(label, 0) + 1
    return counts


def main() -> None:
    """Run visual dataset alignment diagnostics."""
    args = parse_args()
    config = config_from_args(args)
    config.prepare_directories()
    output_root = Path(args.output_dir) if args.output_dir is not None else config.output_dir / "visual_sanity"
    output_root.mkdir(parents=True, exist_ok=True)
    if args.mode == "next_day":
        run_next_day(args, config, output_root)
    else:
        run_short_horizon(args, config, output_root)
    ArtifactMirror(
        checkpoint_dir=config.checkpoint_dir,
        output_dir=config.output_dir,
        enabled=config.mirror_artifacts,
    ).mirror_output_tree(output_root)


if __name__ == "__main__":
    main()
