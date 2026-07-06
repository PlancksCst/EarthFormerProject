"""Inspect one SEVIRI dataset item and print keys, shapes, and candidate fields."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
EARTHFORMER_DIR = SCRIPT_DIR.parents[1]
PROJECT_ROOT = EARTHFORMER_DIR.parent
for candidate in (PROJECT_ROOT, EARTHFORMER_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from configs.config import build_arg_parser, config_from_args  # noqa: E402
from datasets.seviri_dataset import build_dataset  # noqa: E402


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add inspection-specific CLI arguments."""
    parser.add_argument("--split", default="val")
    parser.add_argument("--index", type=int, default=0)
    return parser


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = build_arg_parser()
    parser.description = "Inspect one SEVIRI dataset item."
    add_args(parser)
    return parser.parse_args()


def numeric_array(value: Any) -> np.ndarray | None:
    """Return a numeric numpy array when possible."""
    if isinstance(value, torch.Tensor):
        if not torch.is_floating_point(value) and not value.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.bool):
            return None
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
        return value
    if isinstance(value, (list, tuple)):
        try:
            array = np.asarray(value)
        except Exception:
            return None
        if np.issubdtype(array.dtype, np.number) or array.dtype == np.bool_:
            return array
    return None


def value_shape(value: Any) -> Any:
    """Return shape/length description."""
    if isinstance(value, torch.Tensor):
        return tuple(value.shape)
    if isinstance(value, np.ndarray):
        return tuple(value.shape)
    if isinstance(value, (list, tuple)):
        return f"len={len(value)}"
    return None


def stats(array: np.ndarray | None) -> dict[str, float] | None:
    """Return numeric stats for finite arrays."""
    if array is None:
        return None
    try:
        values = np.asarray(array, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan")}
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
    }


def preview(value: Any, max_items: int = 8) -> Any:
    """Return a compact metadata preview."""
    if isinstance(value, torch.Tensor):
        flat = value.detach().cpu().reshape(-1)
        return flat[:max_items].tolist()
    if isinstance(value, np.ndarray):
        return value.reshape(-1)[:max_items].tolist()
    if isinstance(value, (list, tuple)):
        return list(value[:max_items])
    text = str(value)
    return text if len(text) <= 160 else text[:157] + "..."


def classify_candidates(sample: dict[str, Any]) -> dict[str, list[str]]:
    """Identify likely semantic fields from names and shapes."""
    candidates = {
        "images": [],
        "target_csi": [],
        "clear_sky_ghi": [],
        "ghi": [],
        "valid_mask": [],
        "location": [],
        "date": [],
        "solar_elevation": [],
        "previous_day_csi": [],
    }
    for key, value in sample.items():
        lower = key.lower()
        shape = value_shape(value)
        if lower in {"satellite", "images", "image", "input", "x"} or (isinstance(shape, tuple) and len(shape) >= 3 and "mask" not in lower):
            if "target" not in lower and "ghi" not in lower and "csi" not in lower:
                candidates["images"].append(key)
        if "csi" in lower or lower == "target":
            if "input" in lower or "previous" in lower or "prev" in lower:
                candidates["previous_day_csi"].append(key)
            else:
                candidates["target_csi"].append(key)
        if "clear" in lower:
            candidates["clear_sky_ghi"].append(key)
        if "ghi" in lower and "clear" not in lower:
            candidates["ghi"].append(key)
        if "mask" in lower or "valid" in lower:
            candidates["valid_mask"].append(key)
        if "location" in lower or lower in {"site", "station"}:
            candidates["location"].append(key)
        if "day" in lower or "date" in lower or "time" in lower or "timestamp" in lower:
            candidates["date"].append(key)
        if "solar" in lower or "elevation" in lower:
            candidates["solar_elevation"].append(key)
    return candidates


def main() -> None:
    """Inspect one dataset item."""
    args = parse_args()
    config = config_from_args(args)
    dataset = build_dataset(config, split=args.split, include_target=True)
    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(f"index={args.index} outside dataset length {len(dataset)}")
    sample = dataset[args.index]

    print(f"dataset_root: {config.dataset_root}")
    print(f"split: {args.split}")
    print(f"dataset_length: {len(dataset)}")
    print(f"index: {args.index}")
    print("\nKEYS")
    for key in sorted(sample.keys()):
        value = sample[key]
        array = numeric_array(value)
        report = {
            "key": key,
            "type": type(value).__name__,
            "shape": value_shape(value),
            "dtype": str(value.dtype) if isinstance(value, torch.Tensor) else str(value.dtype) if isinstance(value, np.ndarray) else None,
            "stats": stats(array),
            "preview": preview(value),
        }
        print(json.dumps(report, indent=2, default=str))

    print("\nDETECTED CANDIDATE FIELDS")
    print(json.dumps(classify_candidates(sample), indent=2))

    meta = getattr(dataset, "meta", None)
    if meta is not None:
        row = meta.iloc[args.index]
        print("\nMETADATA ROW PREVIEW")
        preview_items = {str(key): preview(row[key]) for key in list(row.index)[:40]}
        print(json.dumps(preview_items, indent=2, default=str))


if __name__ == "__main__":
    main()
