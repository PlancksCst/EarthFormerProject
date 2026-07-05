"""Strict metadata and target-alignment verification for SEVIRI CSI samples."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import torch

try:
    from .diagnostic_common import (
        build_context,
        dataset_for_split,
        dataset_row,
        mirror_outputs,
        parse_common_args,
        write_csv,
        write_json,
    )
except ImportError:
    from diagnostic_common import (  # type: ignore
        build_context,
        dataset_for_split,
        dataset_row,
        mirror_outputs,
        parse_common_args,
        write_csv,
        write_json,
    )


def tensor_list(value: Any) -> list[Any]:
    """Return a JSON-friendly list for tensors/sequences."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, np.ndarray):
        return value.reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    if value is None:
        return []
    return [value]


def parse_day(value: Any) -> datetime | None:
    """Parse date strings when possible."""
    if value is None:
        return None
    try:
        return pd.to_datetime(value).to_pydatetime()
    except Exception:
        return None


def row_value(row: Any, key: str) -> Any:
    """Return a metadata row field if present."""
    if row is None:
        return None
    try:
        if key in row.index:
            return row[key]
    except Exception:
        return None
    return None


def item_report(item: dict[str, Any], row: Any, index: int, threshold: float) -> tuple[dict[str, Any], list[str]]:
    """Return one CSV report row plus severe issue labels."""
    severe: list[str] = []
    satellite = item.get("satellite")
    target = item.get("target", item.get("target_csi"))
    clear = item.get("clear_sky_ghi", item.get("clear_ghi"))
    target_ghi = item.get("target_ghi")
    target_mask = item.get("target_mask")
    image_mask = item.get("image_mask")

    sat_shape = tuple(satellite.shape) if isinstance(satellite, torch.Tensor) else None
    target_shape = tuple(target.shape) if isinstance(target, torch.Tensor) else None
    clear_shape = tuple(clear.shape) if isinstance(clear, torch.Tensor) else None
    target_ghi_shape = tuple(target_ghi.shape) if isinstance(target_ghi, torch.Tensor) else None

    if sat_shape != (13, 7, 200, 200):
        severe.append("bad_satellite_shape")
    if target_shape != (13,):
        severe.append("bad_target_shape")
    if clear_shape != (13,):
        severe.append("bad_clear_sky_shape")
    if target_ghi_shape != (13,):
        severe.append("bad_target_ghi_shape")

    target_np = np.asarray(tensor_list(target), dtype=np.float64)
    clear_np = np.asarray(tensor_list(clear), dtype=np.float64)
    ghi_np = np.asarray(tensor_list(target_ghi), dtype=np.float64)
    mask_np = np.asarray(tensor_list(target_mask), dtype=bool) if target_mask is not None else np.zeros_like(target_np, dtype=bool)
    valid = (~mask_np) & np.isfinite(clear_np) & (clear_np > threshold)
    reconstructed = target_np * clear_np
    diff = ghi_np - reconstructed if ghi_np.shape == reconstructed.shape else np.asarray([np.nan])
    max_abs_diff_valid = float(np.nanmax(np.abs(diff[valid]))) if valid.any() and diff.shape == valid.shape else float("nan")
    if np.isfinite(max_abs_diff_valid) and max_abs_diff_valid > 1.0e-2:
        severe.append("target_ghi_mismatch")

    input_day = item.get("input_day")
    target_day = item.get("target_day")
    parsed_input = parse_day(input_day)
    parsed_target = parse_day(target_day)
    target_is_next_day = None
    if parsed_input is not None and parsed_target is not None:
        target_is_next_day = (parsed_target.date() - parsed_input.date()).days == 1
        if not target_is_next_day:
            severe.append("target_day_not_next_day")

    input_length = row_value(row, "input_length")
    target_length = row_value(row, "target_length")
    if input_length is not None and int(input_length) != 13:
        severe.append("bad_input_length")
    if target_length is not None and int(target_length) != 13:
        severe.append("bad_target_length")

    csi_outside = int(np.sum(np.isfinite(target_np) & ((target_np < 0.0) | (target_np > 1.3))))
    report = {
        "dataset_index": index,
        "sample_id": item.get("sample_id"),
        "location": item.get("location"),
        "input_day": input_day,
        "target_day": target_day,
        "target_is_next_day": target_is_next_day,
        "input_indices": json.dumps(tensor_list(row_value(row, "input_indices"))),
        "target_indices": json.dumps(tensor_list(row_value(row, "target_indices"))),
        "image_mask": json.dumps(tensor_list(image_mask)),
        "target_mask": json.dumps(tensor_list(target_mask)),
        "input_length": input_length,
        "target_length": target_length,
        "satellite_shape": str(sat_shape),
        "target_csi_shape": str(target_shape),
        "clear_sky_ghi_shape": str(clear_shape),
        "target_ghi_shape": str(target_ghi_shape),
        "target_csi": json.dumps(target_np.tolist()),
        "clear_sky_ghi": json.dumps(clear_np.tolist()),
        "target_ghi": json.dumps(ghi_np.tolist()),
        "reconstructed_ghi": json.dumps(reconstructed.tolist()),
        "abs_ghi_diff": json.dumps(np.abs(diff).tolist()),
        "max_abs_ghi_diff_valid": max_abs_diff_valid,
        "valid_mask": json.dumps(valid.tolist()),
        "valid_count": int(valid.sum()),
        "valid_fraction": float(valid.mean()) if valid.size else 0.0,
        "csi_outside_0_1p3_count": csi_outside,
        "severe_issues": ";".join(severe),
    }
    return report, severe


def main() -> None:
    """Run metadata alignment checks."""
    args = parse_common_args("Strict metadata alignment check.")
    context = build_context(args, default_subdir="metadata_alignment")
    dataset = dataset_for_split(context.config, args.split, include_target=True, max_samples=None)
    sample_count = min(args.max_samples or 20, len(dataset))
    rng = np.random.default_rng(context.config.random_seed)
    indices = sorted(rng.choice(len(dataset), size=sample_count, replace=False).tolist())
    reports: list[dict[str, Any]] = []
    severe_counts: dict[str, int] = {}

    for position, index in enumerate(indices):
        item = dataset[index]
        row = dataset_row(dataset, index)
        report, severe = item_report(
            item=item,
            row=row,
            index=index,
            threshold=context.config.clear_sky_threshold,
        )
        reports.append(report)
        for issue in severe:
            severe_counts[issue] = severe_counts.get(issue, 0) + 1
        print(
            f"[{position + 1}/{sample_count}] index={index} "
            f"sample_id={report['sample_id']} location={report['location']} "
            f"valid={report['valid_count']} issues={report['severe_issues'] or 'none'}"
        )

    csv_path = context.output_dir / "metadata_alignment_check.csv"
    summary_path = context.output_dir / "metadata_alignment_summary.json"
    write_csv(csv_path, reports)
    summary = {
        "dataset_root": str(context.config.dataset_root),
        "split": args.split,
        "checked_samples": len(reports),
        "clear_sky_threshold": context.config.clear_sky_threshold,
        "metadata_ok": len(severe_counts) == 0,
        "severe_issue_counts": severe_counts,
        "csi_outside_0_1p3_total": int(sum(row["csi_outside_0_1p3_count"] for row in reports)),
        "csv": str(csv_path),
    }
    write_json(summary_path, summary)
    mirror_outputs(context)
    if severe_counts:
        raise RuntimeError(f"Severe metadata alignment issues detected: {severe_counts}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
