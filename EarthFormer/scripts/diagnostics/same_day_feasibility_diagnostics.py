"""Inspect day alignment and optionally run a tiny non-EarthFormer CSI probe."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from .diagnostic_common import (
        build_context,
        dataloader_for_split,
        metadata_value_from_batch,
        mirror_outputs,
        parse_common_args,
        regression_metrics,
        target_ghi_tensor,
        target_tensor,
        clear_sky_tensor,
        valid_mask_tensor,
        write_csv,
        write_json,
    )
except ImportError:
    from diagnostic_common import (  # type: ignore
        build_context,
        dataloader_for_split,
        metadata_value_from_batch,
        mirror_outputs,
        parse_common_args,
        regression_metrics,
        target_ghi_tensor,
        target_tensor,
        clear_sky_tensor,
        valid_mask_tensor,
        write_csv,
        write_json,
    )


def input_csi_from_batch(batch: dict[str, Any]) -> torch.Tensor | None:
    """Return input-day CSI if a dataset item exposes it."""
    for key in ("input_csi", "previous_day_csi", "previous_csi", "prev_csi", "input_day_csi"):
        value = batch.get(key)
        if isinstance(value, torch.Tensor) and value.ndim == 2:
            return value.float()
    return None


def inspect_alignment(context: Any) -> list[dict[str, Any]]:
    """Collect day/target consistency records."""
    loader = dataloader_for_split(
        context.config,
        split=context.args.split,
        include_target=True,
        shuffle=False,
        max_samples=context.args.max_samples,
    )
    rows: list[dict[str, Any]] = []
    sample_index = 0
    for batch in loader:
        target = target_tensor(batch, context.device)
        clear = clear_sky_tensor(batch, context.device)
        target_ghi = target_ghi_tensor(batch, target, clear)
        valid = valid_mask_tensor(batch, target, clear, context.config.clear_sky_threshold)
        reconstructed = target * clear
        diff = (target_ghi - reconstructed).detach().abs().cpu()
        target_cpu = target.detach().cpu()
        clear_cpu = clear.detach().cpu()
        target_ghi_cpu = target_ghi.detach().cpu()
        valid_cpu = valid.detach().cpu()
        for local_index in range(target.shape[0]):
            rows.append(
                {
                    "sample_index": sample_index,
                    "sample_id": metadata_value_from_batch(batch, "sample_id", local_index),
                    "location": metadata_value_from_batch(batch, "location", local_index),
                    "input_day": metadata_value_from_batch(batch, "input_day", local_index),
                    "target_day": metadata_value_from_batch(batch, "target_day", local_index),
                    "target_csi": json.dumps(target_cpu[local_index].tolist()),
                    "target_ghi": json.dumps(target_ghi_cpu[local_index].tolist()),
                    "clear_sky_ghi": json.dumps(clear_cpu[local_index].tolist()),
                    "valid_mask": json.dumps(valid_cpu[local_index].tolist()),
                    "max_abs_target_ghi_minus_csi_times_clear": float(diff[local_index][valid_cpu[local_index]].max())
                    if bool(valid_cpu[local_index].any())
                    else float("nan"),
                    "input_csi_available": input_csi_from_batch(batch) is not None,
                }
            )
            sample_index += 1
    return rows


def run_optional_probe(context: Any) -> dict[str, Any]:
    """Train a tiny input-CSI to target-CSI probe if input CSI is available."""
    loader = dataloader_for_split(
        context.config,
        split=context.args.split,
        include_target=True,
        shuffle=False,
        max_samples=context.args.max_samples,
    )
    features: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for batch in loader:
        input_csi = input_csi_from_batch(batch)
        if input_csi is None:
            return {"available": False, "reason": "input-day CSI is not exposed by dataset items"}
        target = target_tensor(batch, torch.device("cpu"))
        clear = clear_sky_tensor(batch, torch.device("cpu"))
        valid = valid_mask_tensor(batch, target, clear, context.config.clear_sky_threshold)
        features.append(input_csi.cpu())
        targets.append(target.cpu())
        masks.append(valid.cpu())

    x = torch.cat(features, dim=0)
    y = torch.cat(targets, dim=0)
    valid = torch.cat(masks, dim=0)
    if x.numel() == 0 or not bool(valid.any()):
        return {"available": False, "reason": "no valid probe samples"}

    model = nn.Linear(x.shape[1], y.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-2, weight_decay=1.0e-3)
    dataset = TensorDataset(x, y, valid)
    probe_loader = DataLoader(dataset, batch_size=min(32, len(dataset)), shuffle=True)
    log: list[dict[str, float]] = []
    for epoch in range(1, 101):
        total = 0.0
        count = 0
        for batch_x, batch_y, batch_valid in probe_loader:
            pred = model(batch_x)
            squared = (pred - batch_y) ** 2
            loss = squared.masked_fill(~batch_valid, 0.0).sum() / batch_valid.sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * int(batch_valid.sum())
            count += int(batch_valid.sum())
        log.append({"epoch": epoch, "loss": total / max(count, 1)})

    with torch.no_grad():
        pred = model(x)
    metrics = regression_metrics(pred[valid].numpy(), y[valid].numpy())
    write_csv(context.output_dir / "same_day_probe_training_log.csv", log)
    return {"available": True, "metrics": metrics, "training_log": str(context.output_dir / "same_day_probe_training_log.csv")}


def main() -> None:
    """Run same-day/date feasibility diagnostics."""
    args = parse_common_args("Inspect date alignment and optional simple CSI feasibility.")
    context = build_context(args, default_subdir="same_day_feasibility")
    rows = inspect_alignment(context)
    report_csv = context.output_dir / "metadata_alignment_report.csv"
    write_csv(report_csv, rows)
    probe = run_optional_probe(context)
    summary = {
        "dataset_root": str(context.config.dataset_root),
        "split": args.split,
        "max_samples": args.max_samples,
        "records": len(rows),
        "input_csi_available": any(bool(row["input_csi_available"]) for row in rows),
        "probe": probe,
        "metadata_alignment_report": str(report_csv),
    }
    write_json(context.output_dir / "same_day_feasibility_summary.json", summary)
    mirror_outputs(context)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
