"""Run the full image-only model failure diagnostic suite."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .diagnostic_common import (
        add_common_diagnostic_args,
        build_context,
        mirror_outputs,
        write_json,
    )
except ImportError:
    from diagnostic_common import (  # type: ignore
        add_common_diagnostic_args,
        build_context,
        mirror_outputs,
        write_json,
    )

from configs.config import build_arg_parser  # noqa: E402


def parse_args() -> Any:
    """Parse runner arguments."""
    parser = build_arg_parser()
    parser.description = "Run all failure-analysis diagnostics."
    add_common_diagnostic_args(parser)
    parser.add_argument("--run-latent-probe", action="store_true")
    return parser.parse_args()


def common_cli(args: Any) -> list[str]:
    """Return common CLI flags to pass to child scripts."""
    pairs = [
        ("--dataset-root", args.dataset_root),
        ("--checkpoint", args.checkpoint),
        ("--split", args.split),
        ("--batch-size", args.batch_size),
        ("--num-workers", args.num_workers),
        ("--device", args.device),
        ("--output-dir", args.output_dir),
        ("--clear-sky-threshold", args.clear_sky_threshold),
        ("--max-samples", args.max_samples),
        ("--checkpoint-dir", args.checkpoint_dir),
        ("--hourly-csv", args.hourly_csv),
        ("--elevation-csv", args.elevation_csv),
    ]
    cli: list[str] = []
    for flag, value in pairs:
        if value is not None:
            cli.extend([flag, str(value)])
    if getattr(args, "no_artifact_mirror", False):
        cli.append("--no-artifact-mirror")
    return cli


def run_child(script_name: str, args: Any) -> dict[str, Any]:
    """Run one diagnostic child script."""
    script_path = Path(__file__).resolve().parent / script_name
    command = [sys.executable, str(script_path), *common_cli(args)]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return {
        "script": script_name,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
        "ok": result.returncode == 0,
    }


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON when it exists."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_metrics(path: Path) -> pd.DataFrame:
    """Read a metrics CSV when available."""
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def interpret(root: Path, results: list[dict[str, Any]]) -> dict[str, Any]:
    """Build rule-based failure interpretation."""
    metadata_summary = load_json(root / "metadata_alignment" / "metadata_alignment_summary.json")
    baseline_metrics = read_metrics(root / "baselines" / "baseline_metrics.csv")
    sensitivity_metrics = read_metrics(root / "image_sensitivity" / "image_sensitivity_metrics.csv")
    latent_metrics = read_metrics(root / "latent_probe" / "latent_probe_metrics.csv")

    metadata_ok = bool(metadata_summary.get("metadata_ok", False))
    model_row = sensitivity_metrics[sensitivity_metrics.get("perturbation", pd.Series(dtype=str)) == "real"]
    model_rmse = float(model_row["CSI_RMSE"].iloc[0]) if not model_row.empty and "CSI_RMSE" in model_row else float("nan")
    hourly_row = baseline_metrics[baseline_metrics.get("baseline", pd.Series(dtype=str)) == "hourly_climatology"]
    location_row = baseline_metrics[baseline_metrics.get("baseline", pd.Series(dtype=str)) == "location_hour_climatology"]
    persistence_row = baseline_metrics[
        baseline_metrics.get("baseline", pd.Series(dtype=str)) == "previous_day_csi_persistence"
    ]
    climatology_rmse = float("nan")
    if not location_row.empty:
        climatology_rmse = float(location_row["CSI_RMSE"].iloc[0])
    elif not hourly_row.empty:
        climatology_rmse = float(hourly_row["CSI_RMSE"].iloc[0])

    model_beats_climatology = bool(model_rmse < climatology_rmse) if pd.notna(model_rmse) and pd.notna(climatology_rmse) else False
    model_beats_persistence = None
    if not persistence_row.empty and pd.notna(model_rmse):
        model_beats_persistence = bool(model_rmse < float(persistence_row["CSI_RMSE"].iloc[0]))

    non_real = sensitivity_metrics[
        sensitivity_metrics.get("perturbation", pd.Series(dtype=str)).astype(str) != "real"
    ]
    max_delta = float(non_real["delta_mean_abs"].max()) if not non_real.empty and "delta_mean_abs" in non_real else float("nan")
    image_sensitivity_detected = bool(max_delta > 0.02) if pd.notna(max_delta) else False

    latent_probe_better = False
    if not latent_metrics.empty and "method" in latent_metrics and "CSI_RMSE" in latent_metrics:
        perceiver = latent_metrics[latent_metrics["method"] == "perceiver_checkpoint"]
        probes = latent_metrics[latent_metrics["method"].isin(["latent_linear", "latent_mlp"])]
        if not perceiver.empty and not probes.empty:
            latent_probe_better = bool(probes["CSI_RMSE"].min() < 0.95 * float(perceiver["CSI_RMSE"].iloc[0]))

    if not metadata_ok:
        suspected = "metadata/index/target alignment problem"
    elif not image_sensitivity_detected:
        suspected = "model is ignoring satellite images"
    elif not model_beats_climatology:
        suspected = "model does not beat simple baselines"
    elif latent_probe_better:
        suspected = "Perceiver readout bottleneck"
    else:
        suspected = "next-day satellite-only signal may be weak or dataset too small"

    return {
        "metadata_ok": metadata_ok,
        "model_csi_rmse": model_rmse,
        "climatology_csi_rmse": climatology_rmse,
        "model_beats_climatology": model_beats_climatology,
        "model_beats_persistence_if_available": model_beats_persistence,
        "image_sensitivity_detected": image_sensitivity_detected,
        "max_prediction_delta_vs_real": max_delta,
        "latent_probe_better_than_perceiver": latent_probe_better,
        "suspected_failure_mode": suspected,
        "child_results": results,
    }


def main() -> None:
    """Run all diagnostics and write combined summary."""
    args = parse_args()
    context = build_context(args, default_subdir=None)
    scripts = [
        "metadata_alignment_check.py",
        "baseline_diagnostics.py",
        "image_sensitivity_diagnostics.py",
    ]
    if args.run_latent_probe:
        scripts.append("latent_probe_diagnostics.py")
    scripts.append("same_day_feasibility_diagnostics.py")

    results = [run_child(script, args) for script in scripts]
    summary = interpret(context.output_dir, results)
    summary_path = context.output_dir / "failure_diagnostics_summary.json"
    write_json(summary_path, summary)
    mirror_outputs(context)
    print(json.dumps(summary, indent=2))
    failed = [result for result in results if not result["ok"]]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
