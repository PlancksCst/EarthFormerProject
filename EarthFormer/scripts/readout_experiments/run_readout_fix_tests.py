"""Run the controlled readout-fix diagnostic suite."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from readout_common import (  # type: ignore
    SCRIPT_DIR,
    build_experiment_context,
    parse_readout_args,
    run_script,
    write_summary,
)


DIAGNOSTIC_DIR = SCRIPT_DIR.parent / "diagnostics"


def read_json(path: Path) -> dict[str, Any]:
    """Read JSON if present."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> pd.DataFrame:
    """Read CSV if present."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def rmse_for(metrics: pd.DataFrame, method: str) -> float:
    """Return CSI RMSE for one method."""
    if metrics.empty or "method" not in metrics.columns or "CSI_RMSE" not in metrics.columns:
        return float("nan")
    rows = metrics[metrics["method"] == method]
    if rows.empty:
        return float("nan")
    return float(rows["CSI_RMSE"].iloc[0])


def build_command_output_root(args: Any) -> Path:
    """Return the wrapper's output root."""
    if args.output_dir is not None:
        return Path(args.output_dir)
    return Path(args.checkpoint_dir or "checkpoints").parent / "outputs" / "readout_experiments"


def interpret(root: Path, child_results: list[dict[str, Any]], run_penalty: bool) -> dict[str, Any]:
    """Create rule-based interpretation from child outputs."""
    metadata_summary = read_json(root / "metadata_alignment" / "metadata_alignment_summary.json")
    query_summary = read_json(root / "query_shortcut" / "prediction_delta_summary.json")
    latent_metrics = read_csv(root / "latent_dependent_readout" / "readout_experiment_metrics.csv")
    penalty_summary = read_json(root / "image_dependence_penalty" / "image_dependence_penalty_summary.json")

    metadata_ok = bool(metadata_summary.get("metadata_ok", False))
    current_readout_ignores_latents = bool(query_summary.get("current_readout_ignores_latents", False))

    perceiver_rmse = rmse_for(latent_metrics, "current_perceiver")
    temporal_pool_rmse = rmse_for(latent_metrics, "temporal_pool_mlp")
    attention_pool_rmse = rmse_for(latent_metrics, "temporal_attention_pool")
    summary_plus_rmse = rmse_for(latent_metrics, "latent_summary_plus_query")
    temporal_pool_beats = bool(np.isfinite(perceiver_rmse) and np.isfinite(temporal_pool_rmse) and temporal_pool_rmse < perceiver_rmse)
    attention_pool_beats = bool(np.isfinite(perceiver_rmse) and np.isfinite(attention_pool_rmse) and attention_pool_rmse < perceiver_rmse)
    summary_plus_beats = bool(np.isfinite(perceiver_rmse) and np.isfinite(summary_plus_rmse) and summary_plus_rmse < perceiver_rmse)
    penalty_helped = bool(penalty_summary.get("image_dependence_penalty_helped", False)) if run_penalty else None

    if not metadata_ok:
        recommendation = "metadata_or_target_alignment_must_be_fixed_before_readout_changes"
    elif current_readout_ignores_latents and (temporal_pool_beats or attention_pool_beats or summary_plus_beats):
        recommendation = "promote_a_latent_dependent_readout_candidate_to_a_controlled_production_experiment"
    elif current_readout_ignores_latents:
        recommendation = "readout_has_query_only_shortcut_but_latent_probe_did_not_clearly_fix_it"
    elif temporal_pool_beats or attention_pool_beats or summary_plus_beats:
        recommendation = "latent_information_is_useful_and_current_readout_is_likely_bottleneck"
    elif penalty_helped:
        recommendation = "image_dependence_regularization_is_promising_for_a_future_ablation"
    else:
        recommendation = "no_single_readout_fix_was_decisive_on_this_subset"

    return {
        "metadata_ok": metadata_ok,
        "current_readout_ignores_latents": current_readout_ignores_latents,
        "temporal_pool_mlp_beats_perceiver": temporal_pool_beats,
        "attention_pool_beats_perceiver": attention_pool_beats,
        "latent_summary_plus_query_beats_perceiver": summary_plus_beats,
        "image_dependence_penalty_helped": penalty_helped,
        "current_perceiver_CSI_RMSE": perceiver_rmse,
        "temporal_pool_mlp_CSI_RMSE": temporal_pool_rmse,
        "temporal_attention_pool_CSI_RMSE": attention_pool_rmse,
        "latent_summary_plus_query_CSI_RMSE": summary_plus_rmse,
        "recommended_next_fix": recommendation,
        "child_results": child_results,
    }


def optional_cli_pairs(pairs: list[tuple[str, Any]]) -> list[str]:
    """Return CLI flags for non-None experiment-only values."""
    cli: list[str] = []
    for flag, value in pairs:
        if value is not None:
            cli.extend([flag, str(value)])
    return cli


def main() -> None:
    """Run metadata, shortcut, latent-readout, and optional penalty tests."""
    args = parse_readout_args("Run controlled readout-fix diagnostics.")
    context = build_experiment_context(args, subdir=None)
    root = context.output_dir
    args.output_dir = root

    child_results: list[dict[str, Any]] = []
    metadata_script = DIAGNOSTIC_DIR / "metadata_alignment_check.py"
    query_script = SCRIPT_DIR / "test_query_only_shortcut.py"
    latent_script = SCRIPT_DIR / "test_latent_dependent_readout.py"
    penalty_script = SCRIPT_DIR / "test_image_dependence_penalty.py"

    print("Running metadata alignment check...")
    child_results.append(run_script(metadata_script, args))
    print("Running query-only shortcut test...")
    child_results.append(run_script(query_script, args))
    print("Running latent-dependent readout experiments...")
    readout_extra = optional_cli_pairs(
        [
            ("--experiment-epochs", args.experiment_epochs),
            ("--latent-token-stride", args.latent_token_stride),
            ("--readout-types", args.readout_types),
        ]
    )
    child_results.append(run_script(latent_script, args, extra=readout_extra))
    if args.run_image_dependence_penalty:
        print("Running optional image-dependence penalty test...")
        penalty_extra = optional_cli_pairs(
            [
                ("--experiment-epochs", args.experiment_epochs),
                ("--image-dependence-weight", args.image_dependence_weight),
                ("--image-dependence-margin", args.image_dependence_margin),
            ]
        )
        child_results.append(run_script(penalty_script, args, extra=penalty_extra))

    summary = interpret(root, child_results, run_penalty=bool(args.run_image_dependence_penalty))
    summary_path = root / "readout_fix_test_summary.json"
    write_summary(summary_path, summary)
    context.artifact_mirror.mirror_output_tree(root)
    print(json.dumps(summary, indent=2))

    if any(not result["ok"] for result in child_results):
        print("One or more diagnostics failed; inspect child stderr tails in the summary.", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
