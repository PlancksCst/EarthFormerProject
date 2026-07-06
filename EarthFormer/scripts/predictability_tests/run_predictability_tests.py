"""Run the full controlled predictability-test suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from predictability_common import (  # type: ignore
    SCRIPT_DIR,
    build_context,
    parse_args,
    run_child,
    write_json,
)


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file when available."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file when available."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def recommend(
    image_summary: dict[str, Any],
    short_summary: dict[str, Any],
    child_results: list[dict[str, Any]],
) -> str:
    """Return the combined predictability recommendation."""
    if any(not result["ok"] for result in child_results):
        return "baselines_unavailable_or_test_failed"

    image_beats_hourly = image_summary.get("image_model_beats_hourly_climatology")
    image_beats_location = image_summary.get("image_model_beats_location_hour_climatology")
    image_beats_persistence = image_summary.get("image_model_beats_previous_day_persistence")
    image_beats_available = (
        image_beats_hourly is True
        and image_beats_location in (True, None)
        and image_beats_persistence in (True, None)
    )
    if image_beats_available:
        return "image_model_beats_baselines_next_day_continue_image_model"

    short_flags = short_summary.get("satellite_model_beats_climatology", {})
    short_works = any(value is True for value in short_flags.values()) if isinstance(short_flags, dict) else False
    if short_works:
        return "image_model_does_not_beat_baselines_but_short_horizon_works_horizon_limitation"
    return "image_model_does_not_beat_baselines_and_short_horizon_fails_check_data_pipeline"


def main() -> None:
    """Run climatology, image-vs-baseline, and short-horizon tests."""
    args = parse_args("Run all predictability tests.")
    root_context = build_context(args, default_subdir="predictability_tests")
    root = root_context.output_dir
    args.output_dir = root

    climatology_dir = root / "climatology_baselines"
    image_dir = root / "image_only_vs_baselines"
    short_dir = root / "short_horizon"

    scripts = {
        "climatology": SCRIPT_DIR / "test_climatology_baselines.py",
        "image": SCRIPT_DIR / "test_image_only_model_against_baselines.py",
        "short": SCRIPT_DIR / "test_short_horizon_satellite_predictability.py",
    }
    child_results = [
        run_child(scripts["climatology"], args, climatology_dir),
        run_child(scripts["image"], args, image_dir),
        run_child(scripts["short"], args, short_dir),
    ]

    climatology_summary = read_json(climatology_dir / "climatology_baseline_summary.json")
    image_summary = read_json(image_dir / "image_only_vs_baseline_summary.json")
    short_summary = read_json(short_dir / "short_horizon_summary.json")
    combined = {
        "climatology_summary": climatology_summary,
        "image_only_vs_baseline_summary": image_summary,
        "short_horizon_summary": short_summary,
        "child_results": child_results,
        "recommendation": recommend(image_summary, short_summary, child_results),
    }
    summary_path = root / "predictability_test_summary.json"
    write_json(summary_path, combined)
    root_context.artifact_mirror.mirror_output_tree(root)
    print(json.dumps(combined, indent=2))
    if any(not result["ok"] for result in child_results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
