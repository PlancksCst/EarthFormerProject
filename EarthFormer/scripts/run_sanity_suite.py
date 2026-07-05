"""Run the full Perceiver forecasting sanity suite."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from .diagnostic_utils import (
        Timer,
        append_csv_row,
        diagnostics_dir,
        prepare_config,
        print_json,
        save_json_report,
    )
except ImportError:
    from diagnostic_utils import (  # type: ignore
        Timer,
        append_csv_row,
        diagnostics_dir,
        prepare_config,
        print_json,
        save_json_report,
    )

from configs.config import TrainingConfig, build_arg_parser, config_from_args

SCRIPT_DIR = Path(__file__).resolve().parent


def optional_value_arg(value: Any, flag: str) -> list[str]:
    """Return a CLI flag/value pair when a resolved value is available."""
    if value is None:
        return []
    return [flag, str(value)]


def common_child_args(config: TrainingConfig) -> list[str]:
    """Forward the resolved config to child scripts.

    Child diagnostics run in separate Python processes. Passing resolved config
    values prevents them from constructing a fresh default `TrainingConfig` that
    might fall back to `EarthFormer/data` instead of the caller's dataset path.
    """
    forwarded: list[str] = []
    for value, flag in (
        (config.dataset_root, "--dataset-root"),
        (config.metadata_filename, "--metadata-filename"),
        (config.hourly_csv, "--hourly-csv"),
        (config.elevation_csv, "--elevation-csv"),
        (config.batch_size, "--batch-size"),
        (config.learning_rate, "--learning-rate"),
        (config.backbone_learning_rate, "--backbone-learning-rate"),
        (config.head_learning_rate, "--head-learning-rate"),
        (config.weight_decay, "--weight-decay"),
        (config.epochs, "--epochs"),
        (config.num_workers, "--num-workers"),
        (config.device, "--device"),
        (config.checkpoint_dir, "--checkpoint-dir"),
        (config.output_dir, "--output-dir"),
        (config.pretrained_checkpoint, "--pretrained-checkpoint"),
        (config.resume_checkpoint, "--resume-checkpoint"),
        (config.random_seed, "--seed"),
        (config.amp_dtype, "--amp-dtype"),
        (config.gradient_clip, "--gradient-clip"),
        (config.warmup_epochs, "--warmup-epochs"),
        (config.early_stopping_patience, "--early-stopping-patience"),
        (config.scheduler_t_max, "--scheduler-t-max"),
        (config.scheduler_eta_min, "--scheduler-eta-min"),
        (config.input_length, "--input-length"),
        (config.output_length, "--output-length"),
        (config.target_channel_index, "--target-channel-index"),
        (config.readout_type, "--readout-type"),
        (config.readout_latent_dim, "--readout-latent-dim"),
        (config.query_dimension, "--query-dimension"),
        (config.num_output_queries, "--num-output-queries"),
        (config.num_attention_heads, "--num-attention-heads"),
        (config.readout_dropout, "--readout-dropout"),
        (config.regression_hidden_dim, "--regression-hidden-dim"),
    ):
        forwarded.extend(optional_value_arg(value, flag))
    if config.mixed_precision:
        forwarded.append("--amp")
    else:
        forwarded.append("--no-amp")
    if not config.normalize:
        forwarded.append("--no-normalize")
    if config.freeze_earthformer:
        forwarded.append("--freeze-earthformer")
    if not config.mirror_artifacts:
        forwarded.append("--no-artifact-mirror")
    return forwarded


def run_child(
    script_name: str,
    report_name: str,
    common_args: list[str],
    extra_args: list[str],
    timeout_seconds: int,
) -> dict[str, Any]:
    """Run one diagnostic script and return a suite result row."""
    timer = Timer()
    script_path = SCRIPT_DIR / script_name
    command = [
        sys.executable,
        str(script_path),
        *common_args,
        "--report-name",
        report_name,
        *extra_args,
    ]
    completed = subprocess.run(
        command,
        cwd=str(SCRIPT_DIR.parent),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    elapsed = timer.elapsed()
    status = "PASS" if completed.returncode == 0 else "FAIL"
    warning_text = completed.stderr.strip()
    return {
        "script": script_name,
        "report_name": report_name,
        "command": command,
        "status": status,
        "return_code": completed.returncode,
        "execution_time_seconds": elapsed,
        "warnings": warning_text,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def main() -> None:
    """Run all sanity checks and write a suite summary."""
    parser = build_arg_parser()
    parser.description = "Run all Perceiver forecasting sanity diagnostics."
    parser.add_argument("--split", default="train")
    parser.add_argument("--target-mode", choices=["satellite_mean", "zeros"], default="satellite_mean")
    parser.add_argument("--overfit-samples", type=int, default=8)
    parser.add_argument("--overfit-max-epochs", type=int, default=50)
    parser.add_argument("--overfit-threshold", type=float, default=1.0e-3)
    parser.add_argument("--resume-steps-before-save", type=int, default=1)
    parser.add_argument("--resume-steps-after-resume", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--report-name", default="run_sanity_suite")
    parser.add_argument(
        "--tests",
        nargs="+",
        default=["all"],
        choices=[
            "all",
            "pipeline",
            "inspect",
            "attention",
            "one_batch",
            "overfit",
            "resume",
        ],
        help="Which sanity tests to execute.",
    )
    args = parser.parse_args()

    config = prepare_config(config_from_args(args))
    timer = Timer()
    common_args = common_child_args(config)
    split_args = ["--split", args.split]
    target_args = ["--target-mode", args.target_mode]

    suite = [
        (
            "verify_perceiver_pipeline.py",
            "suite_verify_perceiver_pipeline",
            split_args,
        ),
        (
            "inspect_perceiver.py",
            "suite_inspect_perceiver",
            split_args,
        ),
        (
            "check_attention.py",
            "suite_check_attention",
            split_args,
        ),
        (
            "test_one_batch.py",
            "suite_test_one_batch",
            [*split_args, *target_args],
        ),
        (
            "test_overfit.py",
            "suite_test_overfit",
            [
                *split_args,
                *target_args,
                "--samples",
                str(args.overfit_samples),
                "--max-epochs",
                str(args.overfit_max_epochs),
                "--threshold",
                str(args.overfit_threshold),
            ],
        ),
        (
            "test_resume.py",
            "suite_test_resume",
            [
                *split_args,
                *target_args,
                "--steps-before-save",
                str(args.resume_steps_before_save),
                "--steps-after-resume",
                str(args.resume_steps_after_resume),
            ],
        ),
    ]

    TEST_MAP = {
        "pipeline": "verify_perceiver_pipeline.py",
        "inspect": "inspect_perceiver.py",
        "attention": "check_attention.py",
        "one_batch": "test_one_batch.py",
        "overfit": "test_overfit.py",
        "resume": "test_resume.py",
    }

    results: list[dict[str, Any]] = []
    csv_path = diagnostics_dir(config) / "sanity_suite_summary.csv"
    
    if "all" in args.tests:
        selected_suite = suite
    else:
        selected_scripts = {TEST_MAP[name] for name in args.tests}
        selected_suite = [
            item
            for item in suite
            if item[0] in selected_scripts
        ]
    
    for script_name, child_report_name, extra_args in selected_suite:
        print(f"Running {script_name}...")
        result = run_child(
            script_name=script_name,
            report_name=child_report_name,
            common_args=common_args,
            extra_args=extra_args,
            timeout_seconds=args.timeout_seconds,
        )
        report_path = diagnostics_dir(config) / f"{child_report_name}.json"
        if report_path.exists():
            with report_path.open("r", encoding="utf-8") as handle:
                child_report = json.load(handle)
            result["status"] = child_report.get("status", result["status"])
            result["report_path"] = str(report_path)
        else:
            result["report_path"] = None

        results.append(result)
        append_csv_row(
            csv_path,
            {
                "script": script_name,
                "status": result["status"],
                "return_code": result["return_code"],
                "execution_time_seconds": result["execution_time_seconds"],
                "report_path": result["report_path"],
            },
            config=config,
        )
        print(
            f"{script_name}: {result['status']} "
            f"({result['execution_time_seconds']:.1f}s)"
        )

    passed = all(result["status"] == "PASS" for result in results)
    payload = {
        "status": "PASS" if passed else "FAIL",
        "dataset_root": str(config.dataset_root),
        "checkpoint_dir": str(config.checkpoint_dir),
        "output_dir": str(config.output_dir),
        "split": args.split,
        "results": results,
        "summary_csv": str(csv_path),
        "elapsed_seconds": timer.elapsed(),
    }
    report_path = save_json_report(config, args.report_name, payload)
    payload["report_path"] = str(report_path)

    print("\nSanity suite summary")
    for result in results:
        print(
            f"- {result['script']}: {result['status']} "
            f"({result['execution_time_seconds']:.1f}s)"
        )
    print_json(payload)


if __name__ == "__main__":
    main()
