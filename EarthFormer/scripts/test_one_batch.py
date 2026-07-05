"""One-batch optimization sanity test for the forecasting model."""

from __future__ import annotations

try:
    from .diagnostic_utils import (
        Timer,
        append_csv_row,
        autocast_dtype,
        build_model,
        build_optimizer,
        build_scaler,
        capture_trainable_parameters,
        count_updated_parameters,
        diagnostics_dir,
        load_batch,
        make_sanity_target,
        prepare_config,
        print_json,
        resolve_device,
        save_json_report,
        train_one_batch,
        use_amp,
    )
except ImportError:
    from diagnostic_utils import (  # type: ignore
        Timer,
        append_csv_row,
        autocast_dtype,
        build_model,
        build_optimizer,
        build_scaler,
        capture_trainable_parameters,
        count_updated_parameters,
        diagnostics_dir,
        load_batch,
        make_sanity_target,
        prepare_config,
        print_json,
        resolve_device,
        save_json_report,
        train_one_batch,
        use_amp,
    )

from configs.config import build_arg_parser, config_from_args


def main() -> None:
    """Run one forward, backward, and optimizer step."""
    parser = build_arg_parser()
    parser.description = "Run a one-batch forecasting sanity test."
    parser.add_argument("--split", default="train")
    parser.add_argument("--target-mode", choices=["satellite_mean", "zeros"], default="satellite_mean")
    parser.add_argument("--report-name", default="test_one_batch")
    args = parser.parse_args()

    config = prepare_config(config_from_args(args))
    device = resolve_device(config)
    amp_enabled = use_amp(config, device)
    amp_dtype = autocast_dtype(config, device)
    timer = Timer()

    batch = load_batch(config=config, split=args.split, device=device, include_target=False)
    inputs = batch["satellite"]
    target = make_sanity_target(inputs, output_length=config.output_length, mode=args.target_mode)

    model = build_model(config, device)
    optimizer = build_optimizer(config, model)
    scaler = build_scaler(amp_enabled, amp_dtype)
    before = capture_trainable_parameters(model)

    step = train_one_batch(
        model=model,
        inputs=inputs,
        target=target,
        optimizer=optimizer,
        scaler=scaler,
        config=config,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
    )
    update_report = count_updated_parameters(before, model)
    pass_status = all(
        [
            step["loss_finite"],
            step["prediction_finite"],
            step["gradient_summary"]["all_finite"],
            update_report["updated_parameter_tensors"] > 0,
        ]
    )

    payload = {
        "status": "PASS" if pass_status else "FAIL",
        "dataset_root": str(config.dataset_root),
        "split": args.split,
        "target_mode": args.target_mode,
        "device": str(device),
        "loss": step["loss"],
        "loss_finite": step["loss_finite"],
        "prediction_finite": step["prediction_finite"],
        "prediction_variance": step["prediction_variance"],
        "gradient_summary": step["gradient_summary"],
        "updated_parameters": update_report,
        "optimizer_step_successful": update_report["updated_parameter_tensors"] > 0,
        "elapsed_seconds": timer.elapsed(),
    }
    report_path = save_json_report(config, args.report_name, payload)
    append_csv_row(
        diagnostics_dir(config) / "sanity_summary.csv",
        {
            "test": args.report_name,
            "status": payload["status"],
            "loss": payload["loss"],
            "gradient_norm": payload["gradient_summary"]["total_norm"],
            "updated_parameter_tensors": update_report["updated_parameter_tensors"],
            "elapsed_seconds": payload["elapsed_seconds"],
        },
    )
    payload["report_path"] = str(report_path)
    print_json(payload)


if __name__ == "__main__":
    main()
