"""Tiny-dataset overfit test for the forecasting model."""

from __future__ import annotations
from tqdm.auto import tqdm
try:
    from .diagnostic_utils import (
        Timer,
        append_csv_row,
        build_model,
        build_optimizer,
        build_scaler,
        build_scheduler,
        diagnostics_dir,
        make_sanity_target,
        prepare_config,
        print_json,
        resolve_device,
        save_json_report,
        tiny_dataloader,
        train_one_batch,
        use_amp,
    )
except ImportError:
    from diagnostic_utils import (  # type: ignore
        Timer,
        append_csv_row,
        build_model,
        build_optimizer,
        build_scaler,
        build_scheduler,
        diagnostics_dir,
        make_sanity_target,
        prepare_config,
        print_json,
        resolve_device,
        save_json_report,
        tiny_dataloader,
        train_one_batch,
        use_amp,
    )

from configs.config import build_arg_parser, config_from_args


def main() -> None:
    """Train on a tiny subset until the sanity loss is nearly zero."""
    parser = build_arg_parser()
    parser.description = "Run a tiny overfit test for the forecasting model."
    parser.add_argument("--split", default="train")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=1.0e-3)
    parser.add_argument("--target-mode", choices=["satellite_mean", "zeros"], default="satellite_mean")
    parser.add_argument("--report-name", default="test_overfit")
    args = parser.parse_args()

    config = prepare_config(config_from_args(args))
    device = resolve_device(config)
    amp_enabled = use_amp(config, device)
    timer = Timer()

    loader = tiny_dataloader(config=config, split=args.split, samples=args.samples)
    model = build_model(config, device)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer, epochs=args.max_epochs)
    scaler = build_scaler(amp_enabled)

    epoch_rows: list[dict[str, float | int | str]] = []
    passed = False
    final_loss = float("inf")
    final_prediction_variance = 0.0
    final_gradient_norm = 0.0
    csv_path = diagnostics_dir(config) / "overfit_history.csv"

    epoch_bar = tqdm(
        range(1, args.max_epochs + 1),
        desc="Overfit",
        dynamic_ncols=True,
    )
    
    for epoch in epoch_bar:
        total_loss = 0.0
        total_samples = 0
        prediction_variance = 0.0
        gradient_norm = 0.0
        all_finite = True

        for batch_idx, batch in enumerate(loader):
            inputs = batch["satellite"].to(device, non_blocking=True)
            target = make_sanity_target(inputs, output_length=config.output_length, mode=args.target_mode)
            step = train_one_batch(
                model=model,
                inputs=inputs,
                target=target,
                optimizer=optimizer,
                scaler=scaler,
                config=config,
                device=device,
                amp_enabled=amp_enabled,
            )
            batch_size = inputs.shape[0]
            total_loss += step["loss"] * batch_size
            total_samples += batch_size
            prediction_variance = step["prediction_variance"]
            gradient_norm = step["gradient_summary"]["total_norm"]
            all_finite = all_finite and step["loss_finite"] and step["prediction_finite"]
            all_finite = all_finite and step["gradient_summary"]["all_finite"]

        scheduler.step()
        final_loss = total_loss / max(1, total_samples)
        final_prediction_variance = prediction_variance
        final_gradient_norm = gradient_norm

        epoch_bar.set_postfix(
            loss=f"{final_loss:.5f}",
            grad=f"{gradient_norm:.2f}",
            var=f"{prediction_variance:.5f}",
        )
        
        row = {
            "epoch": epoch,
            "loss": final_loss,
            "prediction_variance": final_prediction_variance,
            "gradient_norm": final_gradient_norm,
            "all_finite": str(all_finite),
        }
        epoch_rows.append(row)
        append_csv_row(csv_path, row)
        print(
            f"epoch={epoch:03d} loss={final_loss:.6f} "
            f"prediction_variance={final_prediction_variance:.6f} "
            f"gradient_norm={final_gradient_norm:.6f}"
        )

        if all_finite and final_loss < args.threshold:
            passed = True
            epoch_bar.close()
            break
        if not all_finite:
            break

    payload = {
        "status": "PASS" if passed else "FAIL",
        "dataset_root": str(config.dataset_root),
        "split": args.split,
        "samples": args.samples,
        "target_mode": args.target_mode,
        "threshold": args.threshold,
        "max_epochs": args.max_epochs,
        "epochs_run": len(epoch_rows),
        "final_loss": final_loss,
        "final_prediction_variance": final_prediction_variance,
        "final_gradient_norm": final_gradient_norm,
        "history_csv": str(csv_path),
        "elapsed_seconds": timer.elapsed(),
    }
    report_path = save_json_report(config, args.report_name, payload)
    append_csv_row(
        diagnostics_dir(config) / "sanity_summary.csv",
        {
            "test": args.report_name,
            "status": payload["status"],
            "loss": final_loss,
            "gradient_norm": final_gradient_norm,
            "updated_parameter_tensors": "",
            "elapsed_seconds": payload["elapsed_seconds"],
        },
    )
    payload["report_path"] = str(report_path)
    print_json(payload)


if __name__ == "__main__":
    main()
