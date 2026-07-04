"""Checkpoint resume sanity test for the forecasting model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from .diagnostic_utils import (
        Timer,
        append_csv_row,
        build_model,
        build_optimizer,
        build_scaler,
        build_scheduler,
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
        build_model,
        build_optimizer,
        build_scaler,
        build_scheduler,
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
from training.checkpoint import resume_checkpoint, save_checkpoint


def optimizer_summary(optimizer: Any) -> dict[str, Any]:
    """Return a compact optimizer-state summary."""
    state = optimizer.state_dict()
    return {
        "param_groups": len(state["param_groups"]),
        "learning_rates": [float(group["lr"]) for group in state["param_groups"]],
        "state_entries": len(state["state"]),
    }


def scheduler_summary(scheduler: Any) -> dict[str, Any]:
    """Return a compact scheduler-state summary."""
    state = scheduler.state_dict()
    return {
        "last_epoch": int(state.get("last_epoch", -1)),
        "state_keys": sorted(state.keys()),
    }


def scaler_summary(scaler: Any) -> dict[str, Any]:
    """Return a compact AMP scaler-state summary."""
    state = scaler.state_dict()
    return {
        "enabled_state_keys": sorted(state.keys()),
        "scale": float(state["scale"]) if "scale" in state else None,
        "growth_tracker": int(state["_growth_tracker"]) if "_growth_tracker" in state else None,
    }


def main() -> None:
    """Train, save, reload, continue, and validate restored state."""
    parser = build_arg_parser()
    parser.description = "Run a checkpoint resume sanity test."
    parser.add_argument("--split", default="train")
    parser.add_argument("--steps-before-save", type=int, default=1)
    parser.add_argument("--steps-after-resume", type=int, default=1)
    parser.add_argument("--target-mode", choices=["satellite_mean", "zeros"], default="satellite_mean")
    parser.add_argument("--report-name", default="test_resume")
    args = parser.parse_args()

    config = prepare_config(config_from_args(args))
    device = resolve_device(config)
    amp_enabled = use_amp(config, device)
    timer = Timer()

    batch = load_batch(config=config, split=args.split, device=device, include_target=False)
    inputs = batch["satellite"]
    target = make_sanity_target(inputs, output_length=config.output_length, mode=args.target_mode)

    model = build_model(config, device)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer, epochs=args.steps_before_save + args.steps_after_resume + 1)
    scaler = build_scaler(amp_enabled)

    pre_save_loss = float("inf")
    for _step in range(args.steps_before_save):
        step_report = train_one_batch(
            model=model,
            inputs=inputs,
            target=target,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            device=device,
            amp_enabled=amp_enabled,
        )
        pre_save_loss = step_report["loss"]
        scheduler.step()

    saved_optimizer = optimizer_summary(optimizer)
    saved_scheduler = scheduler_summary(scheduler)
    saved_scaler = scaler_summary(scaler)
    checkpoint_path = diagnostics_dir(config) / "resume_test_checkpoint.pt"
    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=args.steps_before_save,
        best_loss=pre_save_loss,
    )

    resumed_model = build_model(config, device)
    resumed_optimizer = build_optimizer(config, resumed_model)
    resumed_scheduler = build_scheduler(
        config,
        resumed_optimizer,
        epochs=args.steps_before_save + args.steps_after_resume + 1,
    )
    resumed_scaler = build_scaler(amp_enabled)
    next_epoch, best_loss = resume_checkpoint(
        path=checkpoint_path,
        model=resumed_model,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        scaler=resumed_scaler,
        map_location=device,
    )

    restored_optimizer = optimizer_summary(resumed_optimizer)
    restored_scheduler = scheduler_summary(resumed_scheduler)
    restored_scaler = scaler_summary(resumed_scaler)

    post_resume_loss = float("inf")
    post_resume_gradient_norm = 0.0
    for _step in range(args.steps_after_resume):
        resumed_step = train_one_batch(
            model=resumed_model,
            inputs=inputs,
            target=target,
            optimizer=resumed_optimizer,
            scaler=resumed_scaler,
            config=config,
            device=device,
            amp_enabled=amp_enabled,
        )
        post_resume_loss = resumed_step["loss"]
        post_resume_gradient_norm = resumed_step["gradient_summary"]["total_norm"]
        resumed_scheduler.step()

    optimizer_restored = restored_optimizer == saved_optimizer
    scheduler_restored = restored_scheduler == saved_scheduler
    scaler_restored = restored_scaler == saved_scaler
    epoch_restored = next_epoch == args.steps_before_save + 1
    smooth_loss = post_resume_loss <= max(pre_save_loss * 10.0, pre_save_loss + 1.0, 1.0e-6)
    finite_loss = post_resume_loss == post_resume_loss and post_resume_loss != float("inf")
    passed = all(
        [
            optimizer_restored,
            scheduler_restored,
            scaler_restored,
            epoch_restored,
            smooth_loss,
            finite_loss,
        ]
    )

    payload = {
        "status": "PASS" if passed else "FAIL",
        "dataset_root": str(config.dataset_root),
        "split": args.split,
        "target_mode": args.target_mode,
        "checkpoint_path": str(Path(checkpoint_path)),
        "pre_save_loss": pre_save_loss,
        "post_resume_loss": post_resume_loss,
        "post_resume_gradient_norm": post_resume_gradient_norm,
        "best_loss_restored": best_loss,
        "optimizer_restored": optimizer_restored,
        "scheduler_restored": scheduler_restored,
        "grad_scaler_restored": scaler_restored,
        "epoch_restored": epoch_restored,
        "next_epoch": next_epoch,
        "loss_continues_smoothly": smooth_loss,
        "saved_optimizer": saved_optimizer,
        "restored_optimizer": restored_optimizer,
        "saved_scheduler": saved_scheduler,
        "restored_scheduler": restored_scheduler,
        "saved_scaler": saved_scaler,
        "restored_scaler": restored_scaler,
        "elapsed_seconds": timer.elapsed(),
    }
    report_path = save_json_report(config, args.report_name, payload)
    append_csv_row(
        diagnostics_dir(config) / "sanity_summary.csv",
        {
            "test": args.report_name,
            "status": payload["status"],
            "loss": post_resume_loss,
            "gradient_norm": post_resume_gradient_norm,
            "updated_parameter_tensors": "",
            "elapsed_seconds": payload["elapsed_seconds"],
        },
    )
    payload["report_path"] = str(report_path)
    print_json(payload)


if __name__ == "__main__":
    main()
