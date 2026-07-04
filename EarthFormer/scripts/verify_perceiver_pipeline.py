"""Verify one complete EarthFormer -> Perceiver forecasting forward pass."""

from __future__ import annotations

from argparse import ArgumentParser

import torch

try:
    from .diagnostic_utils import (
        Timer,
        attention_tensors,
        build_model,
        forward_debug,
        load_batch,
        prepare_config,
        print_json,
        resolve_device,
        save_json_report,
        tensor_finite,
        use_amp,
    )
except ImportError:
    from diagnostic_utils import (  # type: ignore
        Timer,
        attention_tensors,
        build_model,
        forward_debug,
        load_batch,
        prepare_config,
        print_json,
        resolve_device,
        save_json_report,
        tensor_finite,
        use_amp,
    )

from configs.config import build_arg_parser, config_from_args


def parse_args() -> tuple[ArgumentParser, object]:
    """Parse command-line arguments."""
    parser = build_arg_parser()
    parser.description = "Verify the complete EarthFormer + Perceiver forecasting pipeline."
    parser.add_argument("--split", default="train")
    parser.add_argument("--report-name", default="verify_perceiver_pipeline")
    return parser, parser.parse_args()


def main() -> None:
    """Run the verifier and write a JSON report."""
    _parser, args = parse_args()
    config = prepare_config(config_from_args(args))
    device = resolve_device(config)
    amp_enabled = use_amp(config, device)
    timer = Timer()

    batch = load_batch(config=config, split=args.split, device=device, include_target=False)
    inputs = batch["satellite"]
    model = build_model(config, device)
    model.eval()

    with torch.no_grad():
        result = forward_debug(model, inputs, device=device, amp_enabled=amp_enabled)
        attention = attention_tensors(model, result["pre_head_latent"])

    readout = result["readout"]
    payload = {
        "status": "PASS",
        "dataset_root": str(config.dataset_root),
        "split": args.split,
        "device": str(device),
        "input_tensor_shape": list(inputs.shape),
        "earthformer_latent_shape": list(result["earthformer_trace"]["after_cuboid_decoder"]),
        "pre_head_latent_shape": list(result["pre_head_latent"].shape),
        "perceiver_query_shape": list(readout["queries"].shape),
        "cross_attention_output_shape": list(readout["attention_output"].shape),
        "attention_weight_shape": list(attention["attention_weights"].shape),
        "final_prediction_shape": list(result["prediction"].shape),
        "prediction_dtype": str(result["prediction"].dtype),
        "prediction_finite": tensor_finite(result["prediction"]),
        "latent_finite": tensor_finite(result["pre_head_latent"]),
        "attention_finite": tensor_finite(readout["attention_output"]),
        "elapsed_seconds": timer.elapsed(),
    }
    if not all(
        [
            payload["prediction_finite"],
            payload["latent_finite"],
            payload["attention_finite"],
        ]
    ):
        payload["status"] = "FAIL"

    report_path = save_json_report(config, args.report_name, payload)
    payload["report_path"] = str(report_path)
    print_json(payload)


if __name__ == "__main__":
    main()
