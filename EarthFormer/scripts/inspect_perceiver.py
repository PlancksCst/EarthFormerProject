"""Inspect tensor statistics inside the Perceiver forecasting readout."""

from __future__ import annotations

import torch

try:
    from .diagnostic_utils import (
        Timer,
        attention_tensors,
        autocast_dtype,
        build_model,
        forward_debug,
        load_batch,
        prepare_config,
        print_json,
        resolve_device,
        save_json_report,
        tensor_stats,
        use_amp,
    )
except ImportError:
    from diagnostic_utils import (  # type: ignore
        Timer,
        attention_tensors,
        autocast_dtype,
        build_model,
        forward_debug,
        load_batch,
        prepare_config,
        print_json,
        resolve_device,
        save_json_report,
        tensor_stats,
        use_amp,
    )

from configs.config import build_arg_parser, config_from_args


def main() -> None:
    """Run tensor inspection and write a JSON report."""
    parser = build_arg_parser()
    parser.description = "Inspect Perceiver readout tensor statistics."
    parser.add_argument("--split", default="train")
    parser.add_argument("--report-name", default="inspect_perceiver")
    args = parser.parse_args()

    config = prepare_config(config_from_args(args))
    device = resolve_device(config)
    amp_enabled = use_amp(config, device)
    amp_dtype = autocast_dtype(config, device)
    timer = Timer()

    batch = load_batch(config=config, split=args.split, device=device, include_target=False)
    inputs = batch["satellite"]
    model = build_model(config, device)
    model.eval()

    with torch.no_grad():
        result = forward_debug(
            model,
            inputs,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        attention = attention_tensors(model, result["pre_head_latent"])

    payload = {
        "status": "PASS",
        "dataset_root": str(config.dataset_root),
        "split": args.split,
        "device": str(device),
        "tensors": {
            "latent": tensor_stats(result["pre_head_latent"]),
            "query": tensor_stats(attention["query"]),
            "key": tensor_stats(attention["key"]),
            "value": tensor_stats(attention["value"]),
            "prediction": tensor_stats(result["prediction"]),
        },
        "elapsed_seconds": timer.elapsed(),
    }
    if not all(item["finite"] for item in payload["tensors"].values()):
        payload["status"] = "FAIL"

    report_path = save_json_report(config, args.report_name, payload)
    payload["report_path"] = str(report_path)
    print_json(payload)


if __name__ == "__main__":
    main()
