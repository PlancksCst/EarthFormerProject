"""Evaluate the local crop model against simple CSI baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREP_MODELS_ROOT = PROJECT_ROOT.parent
for path in (PROJECT_ROOT, PREP_MODELS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from configs.config import TrainingConfig  # noqa: E402
from local_crop_pipeline.local_crop_dataset import (  # noqa: E402
    LocalCropDataset,
    build_local_crop_dataloader,
)
from local_crop_pipeline.local_crop_model import LocalCropCNNGRU  # noqa: E402
from local_crop_pipeline.station_crop_mapping import CropBounds  # noqa: E402
from local_crop_pipeline.train_local_crop_model import (  # noqa: E402
    metadata_value,
    prediction_rows_from_batch,
    resolve_device,
    save_csv,
    scalar_metrics,
)
from models.model import build_perceiver_readout_model  # noqa: E402
from training.baselines import ClimatologyBaseline  # noqa: E402
from training.losses import valid_hour_mask  # noqa: E402
from training.validate import ensure_forecast_target, reconstruct_ghi  # noqa: E402


def make_dataset(args: argparse.Namespace, split: str, include_auxiliary_features: bool) -> LocalCropDataset:
    bounds = CropBounds(
        lat_min=args.crop_lat_min,
        lat_max=args.crop_lat_max,
        lon_min=args.crop_lon_min,
        lon_max=args.crop_lon_max,
    )
    return LocalCropDataset(
        dataset_root=args.dataset_root,
        split=split,
        local_crop_size=args.local_crop_size,
        crop_padding_mode=args.crop_padding_mode,
        crop_bounds=bounds,
        locations_csv=args.locations_csv,
        include_target=True,
        include_auxiliary_features=include_auxiliary_features,
        metadata_filename=args.metadata_filename,
        hourly_csv=args.hourly_csv,
        elevation_csv=args.elevation_csv,
        normalize=not args.no_normalize,
    )


def checkpoint_model_args(checkpoint: dict[str, Any]) -> dict[str, Any]:
    raw_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    return dict(raw_args) if isinstance(raw_args, dict) else {}


def load_local_model(path: Path, device: torch.device, cli_args: argparse.Namespace) -> LocalCropCNNGRU:
    checkpoint = torch.load(path, map_location=device)
    saved_args = checkpoint_model_args(checkpoint)
    model = LocalCropCNNGRU(
        input_channels=7,
        output_length=13,
        cnn_feature_dim=int(saved_args.get("cnn_feature_dim", cli_args.cnn_feature_dim)),
        gru_hidden_dim=int(saved_args.get("gru_hidden_dim", cli_args.gru_hidden_dim)),
        use_auxiliary_features=str(saved_args.get("use_auxiliary_features", cli_args.use_auxiliary_features))
        in {"True", "true", "1"},
        auxiliary_dim=int(saved_args.get("auxiliary_dim", 9)),
        dropout=float(saved_args.get("dropout", cli_args.dropout)),
    ).to(device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def evaluate_local_model(
    model: LocalCropCNNGRU,
    dataloader,
    device: torch.device,
    clear_sky_threshold: float,
    use_auxiliary_features: bool,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    pred: list[torch.Tensor] = []
    target: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    for batch in dataloader:
        inputs = batch["satellite"].to(device).float()
        targets = ensure_forecast_target(batch["target"]).to(device)
        clear = ensure_forecast_target(batch["clear_sky_ghi"], "clear_sky_ghi").to(device)
        target_ghi_value = batch.get("target_ghi")
        target_ghi = (
            reconstruct_ghi(targets, clear)
            if target_ghi_value is None
            else ensure_forecast_target(target_ghi_value, "target_ghi").to(device)
        )
        target_mask = batch.get("target_mask")
        if isinstance(target_mask, torch.Tensor):
            target_mask = target_mask.to(device)
        valid = valid_hour_mask(target_mask, targets, clear, clear_sky_threshold)
        aux = None
        if use_auxiliary_features:
            aux_value = batch.get("auxiliary_features", batch.get("aux_features"))
            if not isinstance(aux_value, torch.Tensor):
                raise KeyError("Local crop checkpoint expects auxiliary features.")
            aux = aux_value.to(device).float()
        prediction = model(inputs, auxiliary_features=aux)
        valid_cpu = valid.detach().cpu()
        if int(valid_cpu.sum()) > 0:
            pred.append(prediction.detach().cpu()[valid_cpu])
            target.append(targets.detach().cpu()[valid_cpu])
        rows.extend(prediction_rows_from_batch(batch, prediction, targets, clear, target_ghi, valid))
    metrics = scalar_metrics(torch.cat(pred), torch.cat(target), "CSI")
    return metrics, rows


@torch.no_grad()
def evaluate_prediction_tensor(
    dataloader,
    device: torch.device,
    clear_sky_threshold: float,
    predictor,
) -> dict[str, float]:
    pred: list[torch.Tensor] = []
    target: list[torch.Tensor] = []
    for batch in dataloader:
        targets = ensure_forecast_target(batch["target"]).to(device)
        clear = ensure_forecast_target(batch["clear_sky_ghi"], "clear_sky_ghi").to(device)
        target_mask = batch.get("target_mask")
        if isinstance(target_mask, torch.Tensor):
            target_mask = target_mask.to(device)
        valid = valid_hour_mask(target_mask, targets, clear, clear_sky_threshold)
        prediction = predictor(batch, targets, clear).to(device)
        valid_cpu = valid.detach().cpu()
        if int(valid_cpu.sum()) > 0:
            pred.append(prediction.detach().cpu()[valid_cpu])
            target.append(targets.detach().cpu()[valid_cpu])
    return scalar_metrics(torch.cat(pred), torch.cat(target), "CSI")


def evaluate_persistence(dataloader, device: torch.device, clear_sky_threshold: float) -> dict[str, float] | None:
    def predictor(batch, targets, clear):
        value = batch.get("previous_day_csi")
        if not isinstance(value, torch.Tensor):
            raise KeyError
        return value.to(device=device, dtype=targets.dtype)

    try:
        return evaluate_prediction_tensor(dataloader, device, clear_sky_threshold, predictor)
    except KeyError:
        return None


def evaluate_earthformer_checkpoint(args: argparse.Namespace, device: torch.device) -> dict[str, float] | None:
    if args.earthformer_checkpoint is None:
        return None
    cfg = TrainingConfig(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=str(device),
        checkpoint_dir=args.output_dir / "earthformer_unused_checkpoints",
        output_dir=args.output_dir / "earthformer_unused_outputs",
    )
    cfg.use_auxiliary_features = args.earthformer_use_auxiliary_features
    cfg.pretrained_checkpoint = args.pretrained_earthformer_checkpoint
    base_dataset = cfg.dataset_root
    del base_dataset
    model = build_perceiver_readout_model(cfg).to(device)
    checkpoint = torch.load(args.earthformer_checkpoint, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state, strict=False)
    model.eval()
    eval_dataset = make_dataset(args, args.eval_split, include_auxiliary_features=args.earthformer_use_auxiliary_features)
    dataloader = build_local_crop_dataloader(
        eval_dataset.base_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device.type,
    )

    def predictor(batch, targets, clear):
        inputs = batch["satellite"].to(device).float()
        aux = None
        if args.earthformer_use_auxiliary_features:
            aux_value = batch.get("auxiliary_features", batch.get("aux_features"))
            aux = aux_value.to(device).float() if isinstance(aux_value, torch.Tensor) else None
        return model(inputs, auxiliary_features=aux)

    return evaluate_prediction_tensor(dataloader, device, args.clear_sky_threshold, predictor)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate local crop model against baselines.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--eval-split", type=str, default="val")
    parser.add_argument("--metadata-filename", type=str, default=None)
    parser.add_argument("--hourly-csv", type=Path, default=None)
    parser.add_argument("--elevation-csv", type=Path, default=None)
    parser.add_argument("--locations-csv", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--local-crop-size", type=int, default=64)
    parser.add_argument("--crop-padding-mode", choices=("edge", "reflect"), default="edge")
    parser.add_argument("--crop-lat-min", type=float, default=CropBounds.lat_min)
    parser.add_argument("--crop-lat-max", type=float, default=CropBounds.lat_max)
    parser.add_argument("--crop-lon-min", type=float, default=CropBounds.lon_min)
    parser.add_argument("--crop-lon-max", type=float, default=CropBounds.lon_max)
    parser.add_argument("--clear-sky-threshold", type=float, default=20.0)
    parser.add_argument("--earthformer-checkpoint", type=Path, default=None)
    parser.add_argument("--pretrained-earthformer-checkpoint", type=Path, default=None)
    parser.add_argument("--earthformer-use-auxiliary-features", action="store_true")
    parser.add_argument("--cnn-feature-dim", type=int, default=128)
    parser.add_argument("--gru-hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use-auxiliary-features", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    saved_args = checkpoint_model_args(checkpoint)
    use_aux = str(saved_args.get("use_auxiliary_features", args.use_auxiliary_features)) in {"True", "true", "1"}
    train_dataset = make_dataset(args, args.train_split, include_auxiliary_features=True)
    eval_dataset = make_dataset(args, args.eval_split, include_auxiliary_features=True or use_aux)
    eval_loader = build_local_crop_dataloader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device.type,
    )
    model = load_local_model(args.checkpoint, device, args)
    local_metrics, rows = evaluate_local_model(
        model,
        eval_loader,
        device,
        args.clear_sky_threshold,
        use_auxiliary_features=use_aux,
    )
    save_csv(args.output_dir / "prediction.csv", rows)

    hourly = ClimatologyBaseline.from_dataset(
        train_dataset,
        output_length=13,
        mode="hourly_climatology",
        clear_sky_threshold=args.clear_sky_threshold,
    )
    location_hour = ClimatologyBaseline.from_dataset(
        train_dataset,
        output_length=13,
        mode="location_hour_climatology",
        clear_sky_threshold=args.clear_sky_threshold,
    )
    hourly_metrics = evaluate_prediction_tensor(
        eval_loader,
        device,
        args.clear_sky_threshold,
        lambda batch, targets, clear: hourly.predict(batch, targets.shape[1], device, targets.dtype),
    )
    location_metrics = evaluate_prediction_tensor(
        eval_loader,
        device,
        args.clear_sky_threshold,
        lambda batch, targets, clear: location_hour.predict(batch, targets.shape[1], device, targets.dtype),
    )
    persistence_metrics = evaluate_persistence(eval_loader, device, args.clear_sky_threshold)
    earthformer_metrics = evaluate_earthformer_checkpoint(args, device)

    rows_for_csv = [
        {"method": "local_crop", **local_metrics},
        {"method": "hourly_climatology", **hourly_metrics},
        {"method": "location_hour_climatology", **location_metrics},
    ]
    if persistence_metrics is not None:
        rows_for_csv.append({"method": "previous_day_csi_persistence", **persistence_metrics})
    if earthformer_metrics is not None:
        rows_for_csv.append({"method": "earthformer_200px", **earthformer_metrics})
    save_csv(args.output_dir / "baseline_comparison_metrics.csv", rows_for_csv)

    local_rmse = float(local_metrics["CSI_RMSE"])
    baseline_rmses = [
        float(hourly_metrics["CSI_RMSE"]),
        float(location_metrics["CSI_RMSE"]),
    ]
    if persistence_metrics is not None:
        baseline_rmses.append(float(persistence_metrics["CSI_RMSE"]))
    beats_baselines = all(local_rmse < value for value in baseline_rmses)
    beats_earthformer = (
        None if earthformer_metrics is None else local_rmse < float(earthformer_metrics["CSI_RMSE"])
    )
    if beats_baselines and (beats_earthformer is None or beats_earthformer):
        interpretation = "local_crop_signal_is_promising"
    elif beats_baselines:
        interpretation = "local_crop_beats_simple_baselines_but_not_200px_earthformer"
    else:
        interpretation = "local_crop_does_not_yet_beat_simple_baselines"
    summary = {
        "local_crop_CSI_RMSE": local_rmse,
        "hourly_climatology_CSI_RMSE": float(hourly_metrics["CSI_RMSE"]),
        "location_hour_climatology_CSI_RMSE": float(location_metrics["CSI_RMSE"]),
        "persistence_CSI_RMSE": None
        if persistence_metrics is None
        else float(persistence_metrics["CSI_RMSE"]),
        "earthformer_200px_CSI_RMSE": None
        if earthformer_metrics is None
        else float(earthformer_metrics["CSI_RMSE"]),
        "local_crop_beats_baselines": bool(beats_baselines),
        "local_crop_beats_200px_earthformer": beats_earthformer,
        "recommended_interpretation": interpretation,
    }
    with (args.output_dir / "local_crop_baseline_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
