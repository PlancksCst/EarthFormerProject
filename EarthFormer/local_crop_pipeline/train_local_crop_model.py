"""Train a standalone CNN-GRU model on station-centered local crops."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREP_MODELS_ROOT = PROJECT_ROOT.parent
for path in (PROJECT_ROOT, PREP_MODELS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from local_crop_pipeline.local_crop_dataset import (  # noqa: E402
    LocalCropDataset,
    build_local_crop_dataloader,
)
from local_crop_pipeline.local_crop_model import LocalCropCNNGRU  # noqa: E402
from local_crop_pipeline.station_crop_mapping import CropBounds  # noqa: E402
from training.losses import (  # noqa: E402
    cloudy_csi_weights,
    masked_huber_loss,
    masked_mse_loss,
    masked_weighted_huber_loss,
    valid_hour_mask,
)
from training.validate import ensure_forecast_target, reconstruct_ghi  # noqa: E402
from utils.metrics import forecast_metrics  # noqa: E402


LOSS_CHOICES = ("masked_mse", "masked_huber", "masked_weighted_huber")


def make_grad_scaler(device: torch.device, enabled: bool):
    """Create an AMP grad scaler without deprecated CUDA-only calls."""
    use_amp = bool(enabled and device.type == "cuda")
    try:
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=use_amp)


def amp_autocast(device: torch.device, enabled: bool):
    """Return an autocast context without deprecated CUDA-only calls."""
    use_amp = bool(enabled and device.type == "cuda")
    try:
        return torch.amp.autocast(device_type=device.type, enabled=use_amp)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=use_amp)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def metadata_value(batch: dict[str, Any], key: str, index: int) -> Any:
    value = batch.get(key)
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        item = value[index]
        return item.item() if item.numel() == 1 else item.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return value[index] if index < len(value) else None
    return value


def compute_loss(
    loss_name: str,
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    huber_beta: float,
    cloudy_weight: float,
) -> torch.Tensor:
    if loss_name == "masked_mse":
        return masked_mse_loss(prediction, target, valid_mask=valid_mask)
    if loss_name == "masked_huber":
        return masked_huber_loss(prediction, target, valid_mask=valid_mask, beta=huber_beta)
    if loss_name == "masked_weighted_huber":
        weights = cloudy_csi_weights(target, cloudy_weight)
        return masked_weighted_huber_loss(
            prediction,
            target,
            valid_mask=valid_mask,
            weights=weights,
            beta=huber_beta,
        )
    raise ValueError(f"Unsupported loss: {loss_name}")


def scalar_metrics(prediction: torch.Tensor, target: torch.Tensor, prefix: str = "CSI") -> dict[str, float]:
    return forecast_metrics(prediction.detach().cpu(), target.detach().cpu(), prefix=prefix)


def make_datasets(args: argparse.Namespace) -> tuple[LocalCropDataset, LocalCropDataset]:
    bounds = CropBounds(
        lat_min=args.crop_lat_min,
        lat_max=args.crop_lat_max,
        lon_min=args.crop_lon_min,
        lon_max=args.crop_lon_max,
    )
    common = {
        "dataset_root": args.dataset_root,
        "local_crop_size": args.local_crop_size,
        "crop_padding_mode": args.crop_padding_mode,
        "crop_bounds": bounds,
        "locations_csv": args.locations_csv,
        "include_target": True,
        "include_auxiliary_features": args.use_auxiliary_features,
        "metadata_filename": args.metadata_filename,
        "hourly_csv": args.hourly_csv,
        "elevation_csv": args.elevation_csv,
        "normalize": not args.no_normalize,
    }
    return (
        LocalCropDataset(split=args.train_split, **common),
        LocalCropDataset(split=args.val_split, **common),
    )


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prediction_rows_from_batch(
    batch: dict[str, Any],
    prediction: torch.Tensor,
    target: torch.Tensor,
    clear_sky_ghi: torch.Tensor,
    target_ghi: torch.Tensor,
    valid_mask: torch.Tensor,
) -> list[dict[str, Any]]:
    pred_cpu = prediction.detach().float().cpu()
    target_cpu = target.detach().float().cpu()
    clear_cpu = clear_sky_ghi.detach().float().cpu()
    target_ghi_cpu = target_ghi.detach().float().cpu()
    pred_ghi_cpu = reconstruct_ghi(prediction, clear_sky_ghi).detach().float().cpu()
    valid_cpu = valid_mask.detach().cpu().bool()
    rows: list[dict[str, Any]] = []
    batch_size, horizon = pred_cpu.shape
    for sample_index in range(batch_size):
        for hour_index in range(horizon):
            rows.append(
                {
                    "sample_id": metadata_value(batch, "sample_id", sample_index),
                    "location": metadata_value(batch, "location", sample_index),
                    "hour_index": hour_index,
                    "target_csi": float(target_cpu[sample_index, hour_index]),
                    "pred_csi": float(pred_cpu[sample_index, hour_index]),
                    "clear_sky_ghi": float(clear_cpu[sample_index, hour_index]),
                    "target_ghi": float(target_ghi_cpu[sample_index, hour_index]),
                    "pred_ghi": float(pred_ghi_cpu[sample_index, hour_index]),
                    "valid_mask": bool(valid_cpu[sample_index, hour_index]),
                    "local_crop_center_y": metadata_value(batch, "local_crop_center_y", sample_index),
                    "local_crop_center_x": metadata_value(batch, "local_crop_center_x", sample_index),
                }
            )
    return rows


def aggregate_prediction_metrics(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_location: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_hour: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row["valid_mask"]:
            continue
        by_location[str(row["location"])].append(row)
        by_hour[int(row["hour_index"])].append(row)

    def metrics_for(group_rows: list[dict[str, Any]]) -> dict[str, float]:
        pred = torch.tensor([float(row["pred_csi"]) for row in group_rows])
        target = torch.tensor([float(row["target_csi"]) for row in group_rows])
        return scalar_metrics(pred, target, "CSI")

    location_rows = [
        {"location": location, "count": len(group), **metrics_for(group)}
        for location, group in sorted(by_location.items())
        if group
    ]
    hour_rows = [
        {"hour_index": hour, "count": len(group), **metrics_for(group)}
        for hour, group in sorted(by_hour.items())
        if group
    ]
    return location_rows, hour_rows


def save_plots(output_dir: Path, history: list[dict[str, Any]], prediction_rows: list[dict[str, Any]]) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, [row["train_loss"] for row in history], label="train")
    plt.plot(epochs, [row["val_loss"] for row in history], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / "loss_curve.png", dpi=150)
    plt.close()

    valid = [row for row in prediction_rows if row["valid_mask"]]
    if not valid:
        return
    target = torch.tensor([float(row["target_csi"]) for row in valid])
    pred = torch.tensor([float(row["pred_csi"]) for row in valid])
    residual = pred - target

    plt.figure(figsize=(5, 5))
    plt.scatter(target.numpy(), pred.numpy(), s=5, alpha=0.4)
    lim = [0.0, max(1.3, float(target.max()), float(pred.max()))]
    plt.plot(lim, lim, color="black", linewidth=1)
    plt.xlabel("target CSI")
    plt.ylabel("predicted CSI")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / "scatter_csi.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(target.numpy(), bins=40, alpha=0.5, label="target")
    plt.hist(pred.numpy(), bins=40, alpha=0.5, label="prediction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "csi_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(residual.numpy(), bins=40, alpha=0.8)
    plt.xlabel("predicted CSI - target CSI")
    plt.tight_layout()
    plt.savefig(plot_dir / "residual_histogram.png", dpi=150)
    plt.close()

    first_sample = valid[0]["sample_id"]
    sample_rows = [row for row in prediction_rows if row["sample_id"] == first_sample]
    sample_rows = sorted(sample_rows, key=lambda row: int(row["hour_index"]))
    plt.figure(figsize=(7, 4))
    plt.plot([row["hour_index"] for row in sample_rows], [row["target_csi"] for row in sample_rows], label="target")
    plt.plot([row["hour_index"] for row in sample_rows], [row["pred_csi"] for row in sample_rows], label="prediction")
    plt.xlabel("hour index")
    plt.ylabel("CSI")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / "target_vs_prediction_curves.png", dpi=150)
    plt.close()


class LocalCropTrainer:
    """Standalone trainer for the local crop CNN-GRU."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = resolve_device(args.device)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        train_dataset, val_dataset = make_datasets(args)
        self.train_loader = build_local_crop_dataloader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            device=self.device.type,
        )
        self.val_loader = build_local_crop_dataloader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            device=self.device.type,
        )
        self.model = LocalCropCNNGRU(
            input_channels=7,
            output_length=13,
            cnn_feature_dim=args.cnn_feature_dim,
            gru_hidden_dim=args.gru_hidden_dim,
            use_auxiliary_features=args.use_auxiliary_features,
            auxiliary_dim=args.auxiliary_dim,
            dropout=args.dropout,
        ).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.scaler = make_grad_scaler(self.device, enabled=args.amp)
        self.history: list[dict[str, Any]] = []
        self.best_val = float("inf")

    def _batch_tensors(self, batch: dict[str, Any]) -> tuple[torch.Tensor, ...]:
        inputs = batch["satellite"].to(self.device, non_blocking=True).float()
        targets = ensure_forecast_target(batch["target"]).to(self.device, non_blocking=True)
        clear = ensure_forecast_target(batch["clear_sky_ghi"], "clear_sky_ghi").to(self.device, non_blocking=True)
        target_ghi_value = batch.get("target_ghi")
        if target_ghi_value is None:
            target_ghi = reconstruct_ghi(targets, clear)
        else:
            target_ghi = ensure_forecast_target(target_ghi_value, "target_ghi").to(self.device, non_blocking=True)
        target_mask = batch.get("target_mask")
        if isinstance(target_mask, torch.Tensor):
            target_mask = target_mask.to(self.device, non_blocking=True)
        valid = valid_hour_mask(
            target_mask=target_mask,
            reference=targets,
            clear_sky_ghi=clear,
            clear_sky_threshold=self.args.clear_sky_threshold,
        )
        return inputs, targets, clear, target_ghi, valid

    def _auxiliary(self, batch: dict[str, Any]) -> torch.Tensor | None:
        if not self.args.use_auxiliary_features:
            return None
        value = batch.get("auxiliary_features", batch.get("aux_features"))
        if not isinstance(value, torch.Tensor):
            raise KeyError("Auxiliary features enabled but missing from batch.")
        return value.to(self.device, non_blocking=True).float()

    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total = 0.0
        count = 0
        progress = tqdm(self.train_loader, desc=f"Local crop epoch {epoch}/{self.args.epochs}", leave=False)
        for batch in progress:
            inputs, targets, _clear, _target_ghi, valid = self._batch_tensors(batch)
            valid_count = int(valid.sum().detach().cpu())
            if valid_count == 0:
                continue
            self.optimizer.zero_grad(set_to_none=True)
            with amp_autocast(self.device, enabled=self.args.amp):
                prediction = self.model(inputs, auxiliary_features=self._auxiliary(batch))
                loss = compute_loss(
                    self.args.loss,
                    prediction,
                    targets,
                    valid,
                    self.args.huber_beta,
                    self.args.cloudy_weight,
                )
            self.scaler.scale(loss).backward()
            if self.args.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total += float(loss.detach().cpu()) * valid_count
            count += valid_count
            progress.set_postfix(loss=f"{float(loss.detach().cpu()):.5f}")
        if count == 0:
            raise ValueError("Training dataloader produced no valid target hours.")
        return total / count

    @torch.no_grad()
    def validate(self) -> tuple[dict[str, float], list[dict[str, Any]]]:
        self.model.eval()
        total = 0.0
        count = 0
        pred_csi: list[torch.Tensor] = []
        target_csi: list[torch.Tensor] = []
        pred_ghi: list[torch.Tensor] = []
        target_ghi_values: list[torch.Tensor] = []
        rows: list[dict[str, Any]] = []
        for batch in self.val_loader:
            inputs, targets, clear, target_ghi, valid = self._batch_tensors(batch)
            prediction = self.model(inputs, auxiliary_features=self._auxiliary(batch))
            valid_count = int(valid.sum().detach().cpu())
            if valid_count > 0:
                loss = compute_loss(
                    self.args.loss,
                    prediction,
                    targets,
                    valid,
                    self.args.huber_beta,
                    self.args.cloudy_weight,
                )
                total += float(loss.detach().cpu()) * valid_count
                count += valid_count
                valid_cpu = valid.detach().cpu()
                pred_csi.append(prediction.detach().cpu()[valid_cpu])
                target_csi.append(targets.detach().cpu()[valid_cpu])
                pred_ghi.append(reconstruct_ghi(prediction, clear).detach().cpu()[valid_cpu])
                target_ghi_values.append(target_ghi.detach().cpu()[valid_cpu])
            rows.extend(prediction_rows_from_batch(batch, prediction, targets, clear, target_ghi, valid))
        if count == 0:
            raise ValueError("Validation dataloader produced no valid target hours.")
        metrics = {"val_loss": total / count}
        metrics.update(scalar_metrics(torch.cat(pred_csi), torch.cat(target_csi), "CSI"))
        metrics.update(scalar_metrics(torch.cat(pred_ghi), torch.cat(target_ghi_values), "GHI"))
        return metrics, rows

    def save_checkpoint(self, path: Path, epoch: int, val_loss: float) -> None:
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "args": vars(self.args),
            },
            path,
        )

    def fit(self) -> None:
        for epoch in range(1, self.args.epochs + 1):
            started = time.perf_counter()
            train_loss = self.train_one_epoch(epoch)
            metrics, prediction_rows = self.validate()
            val_loss = float(metrics["val_loss"])
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                **metrics,
                "epoch_time": time.perf_counter() - started,
            }
            self.history.append(row)
            save_csv(self.args.output_dir / "training_history.csv", self.history)
            self.save_checkpoint(self.args.checkpoint_dir / "last.pt", epoch, val_loss)
            if val_loss < self.best_val:
                self.best_val = val_loss
                self.save_checkpoint(self.args.checkpoint_dir / "best.pt", epoch, val_loss)
                save_csv(self.args.output_dir / "prediction.csv", prediction_rows)
                per_location, per_hour = aggregate_prediction_metrics(prediction_rows)
                save_csv(self.args.output_dir / "metrics_per_location.csv", per_location)
                save_csv(self.args.output_dir / "metrics_per_hour.csv", per_hour)
                save_plots(self.args.output_dir, self.history, prediction_rows)
            save_csv(self.args.output_dir / "metrics.csv", [row])
            print(
                f"Epoch {epoch:03d} train={train_loss:.6f} "
                f"val={val_loss:.6f} CSI_RMSE={metrics['CSI_RMSE']:.6f}"
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train local crop CNN-GRU.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--metadata-filename", type=str, default=None)
    parser.add_argument("--hourly-csv", type=Path, default=None)
    parser.add_argument("--elevation-csv", type=Path, default=None)
    parser.add_argument("--locations-csv", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--local-crop-size", type=int, default=64)
    parser.add_argument("--crop-padding-mode", choices=("edge", "reflect"), default="edge")
    parser.add_argument("--crop-lat-min", type=float, default=CropBounds.lat_min)
    parser.add_argument("--crop-lat-max", type=float, default=CropBounds.lat_max)
    parser.add_argument("--crop-lon-min", type=float, default=CropBounds.lon_min)
    parser.add_argument("--crop-lon-max", type=float, default=CropBounds.lon_max)
    parser.add_argument("--loss", choices=LOSS_CHOICES, default="masked_weighted_huber")
    parser.add_argument("--huber-beta", type=float, default=0.1)
    parser.add_argument("--cloudy-weight", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--cnn-feature-dim", type=int, default=128)
    parser.add_argument("--gru-hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use-auxiliary-features", action="store_true")
    parser.add_argument("--auxiliary-dim", type=int, default=9)
    parser.add_argument("--clear-sky-threshold", type=float, default=20.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump({key: str(value) for key, value in vars(args).items()}, handle, indent=2)
    trainer = LocalCropTrainer(args)
    trainer.fit()


if __name__ == "__main__":
    main()
