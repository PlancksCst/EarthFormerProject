"""Train EarthFormer + Perceiver readout for SEVIRI CSI forecasting."""

from __future__ import annotations

import sys
import time
import csv
import math
import shutil
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PREP_MODELS_ROOT = PROJECT_ROOT.parent
if str(PREP_MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(PREP_MODELS_ROOT))

from configs.config import TrainingConfig, build_arg_parser, config_from_args  # noqa: E402
from datasets.seviri_dataset import build_dataloader  # noqa: E402
from models.model import build_perceiver_readout_model  # noqa: E402
from training.checkpoint import load_checkpoint, resume_checkpoint, save_checkpoint  # noqa: E402
from training.debugging import (  # noqa: E402
    assert_finite,
    assert_gradients_finite,
    assert_scalar_finite,
)
from training.losses import MSELoss  # noqa: E402
from training.losses import valid_mask_from_target_mask  # noqa: E402
from training.validate import ensure_forecast_target, reconstruct_ghi, validate  # noqa: E402
from utils.artifacts import ArtifactMirror  # noqa: E402
from utils.logger import CSVLogger  # noqa: E402
from utils.plotting import save_training_plots, save_validation_diagnostic_plots  # noqa: E402
from utils.precision import autocast_context, build_grad_scaler, resolve_amp_dtype  # noqa: E402
from utils.seed import seed_everything  # noqa: E402





class EarthFormerTrainer:
    """Coordinate EarthFormer + Perceiver CSI forecasting fine-tuning."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.config.prepare_directories()
        self.artifacts = ArtifactMirror(
            checkpoint_dir=self.config.checkpoint_dir,
            output_dir=self.config.output_dir,
            enabled=self.config.mirror_artifacts,
        )
        seed_everything(self.config.random_seed)

        self.device = torch.device(self.config.resolved_device())
        self.use_amp = self.config.mixed_precision and self.device.type == "cuda"
        self.amp_dtype = (
            resolve_amp_dtype(self.config.amp_dtype, self.device)
            if self.use_amp
            else None
        )

        self.train_loader = build_dataloader(
            config=self.config,
            split=self.config.train_split,
            include_target=True,
            shuffle=True,
        )
        self.val_loader = build_dataloader(
            config=self.config,
            split=self.config.val_split,
            include_target=True,
            shuffle=False,
        )

        self.model = build_perceiver_readout_model(self.config).to(self.device)
        self.criterion = MSELoss()
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.scaler = build_grad_scaler(enabled=self.use_amp, dtype=self.amp_dtype)
        self.logger = CSVLogger(self.config.output_dir / self.config.log_filename)
        self.history: list[dict[str, float]] = []
        self.start_epoch = 1
        self.best_loss = float("inf")
        self.patience_counter = 0

        if self.config.resume_checkpoint is not None:
            self.start_epoch, self.best_loss = resume_checkpoint(
                path=self.config.resume_checkpoint,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                map_location=self.device,
            )
            checkpoint = load_checkpoint(self.config.resume_checkpoint, map_location="cpu")
            extra_state = checkpoint.get("extra_state", {}) if isinstance(checkpoint, dict) else {}
            self.patience_counter = int(extra_state.get("patience_counter", 0))

    def build_optimizer(self) -> AdamW:
        """Build AdamW with lower LR for EarthFormer and higher LR for readout."""
        return AdamW(
            [
                {
                    "params": list(self.model.earthformer_parameters()),
                    "lr": self.config.backbone_learning_rate,
                    "name": "backbone",
                },
                {
                    "params": list(self.model.readout_parameters()),
                    "lr": self.config.head_learning_rate,
                    "name": "head",
                },
            ],
            weight_decay=self.config.weight_decay,
        )

    def build_scheduler(self) -> LambdaLR:
        """Build linear warmup followed by cosine decay for each LR group."""
        total_epochs = max(1, self.config.scheduler_t_max or self.config.epochs)
        warmup_epochs = max(0, self.config.warmup_epochs)

        def lr_lambda(base_lr: float):
            min_factor = self.config.scheduler_eta_min / max(base_lr, self.config.scheduler_eta_min)

            def schedule(epoch_index: int) -> float:
                if warmup_epochs <= 0:
                    progress = min(max(epoch_index, 0), total_epochs) / total_epochs
                    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                    return min_factor + (1.0 - min_factor) * cosine
                epoch_number = epoch_index + 1
                if epoch_number <= warmup_epochs:
                    return max(epoch_number / warmup_epochs, min_factor)
                cosine_epochs = max(1, total_epochs - warmup_epochs)
                progress = min(max(epoch_number - warmup_epochs, 0), cosine_epochs) / cosine_epochs
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_factor + (1.0 - min_factor) * cosine

            return schedule

        return LambdaLR(
            self.optimizer,
            lr_lambda=[lr_lambda(group["lr"]) for group in self.optimizer.param_groups],
        )

    def current_lrs(self) -> dict[str, float]:
        """Return current learning rates by optimizer group."""
        return {
            str(group.get("name", index)): float(group["lr"])
            for index, group in enumerate(self.optimizer.param_groups)
        }

    def current_lr(self) -> float:
        """Return the head learning rate for backwards-compatible logging."""
        return float(self.current_lrs().get("head", self.optimizer.param_groups[-1]["lr"]))

    def train_one_epoch(self, epoch: int) -> float:
        """Run one training epoch and return average training loss."""
        self.model.train()
        total_loss = 0.0
        total_valid_positions = 0
        progress = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.epochs}", leave=False)

        for batch_index, batch in enumerate(progress):
            inputs = batch["satellite"].to(self.device, non_blocking=True)
            targets = ensure_forecast_target(batch["target"]).to(
                self.device,
                non_blocking=True,
            )
            clear_sky_ghi = ensure_forecast_target(batch["clear_sky_ghi"], "clear_sky_ghi").to(
                self.device,
                non_blocking=True,
            )
            target_mask = batch.get("target_mask")
            if isinstance(target_mask, torch.Tensor):
                target_mask = target_mask.to(self.device, non_blocking=True)
                assert_finite(
                    "target_mask",
                    target_mask.float(),
                    batch=batch,
                    batch_index=batch_index,
                )
            valid_mask = valid_mask_from_target_mask(target_mask, targets)
            valid_count = int(valid_mask.sum().detach().cpu())
            if valid_count == 0:
                raise RuntimeError(
                    "No valid target positions in training batch. "
                    "Mask convention is target_mask=0 valid, target_mask=1 invalid."
                )
            assert_finite("inputs", inputs, batch=batch, batch_index=batch_index)
            assert_finite("targets", targets, batch=batch, batch_index=batch_index)
            assert_finite(
                "clear_sky_ghi",
                clear_sky_ghi,
                batch=batch,
                batch_index=batch_index,
            )
            self.optimizer.zero_grad(set_to_none=True)
            with autocast_context(
                device=self.device,
                enabled=self.use_amp,
                dtype=self.amp_dtype,
            ):
                predictions = self.model(inputs)
                if predictions.shape != targets.shape:
                    raise ValueError(
                        "Prediction and target shapes differ: "
                        f"{tuple(predictions.shape)} vs {tuple(targets.shape)}"
                    )
                assert_finite(
                    "predictions",
                    predictions,
                    batch=batch,
                    batch_index=batch_index,
                )
                predicted_ghi = reconstruct_ghi(predictions, clear_sky_ghi)
                assert_finite(
                    "predicted_ghi",
                    predicted_ghi,
                    batch=batch,
                    batch_index=batch_index,
                )
                loss = self.criterion(predictions, targets, valid_mask=valid_mask)
                assert_scalar_finite(
                    "loss",
                    loss,
                    batch=batch,
                    batch_index=batch_index,
                )

            self.scaler.scale(loss).backward()
            if self.config.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                gradient_stats = assert_gradients_finite(
                    self.model,
                    batch=batch,
                    batch_index=batch_index,
                )
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip,
                )
                if not isinstance(grad_norm, torch.Tensor):
                    grad_norm = torch.tensor(float(grad_norm), device=self.device)
                assert_scalar_finite(
                    "gradient_norm",
                    grad_norm.detach().reshape(()),
                    batch=batch,
                    batch_index=batch_index,
                )
            else:
                gradient_stats = assert_gradients_finite(
                    self.model,
                    batch=batch,
                    batch_index=batch_index,
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += float(loss.item()) * valid_count
            total_valid_positions += valid_count
            progress.set_postfix(
                loss=f"{loss.item():.5f}",
                grad=f"{float(gradient_stats['total_norm']):.3e}",
                lr=f"{self.current_lr():.3e}",
            )

        if total_valid_positions == 0:
            raise ValueError("Training dataloader produced no samples")
        return total_loss / total_valid_positions

    def checkpoint_extra_state(self) -> dict[str, float | int | str]:
        """Return trainer state needed for exact early-stopping resume."""
        return {
            "patience_counter": self.patience_counter,
            "early_stopping_patience": self.config.early_stopping_patience,
            "monitor": "val_loss",
        }

    def save_epoch_checkpoints(self, epoch: int, validation_loss: float, improved: bool) -> None:
        """Save latest and best checkpoints."""
        last_path = self.config.checkpoint_dir / "last.pt"
        save_checkpoint(
            path=last_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            epoch=epoch,
            best_loss=self.best_loss,
            config=self.config,
            best_metric_name="val_loss",
            best_metric=self.best_loss,
            extra_state=self.checkpoint_extra_state(),
        )
        self.artifacts.mirror_checkpoint_file(last_path)

        if improved:
            best_path = self.config.checkpoint_dir / "best.pt"
            save_checkpoint(
                path=best_path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                epoch=epoch,
                best_loss=self.best_loss,
                config=self.config,
                best_metric_name="val_loss",
                best_metric=self.best_loss,
                extra_state=self.checkpoint_extra_state(),
            )
            self.artifacts.mirror_checkpoint_file(best_path)

    def save_validation_predictions(
        self,
        epoch: int,
        rows: list[dict[str, object]],
    ) -> Path | None:
        """Save per-hour validation predictions for one epoch."""
        if not rows:
            return None
        prediction_dir = self.config.output_dir / "predictions"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        path = prediction_dir / f"validation_epoch_{epoch:03d}.csv"
        fieldnames = [
            "sample_id",
            "location",
            "input_day",
            "day",
            "target_day",
            "hour",
            "forecast_hour",
            "valid",
            "target_csi",
            "predicted_csi",
            "target_ghi",
            "predicted_ghi",
            "clear_sky_ghi",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        latest_path = prediction_dir / "validation_latest.csv"
        shutil.copy2(path, latest_path)
        self.artifacts.mirror_output_file(path)
        self.artifacts.mirror_output_file(latest_path)
        return path

    def save_best_epoch_artifacts(self, plot_paths: dict[str, Path]) -> None:
        """Copy current best validation plots into `outputs/best_epoch`."""
        best_dir = self.config.output_dir / "best_epoch"
        best_dir.mkdir(parents=True, exist_ok=True)
        for source in plot_paths.values():
            if source.exists():
                destination = best_dir / source.name
                shutil.copy2(source, destination)
                self.artifacts.mirror_output_file(destination)

    def fit(self) -> None:
        """Run the full training loop."""
        for epoch in range(self.start_epoch, self.config.epochs + 1):
            epoch_start = time.perf_counter()
            train_loss = self.train_one_epoch(epoch)
            validation_metrics = validate(
                model=self.model,
                dataloader=self.val_loader,
                criterion=self.criterion,
                device=self.device,
                use_amp=self.use_amp,
                amp_dtype=self.amp_dtype,
                collect_predictions=True,
            )
            validation_loss = float(validation_metrics["val_loss"])
            lrs = self.current_lrs()
            improved = validation_loss < self.best_loss
            if improved:
                self.best_loss = validation_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            epoch_time = time.perf_counter() - epoch_start
            self.scheduler.step()
            self.save_epoch_checkpoints(epoch, validation_loss, improved=improved)
            prediction_rows = validation_metrics.get("predictions", [])
            self.save_validation_predictions(epoch, prediction_rows)
            row = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": validation_loss,
                "CSI_MAE": float(validation_metrics["CSI_MAE"]),
                "CSI_RMSE": float(validation_metrics["CSI_RMSE"]),
                "CSI_nRMSE": float(validation_metrics["CSI_nRMSE"]),
                "CSI_R2": float(validation_metrics["CSI_R2"]),
                "GHI_MAE": float(validation_metrics["GHI_MAE"]),
                "GHI_RMSE": float(validation_metrics["GHI_RMSE"]),
                "GHI_nRMSE": float(validation_metrics["GHI_nRMSE"]),
                "GHI_R2": float(validation_metrics["GHI_R2"]),
                "learning_rate": float(lrs.get("head", self.current_lr())),
                "lr_backbone": float(lrs.get("backbone", 0.0)),
                "lr_head": float(lrs.get("head", self.current_lr())),
                "best_val_loss": float(self.best_loss),
                "patience_counter": int(self.patience_counter),
                "epoch_time": float(epoch_time),
            }
            self.history.append(row)
            self.logger.log(**row)
            self.artifacts.mirror_output_file(self.logger.path)
            plot_paths = save_training_plots(
                history=self.history,
                sample=validation_metrics.get("sample"),
                output_dir=self.config.output_dir,
                epoch=epoch,
            )
            plot_paths.update(
                save_validation_diagnostic_plots(
                    prediction_rows,
                    output_dir=self.config.output_dir,
                    epoch=epoch,
                )
            )
            for plot_path in plot_paths.values():
                self.artifacts.mirror_output_file(plot_path)
            if improved:
                self.save_best_epoch_artifacts(plot_paths)
            print(
                f"Epoch {epoch:03d}\n"
                f"Train Loss: {train_loss:.6f}\n"
                f"Validation Loss: {validation_loss:.6f}\n"
                f"CSI RMSE: {validation_metrics['CSI_RMSE']:.6f}\n"
                f"CSI MAE: {validation_metrics['CSI_MAE']:.6f}\n"
                f"GHI RMSE: {validation_metrics['GHI_RMSE']:.6f}\n"
                f"LR backbone: {lrs.get('backbone', 0.0):.3e}\n"
                f"LR head: {lrs.get('head', self.current_lr()):.3e}\n"
                f"Best Val: {self.best_loss:.6f}\n"
                f"Patience: {self.patience_counter}/{self.config.early_stopping_patience}\n"
                f"Time: {epoch_time:.1f}s"
            )
            if self.patience_counter >= self.config.early_stopping_patience:
                print("Early stopping triggered")
                break

    def inspect_gradients(self):
        """
        Check for NaN/Inf gradients and report gradient norms.
        """

        print("\nGradient Diagnostics")

        total_norm = 0.0

        for name, param in self.model.named_parameters():

            if param.grad is None:
                continue

            grad = param.grad

            if not torch.isfinite(grad).all():
                raise RuntimeError(
                    f"Non-finite gradient detected in {name}"
                )

            norm = grad.norm().item()

            total_norm += norm ** 2

            print(
                f"{name:60s}"
                f" norm={norm:.6f}"
            )

        total_norm = total_norm ** 0.5

        print(f"\nTotal gradient norm: {total_norm:.6f}")
        

    def experiment_one_batch(self):

        print("=" * 80)
        print("EXPERIMENT 1")
        print("Single Batch Training Step")
        print("=" * 80)

        self.model.train()

        before = {}

        for name, param in self.model.named_parameters():

            before[name] = param.detach().clone()
            
        batch = next(iter(self.train_loader))

        inputs = batch["satellite"].to(self.device)
        targets = ensure_forecast_target(batch["target"]).to(self.device)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast_context(
            device=self.device,
            enabled=self.use_amp,
            dtype=self.amp_dtype,
        ):
            prediction = self.model(inputs)

            loss = self.criterion(
                prediction,
                targets
            )

        print(f"Loss: {loss.item():.6f}")

        self.scaler.scale(loss).backward()

        # If using AMP, unscale gradients before inspecting or clipping so
        # the checks operate on the true fp32 gradients rather than the
        # internally scaled values.
        if hasattr(self, "scaler") and getattr(self, "scaler") is not None:
            try:
                self.scaler.unscale_(self.optimizer)
            except Exception:
                # If unscale_ isn't available or fails, continue and inspect raw grads
                pass

            # Inspect gradients and, if any non-finite values are present, run
            # detailed forward-trace diagnostics to help locate their origin.
            try:
                self.inspect_gradients()
            except RuntimeError as e:
                print("Non-finite gradients detected; running diagnostics...")
                # Check prediction finite status
                with torch.no_grad():
                    try:
                        trace = self.model(inputs, return_debug=True)
                    except Exception as e_trace:
                        print("debug forward failed:", e_trace)
                        raise

                # Print basic stats for traced tensors
                for k, v in trace.get("earthformer_trace", {}).items():
                    if isinstance(v, tuple):
                        print(f"trace {k}: shape={v}")
                    else:
                        print(f"trace {k}: {type(v)} -> {v}")

                # Check pre-head latent and prediction finiteness
                pre = trace.get("pre_head_latent")
                pred = trace.get("prediction")
                try:
                    if isinstance(pre, torch.Tensor):
                        print("pre_head_latent any nan:", torch.isnan(pre).any().item(), "any inf:", torch.isinf(pre).any().item())
                    if isinstance(pred, torch.Tensor):
                        print("prediction any nan:", torch.isnan(pred).any().item(), "any inf:", torch.isinf(pred).any().item())
                except Exception:
                    pass

                # Check init_global_vectors if present
                try:
                    core = getattr(getattr(self.model, "earthformer", None), "model", None)
                    if core is not None and hasattr(core, "init_global_vectors"):
                        ig = core.init_global_vectors
                        print("init_global_vectors any nan:", torch.isnan(ig).any().item(), "any inf:", torch.isinf(ig).any().item())
                        if ig.grad is not None:
                            print("init_global_vectors.grad any nan:", torch.isnan(ig.grad).any().item())
                except Exception:
                    pass

                # Re-raise the original error after diagnostics
                raise

        self.scaler.step(self.optimizer)

        self.scaler.update()

        changed = 0

        for name, param in self.model.named_parameters():

            if not torch.equal(
                before[name],
                param,
            ):

                changed += 1

        print(f"\nParameters updated: {changed}")

        if changed == 0:

            raise RuntimeError(
                "Optimizer step did not modify any parameters."
            )

        print("Optimizer step successful.")
        
    def experiment_overfit(
        self,
        samples=8,
        epochs=100
    ):

        from torch.utils.data import Subset
        from torch.utils.data import DataLoader

        print("=" * 80)
        print("EXPERIMENT 2")
        print("Tiny Dataset Overfit")
        print("=" * 80)

        tiny = Subset(
            self.train_loader.dataset,
            range(samples)
        )

        loader = DataLoader(

            tiny,

            batch_size=min(
                self.config.batch_size,
                samples
            ),

            shuffle=True,

            num_workers=0
        )

        for epoch in range(epochs):

            self.model.train()

            running = 0

            for batch in loader:

                inputs = batch["satellite"].to(self.device)

                targets = ensure_forecast_target(batch["target"]).to(self.device)

                self.optimizer.zero_grad()

                with autocast_context(
                    device=self.device,
                    enabled=self.use_amp,
                    dtype=self.amp_dtype,
                ):

                    prediction = self.model(inputs)

                    loss = self.criterion(
                        prediction,
                        targets
                    )

                self.scaler.scale(loss).backward()

                if self.config.gradient_clip > 0:

                    self.scaler.unscale_(self.optimizer)

                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip,
                    )

                self.scaler.step(self.optimizer)

                self.scaler.update()

                running += loss.item()

            running /= len(loader)

            print(
                f"Epoch {epoch+1:03d}"
                f"  Loss={running:.6f}"
            )

            if running < 1e-3:

                print("\nSUCCESS")
                print("Model can memorize dataset.")

                return

        print("\nWARNING")
        print("Model failed to overfit tiny dataset.")
        
    def experiment_resume(
        self,
        epochs_after_resume=3
    ):

        print("=" * 80)
        print("EXPERIMENT 3")
        print("Checkpoint Resume")
        print("=" * 80)

        print(
            f"Resuming from epoch "
            f"{self.start_epoch}"
        )

        for epoch in range(

            self.start_epoch,

            self.start_epoch
            + epochs_after_resume

        ):

            train = self.train_one_epoch(epoch)

            val_metrics = validate(
                model=self.model,
                dataloader=self.val_loader,
                criterion=self.criterion,
                device=self.device,
                use_amp=self.use_amp,
                amp_dtype=self.amp_dtype,
            )
            val = float(val_metrics["val_loss"])

            improved = val < self.best_loss
            if improved:
                self.best_loss = val
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            self.scheduler.step()

            self.save_epoch_checkpoints(
                epoch,
                val,
                improved=improved,
            )
            lrs = self.current_lrs()

            self.logger.log(
                epoch=epoch,
                train_loss=train,
                val_loss=val,
                CSI_MAE=float(val_metrics["CSI_MAE"]),
                CSI_RMSE=float(val_metrics["CSI_RMSE"]),
                CSI_nRMSE=float(val_metrics["CSI_nRMSE"]),
                CSI_R2=float(val_metrics["CSI_R2"]),
                GHI_MAE=float(val_metrics["GHI_MAE"]),
                GHI_RMSE=float(val_metrics["GHI_RMSE"]),
                GHI_nRMSE=float(val_metrics["GHI_nRMSE"]),
                GHI_R2=float(val_metrics["GHI_R2"]),
                learning_rate=lrs.get("head", self.current_lr()),
                lr_backbone=lrs.get("backbone", 0.0),
                lr_head=lrs.get("head", self.current_lr()),
                best_val_loss=self.best_loss,
                patience_counter=self.patience_counter,
                epoch_time=0.0,
            )
            self.artifacts.mirror_output_file(self.logger.path)

            print(
                f"Epoch {epoch:03d} | "
                f"train={train:.6f} | "
                f"val={val:.6f}"
            )
            
            
def main() -> None:
    """Entry point for command-line training."""
    
    
    parser = build_arg_parser()

    parser.add_argument(

        "--experiment",

        choices=[

            "train",

            "one_batch",

            "overfit",

            "resume"

        ],

        default="train"

    )

    parser.add_argument(

        "--samples",

        type=int,

        default=8
    )

    args = parser.parse_args()

    config = config_from_args(args)

    trainer = EarthFormerTrainer(config)

    if args.experiment == "one_batch":

        trainer.experiment_one_batch()

    elif args.experiment == "overfit":

        trainer.experiment_overfit(
            samples=args.samples
        )

    elif args.experiment == "resume":

        trainer.experiment_resume()

    else:

        trainer.fit()


if __name__ == "__main__":
    main()
