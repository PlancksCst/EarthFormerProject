"""Train EarthFormer + Perceiver readout for SEVIRI CSI forecasting."""

from __future__ import annotations

import sys
import time
import csv
import json
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
from training.baselines import ClimatologyBaseline  # noqa: E402
from training.debugging import (  # noqa: E402
    assert_finite,
    assert_gradients_finite,
    assert_scalar_finite,
)
from training.losses import MSELoss  # noqa: E402
from training.losses import valid_hour_mask  # noqa: E402
from training.residual_losses import ResidualLoss  # noqa: E402
from training.validate import ensure_forecast_target, reconstruct_ghi, validate  # noqa: E402
from utils.artifacts import ArtifactMirror  # noqa: E402
from utils.logger import CSVLogger  # noqa: E402
from utils.precision import autocast_context, build_grad_scaler, resolve_amp_dtype  # noqa: E402
from utils.seed import seed_everything  # noqa: E402

try:
    from utils.plotting import save_query_similarity_heatmap, save_training_plots  # noqa: E402
except ImportError:
    from utils.plotting import save_training_plots  # noqa: E402

    def save_query_similarity_heatmap(
        similarity: torch.Tensor,
        output_dir: str | Path,
        epoch: int,
        plot_dir: str | Path | None = None,
    ) -> Path | None:
        """Gracefully skip query heatmaps when Colab has a stale plotting module."""
        if not getattr(save_query_similarity_heatmap, "_warned", False):
            print(
                "WARNING: query similarity heatmaps are unavailable because "
                "`utils.plotting.save_query_similarity_heatmap` was not found. "
                "Update EarthFormer/utils/plotting.py or restart the runtime after syncing."
            )
            setattr(save_query_similarity_heatmap, "_warned", True)
        return None

try:
    from utils.plotting import save_validation_diagnostic_plots  # noqa: E402
    _DIAGNOSTIC_PLOTS_AVAILABLE = True
except ImportError:
    _DIAGNOSTIC_PLOTS_AVAILABLE = False

    def save_validation_diagnostic_plots(
        prediction_rows: list[dict[str, object]],
        output_dir: str | Path,
        epoch: int,
        plot_dir: str | Path | None = None,
    ) -> dict[str, Path]:
        """Gracefully skip diagnostics when Colab has a stale plotting module."""
        if not getattr(save_validation_diagnostic_plots, "_warned", False):
            print(
                "WARNING: validation diagnostic plots are unavailable because "
                "`utils.plotting.save_validation_diagnostic_plots` was not found. "
                "Update EarthFormer/utils/plotting.py or restart the runtime after syncing."
            )
            setattr(save_validation_diagnostic_plots, "_warned", True)
        return {}





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
        self.backbone_frozen: bool | None = None
        self.set_backbone_trainability(epoch=1, announce=False)
        self.residual_baseline = self.build_residual_baseline()
        self.residual_criterion = ResidualLoss(
            loss_name=self.config.residual_loss,
            beta=self.config.huber_beta,
            cloudy_weight=self.config.cloudy_weight,
            ramp_weight=self.config.ramp_weight,
        )
        self.criterion = MSELoss(
            loss_name=self.config.loss_name,
            low_csi_weight=self.config.low_csi_weight,
            low_csi_threshold=self.config.low_csi_threshold,
            ghi_loss_weight=self.config.ghi_loss_weight,
            huber_beta=self.config.huber_beta,
            cloudy_weight=self.config.cloudy_weight,
            ramp_weight=self.config.ramp_weight,
            lambda_corr=self.config.lambda_corr,
        )
        print(
            "Training loss: "
            f"{self.config.loss_name} "
            f"(residual_loss={self.config.residual_loss}, "
            f"huber_beta={self.config.huber_beta}, "
            f"cloudy_weight={self.config.cloudy_weight}, "
            f"ramp_weight={self.config.ramp_weight}, "
            f"lambda_corr={self.config.lambda_corr}, "
            f"ghi_loss_weight={self.config.ghi_loss_weight})"
        )
        print(
            "Forecast mode: "
            f"fix_preset={self.config.fix_preset}, "
            f"{self.config.forecast_mode} "
            f"(residual_baseline={self.config.residual_baseline}, "
            f"use_auxiliary_features={self.config.use_auxiliary_features}, "
            f"freeze_backbone_epochs={self.config.freeze_backbone_epochs}, "
            f"image_dependence_weight={self.config.image_dependence_weight}, "
            f"image_dependence_margin={self.config.image_dependence_margin})"
        )
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.scaler = build_grad_scaler(enabled=self.use_amp, dtype=self.amp_dtype)
        self.logger = CSVLogger(self.config.output_dir / self.config.log_filename)
        self.history: list[dict[str, float]] = []
        self.start_epoch = 1
        self.best_loss = float("inf")
        self.patience_counter = 0
        self.last_train_metrics: dict[str, float] = {}

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

    def is_explicit_residual_preset(self) -> bool:
        """Return whether the active preset uses explicit residual learning."""
        return self.config.fix_preset in {"explicit_residual_head", "explicit_residual_gated"}

    def build_residual_baseline(self) -> ClimatologyBaseline | None:
        """Build the optional climatology baseline used for residual forecasting."""
        if self.config.forecast_mode == "direct":
            return None
        if self.config.forecast_mode != "residual_climatology":
            raise ValueError(f"Unsupported forecast_mode: {self.config.forecast_mode}")
        print(
            "Building residual climatology baseline from training split "
            f"({self.config.residual_baseline})..."
        )
        baseline = ClimatologyBaseline.from_dataset(
            dataset=self.train_loader.dataset,
            output_length=self.config.output_length,
            mode=self.config.residual_baseline,
            clear_sky_threshold=self.config.clear_sky_threshold,
        )
        print(f"Residual baseline global CSI mean: {baseline.global_mean:.4f}")
        return baseline

    def apply_forecast_mode(
        self,
        batch: dict[str, object],
        model_output: torch.Tensor,
    ) -> torch.Tensor:
        """Convert raw model output into final CSI prediction."""
        if self.config.forecast_mode == "direct":
            return model_output
        if self.residual_baseline is None:
            raise RuntimeError("Residual forecast mode requires a fitted residual baseline.")
        baseline = self.residual_baseline.predict(
            batch=batch,
            horizon=model_output.shape[1],
            device=model_output.device,
            dtype=model_output.dtype,
        )
        return baseline + model_output

    def auxiliary_features_from_batch(
        self,
        batch: dict[str, object],
        batch_index: int | None = None,
    ) -> torch.Tensor | None:
        """Return auxiliary readout-conditioning features when enabled."""
        if not self.config.use_auxiliary_features:
            return None
        value = batch.get("auxiliary_features", batch.get("aux_features"))
        if not isinstance(value, torch.Tensor):
            raise KeyError(
                "Auxiliary features are enabled, but the batch does not contain "
                "'auxiliary_features'."
            )
        auxiliary = value.to(self.device, non_blocking=True).float()
        assert_finite(
            "auxiliary_features",
            auxiliary,
            batch=batch,
            batch_index=-1 if batch_index is None else batch_index,
        )
        return auxiliary

    def set_backbone_trainability(self, epoch: int, announce: bool = True) -> None:
        """Freeze the EarthFormer backbone for the configured warm-start epochs."""
        should_freeze = self.config.freeze_earthformer or (
            self.config.freeze_backbone_epochs > 0
            and epoch <= self.config.freeze_backbone_epochs
        )
        if self.backbone_frozen is not None and self.backbone_frozen == should_freeze:
            return
        if should_freeze:
            self.model.freeze_earthformer()
        else:
            self.model.unfreeze_earthformer()
        self.backbone_frozen = should_freeze
        if announce:
            state = "frozen" if should_freeze else "trainable"
            print(f"EarthFormer backbone is {state} at epoch {epoch}.")

    def image_dependence_regularization(
        self,
        batch: dict[str, object],
        inputs: torch.Tensor,
        auxiliary_features: torch.Tensor | None,
        predictions: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encourage the forecast to change when image evidence is removed."""
        if self.config.image_dependence_weight <= 0.0:
            zero = predictions.new_zeros(())
            return zero, zero

        zero_inputs = torch.zeros_like(inputs)
        zero_outputs = self.model(zero_inputs, auxiliary_features=auxiliary_features)
        zero_predictions = self.apply_forecast_mode(batch, zero_outputs)
        delta = (predictions - zero_predictions).abs()
        valid = valid_mask.to(device=predictions.device, dtype=torch.bool)
        if int(valid.sum().detach().cpu()) == 0:
            zero = predictions.new_zeros(())
            return zero, zero
        mean_delta = delta[valid].mean()
        penalty = torch.relu(predictions.new_tensor(self.config.image_dependence_margin) - mean_delta)
        return self.config.image_dependence_weight * penalty, mean_delta.detach()

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

    def query_diversity_regularization(self, steps: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return weighted and raw query diversity losses."""
        if (
            not self.config.use_query_diversity_loss
            or self.config.query_diversity_weight <= 0.0
            or not hasattr(self.model, "query_diversity_loss")
        ):
            zero = next(self.model.parameters()).new_zeros(())
            return zero, zero
        raw_loss = self.model.query_diversity_loss(steps)
        weighted_loss = self.config.query_diversity_weight * raw_loss
        return weighted_loss, raw_loss

    def query_similarity_diagnostics(self, epoch: int) -> tuple[dict[str, float], Path | None]:
        """Return and plot effective output-query similarity diagnostics."""
        with torch.no_grad():
            similarity = self.model.query_similarity_matrix(self.config.output_length)
            stats = self.model.query_similarity_stats(self.config.output_length)
        heatmap_path = save_query_similarity_heatmap(
            similarity=similarity,
            output_dir=self.config.output_dir,
            epoch=epoch,
        )
        return stats, heatmap_path

    def train_one_epoch(self, epoch: int) -> float:
        """Run one training epoch and return average training loss."""
        if self.is_explicit_residual_preset():
            return self.train_one_explicit_residual_epoch(epoch)
        self.set_backbone_trainability(epoch)
        self.model.train()
        total_loss = 0.0
        total_csi_loss = 0.0
        total_ghi_loss = 0.0
        total_image_dependence_loss = 0.0
        total_image_delta = 0.0
        total_query_diversity_loss = 0.0
        total_valid_positions = 0
        total_positions = 0
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
            target_ghi = batch.get("target_ghi")
            if target_ghi is None:
                target_ghi_tensor = reconstruct_ghi(targets, clear_sky_ghi)
            else:
                target_ghi_tensor = ensure_forecast_target(target_ghi, "target_ghi").to(
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
            valid_mask = valid_hour_mask(
                target_mask=target_mask,
                reference=targets,
                clear_sky_ghi=clear_sky_ghi,
                clear_sky_threshold=self.config.clear_sky_threshold,
            )
            valid_count = int(valid_mask.sum().detach().cpu())
            total_positions += int(targets.numel())
            if valid_count == 0:
                progress.set_postfix(loss="skip", valid="0.000")
                continue
            auxiliary_features = self.auxiliary_features_from_batch(
                batch,
                batch_index=batch_index,
            )
            assert_finite("inputs", inputs, batch=batch, batch_index=batch_index)
            assert_finite("targets", targets, batch=batch, batch_index=batch_index)
            assert_finite(
                "clear_sky_ghi",
                clear_sky_ghi,
                batch=batch,
                batch_index=batch_index,
            )
            assert_finite(
                "target_ghi",
                target_ghi_tensor,
                batch=batch,
                batch_index=batch_index,
            )
            self.optimizer.zero_grad(set_to_none=True)
            with autocast_context(
                device=self.device,
                enabled=self.use_amp,
                dtype=self.amp_dtype,
            ):
                model_outputs = self.model(
                    inputs,
                    auxiliary_features=auxiliary_features,
                )
                predictions = self.apply_forecast_mode(batch, model_outputs)
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
                loss_result = self.criterion(
                    predictions,
                    targets,
                    valid_mask=valid_mask,
                    clear_sky_ghi=clear_sky_ghi,
                    target_ghi=target_ghi_tensor,
                    return_components=True,
                )
                if isinstance(loss_result, dict):
                    loss = loss_result["loss"]
                    csi_loss = loss_result["csi_loss"]
                    ghi_loss = loss_result["ghi_loss"]
                else:
                    loss = loss_result
                    csi_loss = loss_result
                    ghi_loss = loss_result.new_zeros(())
                query_diversity_weighted, query_diversity_raw = (
                    self.query_diversity_regularization(targets.shape[1])
                )
                image_dependence_weighted, image_delta = self.image_dependence_regularization(
                    batch=batch,
                    inputs=inputs,
                    auxiliary_features=auxiliary_features,
                    predictions=predictions,
                    valid_mask=valid_mask,
                )
                loss = loss + query_diversity_weighted + image_dependence_weighted
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
            total_csi_loss += float(csi_loss.detach().cpu()) * valid_count
            total_ghi_loss += float(ghi_loss.detach().cpu()) * valid_count
            total_image_dependence_loss += (
                float(image_dependence_weighted.detach().cpu()) * valid_count
            )
            total_image_delta += float(image_delta.detach().cpu()) * valid_count
            total_query_diversity_loss += float(query_diversity_raw.detach().cpu()) * valid_count
            total_valid_positions += valid_count
            progress.set_postfix(
                loss=f"{loss.item():.5f}",
                csi=f"{float(csi_loss.detach().cpu()):.5f}",
                ghi=f"{float(ghi_loss.detach().cpu()):.5f}",
                img=f"{float(image_dependence_weighted.detach().cpu()):.5f}",
                dimg=f"{float(image_delta.detach().cpu()):.4f}",
                qdiv=f"{float(query_diversity_raw.detach().cpu()):.4f}",
                valid=f"{valid_count / max(targets.numel(), 1):.3f}",
                grad=f"{float(gradient_stats['total_norm']):.3e}",
                lr=f"{self.current_lr():.3e}",
            )

        if total_valid_positions == 0:
            raise ValueError("Training dataloader produced no physically valid target hours")
        self.last_train_metrics = {
            "train_loss": total_loss / total_valid_positions,
            "train_csi_loss": total_csi_loss / total_valid_positions,
            "train_ghi_loss": total_ghi_loss / total_valid_positions,
            "train_image_dependence_loss": total_image_dependence_loss / total_valid_positions,
            "train_image_delta": total_image_delta / total_valid_positions,
            "train_query_diversity_loss": total_query_diversity_loss / total_valid_positions,
            "train_valid_fraction": total_valid_positions / max(total_positions, 1),
        }
        return self.last_train_metrics["train_loss"]

    def train_one_explicit_residual_epoch(self, epoch: int) -> float:
        """Run one epoch training only residual predictions against residual targets."""
        if self.residual_baseline is None:
            raise RuntimeError("Explicit residual training requires a climatology baseline.")
        self.set_backbone_trainability(epoch)
        self.model.train()
        total_loss = 0.0
        total_residual_rmse_num = 0.0
        total_valid_positions = 0
        total_positions = 0
        progress = tqdm(self.train_loader, desc=f"Residual {epoch}/{self.config.epochs}", leave=False)

        for batch_index, batch in enumerate(progress):
            inputs = batch["satellite"].to(self.device, non_blocking=True)
            targets = ensure_forecast_target(batch["target"]).to(self.device, non_blocking=True)
            clear_sky_ghi = ensure_forecast_target(batch["clear_sky_ghi"], "clear_sky_ghi").to(
                self.device,
                non_blocking=True,
            )
            target_mask = batch.get("target_mask")
            if isinstance(target_mask, torch.Tensor):
                target_mask = target_mask.to(self.device, non_blocking=True)
            valid_mask = valid_hour_mask(
                target_mask=target_mask,
                reference=targets,
                clear_sky_ghi=clear_sky_ghi,
                clear_sky_threshold=self.config.clear_sky_threshold,
            )
            valid_count = int(valid_mask.sum().detach().cpu())
            total_positions += int(targets.numel())
            if valid_count == 0:
                progress.set_postfix(loss="skip", valid="0.000")
                continue

            auxiliary_features = self.auxiliary_features_from_batch(batch, batch_index=batch_index)
            baseline_csi = self.residual_baseline.predict(
                batch=batch,
                horizon=targets.shape[1],
                device=self.device,
                dtype=targets.dtype,
            )
            target_residual = targets - baseline_csi
            self.optimizer.zero_grad(set_to_none=True)
            with autocast_context(device=self.device, enabled=self.use_amp, dtype=self.amp_dtype):
                result = self.model(
                    inputs,
                    auxiliary_features=auxiliary_features,
                    return_components=True,
                )
                pred_residual = result["pred_residual"] if isinstance(result, dict) else result
                if pred_residual.shape != targets.shape:
                    raise ValueError(
                        "Residual prediction and target shapes differ: "
                        f"{tuple(pred_residual.shape)} vs {tuple(targets.shape)}"
                    )
                loss = self.residual_criterion(
                    pred_residual,
                    target_residual,
                    valid_mask=valid_mask,
                    target_csi=targets,
                )
                assert_scalar_finite("residual_loss", loss, batch=batch, batch_index=batch_index)

            self.scaler.scale(loss).backward()
            if self.config.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                gradient_stats = assert_gradients_finite(
                    self.model,
                    batch=batch,
                    batch_index=batch_index,
                )
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                if not isinstance(grad_norm, torch.Tensor):
                    grad_norm = torch.tensor(float(grad_norm), device=self.device)
                assert_scalar_finite("gradient_norm", grad_norm.detach().reshape(()), batch=batch, batch_index=batch_index)
            else:
                gradient_stats = assert_gradients_finite(
                    self.model,
                    batch=batch,
                    batch_index=batch_index,
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            with torch.no_grad():
                squared = (pred_residual.detach() - target_residual).pow(2)
                total_residual_rmse_num += float(squared[valid_mask].sum().detach().cpu())
            total_loss += float(loss.item()) * valid_count
            total_valid_positions += valid_count
            progress.set_postfix(
                loss=f"{loss.item():.5f}",
                valid=f"{valid_count / max(targets.numel(), 1):.3f}",
                grad=f"{float(gradient_stats['total_norm']):.3e}",
                lr=f"{self.current_lr():.3e}",
            )

        if total_valid_positions == 0:
            raise ValueError("Training dataloader produced no physically valid target hours")
        residual_rmse = math.sqrt(total_residual_rmse_num / total_valid_positions)
        self.last_train_metrics = {
            "train_loss": total_loss / total_valid_positions,
            "train_csi_loss": total_loss / total_valid_positions,
            "train_ghi_loss": 0.0,
            "train_image_dependence_loss": 0.0,
            "train_image_delta": 0.0,
            "train_query_diversity_loss": 0.0,
            "train_valid_fraction": total_valid_positions / max(total_positions, 1),
            "train_residual_CSI_RMSE": residual_rmse,
        }
        return self.last_train_metrics["train_loss"]

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
        if rows and "baseline_csi" in rows[0]:
            fieldnames = [
                "sample_id",
                "location",
                "hour_index",
                "target_csi",
                "baseline_csi",
                "target_residual",
                "pred_residual",
                "pred_csi",
                "valid_mask",
                "clear_sky_ghi",
                "target_ghi",
                "pred_ghi",
            ]
            if any("gate" in row for row in rows):
                fieldnames.append("gate")
        else:
            fieldnames = [
            "sample_id",
            "location",
            "input_day",
            "day",
            "target_day",
            "hour",
            "forecast_hour",
            "valid",
            "valid_hour",
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
        if rows and "baseline_csi" in rows[0]:
            explicit_path = prediction_dir / "prediction.csv"
            shutil.copy2(path, explicit_path)
            self.artifacts.mirror_output_file(explicit_path)
        self.artifacts.mirror_output_file(path)
        self.artifacts.mirror_output_file(latest_path)
        return path

    @staticmethod
    def _metadata_value(batch: dict[str, object], key: str, index: int) -> object:
        value = batch.get(key)
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            item = value[index]
            return item.item() if item.numel() == 1 else item.detach().cpu().tolist()
        if isinstance(value, (list, tuple)):
            return value[index] if index < len(value) else None
        return value

    @staticmethod
    def _rmse(prediction: torch.Tensor, target: torch.Tensor) -> float:
        if prediction.numel() == 0:
            return float("nan")
        return float(torch.sqrt(torch.mean((prediction - target) ** 2)).item())

    @staticmethod
    def _pearson(prediction: torch.Tensor, target: torch.Tensor) -> float:
        if prediction.numel() < 2:
            return 0.0
        pred = prediction.float() - prediction.float().mean()
        tgt = target.float() - target.float().mean()
        denom = torch.sqrt(pred.pow(2).sum().clamp_min(1.0e-12) * tgt.pow(2).sum().clamp_min(1.0e-12))
        corr = (pred * tgt).sum() / denom
        if not torch.isfinite(corr):
            return 0.0
        return float(corr.clamp(-1.0, 1.0).item())

    @torch.no_grad()
    def validate_explicit_residual(self) -> dict[str, object]:
        """Validate explicit residual presets with baseline/residual diagnostics."""
        if self.residual_baseline is None:
            raise RuntimeError("Explicit residual validation requires a climatology baseline.")
        self.model.eval()
        total_loss = 0.0
        total_valid_positions = 0
        total_positions = 0
        pred_csi_values: list[torch.Tensor] = []
        baseline_csi_values: list[torch.Tensor] = []
        target_csi_values: list[torch.Tensor] = []
        pred_residual_values: list[torch.Tensor] = []
        target_residual_values: list[torch.Tensor] = []
        rows: list[dict[str, object]] = []
        sample: dict[str, object] | None = None

        for batch_index, batch in enumerate(self.val_loader):
            inputs = batch["satellite"].to(self.device, non_blocking=True)
            targets = ensure_forecast_target(batch["target"]).to(self.device, non_blocking=True)
            clear_sky_ghi = ensure_forecast_target(batch["clear_sky_ghi"], "clear_sky_ghi").to(
                self.device,
                non_blocking=True,
            )
            target_ghi_value = batch.get("target_ghi")
            if target_ghi_value is None:
                target_ghi_tensor = reconstruct_ghi(targets, clear_sky_ghi)
            else:
                target_ghi_tensor = ensure_forecast_target(target_ghi_value, "target_ghi").to(
                    self.device,
                    non_blocking=True,
                )
            target_mask = batch.get("target_mask")
            if isinstance(target_mask, torch.Tensor):
                target_mask = target_mask.to(self.device, non_blocking=True)
            valid_mask = valid_hour_mask(
                target_mask=target_mask,
                reference=targets,
                clear_sky_ghi=clear_sky_ghi,
                clear_sky_threshold=self.config.clear_sky_threshold,
            )
            valid_count = int(valid_mask.sum().detach().cpu())
            total_positions += int(targets.numel())
            auxiliary_features = self.auxiliary_features_from_batch(batch, batch_index=batch_index)
            baseline_csi = self.residual_baseline.predict(
                batch=batch,
                horizon=targets.shape[1],
                device=self.device,
                dtype=targets.dtype,
            )
            target_residual = targets - baseline_csi

            with autocast_context(device=self.device, enabled=self.use_amp, dtype=self.amp_dtype):
                result = self.model(
                    inputs,
                    auxiliary_features=auxiliary_features,
                    return_components=True,
                )
                pred_residual = result["pred_residual"] if isinstance(result, dict) else result
                gate = result.get("gate") if isinstance(result, dict) else None
                pred_csi = (baseline_csi + pred_residual).clamp(0.0, 1.3)
                if valid_count > 0:
                    loss = self.residual_criterion(
                        pred_residual,
                        target_residual,
                        valid_mask=valid_mask,
                        target_csi=targets,
                    )
                else:
                    loss = pred_residual.new_zeros(())

            pred_ghi = reconstruct_ghi(pred_csi, clear_sky_ghi)
            if valid_count > 0:
                total_loss += float(loss.item()) * valid_count
                total_valid_positions += valid_count
                valid_cpu = valid_mask.detach().cpu()
                pred_csi_values.append(pred_csi.detach().cpu()[valid_cpu])
                baseline_csi_values.append(baseline_csi.detach().cpu()[valid_cpu])
                target_csi_values.append(targets.detach().cpu()[valid_cpu])
                pred_residual_values.append(pred_residual.detach().cpu()[valid_cpu])
                target_residual_values.append(target_residual.detach().cpu()[valid_cpu])

            batch_size, horizon = targets.shape
            cpu_tensors = {
                "target": targets.detach().float().cpu(),
                "baseline": baseline_csi.detach().float().cpu(),
                "target_residual": target_residual.detach().float().cpu(),
                "pred_residual": pred_residual.detach().float().cpu(),
                "pred_csi": pred_csi.detach().float().cpu(),
                "valid": valid_mask.detach().cpu().bool(),
                "clear": clear_sky_ghi.detach().float().cpu(),
                "target_ghi": target_ghi_tensor.detach().float().cpu(),
                "pred_ghi": pred_ghi.detach().float().cpu(),
                "gate": gate.detach().float().cpu() if isinstance(gate, torch.Tensor) else None,
            }
            for sample_index in range(batch_size):
                for hour_index in range(horizon):
                    row = {
                        "sample_id": self._metadata_value(batch, "sample_id", sample_index),
                        "location": self._metadata_value(batch, "location", sample_index),
                        "hour_index": hour_index,
                        "target_csi": float(cpu_tensors["target"][sample_index, hour_index]),
                        "baseline_csi": float(cpu_tensors["baseline"][sample_index, hour_index]),
                        "target_residual": float(cpu_tensors["target_residual"][sample_index, hour_index]),
                        "pred_residual": float(cpu_tensors["pred_residual"][sample_index, hour_index]),
                        "pred_csi": float(cpu_tensors["pred_csi"][sample_index, hour_index]),
                        "valid_mask": bool(cpu_tensors["valid"][sample_index, hour_index]),
                        "clear_sky_ghi": float(cpu_tensors["clear"][sample_index, hour_index]),
                        "target_ghi": float(cpu_tensors["target_ghi"][sample_index, hour_index]),
                        "pred_ghi": float(cpu_tensors["pred_ghi"][sample_index, hour_index]),
                    }
                    if cpu_tensors["gate"] is not None:
                        row["gate"] = float(cpu_tensors["gate"][sample_index, hour_index])
                    rows.append(row)

            if sample is None:
                sample = {
                    "prediction_csi": pred_csi[0].detach().cpu(),
                    "target_csi": targets[0].detach().cpu(),
                    "prediction_ghi": pred_ghi[0].detach().cpu(),
                    "target_ghi": target_ghi_tensor[0].detach().cpu(),
                    "clear_sky_ghi": clear_sky_ghi[0].detach().cpu(),
                    "valid_mask": valid_mask[0].detach().cpu(),
                    "sample_id": self._metadata_value(batch, "sample_id", 0),
                    "location": self._metadata_value(batch, "location", 0),
                    "input_day": self._metadata_value(batch, "input_day", 0),
                    "target_day": self._metadata_value(batch, "target_day", 0),
                }

        if total_valid_positions == 0:
            raise ValueError("Validation dataloader produced no physically valid target hours")

        all_pred_csi = torch.cat(pred_csi_values)
        all_baseline_csi = torch.cat(baseline_csi_values)
        all_target_csi = torch.cat(target_csi_values)
        all_pred_residual = torch.cat(pred_residual_values)
        all_target_residual = torch.cat(target_residual_values)
        baseline_rmse = self._rmse(all_baseline_csi, all_target_csi)
        final_rmse = self._rmse(all_pred_csi, all_target_csi)
        residual_rmse = self._rmse(all_pred_residual, all_target_residual)
        target_residual_std = float(all_target_residual.std(unbiased=False).item())
        pred_residual_std = float(all_pred_residual.std(unbiased=False).item())
        residual_std_ratio = pred_residual_std / max(target_residual_std, 1.0e-12)
        residual_pearson = self._pearson(all_pred_residual, all_target_residual)
        improvement = 100.0 * (baseline_rmse - final_rmse) / max(baseline_rmse, 1.0e-12)

        metrics: dict[str, object] = {
            "val_loss": total_loss / total_valid_positions,
            "val_csi_loss": total_loss / total_valid_positions,
            "val_ghi_loss": 0.0,
            "valid_fraction": total_valid_positions / max(total_positions, 1),
            "CSI_RMSE": final_rmse,
            "CSI_MAE": float((all_pred_csi - all_target_csi).abs().mean().item()),
            "CSI_nRMSE": final_rmse / max(float(all_target_csi.mean().abs().item()), 1.0e-12),
            "CSI_R2": 1.0 - float(((all_pred_csi - all_target_csi) ** 2).sum().item())
            / max(float(((all_target_csi - all_target_csi.mean()) ** 2).sum().item()), 1.0e-12),
            "CSI_MBE": float((all_pred_csi - all_target_csi).mean().item()),
            "GHI_MAE": 0.0,
            "GHI_RMSE": 0.0,
            "GHI_nRMSE": 0.0,
            "GHI_R2": 0.0,
            "GHI_MBE": 0.0,
            "baseline_CSI_RMSE": baseline_rmse,
            "final_CSI_RMSE": final_rmse,
            "residual_CSI_RMSE": residual_rmse,
            "residual_Pearson": residual_pearson,
            "pred_residual_std": pred_residual_std,
            "target_residual_std": target_residual_std,
            "residual_std_ratio": residual_std_ratio,
            "improvement_over_baseline_percent": improvement,
            "predictions": rows,
            "sample": sample,
        }
        self.save_explicit_residual_summary(metrics)
        return metrics

    def save_explicit_residual_summary(self, metrics: dict[str, object]) -> Path:
        """Write the required explicit residual summary JSON."""
        ratio = float(metrics["residual_std_ratio"])
        pearson = float(metrics["residual_Pearson"])
        improved = float(metrics["final_CSI_RMSE"]) < float(metrics["baseline_CSI_RMSE"])
        active = ratio > 0.2 and pearson > 0.1
        if improved and active:
            interpretation = "satellite_residual_branch_active_and_improves_over_climatology"
        elif active:
            interpretation = "satellite_residual_branch_active_but_not_yet_improving_rmse"
        else:
            interpretation = "satellite_residual_branch_remains_inactive_or_near_mean"
        summary = {
            "baseline_CSI_RMSE": float(metrics["baseline_CSI_RMSE"]),
            "final_CSI_RMSE": float(metrics["final_CSI_RMSE"]),
            "improvement_over_baseline_percent": float(metrics["improvement_over_baseline_percent"]),
            "residual_Pearson": pearson,
            "pred_residual_std": float(metrics["pred_residual_std"]),
            "target_residual_std": float(metrics["target_residual_std"]),
            "residual_std_ratio": ratio,
            "satellite_residual_active": bool(active),
            "recommended_interpretation": interpretation,
        }
        path = self.config.output_dir / "explicit_residual_summary.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        self.artifacts.mirror_output_file(path)
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
            if self.is_explicit_residual_preset():
                validation_metrics = self.validate_explicit_residual()
            else:
                validation_metrics = validate(
                    model=self.model,
                    dataloader=self.val_loader,
                    criterion=self.criterion,
                    device=self.device,
                    use_amp=self.use_amp,
                    amp_dtype=self.amp_dtype,
                    collect_predictions=True,
                    clear_sky_threshold=self.config.clear_sky_threshold,
                    prediction_transform=self.apply_forecast_mode,
                    use_auxiliary_features=self.config.use_auxiliary_features,
                )
            validation_loss = float(validation_metrics["val_loss"])
            if self.is_explicit_residual_preset():
                query_stats = {"mean": 0.0, "min": 0.0, "max": 0.0}
                query_heatmap_path = None
            else:
                query_stats, query_heatmap_path = self.query_similarity_diagnostics(epoch)
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
                "fix_preset": self.config.fix_preset,
                "use_auxiliary_features": int(self.config.use_auxiliary_features),
                "loss_name": self.config.loss_name,
                "forecast_mode": self.config.forecast_mode,
                "residual_baseline": self.config.residual_baseline,
                "huber_beta": float(self.config.huber_beta),
                "cloudy_weight": float(self.config.cloudy_weight),
                "ramp_weight": float(self.config.ramp_weight),
                "lambda_corr": float(self.config.lambda_corr),
                "ghi_loss_weight": float(self.config.ghi_loss_weight),
                "image_dependence_weight": float(self.config.image_dependence_weight),
                "image_dependence_margin": float(self.config.image_dependence_margin),
                "freeze_backbone_epochs": int(self.config.freeze_backbone_epochs),
                "train_loss": float(train_loss),
                "train_csi_loss": float(self.last_train_metrics.get("train_csi_loss", train_loss)),
                "train_ghi_loss": float(self.last_train_metrics.get("train_ghi_loss", 0.0)),
                "train_image_dependence_loss": float(
                    self.last_train_metrics.get("train_image_dependence_loss", 0.0)
                ),
                "train_image_delta": float(self.last_train_metrics.get("train_image_delta", 0.0)),
                "train_query_diversity_loss": float(
                    self.last_train_metrics.get("train_query_diversity_loss", 0.0)
                ),
                "train_valid_fraction": float(self.last_train_metrics.get("train_valid_fraction", 0.0)),
                "val_loss": validation_loss,
                "val_csi_loss": float(validation_metrics["val_csi_loss"]),
                "val_ghi_loss": float(validation_metrics["val_ghi_loss"]),
                "valid_fraction": float(validation_metrics["valid_fraction"]),
                "CSI_MAE": float(validation_metrics["CSI_MAE"]),
                "CSI_RMSE": float(validation_metrics["CSI_RMSE"]),
                "CSI_nRMSE": float(validation_metrics["CSI_nRMSE"]),
                "CSI_R2": float(validation_metrics["CSI_R2"]),
                "CSI_MBE": float(validation_metrics["CSI_MBE"]),
                "GHI_MAE": float(validation_metrics["GHI_MAE"]),
                "GHI_RMSE": float(validation_metrics["GHI_RMSE"]),
                "GHI_nRMSE": float(validation_metrics["GHI_nRMSE"]),
                "GHI_R2": float(validation_metrics["GHI_R2"]),
                "GHI_MBE": float(validation_metrics["GHI_MBE"]),
                "baseline_CSI_RMSE": float(validation_metrics.get("baseline_CSI_RMSE", float("nan"))),
                "final_CSI_RMSE": float(validation_metrics.get("final_CSI_RMSE", validation_metrics["CSI_RMSE"])),
                "residual_CSI_RMSE": float(validation_metrics.get("residual_CSI_RMSE", float("nan"))),
                "residual_Pearson": float(validation_metrics.get("residual_Pearson", float("nan"))),
                "pred_residual_std": float(validation_metrics.get("pred_residual_std", float("nan"))),
                "target_residual_std": float(validation_metrics.get("target_residual_std", float("nan"))),
                "residual_std_ratio": float(validation_metrics.get("residual_std_ratio", float("nan"))),
                "improvement_over_baseline_percent": float(
                    validation_metrics.get("improvement_over_baseline_percent", float("nan"))
                ),
                "query_similarity_mean": float(query_stats["mean"]),
                "query_similarity_min": float(query_stats["min"]),
                "query_similarity_max": float(query_stats["max"]),
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
            if query_heatmap_path is not None:
                plot_paths["query_similarity"] = query_heatmap_path
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
                f"Fix Preset: {self.config.fix_preset}\n"
                f"Auxiliary Features: {self.config.use_auxiliary_features}\n"
                f"Loss: {self.config.loss_name}\n"
                f"Forecast Mode: {self.config.forecast_mode}\n"
                f"Train Loss: {train_loss:.6f}\n"
                f"Train CSI Loss: {self.last_train_metrics.get('train_csi_loss', train_loss):.6f}\n"
                f"Train GHI Loss: {self.last_train_metrics.get('train_ghi_loss', 0.0):.6f}\n"
                f"Image Dependence Loss: "
                f"{self.last_train_metrics.get('train_image_dependence_loss', 0.0):.6f}\n"
                f"Image Delta real-vs-zero: "
                f"{self.last_train_metrics.get('train_image_delta', 0.0):.6f}\n"
                f"Query Diversity Loss: {self.last_train_metrics.get('train_query_diversity_loss', 0.0):.6f}\n"
                f"Validation Loss: {validation_loss:.6f}\n"
                f"Validation CSI Loss: {validation_metrics['val_csi_loss']:.6f}\n"
                f"Validation GHI Loss: {validation_metrics['val_ghi_loss']:.6f}\n"
                f"Valid Fraction: {validation_metrics['valid_fraction']:.3f}\n"
                f"CSI RMSE: {validation_metrics['CSI_RMSE']:.6f}\n"
                f"CSI MAE: {validation_metrics['CSI_MAE']:.6f}\n"
                f"GHI RMSE: {validation_metrics['GHI_RMSE']:.6f}\n"
                f"Baseline CSI RMSE: {float(validation_metrics.get('baseline_CSI_RMSE', float('nan'))):.6f}\n"
                f"Final CSI RMSE: {float(validation_metrics.get('final_CSI_RMSE', validation_metrics['CSI_RMSE'])):.6f}\n"
                f"Residual std ratio: {float(validation_metrics.get('residual_std_ratio', float('nan'))):.6f}\n"
                f"Residual Pearson: {float(validation_metrics.get('residual_Pearson', float('nan'))):.6f}\n"
                f"Query Similarity mean/min/max: "
                f"{query_stats['mean']:.4f}/{query_stats['min']:.4f}/{query_stats['max']:.4f}\n"
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
        clear_sky_ghi = ensure_forecast_target(batch["clear_sky_ghi"], "clear_sky_ghi").to(self.device)
        target_ghi_value = batch.get("target_ghi")
        if target_ghi_value is None:
            target_ghi = reconstruct_ghi(targets, clear_sky_ghi)
        else:
            target_ghi = ensure_forecast_target(target_ghi_value, "target_ghi").to(self.device)
        target_mask = batch.get("target_mask")
        if isinstance(target_mask, torch.Tensor):
            target_mask = target_mask.to(self.device)
        valid_mask = valid_hour_mask(
            target_mask=target_mask,
            reference=targets,
            clear_sky_ghi=clear_sky_ghi,
            clear_sky_threshold=self.config.clear_sky_threshold,
        )
        auxiliary_features = self.auxiliary_features_from_batch(batch)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast_context(
            device=self.device,
            enabled=self.use_amp,
            dtype=self.amp_dtype,
        ):
            model_output = self.model(
                inputs,
                auxiliary_features=auxiliary_features,
            )
            if self.is_explicit_residual_preset():
                if self.residual_baseline is None:
                    raise RuntimeError("Explicit residual one-batch test requires a baseline.")
                baseline = self.residual_baseline.predict(
                    batch=batch,
                    horizon=targets.shape[1],
                    device=self.device,
                    dtype=targets.dtype,
                )
                prediction = (baseline + model_output).clamp(0.0, 1.3)
                loss = self.residual_criterion(
                    model_output,
                    targets - baseline,
                    valid_mask=valid_mask,
                    target_csi=targets,
                )
            else:
                prediction = self.apply_forecast_mode(batch, model_output)
                loss = self.criterion(
                    prediction,
                    targets,
                    valid_mask=valid_mask,
                    clear_sky_ghi=clear_sky_ghi,
                    target_ghi=target_ghi,
                )
            query_diversity_weighted, _query_diversity_raw = (
                self.query_diversity_regularization(targets.shape[1])
            )
            image_dependence_weighted, _image_delta = self.image_dependence_regularization(
                batch=batch,
                inputs=inputs,
                auxiliary_features=auxiliary_features,
                predictions=prediction,
                valid_mask=valid_mask,
            )
            loss = loss + query_diversity_weighted + image_dependence_weighted

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
                        trace = self.model(
                            inputs,
                            auxiliary_features=auxiliary_features,
                            return_debug=True,
                        )
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
            steps = 0

            for batch in loader:

                inputs = batch["satellite"].to(self.device)

                targets = ensure_forecast_target(batch["target"]).to(self.device)
                clear_sky_ghi = ensure_forecast_target(batch["clear_sky_ghi"], "clear_sky_ghi").to(self.device)
                target_ghi_value = batch.get("target_ghi")
                if target_ghi_value is None:
                    target_ghi = reconstruct_ghi(targets, clear_sky_ghi)
                else:
                    target_ghi = ensure_forecast_target(target_ghi_value, "target_ghi").to(self.device)
                target_mask = batch.get("target_mask")
                if isinstance(target_mask, torch.Tensor):
                    target_mask = target_mask.to(self.device)
                valid_mask = valid_hour_mask(
                    target_mask=target_mask,
                    reference=targets,
                    clear_sky_ghi=clear_sky_ghi,
                    clear_sky_threshold=self.config.clear_sky_threshold,
                )
                if int(valid_mask.sum().detach().cpu()) == 0:
                    continue
                auxiliary_features = self.auxiliary_features_from_batch(batch)

                self.optimizer.zero_grad()

                with autocast_context(
                    device=self.device,
                    enabled=self.use_amp,
                    dtype=self.amp_dtype,
                ):

                    model_output = self.model(
                        inputs,
                        auxiliary_features=auxiliary_features,
                    )
                    prediction = self.apply_forecast_mode(batch, model_output)

                    loss = self.criterion(
                        prediction,
                        targets,
                        valid_mask=valid_mask,
                        clear_sky_ghi=clear_sky_ghi,
                        target_ghi=target_ghi,
                    )
                    query_diversity_weighted, _query_diversity_raw = (
                        self.query_diversity_regularization(targets.shape[1])
                    )
                    image_dependence_weighted, _image_delta = self.image_dependence_regularization(
                        batch=batch,
                        inputs=inputs,
                        auxiliary_features=auxiliary_features,
                        predictions=prediction,
                        valid_mask=valid_mask,
                    )
                    loss = loss + query_diversity_weighted + image_dependence_weighted

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
                steps += 1

            if steps == 0:
                raise RuntimeError("Tiny overfit batch contained no physically valid target hours.")
            running /= steps

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
                clear_sky_threshold=self.config.clear_sky_threshold,
                prediction_transform=self.apply_forecast_mode,
                use_auxiliary_features=self.config.use_auxiliary_features,
            )
            val = float(val_metrics["val_loss"])
            query_stats, _query_heatmap_path = self.query_similarity_diagnostics(epoch)
            if _query_heatmap_path is not None:
                self.artifacts.mirror_output_file(_query_heatmap_path)

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
                fix_preset=self.config.fix_preset,
                use_auxiliary_features=int(self.config.use_auxiliary_features),
                loss_name=self.config.loss_name,
                forecast_mode=self.config.forecast_mode,
                residual_baseline=self.config.residual_baseline,
                huber_beta=float(self.config.huber_beta),
                cloudy_weight=float(self.config.cloudy_weight),
                ramp_weight=float(self.config.ramp_weight),
                lambda_corr=float(self.config.lambda_corr),
                ghi_loss_weight=float(self.config.ghi_loss_weight),
                image_dependence_weight=float(self.config.image_dependence_weight),
                image_dependence_margin=float(self.config.image_dependence_margin),
                freeze_backbone_epochs=int(self.config.freeze_backbone_epochs),
                train_loss=train,
                train_csi_loss=self.last_train_metrics.get("train_csi_loss", train),
                train_ghi_loss=self.last_train_metrics.get("train_ghi_loss", 0.0),
                train_image_dependence_loss=self.last_train_metrics.get(
                    "train_image_dependence_loss",
                    0.0,
                ),
                train_image_delta=self.last_train_metrics.get("train_image_delta", 0.0),
                train_query_diversity_loss=self.last_train_metrics.get(
                    "train_query_diversity_loss",
                    0.0,
                ),
                train_valid_fraction=self.last_train_metrics.get("train_valid_fraction", 0.0),
                val_loss=val,
                val_csi_loss=float(val_metrics["val_csi_loss"]),
                val_ghi_loss=float(val_metrics["val_ghi_loss"]),
                valid_fraction=float(val_metrics["valid_fraction"]),
                CSI_MAE=float(val_metrics["CSI_MAE"]),
                CSI_RMSE=float(val_metrics["CSI_RMSE"]),
                CSI_nRMSE=float(val_metrics["CSI_nRMSE"]),
                CSI_R2=float(val_metrics["CSI_R2"]),
                CSI_MBE=float(val_metrics["CSI_MBE"]),
                GHI_MAE=float(val_metrics["GHI_MAE"]),
                GHI_RMSE=float(val_metrics["GHI_RMSE"]),
                GHI_nRMSE=float(val_metrics["GHI_nRMSE"]),
                GHI_R2=float(val_metrics["GHI_R2"]),
                GHI_MBE=float(val_metrics["GHI_MBE"]),
                query_similarity_mean=float(query_stats["mean"]),
                query_similarity_min=float(query_stats["min"]),
                query_similarity_max=float(query_stats["max"]),
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
