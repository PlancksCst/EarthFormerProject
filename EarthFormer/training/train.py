"""Train the migrated EarthFormer backbone on SEVIRI imagery."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PREP_MODELS_ROOT = PROJECT_ROOT.parent
if str(PREP_MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(PREP_MODELS_ROOT))

from configs.config import TrainingConfig, build_arg_parser, config_from_args  # noqa: E402
from datasets.seviri_dataset import build_dataloader  # noqa: E402
from models.model import build_training_model  # noqa: E402
from training.checkpoint import resume_checkpoint, save_checkpoint  # noqa: E402
from training.losses import MSELoss  # noqa: E402
from training.validate import target_to_nthwc, validate  # noqa: E402
from utils.logger import CSVLogger  # noqa: E402
from utils.seed import seed_everything  # noqa: E402





class EarthFormerTrainer:
    """Coordinate EarthFormer backbone fine-tuning."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.config.prepare_directories()
        seed_everything(self.config.random_seed)

        self.device = torch.device(self.config.resolved_device())
        self.use_amp = self.config.mixed_precision and self.device.type == "cuda"

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

        self.model = build_training_model(self.config).to(self.device)
        self.criterion = MSELoss()
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        t_max = self.config.scheduler_t_max or self.config.epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, t_max),
            eta_min=self.config.scheduler_eta_min,
        )
        # Use a reduced initial scale to reduce risk of FP16 overflow/NaN
        # when training with mixed precision on some GPUs.
        try:
            self.scaler = torch.amp.GradScaler(enabled=self.use_amp, init_scale=2.0 ** 8)
        except TypeError:
            # Older PyTorch versions may not accept init_scale kwarg.
            self.scaler = torch.amp.GradScaler(enabled=self.use_amp)
        self.logger = CSVLogger(self.config.output_dir / self.config.log_filename)
        self.start_epoch = 1
        self.best_loss = float("inf")

        if self.config.resume_checkpoint is not None:
            self.start_epoch, self.best_loss = resume_checkpoint(
                path=self.config.resume_checkpoint,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                map_location=self.device,
            )

    def current_lr(self) -> float:
        """Return the learning rate of the first optimizer group."""
        return float(self.optimizer.param_groups[0]["lr"])

    def train_one_epoch(self, epoch: int) -> float:
        """Run one training epoch and return average training loss."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        progress = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.epochs}", leave=False)

        for batch in progress:
            inputs = batch["satellite"].to(self.device, non_blocking=True)
            targets = target_to_nthwc(batch["target"]).to(self.device, non_blocking=True)
            batch_size = inputs.shape[0]

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                predictions = self.model(inputs)
                loss = self.criterion(predictions, targets)

            self.scaler.scale(loss).backward()
            if self.config.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            progress.set_postfix(loss=f"{loss.item():.5f}", lr=f"{self.current_lr():.3e}")

        if total_samples == 0:
            raise ValueError("Training dataloader produced no samples")
        return total_loss / total_samples

    def save_epoch_checkpoints(self, epoch: int, validation_loss: float) -> None:
        """Save latest and best checkpoints."""
        is_best = validation_loss < self.best_loss
        if is_best:
            self.best_loss = validation_loss

        last_path = self.config.checkpoint_dir / "last.pt"
        save_checkpoint(
            path=last_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            epoch=epoch,
            best_loss=self.best_loss,
        )

        if is_best:
            best_path = self.config.checkpoint_dir / "best.pt"
            save_checkpoint(
                path=best_path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                epoch=epoch,
                best_loss=self.best_loss,
            )

    def fit(self) -> None:
        """Run the full training loop."""
        for epoch in range(self.start_epoch, self.config.epochs + 1):
            epoch_start = time.perf_counter()
            train_loss = self.train_one_epoch(epoch)
            validation_loss = validate(
                model=self.model,
                dataloader=self.val_loader,
                criterion=self.criterion,
                device=self.device,
                use_amp=self.use_amp,
            )
            self.scheduler.step()
            epoch_time = time.perf_counter() - epoch_start
            self.save_epoch_checkpoints(epoch, validation_loss)
            self.logger.log(
                epoch=epoch,
                train_loss=train_loss,
                validation_loss=validation_loss,
                learning_rate=self.current_lr(),
                epoch_time=epoch_time,
            )
            print(
                f"Epoch {epoch:03d} | "
                f"train={train_loss:.6f} | "
                f"val={validation_loss:.6f} | "
                f"lr={self.current_lr():.3e} | "
                f"time={epoch_time:.1f}s"
            )

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
        targets = target_to_nthwc(
            batch["target"]
        ).to(self.device)

        self.optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=self.device.type,
            enabled=self.use_amp
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
                        trace = self.model.forward_trace(inputs)
                    except Exception as e_trace:
                        print("forward_trace failed:", e_trace)
                        raise

                # Print basic stats for traced tensors
                for k, v in trace.get("trace", {}).items():
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
                    core = getattr(self.model, "model", None)
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

                targets = target_to_nthwc(
                    batch["target"]
                ).to(self.device)

                self.optimizer.zero_grad()

                with torch.amp.autocast(
                    device_type=self.device.type,
                    enabled=self.use_amp,
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

            val = validate(
                model=self.model,
                dataloader=self.val_loader,
                criterion=self.criterion,
                device=self.device,
                use_amp=self.use_amp,
            )

            self.scheduler.step()

            self.save_epoch_checkpoints(
                epoch,
                val
            )

            self.logger.log(
                epoch=epoch,
                train_loss=train,
                validation_loss=val,
                learning_rate=self.current_lr(),
                epoch_time=0.0,
            )

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
