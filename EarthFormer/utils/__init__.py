"""Utility package."""

from .logger import CSVLogger
from .metrics import forecast_metrics, mae, nrmse, r2_score, rmse
from .plotting import save_training_plots
try:
    from .plotting import save_validation_diagnostic_plots
except ImportError:
    save_validation_diagnostic_plots = None  # type: ignore[assignment]
from .precision import amp_dtype_label, autocast_context, build_grad_scaler, resolve_amp_dtype
from .artifacts import ArtifactMirror, discover_drive_root
from .seed import seed_everything

__all__ = [
    "ArtifactMirror",
    "CSVLogger",
    "amp_dtype_label",
    "autocast_context",
    "build_grad_scaler",
    "discover_drive_root",
    "forecast_metrics",
    "mae",
    "nrmse",
    "r2_score",
    "resolve_amp_dtype",
    "rmse",
    "save_training_plots",
    "save_validation_diagnostic_plots",
    "seed_everything",
]
