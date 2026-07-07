"""Model construction utilities for EarthFormer SEVIRI training."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREP_MODELS_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PREP_MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(PREP_MODELS_ROOT))

from configs.config import TrainingConfig  # noqa: E402
from earthformer_migration.model import (  # noqa: E402
    EarthFormerSEVIRIMigration,
    build_seviri_earthformer,
    ensure_sevir_pretrained_checkpoint,
    load_sevir_pretrained_weights,
)
from models.explicit_residual_model import (  # noqa: E402
    ExplicitResidualGatedModel,
    ExplicitResidualModel,
)
from models.perceiver_model import EarthFormerPerceiverReadoutModel  # noqa: E402
from readout import PerceiverReadout  # noqa: E402


def build_training_model(config: TrainingConfig) -> EarthFormerSEVIRIMigration:
    """Build the migrated EarthFormer model for backbone fine-tuning."""
    base_model = build_seviri_earthformer(
        image_size=config.image_size,
        input_length=config.input_length,
        output_length=config.output_length,
        input_channels=config.input_channels,
        output_channels=config.output_channels,
    )
    checkpoint_path = config.pretrained_checkpoint
    if checkpoint_path is None:
        checkpoint_path = ensure_sevir_pretrained_checkpoint()
    else:
        checkpoint_path = ensure_sevir_pretrained_checkpoint(checkpoint_path)
    load_sevir_pretrained_weights(base_model, checkpoint_path=checkpoint_path)
    return EarthFormerSEVIRIMigration(base_model)


def build_perceiver_readout_model(config: TrainingConfig) -> EarthFormerPerceiverReadoutModel:
    """Build EarthFormer with a Perceiver IO readout after `pre_head_latent`."""
    earthformer = build_training_model(config)
    if config.fix_preset == "explicit_residual_head":
        model = ExplicitResidualModel(
            earthformer=earthformer,
            output_length=config.output_length,
            latent_dim=config.readout_latent_dim,
            hidden_dim=config.regression_hidden_dim,
            residual_scale=config.residual_scale,
        )
        if config.freeze_earthformer:
            model.freeze_earthformer()
        return model
    if config.fix_preset == "explicit_residual_gated":
        model = ExplicitResidualGatedModel(
            earthformer=earthformer,
            output_length=config.output_length,
            latent_dim=config.readout_latent_dim,
            hidden_dim=config.regression_hidden_dim,
            residual_scale=config.residual_scale,
            auxiliary_dim=config.auxiliary_feature_dim,
        )
        if config.freeze_earthformer:
            model.freeze_earthformer()
        return model
    num_queries = config.num_output_queries or config.output_length
    readout = PerceiverReadout(
        latent_dim=config.readout_latent_dim,
        query_dim=config.query_dimension,
        num_queries=num_queries,
        num_attention_heads=config.num_attention_heads,
        dropout=config.readout_dropout,
        regression_hidden_dim=config.regression_hidden_dim,
        use_hour_query_embedding=getattr(config, "use_hour_query_embedding", True),
        query_hour_embedding_dim=getattr(config, "query_hour_embedding_dim", None),
        use_auxiliary_features=getattr(config, "use_auxiliary_features", False),
        auxiliary_dim=getattr(config, "auxiliary_feature_dim", 0),
    )
    model = EarthFormerPerceiverReadoutModel(earthformer=earthformer, readout=readout)
    if config.freeze_earthformer:
        model.freeze_earthformer()
    return model


def move_to_device(model: nn.Module, device: torch.device) -> nn.Module:
    """Move a model to a device and return it for chaining."""
    return model.to(device)
