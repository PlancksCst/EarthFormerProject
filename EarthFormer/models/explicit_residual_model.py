"""Explicit residual heads for EarthFormer SEVIRI experiments."""

from __future__ import annotations

from typing import Any, Iterator

import torch
from torch import nn

from earthformer_migration.model import EarthFormerSEVIRIMigration


class LatentSummaryResidualHead(nn.Module):
    """Force residual prediction through per-timestep latent summary stats."""

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        output_length: int = 13,
        residual_scale: float = 0.3,
    ) -> None:
        super().__init__()
        self.output_length = int(output_length)
        self.residual_scale = float(residual_scale)
        self.gru = nn.GRU(
            input_size=2 * int(latent_dim),
            hidden_size=int(hidden_dim),
            batch_first=True,
        )
        self.proj = nn.Linear(int(hidden_dim), 1)

    def forward_raw(self, latent: torch.Tensor) -> torch.Tensor:
        """Return unbounded residual logits shaped ``(B,T)``."""
        if latent.ndim != 5:
            raise ValueError(f"Expected latent shape (B,T,H,W,C), got {tuple(latent.shape)}")
        mean = latent.mean(dim=(2, 3))
        std = latent.std(dim=(2, 3), unbiased=False)
        summary = torch.cat([mean, std], dim=-1)
        encoded, _ = self.gru(summary)
        raw = self.proj(encoded).squeeze(-1)
        return raw[:, : self.output_length]

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return bounded residual and raw residual logits."""
        raw = self.forward_raw(latent)
        return self.residual_scale * torch.tanh(raw), raw


class ExplicitResidualModel(nn.Module):
    """EarthFormer backbone with a forced latent-summary CSI residual head."""

    def __init__(
        self,
        earthformer: EarthFormerSEVIRIMigration,
        output_length: int = 13,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        residual_scale: float = 0.3,
    ) -> None:
        super().__init__()
        self.earthformer = earthformer
        self.output_length = int(output_length)
        self.residual_head = LatentSummaryResidualHead(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_length=output_length,
            residual_scale=residual_scale,
        )

    def forward(
        self,
        x: torch.Tensor,
        auxiliary_features: torch.Tensor | None = None,
        return_debug: bool = False,
        return_components: bool = False,
    ) -> Any:
        """Return predicted residuals, optionally with diagnostics."""
        del auxiliary_features
        latent = self.earthformer.forward_latent(x, return_trace=False)
        pred_residual, raw_residual = self.residual_head(latent)
        if return_debug or return_components:
            return {
                "pred_residual": pred_residual,
                "raw_residual": raw_residual,
                "pre_head_latent": latent,
                "gate": None,
            }
        return pred_residual

    def earthformer_parameters(self) -> Iterator[nn.Parameter]:
        """Iterate over pretrained EarthFormer parameters."""
        return self.earthformer.parameters()

    def readout_parameters(self) -> Iterator[nn.Parameter]:
        """Iterate over residual-head parameters."""
        return self.residual_head.parameters()

    def freeze_earthformer(self) -> None:
        """Freeze the pretrained EarthFormer backbone."""
        for parameter in self.earthformer_parameters():
            parameter.requires_grad = False

    def unfreeze_earthformer(self) -> None:
        """Unfreeze the pretrained EarthFormer backbone."""
        for parameter in self.earthformer_parameters():
            parameter.requires_grad = True

    def query_similarity_matrix(self, steps: int | None = None) -> torch.Tensor:
        """Return a harmless placeholder for legacy diagnostics."""
        horizon = int(steps or self.output_length)
        device = next(self.parameters()).device
        return torch.eye(horizon, device=device)

    def query_similarity_stats(self, steps: int | None = None) -> dict[str, float]:
        """Return neutral placeholder stats for legacy diagnostics."""
        del steps
        return {"mean": 0.0, "min": 0.0, "max": 0.0}

    def query_diversity_loss(self, steps: int | None = None) -> torch.Tensor:
        """Return zero because this head has no output-query table."""
        del steps
        return next(self.parameters()).new_zeros(())


class ExplicitResidualGatedModel(ExplicitResidualModel):
    """Explicit residual model with an auxiliary per-hour residual gate."""

    def __init__(
        self,
        earthformer: EarthFormerSEVIRIMigration,
        output_length: int = 13,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        residual_scale: float = 0.3,
        auxiliary_dim: int = 9,
    ) -> None:
        super().__init__(
            earthformer=earthformer,
            output_length=output_length,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            residual_scale=residual_scale,
        )
        gate_hidden = max(8, int(hidden_dim) // 2)
        self.aux_gate = nn.Sequential(
            nn.Linear(int(auxiliary_dim), gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        auxiliary_features: torch.Tensor | None = None,
        return_debug: bool = False,
        return_components: bool = False,
    ) -> Any:
        """Return gated predicted residuals, optionally with diagnostics."""
        if auxiliary_features is None:
            raise KeyError("explicit_residual_gated requires auxiliary_features in the batch.")
        latent = self.earthformer.forward_latent(x, return_trace=False)
        image_residual, raw_residual = self.residual_head(latent)
        if auxiliary_features.ndim != 3:
            raise ValueError(
                "Expected auxiliary_features with shape (B,T,F), "
                f"got {tuple(auxiliary_features.shape)}"
            )
        gate = torch.sigmoid(self.aux_gate(auxiliary_features.float()).squeeze(-1))
        gate = gate[:, : image_residual.shape[1]]
        pred_residual = gate * image_residual
        if return_debug or return_components:
            return {
                "pred_residual": pred_residual,
                "raw_residual": raw_residual,
                "image_residual": image_residual,
                "pre_head_latent": latent,
                "gate": gate,
            }
        return pred_residual

    def readout_parameters(self) -> Iterator[nn.Parameter]:
        """Iterate over residual-head and gate parameters."""
        yield from self.residual_head.parameters()
        yield from self.aux_gate.parameters()
