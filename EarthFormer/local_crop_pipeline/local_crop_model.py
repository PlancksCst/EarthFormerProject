"""Lightweight station-local CNN-GRU model for 64x64 SEVIRI crops."""

from __future__ import annotations

import torch
from torch import nn


class LocalCropCNNGRU(nn.Module):
    """Encode each local crop with a CNN, then forecast CSI with a GRU."""

    def __init__(
        self,
        input_channels: int = 7,
        output_length: int = 13,
        cnn_feature_dim: int = 128,
        gru_hidden_dim: int = 128,
        use_auxiliary_features: bool = False,
        auxiliary_dim: int = 9,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.output_length = int(output_length)
        self.use_auxiliary_features = bool(use_auxiliary_features)
        self.auxiliary_dim = int(auxiliary_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, cnn_feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_feature_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.temporal = nn.GRU(
            input_size=cnn_feature_dim,
            hidden_size=gru_hidden_dim,
            batch_first=True,
        )
        head_input_dim = gru_hidden_dim + (self.auxiliary_dim if self.use_auxiliary_features else 0)
        self.head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, gru_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        auxiliary_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return CSI predictions shaped ``(B,T)`` from ``(B,T,C,64,64)`` crops."""
        if x.ndim != 5:
            raise ValueError(f"Expected input shape (B,T,C,H,W), got {tuple(x.shape)}")
        batch_size, steps, channels, height, width = x.shape
        if height != 64 or width != 64:
            raise ValueError(f"LocalCropCNNGRU expects 64x64 crops, got {height}x{width}.")
        encoded = self.encoder(x.reshape(batch_size * steps, channels, height, width))
        features = encoded.flatten(1).reshape(batch_size, steps, -1)
        sequence, _ = self.temporal(features)
        sequence = sequence[:, : self.output_length]
        if self.use_auxiliary_features:
            if auxiliary_features is None:
                raise KeyError("Auxiliary features are enabled but not provided.")
            if auxiliary_features.ndim != 3:
                raise ValueError(
                    "Expected auxiliary_features shape (B,T,F), "
                    f"got {tuple(auxiliary_features.shape)}."
                )
            aux = auxiliary_features[:, : sequence.shape[1]].to(
                device=sequence.device,
                dtype=sequence.dtype,
            )
            sequence = torch.cat([sequence, aux], dim=-1)
        prediction = self.head(sequence).squeeze(-1)
        return prediction.clamp(0.0, 1.3)
