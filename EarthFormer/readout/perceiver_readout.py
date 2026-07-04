"""Perceiver IO style output-query readout for EarthFormer latents."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class PerceiverReadout(nn.Module):
    """Decode per-timestep CSI values from EarthFormer pre-head latents.

    The module implements only the Perceiver IO output-query decoding idea:
    learnable output queries cross-attend to encoded tokens. It does not
    implement a Perceiver encoder, latent bottleneck, Fourier features, or
    iterative latent processing.

    Input shape:
        `(B, T, H, W, C)`

    Output shape:
        `(B, T)`
    """

    def __init__(
        self,
        latent_dim: int = 16,
        query_dim: int = 64,
        num_queries: int = 12,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
        regression_hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if query_dim <= 0:
            raise ValueError("query_dim must be positive")
        if num_queries <= 0:
            raise ValueError("num_queries must be positive")
        if num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if query_dim % num_attention_heads != 0:
            raise ValueError(
                "query_dim must be divisible by num_attention_heads; got "
                f"{query_dim} and {num_attention_heads}"
            )
        if regression_hidden_dim <= 0:
            raise ValueError("regression_hidden_dim must be positive")

        self.latent_dim = int(latent_dim)
        self.query_dim = int(query_dim)
        self.num_queries = int(num_queries)
        self.num_attention_heads = int(num_attention_heads)
        self.dropout = float(dropout)
        self.regression_hidden_dim = int(regression_hidden_dim)

        self.output_queries = nn.Parameter(torch.empty(self.num_queries, self.query_dim))
        self.query_norm = nn.LayerNorm(self.query_dim)
        self.token_norm = nn.LayerNorm(self.latent_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.query_dim,
            num_heads=self.num_attention_heads,
            dropout=self.dropout,
            kdim=self.latent_dim,
            vdim=self.latent_dim,
            batch_first=True,
        )
        self.regression = nn.Sequential(
            nn.Linear(self.query_dim, self.regression_hidden_dim),
            nn.GELU(),
            nn.Linear(self.regression_hidden_dim, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize only the new readout parameters."""
        nn.init.normal_(self.output_queries, mean=0.0, std=0.02)

    def flatten_spatial_tokens(self, pre_head_latent: torch.Tensor) -> torch.Tensor:
        """Flatten `(H, W)` into deterministic row-major spatial tokens."""
        if pre_head_latent.ndim != 5:
            raise ValueError(
                "Expected pre_head_latent with shape (B,T,H,W,C), got "
                f"{tuple(pre_head_latent.shape)}"
            )
        bsz, steps, height, width, channels = pre_head_latent.shape
        if channels != self.latent_dim:
            raise ValueError(f"Expected latent_dim={self.latent_dim}, got {channels}")
        return pre_head_latent.reshape(bsz, steps, height * width, channels).contiguous()

    def timestep_queries(self, batch_size: int, steps: int) -> torch.Tensor:
        """Return the first `steps` learnable queries expanded across the batch."""
        if steps > self.num_queries:
            raise ValueError(
                f"Input has T={steps}, but readout was configured with "
                f"num_queries={self.num_queries}"
            )
        queries = self.output_queries[:steps].unsqueeze(0)
        return queries.expand(batch_size, steps, self.query_dim).contiguous()

    def forward(self, pre_head_latent: torch.Tensor, return_debug: bool = False) -> Any:
        """Run independent per-timestep query cross-attention over spatial tokens."""
        spatial_tokens = self.flatten_spatial_tokens(pre_head_latent)
        bsz, steps, num_tokens, _channels = spatial_tokens.shape

        queries = self.timestep_queries(batch_size=bsz, steps=steps)
        normalized_tokens = self.token_norm(spatial_tokens)
        normalized_queries = self.query_norm(queries)

        tokens_flat = normalized_tokens.reshape(bsz * steps, num_tokens, self.latent_dim)
        queries_flat = normalized_queries.reshape(bsz * steps, 1, self.query_dim)

        attention_output, _ = self.cross_attention(
            query=queries_flat,
            key=tokens_flat,
            value=tokens_flat,
            need_weights=False,
        )
        output_embeddings = attention_output.reshape(bsz, steps, self.query_dim)
        regression_output = self.regression(output_embeddings).squeeze(-1)

        if not return_debug:
            return regression_output

        return {
            "prediction": regression_output,
            "flattened_tokens": spatial_tokens,
            "queries": queries,
            "attention_output": output_embeddings,
            "regression_output": regression_output,
        }
