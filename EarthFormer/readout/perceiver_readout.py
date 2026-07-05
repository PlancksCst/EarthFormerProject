"""Perceiver IO style output-query readout for EarthFormer latents."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


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
        num_queries: int = 13,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
        regression_hidden_dim: int = 32,
        use_hour_query_embedding: bool = True,
        query_hour_embedding_dim: int | None = None,
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
        hour_embedding_dim = query_dim if query_hour_embedding_dim is None else int(query_hour_embedding_dim)
        if hour_embedding_dim <= 0:
            raise ValueError("query_hour_embedding_dim must be positive")

        self.latent_dim = int(latent_dim)
        self.query_dim = int(query_dim)
        self.num_queries = int(num_queries)
        self.num_attention_heads = int(num_attention_heads)
        self.dropout = float(dropout)
        self.regression_hidden_dim = int(regression_hidden_dim)
        self.use_hour_query_embedding = bool(use_hour_query_embedding)
        self.query_hour_embedding_dim = int(hour_embedding_dim)

        self.output_queries = nn.Parameter(torch.empty(self.num_queries, self.query_dim))
        self.hour_embeddings = nn.Parameter(
            torch.empty(self.num_queries, self.query_hour_embedding_dim)
        )
        self.hour_embedding_projection: nn.Module
        if self.query_hour_embedding_dim == self.query_dim:
            self.hour_embedding_projection = nn.Identity()
        else:
            self.hour_embedding_projection = nn.Linear(
                self.query_hour_embedding_dim,
                self.query_dim,
                bias=False,
            )
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
        nn.init.normal_(self.hour_embeddings, mean=0.0, std=0.02)
        if isinstance(self.hour_embedding_projection, nn.Linear):
            nn.init.xavier_uniform_(self.hour_embedding_projection.weight)

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

    def query_basis(self, steps: int | None = None) -> torch.Tensor:
        """Return the per-hour output-query vectors before batch expansion."""
        steps = self.num_queries if steps is None else int(steps)
        if steps > self.num_queries:
            raise ValueError(
                f"Input has T={steps}, but readout was configured with "
                f"num_queries={self.num_queries}"
            )
        queries = self.output_queries[:steps]
        if self.use_hour_query_embedding:
            hour_component = self.hour_embedding_projection(self.hour_embeddings[:steps])
            queries = queries + hour_component
        return queries

    def timestep_queries(self, batch_size: int, steps: int) -> torch.Tensor:
        """Return the first `steps` output queries expanded across the batch."""
        queries = self.query_basis(steps).unsqueeze(0)
        return queries.expand(batch_size, steps, self.query_dim).contiguous()

    def query_similarity_matrix(self, steps: int | None = None) -> torch.Tensor:
        """Return pairwise cosine similarity between effective output queries."""
        queries = self.query_basis(steps)
        normalized = F.normalize(queries.float(), dim=-1, eps=1.0e-8)
        return normalized @ normalized.transpose(0, 1)

    def query_similarity_stats(self, steps: int | None = None) -> dict[str, float]:
        """Return off-diagonal query cosine similarity diagnostics."""
        similarity = self.query_similarity_matrix(steps)
        if similarity.shape[0] <= 1:
            return {"mean": 0.0, "min": 0.0, "max": 0.0}
        off_diagonal = ~torch.eye(
            similarity.shape[0],
            dtype=torch.bool,
            device=similarity.device,
        )
        values = similarity[off_diagonal]
        return {
            "mean": float(values.mean().detach().cpu()),
            "min": float(values.min().detach().cpu()),
            "max": float(values.max().detach().cpu()),
        }

    def query_diversity_loss(self, steps: int | None = None) -> torch.Tensor:
        """Penalize off-diagonal cosine similarity between output queries."""
        similarity = self.query_similarity_matrix(steps)
        if similarity.shape[0] <= 1:
            return similarity.new_zeros(())
        off_diagonal = ~torch.eye(
            similarity.shape[0],
            dtype=torch.bool,
            device=similarity.device,
        )
        return similarity[off_diagonal].pow(2).mean()

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
            "effective_query_basis": self.query_basis(steps),
            "hour_embeddings": self.hour_embeddings[:steps],
            "query_similarity": self.query_similarity_matrix(steps),
            "attention_output": output_embeddings,
            "regression_output": regression_output,
        }
