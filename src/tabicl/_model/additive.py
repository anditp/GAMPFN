from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
from torch import Tensor, nn

from .embedding import ColEmbedding
from .layers import MultiheadAttentionBlock


@dataclass(frozen=True)
class AdditiveContext:
    """Reusable dataset representation for evaluating additive effects."""

    memory: Tensor
    x_mean: Tensor
    x_scale: Tensor
    y_mean: Tensor
    y_scale: Tensor
    effect_center: Tensor
    feature_mask: Tensor


class AdditivePerceiver(nn.Module):
    """Predict with centered feature effects decoded from per-feature memories.

    ``ColEmbedding`` maps context cells from ``(B, T, D)`` to distribution-aware
    embeddings of shape ``(B, T, D, E)``. Row-wise attention contextualizes
    each feature with the other features observed in the same row while retaining
    that shape. Shared latent queries then independently pool the ``T`` embeddings
    of each feature into ``K`` memory tokens, producing ``(B, D, K, E)``.
    Finally, scalar queries cross-attend to the corresponding feature memory.
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        col_num_blocks: int = 3,
        col_num_inds: int = 128,
        num_memory_tokens: int = 8,
        num_row_blocks: int = 2,
        num_decoder_blocks: int = 2,
        dim_feedforward: int | None = None,
        dropout: float = 0.0,
        col_ssmax: bool | str = False,
        activation: str = "gelu",
        norm_first: bool = True,
        bias_free_ln: bool = False,
        zero_init: bool = False,
        recompute: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        if min(col_num_blocks, col_num_inds, num_memory_tokens, num_row_blocks, num_decoder_blocks) < 1:
            raise ValueError("block and memory counts must be positive")

        hidden = dim_feedforward or 4 * d_model
        block_kwargs = dict(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=hidden,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            bias_free_ln=bias_free_ln,
            zero_init=zero_init,
        )
        self.eps = eps
        self.col_embedder = ColEmbedding(
            embed_dim=d_model,
            num_blocks=col_num_blocks,
            nhead=nhead,
            dim_feedforward=hidden,
            num_inds=col_num_inds,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            bias_free_ln=bias_free_ln,
            affine=False,
            feature_group=False,
            target_aware=True,
            max_classes=0,
            reserve_cls_tokens=0,
            ssmax=col_ssmax,
            zero_init=zero_init,
            recompute=recompute,
        )
        self.query_embedding = self._scalar_mlp(1, d_model)
        self.memory_tokens = nn.Parameter(
            torch.randn(num_memory_tokens, d_model) / sqrt(d_model)
        )
        self.row_contextualizer = nn.ModuleList(
            [MultiheadAttentionBlock(**block_kwargs) for _ in range(num_row_blocks)]
        )
        self.memory_cross_attention = MultiheadAttentionBlock(**block_kwargs)
        self.memory_self_attention = MultiheadAttentionBlock(**block_kwargs)
        self.decoder = nn.ModuleList(
            [MultiheadAttentionBlock(**block_kwargs) for _ in range(num_decoder_blocks)]
        )
        self.effect_head = nn.Linear(d_model, 1, bias=False)

    @staticmethod
    def _scalar_mlp(input_dim: int, d_model: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def encode_context(
        self,
        x_context: Tensor,
        y_context: Tensor,
        feature_mask: Tensor | None = None,
    ) -> AdditiveContext:
        """Encode one or more datasets into reusable per-feature memories."""
        self._validate_context(x_context, y_context)
        batch_size, num_rows, num_features = x_context.shape
        feature_mask = self._validate_feature_mask(
            feature_mask, batch_size, num_features, x_context.device
        )

        x_mean = x_context.mean(dim=1, keepdim=True)
        x_scale = x_context.var(dim=1, keepdim=True, unbiased=False).sqrt().clamp_min(self.eps)
        y_mean = y_context.mean(dim=1, keepdim=True)
        y_scale = y_context.var(dim=1, keepdim=True, unbiased=False).sqrt().clamp_min(self.eps)
        x_normalized = (x_context - x_mean) / x_scale
        y_normalized = (y_context - y_mean) / y_scale

        column_embeddings = self.col_embedder(x_normalized, y_normalized).to(self.memory_tokens.dtype)
        feature_padding_mask = (~feature_mask).unsqueeze(1).expand(-1, num_rows, -1)
        for block in self.row_contextualizer:
            column_embeddings = block(
                q=column_embeddings,
                key_padding_mask=feature_padding_mask,
            )
        column_embeddings = column_embeddings * feature_mask[:, None, :, None]

        feature_tokens = column_embeddings.transpose(1, 2).reshape(
            batch_size * num_features, num_rows, -1
        )
        memory = self.memory_tokens.expand(batch_size * num_features, -1, -1)
        memory = self.memory_cross_attention(
            q=memory,
            k=feature_tokens,
            v=feature_tokens,
        )
        memory = self.memory_self_attention(q=memory)
        memory = memory.reshape(batch_size, num_features, memory.shape[-2], memory.shape[-1])
        memory = memory * feature_mask[:, :, None, None]

        raw_context_effects = self._decode_normalized(
            memory, x_normalized.transpose(1, 2)
        )
        effect_center = raw_context_effects.mean(dim=-1, keepdim=True)
        return AdditiveContext(
            memory=memory,
            x_mean=x_mean,
            x_scale=x_scale,
            y_mean=y_mean,
            y_scale=y_scale,
            effect_center=effect_center,
            feature_mask=feature_mask,
        )

    def predict_effects(self, context: AdditiveContext, x_query: Tensor) -> Tensor:
        """Evaluate every centered feature effect at its corresponding query values."""
        if x_query.ndim != 3:
            raise ValueError("x_query must have shape (batch, rows, features)")
        batch_size, _, num_features = x_query.shape
        if context.memory.shape[:2] != (batch_size, num_features):
            raise ValueError("x_query batch and feature dimensions must match the context")

        x_normalized = (x_query - context.x_mean) / context.x_scale
        raw_effects = self._decode_normalized(
            context.memory, x_normalized.transpose(1, 2)
        )
        effects = (raw_effects - context.effect_center).transpose(1, 2)
        effects = effects * context.y_scale.unsqueeze(-1)
        return effects * context.feature_mask.unsqueeze(1)

    def forward(
        self,
        x_context: Tensor,
        y_context: Tensor,
        x_query: Tensor,
        feature_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return predictions and their feature-wise additive decomposition."""
        context = self.encode_context(x_context, y_context, feature_mask)
        effects = self.predict_effects(context, x_query)
        prediction = context.y_mean + effects.sum(dim=-1)
        return prediction, effects

    def _decode_normalized(self, memory: Tensor, values: Tensor) -> Tensor:
        batch_size, num_features, num_queries = values.shape
        queries = self.query_embedding(values.unsqueeze(-1)).reshape(
            batch_size * num_features, num_queries, -1
        )
        flat_memory = memory.reshape(
            batch_size * num_features, memory.shape[-2], memory.shape[-1]
        )
        for block in self.decoder:
            queries = block(q=queries, k=flat_memory, v=flat_memory)
        return self.effect_head(queries).squeeze(-1).reshape(
            batch_size, num_features, num_queries
        )

    @staticmethod
    def _validate_context(x_context: Tensor, y_context: Tensor) -> None:
        if x_context.ndim != 3:
            raise ValueError("x_context must have shape (batch, rows, features)")
        if y_context.ndim != 2 or y_context.shape != x_context.shape[:2]:
            raise ValueError("y_context must have shape (batch, rows)")
        if x_context.shape[1] == 0 or x_context.shape[2] == 0:
            raise ValueError("context rows and features must be non-empty")

    @staticmethod
    def _validate_feature_mask(
        feature_mask: Tensor | None,
        batch_size: int,
        num_features: int,
        device: torch.device,
    ) -> Tensor:
        if feature_mask is None:
            return torch.ones(batch_size, num_features, dtype=torch.bool, device=device)
        if feature_mask.shape != (batch_size, num_features):
            raise ValueError("feature_mask must have shape (batch, features)")
        feature_mask = feature_mask.to(device=device, dtype=torch.bool)
        if not feature_mask.any(dim=1).all():
            raise ValueError("each dataset must contain at least one feature")
        return feature_mask
