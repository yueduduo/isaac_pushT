"""Transformer-based flow matching model for PushT actions."""

from __future__ import annotations

import torch
import torch.nn as nn

from algo.common.networks import MultimodalTransformerEncoder, TransformerActionHead


class FlowMatchingTransformerModel(nn.Module):
    """Predict velocity field v_t(x_t | obs) for flow matching."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        image_size: int = 224,
        embed_dim: int = 256,
        patch_size: int = 16,
        obs_layers: int = 4,
        head_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        obs_dim_feedforward: int | None = None,
        head_dim_feedforward: int | None = None,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.obs_encoder = MultimodalTransformerEncoder(
            image_size=image_size,
            in_channels=3,
            state_dim=state_dim,
            embed_dim=embed_dim,
            patch_size=patch_size,
            num_layers=obs_layers,
            num_heads=num_heads,
            dropout=dropout,
            dim_feedforward=obs_dim_feedforward,
        )
        self.head = TransformerActionHead(
            action_dim=action_dim,
            embed_dim=embed_dim,
            num_layers=head_layers,
            num_heads=num_heads,
            dropout=dropout,
            dim_feedforward=head_dim_feedforward,
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        obs_tokens, _ = self.obs_encoder(obs)
        return self.head(x_t, t, obs_tokens)

    def set_resnet_backbone_trainable(self, trainable: bool) -> None:
        self.obs_encoder.set_resnet_backbone_trainable(trainable)
