"""Shared Transformer networks for multimodal PushT policies."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class SinusoidalTimeEmbedding(nn.Module):
    """Classic sinusoidal embedding for discrete/continuous timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        device = t.device
        t = t.float().view(-1, 1)
        exponent = torch.arange(half_dim, device=device).float()
        exponent = -math.log(10000.0) * exponent / max(half_dim - 1, 1)
        freqs = torch.exp(exponent).view(1, -1)
        angles = t * freqs
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


class ResNet18ImageEncoder(nn.Module):
    """Pretrained ResNet18 encoder that outputs one image token."""

    def __init__(self, embed_dim: int):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # [B, 512, 1, 1]
        self.proj = nn.Linear(512, embed_dim)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input should be [B, 3, H, W] in [0, 1].
        x = (x - self.mean) / self.std
        feat = self.backbone(x).flatten(1)  # [B, 512]
        return self.proj(feat).unsqueeze(1)  # [B, 1, D]

    def set_backbone_trainable(self, trainable: bool) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = trainable


class MultimodalTransformerEncoder(nn.Module):
    """Encode front/back images and state vector into context tokens."""

    def __init__(
        self,
        image_size: int = 224,
        in_channels: int = 3,
        state_dim: int = 21,
        embed_dim: int = 256,
        patch_size: int = 16,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Keep arguments for API compatibility with callers.
        _ = image_size, in_channels, patch_size

        self.embed_dim = embed_dim
        self.image_encoder = ResNet18ImageEncoder(embed_dim)
        self.state_embed = nn.Linear(state_dim, embed_dim)

        self.front_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.back_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.state_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.front_type = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.back_type = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.state_type = nn.Parameter(torch.zeros(1, 1, embed_dim))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.front_pos, std=0.02)
        nn.init.trunc_normal_(self.back_pos, std=0.02)
        nn.init.trunc_normal_(self.state_pos, std=0.02)
        nn.init.trunc_normal_(self.front_type, std=0.02)
        nn.init.trunc_normal_(self.back_type, std=0.02)
        nn.init.trunc_normal_(self.state_type, std=0.02)

    def forward(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        front = obs["observation.front_wrist_camera_image"]
        back = obs["observation.back_wrist_camera_image"]
        state = obs["observation.state"]

        front_tokens = self.image_encoder(front) + self.front_pos + self.front_type
        back_tokens = self.image_encoder(back) + self.back_pos + self.back_type
        state_token = self.state_embed(state).unsqueeze(1) + self.state_pos + self.state_type

        tokens = torch.cat([front_tokens, back_tokens, state_token], dim=1)
        encoded = self.norm(self.encoder(tokens))
        pooled = encoded.mean(dim=1)
        return encoded, pooled

    def set_resnet_backbone_trainable(self, trainable: bool) -> None:
        self.image_encoder.set_backbone_trainable(trainable)


class TransformerActionHead(nn.Module):
    """Transformer head conditioned on obs + action + time tokens."""

    def __init__(
        self,
        action_dim: int,
        embed_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.action_proj = nn.Linear(action_dim, embed_dim)
        self.time_embed = SinusoidalTimeEmbedding(embed_dim)
        self.time_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.out = nn.Linear(embed_dim, action_dim)

    def forward(
        self,
        x_action: torch.Tensor,
        t: torch.Tensor,
        obs_tokens: torch.Tensor,
    ) -> torch.Tensor:
        action_token = self.action_proj(x_action).unsqueeze(1)
        time_token = self.time_proj(self.time_embed(t)).unsqueeze(1)
        tokens = torch.cat([action_token, time_token, obs_tokens], dim=1)
        encoded = self.norm(self.encoder(tokens))
        return self.out(encoded[:, 0])
