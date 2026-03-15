"""Inference wrapper for flow mapping policy."""

from __future__ import annotations

from pathlib import Path

import torch

from algo.flow_mapping.model import FlowMappingTransformerModel
from algo.flow_mapping.trainer import FlowMappingTrainer, FlowMatchingConfig


class FlowMappingPolicy:
    """High-level policy API for train/eval scripts."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: torch.device,
        flow_cfg: FlowMatchingConfig | None = None,
    ):
        self.device = device
        self.model = FlowMappingTransformerModel(state_dim=state_dim, action_dim=action_dim)
        self.trainer = FlowMappingTrainer(self.model, flow_cfg or FlowMatchingConfig(), device=device)

    @torch.no_grad()
    def act(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        if obs["observation.state"].dim() == 1:
            obs = {k: v.unsqueeze(0) for k, v in obs.items()}
        action = self.trainer.sample_actions(obs)
        return action.squeeze(0)

    def save(self, ckpt_path: str | Path) -> None:
        ckpt_path = Path(ckpt_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.trainer.state_dict(), ckpt_path)

    def load(self, ckpt_path: str | Path) -> None:
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.trainer.load_state_dict(checkpoint)
