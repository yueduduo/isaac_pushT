"""Inference wrapper for flow matching policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from algo.flow_matching.model import FlowMatchingTransformerModel
from algo.flow_matching.trainer import FlowMatchingConfig, FlowMatchingTrainer


class FlowMatchingPolicy:
    """High-level policy API for train/eval scripts."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: torch.device,
        matching_cfg: FlowMatchingConfig | None = None,
        model_kw: dict[str, Any] | None = None,
    ):
        self.device = device
        mk = model_kw or {}
        self.model = FlowMatchingTransformerModel(state_dim=state_dim, action_dim=action_dim, **mk)
        self.trainer = FlowMatchingTrainer(self.model, matching_cfg or FlowMatchingConfig(), device=device)

    @torch.no_grad()
    def act(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        if obs["observation.state"].dim() == 1:
            obs = {k: v.unsqueeze(0) for k, v in obs.items()}
        action = self.trainer.sample_actions(obs)
        return action.squeeze(0)

    def save(self, ckpt_path: str | Path, extra_state: dict[str, Any] | None = None) -> None:
        ckpt_path = Path(ckpt_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.trainer.state_dict()
        if extra_state:
            payload = {**payload, **extra_state}
        torch.save(payload, ckpt_path)

    def load(self, ckpt_path: str | Path) -> tuple[int | None, float | None]:
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        step = checkpoint.get("step")
        best_metric = checkpoint.get("best_metric")
        self.trainer.load_state_dict(checkpoint)
        st = int(step) if step is not None else None
        bm = float(best_metric) if best_metric is not None else None
        return st, bm
