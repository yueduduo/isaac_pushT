"""Trainer and sampler for flow mapping policy."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from tqdm import tqdm

from  algo.flow_mapping.model import FlowMappingTransformerModel


@dataclass
class FlowMatchingConfig:
    lr: float = 1e-4
    grad_clip_norm: float = 1.0
    ode_steps: int = 50


class FlowMappingTrainer:
    """Train and sample from a flow-matching action model."""

    def __init__(self, model: FlowMappingTransformerModel, config: FlowMatchingConfig, device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr, weight_decay=1e-4)

    def _move_obs(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return obs

    @staticmethod
    def _flatten_action(action: torch.Tensor) -> torch.Tensor:
        if action.dim() <= 2:
            return action
        return action.flatten(1)

    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> float:
        self.model.train()
        losses: list[float] = []
        
        pbar = tqdm(dataloader, desc="  Training", leave=False, mininterval=1.0)
        for i, batch in enumerate(pbar):
            if i == 0:
                torch.cuda.synchronize()
            
            obs = self._move_obs(batch["obs"])
            action = batch["action"]
            action = self._flatten_action(action)

            x0 = torch.randn_like(action)
            t = torch.rand(action.shape[0], device=self.device)
            t_expand = t.unsqueeze(-1)
            x_t = (1.0 - t_expand) * x0 + t_expand * action
            target_v = action - x0

            pred_v = self.model(x_t, t, obs)
            loss = F.mse_loss(pred_v, target_v)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            self.optimizer.step()
            
            loss_val = loss.item()
            losses.append(loss_val)
            pbar.set_postfix(loss=f"{loss_val:.4f}")
            
        return float(sum(losses) / max(len(losses), 1))

    @torch.no_grad()
    def sample_actions(self, obs: dict[str, torch.Tensor], ode_steps: int | None = None) -> torch.Tensor:
        self.model.eval()
        obs = self._move_obs(obs)
        bsz = obs["observation.state"].shape[0]
        action_dim = self.model.action_dim

        steps = self.config.ode_steps if ode_steps is None else ode_steps
        dt = 1.0 / steps
        x = torch.randn(bsz, action_dim, device=self.device)
        for i in range(steps):
            t_val = torch.full((bsz,), i / steps, device=self.device)
            v = self.model(x, t_val, obs)
            x = x + dt * v
        return x

    def state_dict(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.__dict__,
        }

    def load_state_dict(self, checkpoint: dict) -> None:
        self.model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
