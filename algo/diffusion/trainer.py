"""Trainer and sampler for diffusion policy."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from tqdm import tqdm

from algo.diffusion.model import DiffusionTransformerModel


@dataclass
class DiffusionConfig:
    num_diffusion_steps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    lr: float = 1e-4
    grad_clip_norm: float = 1.0
    weight_decay: float = 1e-4


class DiffusionTrainer:
    """Train and sample from a DDPM-style action diffusion model."""

    def __init__(self, model: DiffusionTransformerModel, config: DiffusionConfig, device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

        betas = torch.linspace(config.beta_start, config.beta_end, config.num_diffusion_steps, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars

    def _move_obs(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # 此时 obs 已经在 collate_fn 中搬运到了 GPU
        return obs

    @staticmethod
    def _flatten_action(action: torch.Tensor) -> torch.Tensor:
        if action.dim() <= 2:
            return action
        return action.flatten(1)

    def train_step(self, batch: dict) -> float:
        self.model.train()
        obs = self._move_obs(batch["obs"])
        action = batch["action"]
        action = self._flatten_action(action)
        bsz = action.shape[0]

        t = torch.randint(0, self.config.num_diffusion_steps, (bsz,), device=self.device)
        noise = torch.randn_like(action)
        alpha_bar_t = self.alpha_bars[t].unsqueeze(-1)
        noisy_action = torch.sqrt(alpha_bar_t) * action + torch.sqrt(1.0 - alpha_bar_t) * noise

        pred_noise = self.model(noisy_action, t, obs)
        loss = F.mse_loss(pred_noise, noise)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
        self.optimizer.step()
        return float(loss.item())

    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> float:
        self.model.train()
        losses: list[float] = []
        pbar = tqdm(dataloader, desc="  Training", leave=False, mininterval=1.0)
        for i, batch in enumerate(pbar):
            if i == 0:
                torch.cuda.synchronize()
            loss_val = self.train_step(batch)
            losses.append(loss_val)
            pbar.set_postfix(loss=f"{loss_val:.4f}")
        return float(sum(losses) / max(len(losses), 1))

    @torch.no_grad()
    def sample_actions(
        self,
        obs: dict[str, torch.Tensor],
        num_steps: int | None = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        self.model.eval()
        obs = self._move_obs(obs)
        bsz = obs["observation.state"].shape[0]
        action_dim = self.model.action_dim
        x = torch.randn(bsz, action_dim, device=self.device)

        total_steps = self.config.num_diffusion_steps if num_steps is None else min(num_steps, self.config.num_diffusion_steps)
        for t_idx in reversed(range(total_steps)):
            t = torch.full((bsz,), t_idx, device=self.device, dtype=torch.long)
            pred_noise = self.model(x, t, obs)

            alpha = self.alphas[t_idx]
            alpha_bar = self.alpha_bars[t_idx]
            beta = self.betas[t_idx]
            mean = (x - (beta / torch.sqrt(1.0 - alpha_bar)) * pred_noise) / torch.sqrt(alpha)
            if t_idx > 0:
                # Use posterior variance Var(q(x_{t-1} | x_t, x_0)) rather than raw beta.
                # This matches the DDPM sampling equation for epsilon-prediction parameterization.
                alpha_bar_prev = self.alpha_bars[t_idx - 1]
                posterior_var = beta * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
                posterior_var = torch.clamp(posterior_var, min=1e-20)
                if deterministic:
                    x = mean
                else:
                    x = mean + torch.sqrt(posterior_var) * torch.randn_like(x)
            else:
                x = mean
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
