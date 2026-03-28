"""State/action normalization utilities shared by train/eval/inference."""

from __future__ import annotations

import json
from pathlib import Path

import torch


def _clamp_std(std: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.clamp(std, min=eps)


def compute_state_action_stats(
    all_states: torch.Tensor,
    all_actions: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    state_mean = all_states.mean(dim=0)
    state_std = _clamp_std(all_states.std(dim=0, unbiased=False), eps=eps)
    action_mean = all_actions.mean(dim=0)
    action_std = _clamp_std(all_actions.std(dim=0, unbiased=False), eps=eps)
    return state_mean, state_std, action_mean, action_std


def ckpt_norm_path(ckpt_path: str | Path) -> Path:
    path = Path(ckpt_path)
    return path.with_suffix(".norm.json")


def save_norm_stats(
    norm_path: str | Path,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
) -> None:
    path = Path(norm_path)
    payload = {
        "state_mean": state_mean.detach().cpu().tolist(),
        "state_std": state_std.detach().cpu().tolist(),
        "action_mean": action_mean.detach().cpu().tolist(),
        "action_std": action_std.detach().cpu().tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_norm_stats(
    norm_path: str | Path,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    path = Path(norm_path)
    if not path.exists():
        raise FileNotFoundError(f"Normalization file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    dev = device if device is not None else torch.device("cpu")
    state_mean = torch.tensor(payload["state_mean"], dtype=torch.float32, device=dev)
    state_std = torch.tensor(payload["state_std"], dtype=torch.float32, device=dev)
    action_mean = torch.tensor(payload["action_mean"], dtype=torch.float32, device=dev)
    action_std = torch.tensor(payload["action_std"], dtype=torch.float32, device=dev)
    return state_mean, state_std, action_mean, action_std


def normalize_state(
    state: torch.Tensor,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
) -> torch.Tensor:
    return (state - state_mean) / state_std


def normalize_action(
    action: torch.Tensor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
) -> torch.Tensor:
    if action.shape[-1] == action_mean.shape[0]:
        return (action - action_mean) / action_std
    if action.shape[-1] % action_mean.shape[0] != 0:
        raise ValueError(
            f"Action last dim {action.shape[-1]} is incompatible with base dim {action_mean.shape[0]}"
        )
    base = action_mean.shape[0]
    flat = action.reshape(*action.shape[:-1], -1, base)
    norm = (flat - action_mean) / action_std
    return norm.reshape(*action.shape)


def denormalize_action(
    action: torch.Tensor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
) -> torch.Tensor:
    if action.shape[-1] == action_mean.shape[0]:
        return action * action_std + action_mean
    if action.shape[-1] % action_mean.shape[0] != 0:
        raise ValueError(
            f"Action last dim {action.shape[-1]} is incompatible with base dim {action_mean.shape[0]}"
        )
    base = action_mean.shape[0]
    flat = action.reshape(*action.shape[:-1], -1, base)
    denorm = flat * action_std + action_mean
    return denorm.reshape(*action.shape)

