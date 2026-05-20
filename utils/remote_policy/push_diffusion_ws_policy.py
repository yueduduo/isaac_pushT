"""PushT LeRobot 扩散策略的 WebSocket 服务端推理封装与仿真端远程客户端。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from utils.dataset import STATE_KEY, TOP_CAMERA_KEY, WRIST_CAMERA_KEY
from utils.lerobot_push_diffusion import PushTLerobotTrainer, ensure_batched_obs
from utils.normalization import denormalize_action, normalize_state
from utils.remote_policy.websocket_client import WebsocketClientPolicy


class PushDiffusionWebsocketServicer:
    """
    接收未归一化的观测（与 eval / 数据集一致），在服务端完成归一化、推理、动作反归一化。
    输入 obs 键：WRIST_CAMERA_KEY / TOP_CAMERA_KEY（float32 CHW [0,1]）、STATE_KEY（float32 21 维物理状态）。
    返回 {"flat_action": (horizon * action_dim,) float32}，已为物理空间动作。
    """

    def __init__(
        self,
        trainer: PushTLerobotTrainer,
        *,
        device: torch.device,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
        horizon: int,
        action_dim_per_step: int,
    ) -> None:
        self._trainer = trainer
        self._device = device
        self._state_mean = state_mean
        self._state_std = state_std
        self._action_mean = action_mean
        self._action_std = action_std
        self._horizon = horizon
        self._action_dim_per_step = action_dim_per_step

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        wrist = torch.from_numpy(obs[WRIST_CAMERA_KEY]).to(self._device, dtype=torch.float32)
        top_cam = torch.from_numpy(obs[TOP_CAMERA_KEY]).to(self._device, dtype=torch.float32)
        state = torch.from_numpy(obs[STATE_KEY]).to(self._device, dtype=torch.float32).reshape(-1)
        state = normalize_state(state, self._state_mean, self._state_std)
        obs_dict = ensure_batched_obs(
            {
                WRIST_CAMERA_KEY: wrist,
                TOP_CAMERA_KEY: top_cam,
                STATE_KEY: state,
            }
        )
        with torch.no_grad():
            flat = self._trainer.sample_action_chunk_flat(obs_dict).squeeze(0)
            seq = flat.view(self._horizon, self._action_dim_per_step)
            seq = denormalize_action(seq, self._action_mean, self._action_std)
        flat_np = seq.reshape(-1).detach().cpu().numpy().astype(np.float32)
        return {"flat_action": flat_np}


class RemotePushDiffusionPolicy:
    """仿真进程内占位策略：`act` 将 torch 观测发到远端并取回反归一化后的 flat 动作。"""

    def __init__(self, host: str, port: int, *, out_device: torch.device) -> None:
        self._client = WebsocketClientPolicy(host, port)
        self.device = out_device
        self.server_metadata = self._client.get_server_metadata()

    def act(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        payload = {
            k: v.detach().cpu().contiguous().numpy().astype(np.float32) for k, v in obs.items()
        }
        out = self._client.infer(payload)
        return torch.from_numpy(out["flat_action"]).to(device=self.device, dtype=torch.float32)

    def close(self) -> None:
        self._client.close()


def build_server_metadata(*, horizon: int, action_dim_per_step: int, checkpoint: Path) -> dict[str, Any]:
    return {
        "policy": "isaac_pusht_lerobot_diffusion",
        "horizon": horizon,
        "action_dim_per_step": action_dim_per_step,
        "checkpoint": str(checkpoint.resolve()),
        "observation_keys": [WRIST_CAMERA_KEY, TOP_CAMERA_KEY, STATE_KEY],
    }
