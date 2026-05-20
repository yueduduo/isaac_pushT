"""PushT LeRobot 扩散策略的 WebSocket 服务端推理封装与仿真端远程客户端。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from utils.dataset import STATE_KEY, TOP_CAMERA_KEY, WRIST_CAMERA_KEY
from utils.lerobot_push_diffusion import PushTLerobotTrainer, ensure_batched_obs
from utils.remote_policy.websocket_client import WebsocketClientPolicy


class PushDiffusionWebsocketServicer:
    """
    接收未归一化的观测（与 eval / 数据集一致），在服务端用 LeRobot Normalizer 预处理、
    推理后再 Unnormalizer 反归一化动作。
    """

    def __init__(
        self,
        trainer: PushTLerobotTrainer,
        *,
        device: torch.device,
        horizon: int,
        action_dim_per_step: int,
    ) -> None:
        self._trainer = trainer
        self._device = device
        self._horizon = horizon
        self._action_dim_per_step = action_dim_per_step

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        wrist = torch.from_numpy(obs[WRIST_CAMERA_KEY]).to(self._device, dtype=torch.float32)
        top_cam = torch.from_numpy(obs[TOP_CAMERA_KEY]).to(self._device, dtype=torch.float32)
        state = torch.from_numpy(obs[STATE_KEY]).to(self._device, dtype=torch.float32).reshape(-1)
        obs_dict = ensure_batched_obs(
            {
                WRIST_CAMERA_KEY: wrist,
                TOP_CAMERA_KEY: top_cam,
                STATE_KEY: state,
            }
        )
        with torch.no_grad():
            flat = self._trainer.sample_action_chunk_flat(obs_dict).squeeze(0)
        flat_np = flat.detach().cpu().numpy().astype(np.float32)
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
        "normalization": "lerobot_normalizer_processor",
    }
