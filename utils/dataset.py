"""LeRobot dataset adapters for multimodal PushT policies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader, Dataset, random_split


FRONT_KEY = "observation.front_wrist_camera_image"
BACK_KEY = "observation.back_wrist_camera_image"
STATE_KEY = "observation.state"
ACTION_KEY = "action"


def _to_tensor_chw_float(image: Any) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        tensor = image.float()
    else:
        tensor = torch.as_tensor(np.array(image), dtype=torch.float32)

    if tensor.ndim != 3:
        raise ValueError(f"Image must be rank-3, got shape: {tuple(tensor.shape)}")

    # Support both CHW and HWC.
    if tensor.shape[0] in (1, 3):
        chw = tensor
    else:
        chw = tensor.permute(2, 0, 1)

    # Normalize uint8-like image to [0, 1].
    if chw.max() > 1.0:
        chw = chw / 255.0
    return chw.contiguous()


@dataclass
class DatasetConfig:
    repo_id: str = "isaac_pusht"
    root: str = "data/isaac_pusht"
    front_key: str = FRONT_KEY
    back_key: str = BACK_KEY
    state_key: str = STATE_KEY
    action_key: str = ACTION_KEY
    horizon: int = 16
    preload_in_memory: bool = True


class LeRobotPushTDataset(Dataset):
    """Torch dataset wrapper over LeRobot v3.0 frames."""

    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        self.dataset = LeRobotDataset(repo_id=cfg.repo_id, root=Path(cfg.root))
        self.valid_start_indices = self._build_valid_start_indices()
        self.memory_data: list[dict[str, torch.Tensor | dict[str, torch.Tensor]]] = []
        if self.cfg.preload_in_memory:
            self._preload_all_to_memory()

    def __len__(self) -> int:
        if self.cfg.preload_in_memory:
            return len(self.memory_data)
        return len(self.valid_start_indices)

    @staticmethod
    def _to_episode_id(value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            return int(value.item())
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return None
            return int(value.item())
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _is_same_episode_window(self, start_idx: int) -> bool:
        if self.cfg.horizon <= 1:
            return True
        first_frame = self.dataset[start_idx]
        first_episode = self._to_episode_id(first_frame.get("episode_index"))
        if first_episode is None:
            # If episode id is unavailable, keep behavior compatible with prior version.
            return True
        for step in range(1, self.cfg.horizon):
            frame = self.dataset[start_idx + step]
            if self._to_episode_id(frame.get("episode_index")) != first_episode:
                return False
        return True

    def _build_valid_start_indices(self) -> list[int]:
        if self.cfg.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.cfg.horizon}")
        max_start = len(self.dataset) - self.cfg.horizon + 1
        if max_start <= 0:
            raise ValueError(
                f"Dataset has {len(self.dataset)} frames, cannot build horizon={self.cfg.horizon} samples."
            )
        valid = [idx for idx in range(max_start) if self._is_same_episode_window(idx)]
        if len(valid) == 0:
            raise ValueError("No valid sequence windows found. Check horizon or dataset integrity.")
        return valid

    def _frame_to_sample(self, frame: dict[str, Any]) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        front = _to_tensor_chw_float(frame[self.cfg.front_key])
        back = _to_tensor_chw_float(frame[self.cfg.back_key])
        state = torch.as_tensor(frame[self.cfg.state_key], dtype=torch.float32)
        action = torch.as_tensor(frame[self.cfg.action_key], dtype=torch.float32)
        return {
            "obs": {
                self.cfg.front_key: front,
                self.cfg.back_key: back,
                self.cfg.state_key: state,
            },
            "action": action,
        }

    def _sequence_to_sample(self, start_idx: int) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        frame = self.dataset[start_idx]
        front = _to_tensor_chw_float(frame[self.cfg.front_key])
        back = _to_tensor_chw_float(frame[self.cfg.back_key])
        state = torch.as_tensor(frame[self.cfg.state_key], dtype=torch.float32)
        actions = []
        for step in range(self.cfg.horizon):
            a_t = torch.as_tensor(self.dataset[start_idx + step][self.cfg.action_key], dtype=torch.float32)
            actions.append(a_t)
        action_seq = torch.stack(actions, dim=0)  # [H, action_dim]
        return {
            "obs": {
                self.cfg.front_key: front,
                self.cfg.back_key: back,
                self.cfg.state_key: state,
            },
            "action": action_seq,
        }

    def _preload_all_to_memory(self) -> None:
        total = len(self.valid_start_indices)
        print(f"[Dataset] Preloading {total} horizon samples (H={self.cfg.horizon}) into memory...")
        self.memory_data = []
        for start_idx in self.valid_start_indices:
            self.memory_data.append(self._sequence_to_sample(start_idx))
        print(f"[Dataset] Preload finished. In-memory samples: {len(self.memory_data)}")

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        if self.cfg.preload_in_memory:
            return self.memory_data[idx]
        start_idx = self.valid_start_indices[idx]
        return self._sequence_to_sample(start_idx)


def _collate_fn(batch: list[dict]) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
    front = torch.stack([item["obs"][FRONT_KEY] for item in batch], dim=0)
    back = torch.stack([item["obs"][BACK_KEY] for item in batch], dim=0)
    state = torch.stack([item["obs"][STATE_KEY] for item in batch], dim=0)
    action = torch.stack([item["action"] for item in batch], dim=0)
    return {
        "obs": {
            FRONT_KEY: front,
            BACK_KEY: back,
            STATE_KEY: state,
        },
        "action": action,
    }


def build_dataloaders(
    cfg: DatasetConfig,
    batch_size: int = 32,
    num_workers: int = 0,
    train_ratio: float = 0.95,
) -> tuple[DataLoader, DataLoader]:
    dataset = LeRobotPushTDataset(cfg)
    train_size = int(len(dataset) * train_ratio)
    val_size = max(1, len(dataset) - train_size)
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_collate_fn,
    )
    return train_loader, val_loader
