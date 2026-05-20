"""PushT 按需加载数据集：不把整库图像载入内存，训练时从磁盘读取。

与 ``utils.dataset.LeRobotPushTDataset``（全量 preload）并列使用，不修改原文件。

并行策略
--------
- **DataLoader ``num_workers``**：多进程，每个 worker 在 ``__getitem__`` 里读图（推荐 Linux 上 ``num_workers>=4``）。
- **``io_workers``**：单样本内用线程池并行读 wrist / top 两路图像（``num_workers=0`` 时仍有收益）。

训练脚本接入示例::

    from utils.dataset_on_demand import OnDemandDatasetConfig, build_on_demand_dataloaders

    cfg = OnDemandDatasetConfig(repo_id=..., root=..., horizon=32, io_workers=4)
    train_loader, val_loader = build_on_demand_dataloaders(
        cfg, batch_size=32, num_workers=4, device=device, ...
    )
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from utils.dataset import (
    ACTION_KEY,
    STATE_KEY,
    TOP_CAMERA_KEY,
    WRIST_CAMERA_KEY,
    _episode_level_split_indices,
)


def _numpy_image_to_chw_float(img: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(np.asarray(img))
    if t.dtype == torch.uint8:
        t = t.float().div(255.0)
    elif t.dtype == torch.float64:
        t = t.float()
    if t.ndim == 3 and t.shape[-1] == 3:
        t = t.permute(2, 0, 1)
    return t.contiguous()


def move_batch_to_device(
    batch: dict[str, dict[str, torch.Tensor] | torch.Tensor],
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
    """将 ``collate_fn`` 留在 CPU 上的 batch 搬到训练设备（配合 ``num_workers>0``）。"""
    obs = batch["obs"]
    return {
        "obs": {k: v.to(device, non_blocking=True) for k, v in obs.items()},
        "action": batch["action"].to(device, non_blocking=True),
    }


@dataclass
class OnDemandDatasetConfig:
    repo_id: str = "isaac_pusht"
    root: str = "data/isaac_pusht"
    wrist_camera_key: str = WRIST_CAMERA_KEY
    top_camera_key: str = TOP_CAMERA_KEY
    state_key: str = STATE_KEY
    action_key: str = ACTION_KEY
    horizon: int = 16
    """单样本内并行读图的线程数（wrist + top 等）。"""
    io_workers: int = 4


class LeRobotPushTOnDemandDataset(Dataset):
    """仅预加载 state/action 数值列；图像在 ``__getitem__`` 时按帧从 LeRobot 数据集读取。"""

    def __init__(
        self,
        cfg: OnDemandDatasetConfig,
        device: torch.device = torch.device("cpu"),
        *,
        collate_to_device: bool = True,
    ):
        self.cfg = cfg
        self.device = device
        self._collate_to_device = collate_to_device
        self._io_pool: ThreadPoolExecutor | None = None
        # 多进程 DataLoader 时不在 worker 内再建线程池（fork 不安全且冗余）
        if cfg.io_workers > 1 and collate_to_device:
            self._io_pool = ThreadPoolExecutor(max_workers=cfg.io_workers)

        self.lerobot = LeRobotDataset(repo_id=cfg.repo_id, root=Path(cfg.root))
        self.lerobot._ensure_hf_dataset_loaded()
        self._hf = self.lerobot.hf_dataset

        num_frames = len(self._hf)
        print(f"[OnDemandDataset] Loading numerical data ({num_frames} frames)...")
        self.all_states = torch.from_numpy(
            np.array(self._hf.select_columns([cfg.state_key])[cfg.state_key])
        ).float().clone()
        self.all_actions = torch.from_numpy(
            np.array(self._hf.select_columns([cfg.action_key])[cfg.action_key])
        ).float().clone()

        self.valid_start_indices, self.episode_id_per_window = self._build_valid_start_indices()
        chw = _numpy_image_to_chw_float(self._hf[0][cfg.wrist_camera_key]).shape
        self.image_chw: tuple[int, int, int] = (int(chw[0]), int(chw[1]), int(chw[2]))
        # 与 train.py 中 ``all_wrist_camera_images.shape[1:4]`` 兼容
        self.all_wrist_camera_images = torch.empty((0, *self.image_chw))
        self.all_top_camera_images = torch.empty((0, *self.image_chw))
        print(
            f"[OnDemandDataset] Ready: {len(self.valid_start_indices)} windows, "
            f"image_chw={self.image_chw}, io_workers={cfg.io_workers}, "
            f"num_workers 请由 DataLoader 指定。"
        )

    def __len__(self) -> int:
        return len(self.valid_start_indices)

    def _build_valid_start_indices(self) -> tuple[list[int], list[int]]:
        if self.cfg.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.cfg.horizon}")

        print(f"[OnDemandDataset] Calculating valid start indices (horizon={self.cfg.horizon})...")
        valid_indices: list[int] = []
        episode_id_per_window: list[int] = []
        episodes = self.lerobot.meta.episodes
        for episode_id, ep_info in enumerate(episodes):
            start_i = ep_info["dataset_from_index"]
            end_i = ep_info["dataset_to_index"]
            max_valid_start = end_i - self.cfg.horizon
            if max_valid_start >= start_i:
                for idx in range(start_i, max_valid_start + 1):
                    valid_indices.append(idx)
                    episode_id_per_window.append(episode_id)

        if len(valid_indices) == 0:
            raise ValueError("No valid sequence windows found. Check horizon or dataset integrity.")

        print(f"[OnDemandDataset] Found {len(valid_indices)} valid windows.")
        return valid_indices, episode_id_per_window

    def _read_image(self, frame_idx: int, key: str) -> torch.Tensor:
        return _numpy_image_to_chw_float(self._hf[frame_idx][key])

    def _load_observation_images(self, start_idx: int) -> dict[str, torch.Tensor]:
        wrist_key = self.cfg.wrist_camera_key
        top_key = self.cfg.top_camera_key

        if self._io_pool is None:
            return {
                wrist_key: self._read_image(start_idx, wrist_key),
                top_key: self._read_image(start_idx, top_key),
            }

        fut_wrist = self._io_pool.submit(self._read_image, start_idx, wrist_key)
        fut_top = self._io_pool.submit(self._read_image, start_idx, top_key)
        return {wrist_key: fut_wrist.result(), top_key: fut_top.result()}

    def __getitem__(self, idx: int) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
        start_idx = self.valid_start_indices[idx]
        state = self.all_states[start_idx]
        end_idx = start_idx + self.cfg.horizon
        action = self.all_actions[start_idx:end_idx]
        images = self._load_observation_images(start_idx)
        return {
            "obs": {
                self.cfg.wrist_camera_key: images[self.cfg.wrist_camera_key],
                self.cfg.top_camera_key: images[self.cfg.top_camera_key],
                self.cfg.state_key: state,
            },
            "action": action,
        }

    @staticmethod
    def collate_batch(
        samples: list[dict[str, dict[str, torch.Tensor] | torch.Tensor]],
    ) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
        obs_keys = samples[0]["obs"].keys()
        return {
            "obs": {k: torch.stack([s["obs"][k] for s in samples], dim=0) for k in obs_keys},
            "action": torch.stack([s["action"] for s in samples], dim=0),
        }

    def collate_fn(
        self,
        batch: list[dict[str, dict[str, torch.Tensor] | torch.Tensor]],
    ) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
        out = self.collate_batch(batch)
        if self._collate_to_device and self.device.type != "cpu":
            return move_batch_to_device(out, self.device)
        return out

    def close(self) -> None:
        if self._io_pool is not None:
            self._io_pool.shutdown(wait=True)
            self._io_pool = None


def build_on_demand_dataloaders(
    cfg: OnDemandDatasetConfig,
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.95,
    device: torch.device = torch.device("cpu"),
    split_by_episode: bool = True,
    split_seed: int = 42,
    val_split: bool = False,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
) -> tuple[DataLoader, DataLoader | None]:
    """构建按需加载的 train/val DataLoader。

    ``num_workers>0`` 时 ``collate_fn`` 在 worker 内只组 batch（CPU）；请在训练循环中调用
    ``move_batch_to_device``，或使用 ``num_workers=0`` 让 ``collate_fn`` 直接搬到 ``device``。
    """
    collate_to_device = num_workers == 0
    dataset = LeRobotPushTOnDemandDataset(cfg, device=device, collate_to_device=collate_to_device)
    use_pin = device.type == "cuda" and num_workers > 0
    if num_workers > 0:
        print(
            "[OnDemandDataset] num_workers>0：collate 在 CPU，训练循环需调用 move_batch_to_device。"
        )

    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": use_pin,
        "collate_fn": dataset.collate_fn,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = persistent_workers

    if not val_split:
        print("[OnDemandDataset] val_split=False：全部窗口用于训练，不构建验证集。")
        train_loader = DataLoader(dataset, shuffle=True, **loader_kwargs)
        return train_loader, None

    if split_by_episode:
        train_idx, val_idx = _episode_level_split_indices(
            dataset.episode_id_per_window, train_ratio=train_ratio, seed=split_seed
        )
        if len(train_idx) == 0 and len(val_idx) == 0:
            split_by_episode = False
        else:
            print(
                f"[OnDemandDataset] Episode-level split: {len(train_idx)} train windows, "
                f"{len(val_idx)} val windows."
            )
            train_ds = Subset(dataset, train_idx)
            val_ds = Subset(dataset, val_idx)

    if not split_by_episode:
        train_size = int(len(dataset) * train_ratio)
        val_size = len(dataset) - train_size
        train_ds, val_ds = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(split_seed),
        )
        print(f"[OnDemandDataset] Random window split: train={train_size}, val={val_size}.")

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader
