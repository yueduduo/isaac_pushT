"""LeRobot dataset adapters for multimodal PushT policies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
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
        
        # [优化1] 预加载所有状态和动作 (一次性读取列，避免循环中重复 select_columns)
        num_frames = len(self.dataset.hf_dataset)
        
        print(f"[Dataset] Loading numerical data (Total frames: {num_frames})...")
        # [优化1] 预加载所有状态和动作 (使用 .clone() 彻底断开磁盘映射句柄)
        num_frames = len(self.dataset.hf_dataset)
        print(f"[Dataset] Loading numerical data ({num_frames} frames)...")
        # 直接select_columns获取整列数据，Hugging Face Dataset 对整列访问有高度优化
        self.all_states = torch.from_numpy(np.array(self.dataset.hf_dataset.select_columns([cfg.state_key])[cfg.state_key])).float().clone()
        self.all_actions = torch.from_numpy(np.array(self.dataset.hf_dataset.select_columns([cfg.action_key])[cfg.action_key])).float().clone()
        
        # [优化2] 预分配大张量存储图像
        self.all_front_images = torch.zeros((num_frames, 3, 224, 224), dtype=torch.uint8)
        self.all_back_images = torch.zeros((num_frames, 3, 224, 224), dtype=torch.uint8)
        
        self.valid_start_indices = self._build_valid_start_indices()
        
        if self.cfg.preload_in_memory:
            self._preload_images_to_memory()
            
        # [核心优化] 释放原始 dataset 对象，彻底释放磁盘句柄
        print("[Dataset] Preload complete. Releasing source dataset handles...")
        del self.dataset
        import gc
        gc.collect()

    def __len__(self) -> int:
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
        
        print(f"[Dataset] Calculating valid start indices (horizon={self.cfg.horizon})...")
        
        valid_indices = []
        
        # LeRobotDataset v3.0 存储 episode 边界在 self.dataset.meta.episodes 中
        if hasattr(self.dataset.meta, "episodes"):
            episodes = self.dataset.meta.episodes
            for ep_info in episodes:
                start_i = ep_info["dataset_from_index"]
                end_i = ep_info["dataset_to_index"]
                # 有效起始索引 idx 需满足 [idx, idx + horizon) 都在 [start_i, end_i) 内
                # 即 idx >= start_i 且 idx + horizon <= end_i => idx <= end_i - horizon
                max_valid_start = end_i - self.cfg.horizon
                if max_valid_start >= start_i:
                    valid_indices.extend(range(start_i, max_valid_start + 1))
        else:
            # 备选方案：如果结构不匹配，降级为原始方案
            print("No episodes found in meta. Using original method.")
            max_start = len(self.dataset) - self.cfg.horizon + 1
            if max_start <= 0:
                raise ValueError(f"Dataset has {len(self.dataset)} frames, too small for horizon={self.cfg.horizon}")
            
            for idx in tqdm(range(max_start), desc="Finding valid indices", leave=False):
                if self._is_same_episode_window(idx):
                    valid_indices.append(idx)

        if len(valid_indices) == 0:
            raise ValueError("No valid sequence windows found. Check horizon or dataset integrity.")
        
        print(f"[Dataset] Found {len(valid_indices)} valid windows.")
        return valid_indices

    def _preload_images_to_memory(self) -> None:
        num_frames = len(self.dataset)
        import os
        num_workers = min(32, (os.cpu_count() or 1) * 2)
        
        print(f"[Dataset] Preloading {num_frames} images using {num_workers} threads...")
        
        def load_frame_images(i):
            try:
                frame = self.dataset[i]
                
                # 获取原始数据 (你录制的数据是 CHW float32 [0, 1])
                f_raw = np.array(frame[self.cfg.front_key])
                b_raw = np.array(frame[self.cfg.back_key])
                
                # [核心修复] 如果是 float 且范围在 [0, 1]，我们需要还原到 [0, 255] 以便存入 uint8 张量
                # 这能节省 4x 内存且不损失精度
                if f_raw.dtype == np.float32 or f_raw.dtype == np.float64:
                    if f_raw.max() <= 1.05:
                        f_raw = (f_raw * 255.0).clip(0, 255).astype(np.uint8)
                        b_raw = (b_raw * 255.0).clip(0, 255).astype(np.uint8)
                
                f_tensor = torch.from_numpy(f_raw)
                b_tensor = torch.from_numpy(b_raw)
                
                # 兼容性检查：如果是 HWC 格式，转换为 CHW
                if f_tensor.ndim == 3 and f_tensor.shape[0] != 3:
                    f_tensor = f_tensor.permute(2, 0, 1)
                    b_tensor = b_tensor.permute(2, 0, 1)
                
                self.all_front_images[i] = f_tensor
                self.all_back_images[i] = b_tensor
            except Exception as e:
                print(f"Error loading frame {i}: {e}")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            list(tqdm(
                executor.map(load_frame_images, range(num_frames)),
                total=num_frames,
                desc="Preloading Images",
                unit="frame"
            ))
            
        total_bytes = (self.all_front_images.numel() + self.all_back_images.numel()) * 1 # uint8
        total_bytes += (self.all_states.numel() + self.all_actions.numel()) * 4 # float32
        mem_gb = total_bytes / (1024**3)
        print(f"[Dataset] Preload finished. Total RAM for Tensors: {mem_gb:.2f} GB")

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        start_idx = self.valid_start_indices[idx]
        
        # [优化3] 仅返回 uint8 原始张量切片，不要在这里做除法/浮点转换。
        # 这样 __getitem__ 是真正的 0 延迟，且不产生大量 Python 临时对象。
        front = self.all_front_images[start_idx]
        back = self.all_back_images[start_idx]
        state = self.all_states[start_idx]
        action_seq = self.all_actions[start_idx : start_idx + self.cfg.horizon]
        
        return {
            "obs": {
                self.cfg.front_key: front,
                self.cfg.back_key: back,
                self.cfg.state_key: state,
            },
            "action": action_seq,
        }


def _collate_fn(batch: list[dict]) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
    # [优化4] 批量进行浮点转换和归一化，充分利用 CPU 向量化指令，比 __getitem__ 里逐个转换快得多
    front = torch.stack([item["obs"][FRONT_KEY] for item in batch], dim=0).float() / 255.0
    back = torch.stack([item["obs"][BACK_KEY] for item in batch], dim=0).float() / 255.0
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
    
    # [优化] Windows 下大内存预加载时，num_workers > 0 会导致严重的启动延迟
    # 内存镜像开销：当你拥有 64GB 内存且主进程已经预加载了约 6GB 的数据集时，如果你设置 num_workers=4，Windows 会尝试为这 4个子进程每一个都“克隆”一份父进程的内存布局。
    # 虽然数据在 RAM 里，但 spawn 过程中，操作系统会由于内存页锁定和句柄检查产生严重的 CPU 阻塞。
    # 既然所有数据都在 RAM 里了，单线程（主进程）读取速度已经足够快。通过设为 0，我们彻底避开了 Windows 的进程 spawn 开销。
    import platform
    if platform.system() == "Windows" and cfg.preload_in_memory:
        if num_workers > 0:
            print(f"[Dataset] Windows detected with preloaded RAM. Forcing num_workers=0 to avoid startup hang.")
            num_workers = 0

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
        pin_memory=False, # Windows 上大内存数据集开启 pin_memory 易导致卡顿
        collate_fn=_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=_collate_fn,
    )
    return train_loader, val_loader
