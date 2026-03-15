"""LeRobot dataset adapters for multimodal PushT policies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader, Dataset, random_split


FRONT_KEY = "observation.front_wrist_camera_image"
BACK_KEY = "observation.back_wrist_camera_image"
STATE_KEY = "observation.state"
ACTION_KEY = "action"


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

    def __init__(self, cfg: DatasetConfig, device: torch.device = torch.device("cpu")):
        self.cfg = cfg
        self.device = device
        self.dataset = LeRobotDataset(repo_id=cfg.repo_id, root=Path(cfg.root))
        
        # [优化1] 预加载所有状态和动作 (使用 .clone() 彻底断开磁盘映射句柄)
        num_frames = len(self.dataset.hf_dataset)
        print(f"[Dataset] Loading numerical data ({num_frames} frames)...")
        # 直接select_columns获取整列数据，Hugging Face Dataset 对整列访问有高度优化
        self.all_states = torch.from_numpy(np.array(self.dataset.hf_dataset.select_columns([cfg.state_key])[cfg.state_key])).float().clone()
        self.all_actions = torch.from_numpy(np.array(self.dataset.hf_dataset.select_columns([cfg.action_key])[cfg.action_key])).float().clone()
        
        self.valid_start_indices = self._build_valid_start_indices()
        # 将有效索引转换为张量，以便在 collate 中快速切片
        self.valid_start_indices_ts = torch.tensor(self.valid_start_indices, dtype=torch.long)
        
        if self.cfg.preload_in_memory:
            self._preload_images_to_memory()
            
        # [核心优化] 释放原始 dataset 对象，彻底释放磁盘句柄
        print("[Dataset] Preload complete. Releasing source dataset handles...")
        del self.dataset
        import gc
        gc.collect()

    def __len__(self) -> int:
        return len(self.valid_start_indices)

    def _build_valid_start_indices(self) -> list[int]:
        if self.cfg.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.cfg.horizon}")
        
        print(f"[Dataset] Calculating valid start indices (horizon={self.cfg.horizon})...")
        valid_indices = []
        
        # LeRobotDataset v3.0 存储 episode 边界在 self.dataset.meta.episodes 中
        episodes = self.dataset.meta.episodes
        for ep_info in episodes:
            start_i = ep_info["dataset_from_index"]
            end_i = ep_info["dataset_to_index"]
            # 有效起始索引 idx 需满足 [idx, idx + horizon) 都在 [start_i, end_i) 内
            # 即 idx >= start_i 且 idx + horizon <= end_i => idx <= end_i - horizon
            max_valid_start = end_i - self.cfg.horizon
            if max_valid_start >= start_i:
                valid_indices.extend(range(start_i, max_valid_start + 1))
    

        if len(valid_indices) == 0:
            raise ValueError("No valid sequence windows found. Check horizon or dataset integrity.")
        
        print(f"[Dataset] Found {len(valid_indices)} valid windows.")
        return valid_indices

    def _preload_images_to_memory(self) -> None:
        num_frames = len(self.dataset.hf_dataset)
        print(f"[Dataset] Preloading {num_frames} images (FLOAT32 Pre-normalized Mode)...")
        
        import time
        start_time = time.time()
        
        def process_column(key):
            # 1. 磁盘读取 ([:] 触发内存映射加载)
            with self.dataset.hf_dataset.formatted_as("numpy"):
                print(f"  --> Loading {key}...")
                data_np = self.dataset.hf_dataset[key][:]
            
            # 2. 强制转为 float32 并归一化 (即使原始是 uint8)
            # 这样训练时 __getitem__ 没有任何计算开销
            if data_np.dtype == np.uint8:
                print(f"  --> Normalizing {key} (uint8 -> float32)...")
                data_np = data_np.astype(np.float32) / 255.0
            elif data_np.dtype == np.float64:
                data_np = data_np.astype(np.float32)
            
            # 3. 转换为张量并调整维度为 (N, C, H, W)
            tensor = torch.from_numpy(data_np)
            if tensor.ndim == 4 and tensor.shape[-1] == 3:
                print(f"  --> Permuting {key} to CHW...")
                tensor = tensor.permute(0, 3, 1, 2)
            
            return tensor.contiguous().clone() # clone 确保数据在内存中是独立的

        # 顺序执行，利用单线程全速顺序 I/O (在 Windows 上通常比多线程并发 I/O 更稳定且跑满带宽)
        self.all_front_images = process_column(self.cfg.front_key)
        self.all_back_images = process_column(self.cfg.back_key)
            
        total_time = time.time() - start_time
        # 统计内存 (float32 = 4 bytes per pixel)
        total_bytes = (self.all_front_images.numel() + self.all_back_images.numel()) * 4
        total_bytes += (self.all_states.numel() + self.all_actions.numel()) * 4
        mem_gb = total_bytes / (1024**3)
        print(f"[Dataset] Preload finished in {total_time:.2f}s. Total RAM for Tensors: {mem_gb:.2f} GB")

    def __getitem__(self, idx: int) -> int:
        # [极速模式] 仅仅返回该样本在 valid_start_indices 中的索引位置
        return idx

    def collate_fn(self, batch_indices: list[int]) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
        # 由于 batch_indices 已经是 list(int)，直接传入高级索引即可
        sample_indices = self.valid_start_indices_ts[batch_indices]
        
        # 1. 批量提取动作序列(使用广播索引矩阵一次性提取 [B, H, D])
        # [B, 1] + [H] -> [B, H] 索引网格
        offsets = torch.arange(self.cfg.horizon, device=sample_indices.device)
        action_batch = self.all_actions[sample_indices.unsqueeze(1) + offsets]
        
        # 2. 在返回前立即 to(device)，将 CPU 压力释放给异步 DMA
        dev = self.device
        return {
            "obs": {
                self.cfg.front_key: self.all_front_images[sample_indices].to(dev),
                self.cfg.back_key: self.all_back_images[sample_indices].to(dev),
                self.cfg.state_key: self.all_states[sample_indices].to(dev),
            },
            "action": action_batch.to(dev),
        }


def build_dataloaders(
    cfg: DatasetConfig,
    batch_size: int = 32,
    num_workers: int = 0,
    train_ratio: float = 0.95,
    device: torch.device = torch.device("cpu"),
) -> tuple[DataLoader, DataLoader]:
    dataset = LeRobotPushTDataset(cfg, device=device)
    
    # [优化] Windows 下大内存预加载时，num_workers > 0 会导致严重的启动延迟
    import platform
    if platform.system() == "Windows" and cfg.preload_in_memory:
        if num_workers > 0:
            print(f"[Dataset] Windows detected with preloaded RAM. Forcing num_workers=0 to avoid startup hang.")
            num_workers = 0

    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    
    # 注意：这里需要传入 dataset.collate_fn，它绑定了具体的 dataset 实例
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False, # Windows 上大内存数据集开启 pin_memory 易导致卡顿
        collate_fn=dataset.collate_fn, # 使用绑定的方法
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=dataset.collate_fn,
    )
    return train_loader, val_loader
