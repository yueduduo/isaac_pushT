"""LeRobot dataset adapters for multimodal PushT policies.

Train/Val 划分说明（见 build_dataloaders）：
- 默认 val_split=False：不划分验证集，全部窗口只用于训练（val_loader 为 None）。
- 若 val_split=True：再按 split_by_episode 等在 train/val 间划分；split_by_episode=True 时按整条轨迹划分，
  避免同轨迹跨 train/val。split_by_episode=False 时为窗口级 random_split（旧行为）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader, Dataset, Subset, random_split


WRIST_CAMERA_KEY = "observation.wrist_camera_image"
TOP_CAMERA_KEY = "observation.top_camera_image"
STATE_KEY = "observation.state"
ACTION_KEY = "action"


@dataclass
class DatasetConfig:
    repo_id: str = "isaac_pusht"
    root: str = "data/isaac_pusht"
    wrist_camera_key: str = WRIST_CAMERA_KEY
    top_camera_key: str = TOP_CAMERA_KEY
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
        self.state_mean: torch.Tensor | None = None
        self.state_std: torch.Tensor | None = None
        self.action_mean: torch.Tensor | None = None
        self.action_std: torch.Tensor | None = None
        
        # [优化1] 预加载所有状态和动作 (使用 .clone() 彻底断开磁盘映射句柄)
        num_frames = len(self.dataset.hf_dataset)
        print(f"[Dataset] Loading numerical data ({num_frames} frames)...")
        # 直接select_columns获取整列数据，Hugging Face Dataset 对整列访问有高度优化
        self.all_states = torch.from_numpy(np.array(self.dataset.hf_dataset.select_columns([cfg.state_key])[cfg.state_key])).float().clone()
        self.all_actions = torch.from_numpy(np.array(self.dataset.hf_dataset.select_columns([cfg.action_key])[cfg.action_key])).float().clone()
        
        # valid_start_indices：全局帧下标，每个元素是一个窗口的起点；
        # episode_id_per_window[i]：第 i 个窗口属于哪一条 episode（与 meta.episodes 下标对齐），
        # 供 build_dataloaders(val_split=True, split_by_episode=True) 使用，保证整条轨迹不跨 train/val。
        self.valid_start_indices, self.episode_id_per_window = self._build_valid_start_indices()
        # 将有效索引转换为张量，以便在 collate 中快速切片
        self.valid_start_indices_ts = torch.tensor(self.valid_start_indices, dtype=torch.long)
        
        if self.cfg.preload_in_memory:
            self._preload_images_to_memory()
            
        # [核心优化] 释放原始 dataset 对象，彻底释放磁盘句柄
        print("[Dataset] Preload complete. Releasing source dataset handles...")
        del self.dataset
        import gc
        gc.collect()

    def set_normalization_stats(
        self,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
    ) -> None:
        self.state_mean = state_mean.detach().cpu().float().clone()
        self.state_std = state_std.detach().cpu().float().clone()
        self.action_mean = action_mean.detach().cpu().float().clone()
        self.action_std = action_std.detach().cpu().float().clone()

    def __len__(self) -> int:
        return len(self.valid_start_indices)

    def _build_valid_start_indices(self) -> tuple[list[int], list[int]]:
        """枚举每条轨迹内所有合法窗口起点，并记录窗口所属 episode，用于后续按轨迹划分 train/val。"""
        if self.cfg.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.cfg.horizon}")
        
        print(f"[Dataset] Calculating valid start indices (horizon={self.cfg.horizon})...")
        valid_indices: list[int] = []
        # 与 valid_indices 等长：窗口 i 来自第几条 episode（0..E-1）
        episode_id_per_window: list[int] = []
        
        # LeRobotDataset v3.0 存储 episode 边界在 self.dataset.meta.episodes 中
        episodes = self.dataset.meta.episodes
        for episode_id, ep_info in enumerate(episodes):
            start_i = ep_info["dataset_from_index"]
            end_i = ep_info["dataset_to_index"]
            # 有效起始索引 idx 需满足 [idx, idx + horizon) 都在 [start_i, end_i) 内
            # 即 idx >= start_i 且 idx + horizon <= end_i => idx <= end_i - horizon
            max_valid_start = end_i - self.cfg.horizon
            if max_valid_start >= start_i:
                for idx in range(start_i, max_valid_start + 1):
                    valid_indices.append(idx)
                    # 该窗口完全落在本 episode 的 [start_i, end_i) 内，故归属 episode_id
                    episode_id_per_window.append(episode_id)
    

        if len(valid_indices) == 0:
            raise ValueError("No valid sequence windows found. Check horizon or dataset integrity.")
        
        print(f"[Dataset] Found {len(valid_indices)} valid windows.")
        return valid_indices, episode_id_per_window

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
        self.all_wrist_camera_images = process_column(self.cfg.wrist_camera_key)
        self.all_top_camera_images = process_column(self.cfg.top_camera_key)
            
        total_time = time.time() - start_time
        # 统计内存 (float32 = 4 bytes per pixel)
        total_bytes = (self.all_wrist_camera_images.numel() + self.all_top_camera_images.numel()) * 4
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
        state_batch = self.all_states[sample_indices]
        if self.state_mean is not None and self.state_std is not None:
            state_batch = (state_batch - self.state_mean) / self.state_std
        if self.action_mean is not None and self.action_std is not None:
            action_batch = (action_batch - self.action_mean.view(1, 1, -1)) / self.action_std.view(1, 1, -1)
        
        # 2. 在返回前立即 to(device)，将 CPU 压力释放给异步 DMA
        dev = self.device
        return {
            "obs": {
                self.cfg.wrist_camera_key: self.all_wrist_camera_images[sample_indices].to(dev),
                self.cfg.top_camera_key: self.all_top_camera_images[sample_indices].to(dev),
                self.cfg.state_key: state_batch.to(dev),
            },
            "action": action_batch.to(dev),
        }


def _episode_level_split_indices(
    episode_id_per_window: list[int],
    train_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """按 episode 划分 train/val 的窗口下标列表。

    目的：避免「同一条轨迹 / 同一个 rollout」同时出现在 train 与 val。
    做法：先把属于同一 episode_id 的所有样本窗口索引归为一组，再随机打乱 episode 顺序，
    按 train_ratio 决定多少条 episode 划入 val；每条 episode 要么整组进 train，要么整组进 val，
    不会出现同轨迹部分窗口在 train、部分在 val 的情况。

    train_ratio 在此表示「大约希望保留在训练侧的 episode 比例」（val 至少 1 条、train 至少 1 条，
    当总 episode 数 >= 2 时）；不是严格的「窗口数 95/5」。
    """
    by_ep: dict[int, list[int]] = {}
    for window_i, ep_id in enumerate(episode_id_per_window):
        by_ep.setdefault(ep_id, []).append(window_i)
    episode_ids = sorted(by_ep.keys())
    n_ep = len(episode_ids)
    if n_ep == 0:
        raise ValueError("No episodes in dataset.")
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_ep, generator=g).tolist()
    shuffled = [episode_ids[i] for i in perm]
    if n_ep == 1:
        print(
            "[Dataset] 仅 1 条 episode，无法做 episode 级划分；退回按窗口 random_split（与旧行为一致）。"
        )
        return [], []  # 由调用方改用 random_split
    n_val_ep = max(1, int(round((1.0 - train_ratio) * n_ep)))
    n_val_ep = min(n_val_ep, n_ep - 1)
    # val：最后 n_val_ep 条轨迹的全部窗口；train：其余轨迹的全部窗口 → 轨迹不跨两侧
    val_ep_ids = frozenset(shuffled[-n_val_ep:])
    train_ep_ids = frozenset(shuffled[:-n_val_ep])
    train_idx = [i for i, e in enumerate(episode_id_per_window) if e in train_ep_ids]
    val_idx = [i for i, e in enumerate(episode_id_per_window) if e in val_ep_ids]
    return train_idx, val_idx


def build_dataloaders(
    cfg: DatasetConfig,
    batch_size: int = 32,
    num_workers: int = 0,
    train_ratio: float = 0.95,
    device: torch.device = torch.device("cpu"),
    split_by_episode: bool = True,
    split_seed: int = 42,
    val_split: bool = False,
) -> tuple[DataLoader, DataLoader | None]:
    """构建 train/val DataLoader。

    val_split（默认 False）：
        不划分验证集，全部窗口用于训练，返回 (train_loader, None)。
    若 val_split=True：
        按 train_ratio 等规则划分训练集与验证集。

    split_by_episode（默认 True，仅当 val_split=True 时参与逻辑）：
        按整条轨迹划分，保证同轨迹不跨 train/val；验证指标更接近「未见过的完整 episode」。
    若设为 False：
        按窗口随机划分（random_split），同一条轨迹的 valid 窗口可能同时出现在 train 与 val。
    """
    dataset = LeRobotPushTDataset(cfg, device=device)
    
    # [优化] Windows 下大内存预加载时，num_workers > 0 会导致严重的启动延迟
    import platform
    if platform.system() == "Windows" and cfg.preload_in_memory:
        if num_workers > 0:
            print(f"[Dataset] Windows detected with preloaded RAM. Forcing num_workers=0 to avoid startup hang.")
            num_workers = 0

    if not val_split:
        print("[Dataset] val_split=False：全部窗口用于训练，不构建验证集。")
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=dataset.collate_fn,
        )
        return train_loader, None

    if split_by_episode:
        train_idx, val_idx = _episode_level_split_indices(
            dataset.episode_id_per_window, train_ratio=train_ratio, seed=split_seed
        )
        if len(train_idx) == 0 and len(val_idx) == 0:
            # 例如仅 1 条 episode：无法同时保留 train/val 各至少一条轨迹，退回窗口级划分
            split_by_episode = False
        else:
            print(
                f"[Dataset] Episode-level split: {len(train_idx)} train windows, {len(val_idx)} val windows "
                f"(train_ratio≈{train_ratio} 作用于 episode 条数)."
            )
            train_ds = Subset(dataset, train_idx)
            val_ds = Subset(dataset, val_idx)

    if not split_by_episode:
        # 窗口级随机划分：不保证轨迹不跨 train/val；可能与历史实验一致
        train_size = int(len(dataset) * train_ratio)
        val_size = len(dataset) - train_size
        train_ds, val_ds = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(split_seed),
        )
        print(f"[Dataset] Random window split: train={train_size}, val={val_size}.")
    
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
