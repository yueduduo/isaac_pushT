"""LeRobot 官方 Diffusion 预处理/后处理（NormalizerProcessorStep）桥接。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.processor_diffusion import make_diffusion_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.core import PolicyAction
from lerobot.utils.constants import ACTION

from utils.dataset import STATE_KEY, TOP_CAMERA_KEY, WRIST_CAMERA_KEY

IMAGE_KEYS: tuple[str, ...] = (WRIST_CAMERA_KEY, TOP_CAMERA_KEY)
OBSERVATION_KEYS: tuple[str, ...] = IMAGE_KEYS + (STATE_KEY,)


def processor_stats_path(ckpt_path: str | Path) -> Path:
    """与 checkpoint 同目录的 dataset stats 侧车（用于重建 Normalizer）。"""
    return Path(ckpt_path).with_suffix(".dataset_stats.json")


def load_dataset_stats(repo_id: str, root: str | Path) -> dict[str, Any]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id=repo_id, root=Path(root))
    return {k: v for k, v in ds.meta.stats.items()}


def _stats_to_jsonable(stats: dict[str, Any]) -> dict[str, Any]:
    from lerobot.datasets.utils import serialize_dict

    return serialize_dict(stats)


def _stats_from_jsonable(payload: dict[str, Any]) -> dict[str, Any]:
    from lerobot.datasets.utils import cast_stats_to_numpy

    return cast_stats_to_numpy(payload)


def save_dataset_stats(stats_path: str | Path, stats: dict[str, Any]) -> None:
    path = Path(stats_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_stats_to_jsonable(stats), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_dataset_stats_file(stats_path: str | Path) -> dict[str, Any]:
    path = Path(stats_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset stats file not found: {path}")
    return _stats_from_jsonable(json.loads(path.read_text(encoding="utf-8")))


def collate_to_flat_batch(batch: dict[str, Any]) -> dict[str, Tensor]:
    """将 ``{obs: {...}, action}`` 转为 LeRobot preprocessor 期望的扁平 batch。"""
    obs = batch["obs"]
    flat: dict[str, Tensor] = {}
    for key in OBSERVATION_KEYS:
        flat[key] = obs[key]
    flat[ACTION] = batch["action"]
    return flat


def split_batch_obs_action(batch: dict[str, Any]) -> tuple[dict[str, Tensor], Tensor | None]:
    obs = {key: batch[key] for key in OBSERVATION_KEYS if key in batch}
    action = batch.get(ACTION)
    return obs, action


@dataclass
class PushTProcessorBundle:
    """训练/推理共用的 LeRobot preprocessor + postprocessor。"""

    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]]
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction]
    config: DiffusionConfig
    dataset_stats: dict[str, Any]

    @classmethod
    def from_config(
        cls,
        config: DiffusionConfig,
        dataset_stats: dict[str, Any],
        *,
        device: torch.device | str,
    ) -> PushTProcessorBundle:
        config = _config_with_device(config, device)
        preprocessor, postprocessor = make_diffusion_pre_post_processors(
            config=config,
            dataset_stats=dataset_stats,
        )
        return cls(
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            config=config,
            dataset_stats=dataset_stats,
        )

    def preprocess_training_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self.preprocessor(collate_to_flat_batch(batch))

    def preprocess_observation(self, obs: dict[str, Tensor]) -> dict[str, Tensor]:
        """观测字典键为 WRIST/TOP/STATE；值为 [B,…] 或单样本（由 AddBatchDimension 处理）。"""
        processed = self.preprocessor(dict(obs))
        out, _ = split_batch_obs_action(processed)
        return out

    def postprocess_action(self, action: Tensor) -> Tensor:
        """归一化动作 → 物理空间（MIN_MAX 反变换）。"""
        if action.dim() == 1:
            return self.postprocessor(action)
        if action.dim() == 2:
            horizon = self.config.horizon
            action_feat = self.config.output_features[ACTION]
            d_a = int(action_feat.shape[0])
            if action.shape[-1] == horizon * d_a:
                seq = action.view(action.shape[0], horizon, d_a)
                out = self.postprocessor(seq)
                return out.view(action.shape[0], -1)
            return self.postprocessor(action)
        if action.dim() == 3:
            return self.postprocessor(action)
        raise ValueError(f"Unsupported action shape for postprocess: {tuple(action.shape)}")


def _config_with_device(config: DiffusionConfig, device: torch.device | str) -> DiffusionConfig:
    import dataclasses

    dev_str = str(device)
    if config.device == dev_str:
        return config
    return dataclasses.replace(config, device=dev_str)


def build_processor_bundle(
    config: DiffusionConfig,
    *,
    repo_id: str,
    root: str | Path,
    device: torch.device | str,
    stats: dict[str, Any] | None = None,
) -> PushTProcessorBundle:
    dataset_stats = stats if stats is not None else load_dataset_stats(repo_id, root)
    return PushTProcessorBundle.from_config(config, dataset_stats, device=device)


def load_processor_bundle_for_checkpoint(
    ckpt_path: str | Path,
    config: DiffusionConfig,
    *,
    repo_id: str,
    root: str | Path,
    device: torch.device | str,
) -> PushTProcessorBundle:
    stats_file = processor_stats_path(ckpt_path)
    if stats_file.is_file():
        stats = load_dataset_stats_file(stats_file)
    else:
        stats = load_dataset_stats(repo_id, root)
    return PushTProcessorBundle.from_config(config, stats, device=device)


def save_processor_stats_for_checkpoint(ckpt_path: str | Path, stats: dict[str, Any]) -> None:
    save_dataset_stats(processor_stats_path(ckpt_path), stats)
