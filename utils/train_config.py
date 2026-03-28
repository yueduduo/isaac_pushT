"""训练与 Transformer 结构的 JSON 配置（stdlib json）。示例见 scripts/configs/train_default.json；不加载文件时使用默认 RunBundle。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

from algo.diffusion.trainer import DiffusionConfig
from algo.flow_matching.trainer import FlowMatchingConfig


@dataclass
class TransformerPolicyArchConfig:
    """与 DiffusionTransformerModel / FlowMatchingTransformerModel 一致的形状超参（含动作头 Transformer）。"""

    image_size: int = 224
    embed_dim: int = 256
    patch_size: int = 16
    obs_layers: int = 4
    head_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.1
    obs_dim_feedforward: int | None = None
    head_dim_feedforward: int | None = None

    def to_model_kwargs(self) -> dict[str, Any]:
        return {
            "image_size": self.image_size,
            "embed_dim": self.embed_dim,
            "patch_size": self.patch_size,
            "obs_layers": self.obs_layers,
            "head_layers": self.head_layers,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "obs_dim_feedforward": self.obs_dim_feedforward,
            "head_dim_feedforward": self.head_dim_feedforward,
        }


@dataclass
class TrainRunParams:
    """与 scripts/train.py 命令行一致的训练循环参数（预算以 train_steps / freeze_steps 计）。"""

    algo: str = "diffusion"
    repo_id: str = "isaac_pusht"
    root: str = "data/isaac_pusht"
    horizon: int = 32
    train_steps: int = 5000
    batch_size: int = 32
    lr: float = 1e-4
    num_workers: int = 0
    device: str = "cuda"
    save_dir: str = "checkpoints"
    diffusion_steps: int = 100
    flow_steps: int = 10
    freeze_resnet: bool = False
    freeze_steps: int = 0
    tensorboard: bool = True
    tb_logdir: str = "runs"
    tb_run_name: str | None = None
    action_mse: bool = False
    action_mse_batches: int = 1
    random_window_split: bool = False
    val_split: bool = False
    train_ratio: float = 0.95
    split_seed: int = 42
    resume: str | None = None


@dataclass
class RunBundle:
    train: TrainRunParams
    diffusion_model: TransformerPolicyArchConfig
    flow_matching_model: TransformerPolicyArchConfig
    diffusion_trainer_overrides: dict[str, Any]
    flow_matching_trainer_overrides: dict[str, Any]


def default_run_bundle() -> RunBundle:
    return RunBundle(
        train=TrainRunParams(),
        diffusion_model=TransformerPolicyArchConfig(),
        flow_matching_model=TransformerPolicyArchConfig(),
        diffusion_trainer_overrides={},
        flow_matching_trainer_overrides={},
    )


def _merge_dataclass(cls: type, base: Any, updates: dict[str, Any]) -> Any:
    names = {f.name for f in fields(cls)}
    u = {k: v for k, v in updates.items() if k in names}
    return replace(base, **u)


def trainer_config_from_checkpoint_dict(cls: type, raw: dict[str, Any]) -> Any:
    """将 checkpoint 内 config 字典合并进 trainer dataclass；未出现的键保留类默认值（兼容旧权重）。"""
    return _merge_dataclass(cls, cls(), raw)


def load_run_bundle_json(path: str | Path) -> RunBundle:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"配置文件不存在: {p.resolve()}")
    raw: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
    train = _merge_dataclass(TrainRunParams, TrainRunParams(), raw.get("train", {}))
    dm = _merge_dataclass(
        TransformerPolicyArchConfig, TransformerPolicyArchConfig(), raw.get("diffusion_model", {})
    )
    fm = _merge_dataclass(
        TransformerPolicyArchConfig,
        TransformerPolicyArchConfig(),
        raw.get("flow_matching_model", {}),
    )
    dtxn = raw.get("diffusion_trainer")
    fmtn = raw.get("flow_matching_trainer")
    if dtxn is not None and not isinstance(dtxn, dict):
        raise TypeError("diffusion_trainer 必须为 JSON object")
    if fmtn is not None and not isinstance(fmtn, dict):
        raise TypeError("flow_matching_trainer 必须为 JSON object")
    return RunBundle(
        train=train,
        diffusion_model=dm,
        flow_matching_model=fm,
        diffusion_trainer_overrides=dict(dtxn or {}),
        flow_matching_trainer_overrides=dict(fmtn or {}),
    )


def load_run_bundle(path: str | Path | None) -> RunBundle:
    if path is None:
        return default_run_bundle()
    return load_run_bundle_json(path)


def make_diffusion_config(bundle: RunBundle) -> DiffusionConfig:
    cfg = _merge_dataclass(DiffusionConfig, DiffusionConfig(), bundle.diffusion_trainer_overrides)
    return replace(cfg, num_diffusion_steps=bundle.train.diffusion_steps, lr=bundle.train.lr)


def make_flow_matching_config(bundle: RunBundle) -> FlowMatchingConfig:
    cfg = _merge_dataclass(
        FlowMatchingConfig, FlowMatchingConfig(), bundle.flow_matching_trainer_overrides
    )
    return replace(cfg, lr=bundle.train.lr, ode_steps=bundle.train.flow_steps)


def train_arg_defaults(bundle: RunBundle) -> dict[str, Any]:
    """供 argparse set_defaults 使用（键与 TrainRunParams 字段一致）。"""
    return asdict(bundle.train)


def train_params_from_namespace(ns: argparse.Namespace) -> TrainRunParams:
    return TrainRunParams(**{f.name: getattr(ns, f.name) for f in fields(TrainRunParams)})
