"""训练 JSON 配置（stdlib json）。示例见 scripts/configs/train_default.json。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any


def _diffusion_config_field_names() -> set[str]:
    from dataclasses import fields as dc_fields

    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

    return {f.name for f in dc_fields(DiffusionConfig)}


@dataclass
class TrainRunParams:
    """与 scripts/train.py 命令行一致。"""

    repo_id: str = "isaac_pusht"
    root: str = "data/isaac_pusht"
    horizon: int = 32
    train_steps: int = 5000
    batch_size: int = 32
    lr: float = 1e-4
    num_workers: int = 12
    device: str = "cuda"
    save_dir: str = "checkpoints"
    diffusion_steps: int = 100
    n_obs_steps: int = 1
    n_action_steps: int | None = None
    freeze_resnet: bool = False
    freeze_steps: int = 0
    grad_clip_norm: float = 1.0
    optimizer_weight_decay: float | None = None
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
    on_demand_dataset: bool = True
    io_workers: int = 8


@dataclass
class RunBundle:
    train: TrainRunParams
    diffusion_config_overrides: dict[str, Any]


def default_run_bundle() -> RunBundle:
    return RunBundle(train=TrainRunParams(), diffusion_config_overrides={})


def _merge_dataclass(cls: type, base: Any, updates: dict[str, Any]) -> Any:
    names = {f.name for f in fields(cls)}
    u = {k: v for k, v in updates.items() if k in names}
    return replace(base, **u)


def load_run_bundle_json(path: str | Path) -> RunBundle:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"配置文件不存在: {p.resolve()}")
    raw: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
    train_raw = dict(raw.get("train", {}))

    dtxn = dict(raw.get("diffusion_trainer", {}))
    lero = dict(raw.get("lerobot_diffusion", {}))

    if "grad_clip_norm" in dtxn:
        train_raw["grad_clip_norm"] = dtxn.pop("grad_clip_norm")
    if "weight_decay" in dtxn:
        wd = dtxn.pop("weight_decay")
        train_raw["optimizer_weight_decay"] = wd
        dtxn["optimizer_weight_decay"] = wd

    known = _diffusion_config_field_names()
    diffusion_overrides: dict[str, Any] = {}
    for k, v in {**dtxn, **lero}.items():
        if k in known:
            diffusion_overrides[k] = v

    train = _merge_dataclass(TrainRunParams, TrainRunParams(), train_raw)
    return RunBundle(train=train, diffusion_config_overrides=diffusion_overrides)


def load_run_bundle(path: str | Path | None) -> RunBundle:
    if path is None:
        return default_run_bundle()
    return load_run_bundle_json(path)


def train_arg_defaults(bundle: RunBundle) -> dict[str, Any]:
    return asdict(bundle.train)


def train_params_from_namespace(ns: argparse.Namespace) -> TrainRunParams:
    return TrainRunParams(**{f.name: getattr(ns, f.name) for f in fields(TrainRunParams)})
