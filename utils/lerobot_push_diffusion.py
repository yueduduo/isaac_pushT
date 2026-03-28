"""LeRobot DiffusionPolicy 与本地 PushT 数据集 collate 的桥接（无自研 algo 骨干）。"""

from __future__ import annotations

import dataclasses
import pickle
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW

# 必须在 import lerobot 之前注入 vendored 源码路径，并 stub policies 包
_LEROOT = Path(__file__).resolve().parents[1]
_LSRC = _LEROOT / "lerobot" / "src"
if str(_LSRC) not in sys.path:
    sys.path.insert(0, str(_LSRC))
if str(_LEROOT) not in sys.path:
    sys.path.append(str(_LEROOT))

from utils.lerobot_policy_stub import stub_lerobot_policies_packages

stub_lerobot_policies_packages(_LEROOT)

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from utils.dataset import BACK_KEY, FRONT_KEY, STATE_KEY

# 与 build_push_diffusion_config 中 input_features 的 VISUAL 键顺序一致
IMAGE_KEYS: tuple[str, ...] = (FRONT_KEY, BACK_KEY)


def ensure_lerobot_on_path(project_root: Path | None = None) -> Path:
    """将 vendored `lerobot/src` 加入 sys.path（若其它脚本先于本模块 import，可显式调用）。"""
    root = project_root if project_root is not None else _LEROOT
    src = root / "lerobot" / "src"
    s = str(src.resolve())
    if s not in sys.path:
        sys.path.insert(0, s)
    return root


def _downsample_factor(cfg: DiffusionConfig) -> int:
    return 2 ** len(cfg.down_dims)


def build_push_diffusion_config(
    *,
    state_dim: int,
    action_dim_per_step: int,
    horizon: int,
    image_chw: tuple[int, int, int],
    n_obs_steps: int = 1,
    n_action_steps: int | None = None,
    num_train_timesteps: int = 100,
    overrides: dict[str, Any] | None = None,
) -> DiffusionConfig:
    """构造与本地数据集键名、形状一致的 DiffusionConfig。"""
    c, h, w = image_chw
    if n_action_steps is None:
        n_action_steps = horizon
    if n_action_steps > horizon - n_obs_steps + 1:
        raise ValueError(
            f"n_action_steps={n_action_steps} 必须 ≤ horizon - n_obs_steps + 1 = {horizon - n_obs_steps + 1}"
        )
    cfg = DiffusionConfig(
        n_obs_steps=n_obs_steps,
        horizon=horizon,
        n_action_steps=n_action_steps,
        input_features={
            STATE_KEY: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
            FRONT_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(c, h, w)),
            BACK_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(c, h, w)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim_per_step,)),
        },
        num_train_timesteps=num_train_timesteps,
    )
    factor = _downsample_factor(cfg)
    if horizon % factor != 0:
        raise ValueError(
            f"horizon={horizon} 必须能被 U-Net 时间下采样因子 {factor} 整除（len(down_dims)={len(cfg.down_dims)}）。"
        )
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)
    cfg.validate_features()
    return cfg


def expand_single_timestep_obs(
    obs: dict[str, torch.Tensor],
    n_obs_steps: int,
    *,
    image_keys: tuple[str, ...] = IMAGE_KEYS,
    state_key: str = STATE_KEY,
) -> dict[str, torch.Tensor]:
    """将 collate 的单时刻观测扩成 (B, n_obs_steps, …)。当前实现：沿时间维重复同一帧。"""
    out: dict[str, torch.Tensor] = {}
    st = obs[state_key]
    b = st.shape[0]
    if n_obs_steps == 1:
        out[state_key] = st.unsqueeze(1)
        for k in image_keys:
            out[k] = obs[k].unsqueeze(1)
        return out
    out[state_key] = st.unsqueeze(1).expand(b, n_obs_steps, -1).contiguous()
    for k in image_keys:
        img = obs[k]
        out[k] = img.unsqueeze(1).expand(b, n_obs_steps, *img.shape[1:]).contiguous()
    return out


def dataset_batch_to_lerobot(
    batch: dict[str, Any],
    horizon: int,
    n_obs_steps: int,
    *,
    image_keys: tuple[str, ...] = IMAGE_KEYS,
    state_key: str = STATE_KEY,
) -> dict[str, torch.Tensor]:
    """将 `LeRobotPushTDataset.collate_fn` 输出转为 DiffusionPolicy.forward 所需字典。"""
    obs = batch["obs"]
    action = batch["action"]
    if action.dim() != 3 or action.shape[1] != horizon:
        raise ValueError(f"期望 action 形状 [B,{horizon},D_a]，得到 {tuple(action.shape)}")
    exp = expand_single_timestep_obs(obs, n_obs_steps, image_keys=image_keys, state_key=state_key)
    b = action.shape[0]
    dev = action.device
    pad = torch.zeros(b, horizon, dtype=torch.bool, device=dev)
    return {
        OBS_STATE: exp[state_key],
        **{k: exp[k] for k in image_keys},
        ACTION: action,
        "action_is_pad": pad,
    }


def set_rgb_backbone_trainable(policy: DiffusionPolicy, trainable: bool) -> None:
    enc = policy.diffusion.rgb_encoder
    if isinstance(enc, nn.ModuleList):
        for module in enc:
            for p in module.backbone.parameters():
                p.requires_grad = trainable
    else:
        for p in enc.backbone.parameters():
            p.requires_grad = trainable


def ensure_batched_obs(obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """state (D,) → (1,D)；图像 (3,H,W) → (1,3,H,W)。"""
    out = dict(obs)
    st = out[STATE_KEY]
    if st.dim() == 1:
        out[STATE_KEY] = st.unsqueeze(0)
    for k in IMAGE_KEYS:
        t = out[k]
        if t.dim() == 3:
            out[k] = t.unsqueeze(0)
    return out


def _obs_batch_for_generate(
    policy: DiffusionPolicy,
    obs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """generate_actions / _prepare_global_conditioning 需要 OBS_STATE 与 OBS_IMAGES。"""
    obs_b = ensure_batched_obs(obs)
    exp = expand_single_timestep_obs(obs_b, policy.config.n_obs_steps)
    cam_keys = tuple(policy.config.image_features.keys())
    stacked = torch.stack([exp[k] for k in cam_keys], dim=-4)
    return {OBS_STATE: exp[STATE_KEY], OBS_IMAGES: stacked}


class PushTLerobotTrainer:
    """与 scripts/train.py 对接：单步训练、验证 loss、采样、存盘。"""

    def __init__(
        self,
        policy: DiffusionPolicy,
        *,
        device: torch.device,
        lr: float,
        grad_clip_norm: float,
        weight_decay: float | None = None,
    ):
        self.policy = policy.to(device)
        self.device = device
        self.grad_clip_norm = grad_clip_norm
        wd = policy.config.optimizer_weight_decay if weight_decay is None else weight_decay
        betas = tuple(policy.config.optimizer_betas)
        self.optimizer = AdamW(
            policy.get_optim_params(),
            lr=lr,
            betas=betas,
            eps=policy.config.optimizer_eps,
            weight_decay=wd,
        )

    @property
    def config(self) -> DiffusionConfig:
        return self.policy.config

    def train_step(self, batch: dict[str, Any]) -> float:
        self.policy.train()
        lb = dataset_batch_to_lerobot(
            batch,
            self.config.horizon,
            self.config.n_obs_steps,
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss, _ = self.policy.forward(lb)
        loss.backward()
        if self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        return float(loss.item())

    @torch.no_grad()
    def eval_loss(self, batch: dict[str, Any]) -> float:
        self.policy.eval()
        lb = dataset_batch_to_lerobot(
            batch,
            self.config.horizon,
            self.config.n_obs_steps,
        )
        loss, _ = self.policy.forward(lb)
        return float(loss.item())

    @torch.no_grad()
    def sample_action_chunk_flat(
        self,
        obs: dict[str, torch.Tensor],
        *,
        num_inference_steps: int | None = None,
    ) -> torch.Tensor:
        """返回 [B, horizon * D_a]，与旧管线 flatten chunk 一致。"""
        self.policy.eval()
        if num_inference_steps is not None:
            self.policy.diffusion.num_inference_steps = int(num_inference_steps)
        gb = _obs_batch_for_generate(self.policy, obs)
        pred = self.policy.diffusion.generate_actions(gb, noise=None)
        bsz = pred.shape[0]
        return pred.reshape(bsz, -1)

    def state_dict(self) -> dict[str, Any]:
        return {
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config_pickle": pickle.dumps(self.policy.config),
        }


def save_checkpoint(
    path: Path,
    trainer: PushTLerobotTrainer,
    *,
    step: int,
    best_metric: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "best_metric": best_metric,
        "policy_type": "lerobot_diffusion",
        **trainer.state_dict(),
    }
    torch.save(payload, path)


def load_trainer_from_checkpoint(
    path: Path,
    *,
    device: torch.device,
    lr: float,
    grad_clip_norm: float,
) -> tuple[PushTLerobotTrainer, int, float]:
    ckpt = torch.load(path, map_location=device)
    if ckpt.get("policy_type") != "lerobot_diffusion":
        raise ValueError(
            "该 checkpoint 不是 lerobot_diffusion 格式；旧版自研 algo 权重无法加载到新管线。"
        )
    cfg = pickle.loads(ckpt["config_pickle"])
    if not isinstance(cfg, DiffusionConfig):
        raise TypeError(f"config_pickle 类型错误: {type(cfg)}")
    policy = DiffusionPolicy(cfg).to(device)
    policy.load_state_dict(ckpt["policy_state_dict"], strict=True)
    trainer = PushTLerobotTrainer(
        policy, device=device, lr=lr, grad_clip_norm=grad_clip_norm
    )
    trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    for g in trainer.optimizer.param_groups:
        g["lr"] = lr
    step = int(ckpt.get("step", 0))
    best = float(ckpt.get("best_metric", float("inf")))
    return trainer, step, best


class PushTLerobotPolicyFacade:
    """仿真 / 离线脚本：`policy.act(obs)` 与 `policy.device`。"""

    def __init__(self, trainer: PushTLerobotTrainer):
        self.trainer = trainer
        self.device = trainer.device

    def act(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.trainer.sample_action_chunk_flat(obs).squeeze(0)
