"""
离线评估：在数据集观测上采样得到的动作 chunk 与监督序列的差距（LeRobot DiffusionPolicy）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LERO_SRC = PROJECT_ROOT / "lerobot" / "src"
if str(_LERO_SRC) not in sys.path:
    sys.path.insert(0, str(_LERO_SRC))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.dataset import DatasetConfig, build_dataloaders
from utils.lerobot_push_diffusion import PushTLerobotTrainer, load_trainer_from_checkpoint
from utils.normalization import ckpt_norm_path, denormalize_action, load_norm_stats

ACTION_DIM_PER_STEP = 8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Dataset action gap (LeRobot diffusion).")
    p.add_argument("--ckpt", type=str, default="checkpoints/best_diffusion.pt")
    p.add_argument("--repo-id", type=str, default="isaac_pusht")
    p.add_argument("--root", type=str, default="data/isaac_pusht")
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-batches", type=int, default=0)
    p.add_argument("--split", type=str, choices=["train", "val", "both"], default="both")
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument(
        "--sample-steps",
        type=int,
        default=0,
        help="推理扩散步数；0 表示使用 checkpoint 内默认（通常等于训练步数）。",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--deterministic-sampling",
        action="store_true",
        help="LeRobot DDPM 仍带随机性；此开关当前仅打印提示，不改变采样。",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    return p.parse_args()


def _device_from_arg(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_str)
    return torch.device("cpu")


@torch.no_grad()
def eval_split(
    trainer: PushTLerobotTrainer,
    loader: torch.utils.data.DataLoader,
    horizon: int,
    num_batches: int,
    num_samples: int,
    sample_steps: int,
    deterministic_sampling: bool,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
) -> dict[str, float]:
    if deterministic_sampling:
        print("[ActionGap] 提示：--deterministic-sampling 对 LeRobot DDPM 未实现零方差采样，结果仍含随机性。")

    trainer.policy.eval()
    ninf = int(sample_steps) if sample_steps > 0 else None

    sum_sq = 0.0
    sum_l2 = 0.0
    sum_l20 = 0.0
    total_windows = 0
    total_elems = 0

    for i, batch in enumerate(loader):
        if num_batches > 0 and i >= num_batches:
            break

        obs = batch["obs"]
        gt = batch["action"]
        if gt.dim() == 2:
            gt_flat = gt
        else:
            gt_flat = gt.reshape(gt.shape[0], -1)

        bsz = gt_flat.shape[0]

        if num_samples <= 1:
            pred_flat = trainer.sample_action_chunk_flat(obs, num_inference_steps=ninf)
        else:
            acc = torch.zeros_like(gt_flat)
            for _ in range(num_samples):
                acc = acc + trainer.sample_action_chunk_flat(obs, num_inference_steps=ninf)
            pred_flat = acc / float(num_samples)

        pred_flat_raw = denormalize_action(pred_flat, action_mean, action_std)
        gt_flat_raw = denormalize_action(gt_flat, action_mean, action_std)
        diff = pred_flat_raw - gt_flat_raw
        sum_sq += float((diff * diff).sum().item())
        total_elems += int(diff.numel())

        sum_l2 += float(torch.norm(diff, p=2, dim=1).sum().item())
        pred_steps_raw = pred_flat_raw.view(bsz, horizon, ACTION_DIM_PER_STEP)
        gt_steps_raw = gt_flat_raw.view(bsz, horizon, ACTION_DIM_PER_STEP)
        sum_l20 += float(torch.norm(pred_steps_raw[:, 0, :] - gt_steps_raw[:, 0, :], p=2, dim=1).sum().item())
        total_windows += bsz

    if total_windows == 0 or total_elems == 0:
        raise RuntimeError("No batches evaluated; check DataLoader or num-batches.")

    return {
        "action_mse": sum_sq / total_elems,
        "mean_l2_full_chunk": sum_l2 / total_windows,
        "mean_l2_step0": sum_l20 / total_windows,
        "num_windows": float(total_windows),
    }


def main() -> None:
    args = parse_args()
    device = _device_from_arg(args.device)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path

    state_mean, state_std, action_mean, action_std = load_norm_stats(ckpt_norm_path(ckpt_path), device=device)

    dataset_cfg = DatasetConfig(
        repo_id=args.repo_id,
        root=args.root,
        horizon=args.horizon,
        preload_in_memory=True,
    )
    train_loader, val_loader = build_dataloaders(
        dataset_cfg,
        batch_size=args.batch_size,
        num_workers=0,
        train_ratio=0.95,
        device=device,
        val_split=True,
    )
    train_ds = train_loader.dataset
    while hasattr(train_ds, "dataset"):
        train_ds = train_ds.dataset
    train_ds.set_normalization_stats(state_mean, state_std, action_mean, action_std)

    trainer, _, _ = load_trainer_from_checkpoint(
        ckpt_path,
        device=device,
        lr=args.lr,
        grad_clip_norm=args.grad_clip_norm,
    )

    print(
        f"[ActionGap] ckpt={ckpt_path.name} horizon={args.horizon} num_samples={args.num_samples} "
        f"sample_steps={args.sample_steps}"
    )

    if args.split in ("train", "both"):
        m = eval_split(
            trainer,
            train_loader,
            horizon=args.horizon,
            num_batches=args.num_batches,
            num_samples=args.num_samples,
            sample_steps=args.sample_steps,
            deterministic_sampling=args.deterministic_sampling,
            action_mean=action_mean,
            action_std=action_std,
        )
        print(
            f"[ActionGap] TRAIN  windows={int(m['num_windows'])}  action_mse={m['action_mse']:.6f}  "
            f"mean_l2_full={m['mean_l2_full_chunk']:.6f}  mean_l2_step0={m['mean_l2_step0']:.6f}"
        )

    if args.split in ("val", "both"):
        m = eval_split(
            trainer,
            val_loader,
            horizon=args.horizon,
            num_batches=args.num_batches,
            num_samples=args.num_samples,
            sample_steps=args.sample_steps,
            deterministic_sampling=args.deterministic_sampling,
            action_mean=action_mean,
            action_std=action_std,
        )
        print(
            f"[ActionGap] VAL    windows={int(m['num_windows'])}  action_mse={m['action_mse']:.6f}  "
            f"mean_l2_full={m['mean_l2_full_chunk']:.6f}  mean_l2_step0={m['mean_l2_step0']:.6f}"
        )


if __name__ == "__main__":
    main()
