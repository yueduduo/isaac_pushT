"""
离线拟合诊断：与训练一致的 LeRobot DiffusionPolicy.forward 标量 loss（非逆扩散采样）。
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
from utils.lerobot_push_diffusion import load_trainer_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Offline diffusion fit (LeRobot forward loss).")
    parser.add_argument("--ckpt", type=str, default="checkpoints/best_diffusion.pt")
    parser.add_argument("--repo-id", type=str, default="isaac_pusht")
    parser.add_argument("--root", type=str, default="data/isaac_pusht")
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-batches", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    return parser.parse_args()


@torch.no_grad()
def mean_forward_loss(trainer, loader, num_batches: int) -> float:
    trainer.policy.eval()
    total_loss = 0.0
    total_count = 0
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        bsz = batch["action"].shape[0]
        loss = trainer.eval_loss(batch)
        total_loss += loss * bsz
        total_count += bsz
    return total_loss / max(total_count, 1)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path

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

    trainer, _, _ = load_trainer_from_checkpoint(
        ckpt_path,
        device=device,
        lr=args.lr,
        grad_clip_norm=args.grad_clip_norm,
    )

    train_m = mean_forward_loss(trainer, train_loader, args.num_batches)
    val_m = mean_forward_loss(trainer, val_loader, args.num_batches)
    print(f"[Diagnose] ckpt={ckpt_path.name} train_mean_loss={train_m:.6f} val_mean_loss={val_m:.6f}")


if __name__ == "__main__":
    main()
