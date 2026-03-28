"""
离线拟合诊断：复现 DiffusionTrainer 的训练目标，检查 checkpoint 是否学到了数据分布。

该脚本不会做扩散逆过程采样（sample_actions），因此速度比 action MSE 评估快得多。
它直接计算：pred_noise 与真实 noise 的 MSE（与 train_epoch 的 loss 一致）。
"""

import argparse
from pathlib import Path

import sys

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from algo.diffusion.policy import DiffusionPolicy
from algo.diffusion.trainer import DiffusionConfig
from utils.dataset import DatasetConfig, build_dataloaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Offline diffusion fit diagnosis (noise-pred MSE).")
    parser.add_argument("--task", type=str, default="isaac_pusht")
    parser.add_argument("--ckpt", type=str, default="checkpoints/best_diffusion.pt")
    parser.add_argument("--repo-id", type=str, default="isaac_pusht")
    parser.add_argument("--root", type=str, default="data/isaac_pusht")
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-batches", type=int, default=20, help="How many batches to evaluate (per split).")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


@torch.no_grad()
def diffusion_noise_mse(policy: DiffusionPolicy, loader, num_batches: int) -> float:
    model = policy.trainer.model
    trainer = policy.trainer
    model.eval()

    total_loss = 0.0
    total_count = 0

    for i, batch in enumerate(loader):
        if i >= num_batches:
            break

        obs = batch["obs"]
        action = batch["action"]
        if action.dim() > 2:
            action = action.flatten(1)  # [B, horizon*8]

        bsz = action.shape[0]
        t = torch.randint(0, trainer.config.num_diffusion_steps, (bsz,), device=policy.device)
        noise = torch.randn_like(action)
        alpha_bar_t = trainer.alpha_bars[t].unsqueeze(-1)
        noisy_action = torch.sqrt(alpha_bar_t) * action + torch.sqrt(1.0 - alpha_bar_t) * noise

        pred_noise = model(noisy_action, t, obs)
        loss = F.mse_loss(pred_noise, noise, reduction="mean")

        total_loss += float(loss.item()) * bsz
        total_count += bsz

    return total_loss / max(total_count, 1)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    ckpt_path = Path(args.ckpt)
    ckpt_obj = torch.load(ckpt_path, map_location="cpu")
    ckpt_cfg = ckpt_obj.get("config", {})
    diffusion_cfg = DiffusionConfig(**ckpt_cfg) if ckpt_cfg else DiffusionConfig()

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

    state_dim = 21
    action_dim = args.horizon * 8
    policy = DiffusionPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        diffusion_cfg=diffusion_cfg,
    )
    policy.load(ckpt_path)

    train_loss = diffusion_noise_mse(policy, train_loader, num_batches=args.num_batches)
    val_loss = diffusion_noise_mse(policy, val_loader, num_batches=args.num_batches)

    print(f"[Diagnose] ckpt={ckpt_path.name} horizon={args.horizon}")
    print(f"[Diagnose] diffusion_cfg={diffusion_cfg.__dict__}")
    print(f"[Diagnose] noise_pred_mse: train={train_loss:.6f}  val={val_loss:.6f}")


if __name__ == "__main__":
    main()

