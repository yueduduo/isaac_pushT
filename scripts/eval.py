"""Offline evaluation script for Transformer-based PushT policies."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from algo.diffusion.policy import DiffusionPolicy
from algo.flow_matching.policy import FlowMatchingPolicy
from utils.dataset import DatasetConfig, build_dataloaders

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - optional dependency guard
    SummaryWriter = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate PushT policy checkpoint on LeRobot dataset.")
    parser.add_argument("--algo", type=str, choices=["diffusion", "flow_matching"], required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--repo-id", type=str, default="isaac_pusht")
    parser.add_argument("--root", type=str, default="data/isaac_pusht")
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tensorboard", action="store_true", default=False, help="Enable TensorBoard logging.")
    parser.add_argument("--tb-logdir", type=str, default="runs", help="TensorBoard root directory.")
    parser.add_argument("--tb-run-name", type=str, default=None, help="Optional TensorBoard run name.")
    parser.add_argument("--tb-step", type=int, default=0, help="Global step for TensorBoard scalar logging.")
    return parser.parse_args()


@torch.no_grad()
def evaluate_action_mse(policy, dataloader: torch.utils.data.DataLoader) -> float:
    errors = []
    for batch in dataloader:
        obs = {k: v.to(policy.device) for k, v in batch["obs"].items()}
        gt_action = batch["action"].to(policy.device)
        if gt_action.dim() > 2:
            gt_action = gt_action.reshape(gt_action.shape[0], -1)
        pred_action = policy.trainer.sample_actions(obs)
        errors.append(F.mse_loss(pred_action, gt_action).item())
    return float(sum(errors) / max(len(errors), 1))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset_cfg = DatasetConfig(repo_id=args.repo_id, root=args.root, horizon=args.horizon, preload_in_memory=True)
    _, val_loader = build_dataloaders(
        dataset_cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=0.95,
        val_split=True,
    )

    sample = next(iter(val_loader))
    state_dim = sample["obs"]["observation.state"].shape[-1]
    action_dim = sample["action"][0].numel()

    if args.algo == "diffusion":
        policy = DiffusionPolicy(state_dim=state_dim, action_dim=action_dim, device=device)
    else:
        policy = FlowMatchingPolicy(state_dim=state_dim, action_dim=action_dim, device=device)

    policy.load(args.ckpt)
    mse = evaluate_action_mse(policy, val_loader)
    print(f"[Eval] algo={args.algo} ckpt={Path(args.ckpt).name} action_mse={mse:.6f}")

    if args.tensorboard:
        if SummaryWriter is None:
            raise ImportError("TensorBoard is not available. Please install it with: pip install tensorboard")
        run_name = args.tb_run_name or f"{args.algo}_h{args.horizon}"
        tb_dir = Path(args.tb_logdir) / run_name
        writer = SummaryWriter(log_dir=str(tb_dir))
        writer.add_scalar("eval/action_mse", mse, args.tb_step)
        writer.add_text("eval/ckpt", str(args.ckpt), args.tb_step)
        writer.add_text("eval/repo_id", args.repo_id, args.tb_step)
        writer.add_text("eval/root", args.root, args.tb_step)
        writer.close()
        print(f"[TensorBoard] Logged eval/action_mse to: {tb_dir} (step={args.tb_step})")


if __name__ == "__main__":
    main()
