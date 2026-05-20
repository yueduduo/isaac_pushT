"""离线评估：LeRobot DiffusionPolicy 在数据集上的 action MSE。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LERO_SRC = PROJECT_ROOT / "lerobot" / "src"
if str(_LERO_SRC) not in sys.path:
    sys.path.insert(0, str(_LERO_SRC))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.dataset import DatasetConfig, build_dataloaders
from utils.lerobot_push_diffusion import load_trainer_from_checkpoint

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate LeRobot diffusion checkpoint on LeRobot dataset.")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--repo-id", type=str, default="isaac_pusht")
    parser.add_argument("--root", type=str, default="data/isaac_pusht")
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lr", type=float, default=1e-4, help="仅用于恢复优化器占位，不影响评估。")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--tensorboard", action="store_true", default=False)
    parser.add_argument("--tb-logdir", type=str, default="runs")
    parser.add_argument("--tb-run-name", type=str, default=None)
    parser.add_argument("--tb-step", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def evaluate_action_mse(trainer, dataloader: torch.utils.data.DataLoader) -> float:
    errors = []
    for batch in dataloader:
        obs = batch["obs"]
        gt_action = batch["action"]
        if gt_action.dim() > 2:
            gt_action = gt_action.reshape(gt_action.shape[0], -1)
        pred_action = trainer.sample_action_chunk_flat(obs)
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
        device=device,
        val_split=True,
    )

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path

    trainer, _, _ = load_trainer_from_checkpoint(
        ckpt_path,
        device=device,
        lr=args.lr,
        grad_clip_norm=args.grad_clip_norm,
    )

    mse = evaluate_action_mse(trainer, val_loader)
    print(f"[Eval] lerobot_diffusion ckpt={ckpt_path.name} action_mse={mse:.6f}")

    if args.tensorboard:
        if SummaryWriter is None:
            raise ImportError("请安装 tensorboard：pip install tensorboard")
        run_name = args.tb_run_name or f"lerobot_dp_h{args.horizon}"
        tb_dir = Path(args.tb_logdir) / run_name
        writer = SummaryWriter(log_dir=str(tb_dir))
        writer.add_scalar("eval/action_mse", mse, args.tb_step)
        writer.add_text("eval/ckpt", str(ckpt_path), args.tb_step)
        writer.close()
        print(f"[TensorBoard] Logged to: {tb_dir}")


if __name__ == "__main__":
    main()
