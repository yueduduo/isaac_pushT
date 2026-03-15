"""Unified trainer for Transformer-based Diffusion / Flow Mapping policies."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

# Make sure local extension package is importable even without editable install.
PROJECT_ROOT = Path(__file__).parent.parent
EXT_ROOT = PROJECT_ROOT 
if str(EXT_ROOT) not in sys.path:
    sys.path.append(str(EXT_ROOT))

from algo.diffusion.policy import DiffusionPolicy
from algo.diffusion.trainer import DiffusionConfig
from algo.flow_mapping.policy import FlowMappingPolicy
from algo.flow_mapping.trainer import FlowMatchingConfig
from utils.dataset import DatasetConfig, build_dataloaders

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - optional dependency guard
    SummaryWriter = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train PushT policy with Transformer backbone.")
    parser.add_argument("--algo", type=str, choices=["diffusion", "flow_mapping"], required=True)
    parser.add_argument("--repo-id", type=str, default="isaac_pusht")
    parser.add_argument("--root", type=str, default="data/isaac_pusht")
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--flow-steps", type=int, default=50)
    parser.add_argument("--freeze-resnet", action="store_true", default=False)
    parser.add_argument("--freeze-epochs", type=int, default=5)
    parser.add_argument("--tensorboard", action="store_true", default=True, help="Enable TensorBoard logging.")
    parser.add_argument("--tb-logdir", type=str, default="runs", help="TensorBoard root directory.")
    parser.add_argument("--tb-run-name", type=str, default=None, help="Optional TensorBoard run name.")
    return parser.parse_args()


@torch.no_grad()
def evaluate_diffusion(policy: DiffusionPolicy, dataloader: torch.utils.data.DataLoader) -> float:
    model = policy.trainer.model
    trainer = policy.trainer
    model.eval()
    losses = []
    for batch in dataloader:
        obs = {k: v.to(policy.device) for k, v in batch["obs"].items()}
        action = batch["action"].to(policy.device)
        if action.dim() > 2:
            action = action.reshape(action.shape[0], -1)
        bsz = action.shape[0]
        t = torch.randint(0, trainer.config.num_diffusion_steps, (bsz,), device=policy.device)
        noise = torch.randn_like(action)
        alpha_bar_t = trainer.alpha_bars[t].unsqueeze(-1)
        noisy_action = torch.sqrt(alpha_bar_t) * action + torch.sqrt(1.0 - alpha_bar_t) * noise
        pred = model(noisy_action, t, obs)
        losses.append(F.mse_loss(pred, noise).item())
    return float(sum(losses) / max(len(losses), 1))


@torch.no_grad()
def evaluate_flow(policy: FlowMappingPolicy, dataloader: torch.utils.data.DataLoader) -> float:
    model = policy.trainer.model
    model.eval()
    losses = []
    for batch in dataloader:
        obs = {k: v.to(policy.device) for k, v in batch["obs"].items()}
        action = batch["action"].to(policy.device)
        if action.dim() > 2:
            action = action.reshape(action.shape[0], -1)
        x0 = torch.randn_like(action)
        t = torch.rand(action.shape[0], device=policy.device)
        x_t = (1.0 - t.unsqueeze(-1)) * x0 + t.unsqueeze(-1) * action
        target_v = action - x0
        pred_v = model(x_t, t, obs)
        losses.append(F.mse_loss(pred_v, target_v).item())
    return float(sum(losses) / max(len(losses), 1))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset_cfg = DatasetConfig(repo_id=args.repo_id, root=args.root, horizon=args.horizon, preload_in_memory=True)
    train_loader, val_loader = build_dataloaders(
        dataset_cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=0.95,
    )

    sample = next(iter(train_loader))
    state_dim = sample["obs"]["observation.state"].shape[-1]
    action_dim = sample["action"][0].numel()
    print(
        f"[Init] algo={args.algo}, device={device}, state_dim={state_dim}, "
        f"action_dim(flat)={action_dim}, horizon={args.horizon}"
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    writer = None

    if args.tensorboard:
        if SummaryWriter is None:
            raise ImportError("TensorBoard is not available. Please install it with: pip install tensorboard")
        run_name = args.tb_run_name or f"{args.algo}_h{args.horizon}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        tb_dir = Path(args.tb_logdir) / run_name
        writer = SummaryWriter(log_dir=str(tb_dir))
        writer.add_text("config/algo", args.algo, 0)
        writer.add_text("config/device", str(device), 0)
        writer.add_text("config/repo_id", args.repo_id, 0)
        writer.add_text("config/root", args.root, 0)
        writer.add_text("config/horizon", str(args.horizon), 0)
        print(f"[TensorBoard] Logging to: {tb_dir}")

    if args.algo == "diffusion":
        policy = DiffusionPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            diffusion_cfg=DiffusionConfig(num_diffusion_steps=args.diffusion_steps, lr=args.lr),
        )
        if args.freeze_resnet and args.freeze_epochs > 0:
            policy.trainer.model.set_resnet_backbone_trainable(False)
            print(f"[Freeze] ResNet18 backbone frozen for first {args.freeze_epochs} epochs.")
        for epoch in range(1, args.epochs + 1):
            if args.freeze_resnet and epoch == args.freeze_epochs + 1:
                policy.trainer.model.set_resnet_backbone_trainable(True)
                print("[Freeze] ResNet18 backbone unfrozen. Start fine-tuning.")
            train_loss = policy.trainer.train_epoch(train_loader)
            val_loss = evaluate_diffusion(policy, val_loader)
            print(f"[Epoch {epoch:03d}] train={train_loss:.6f} val={val_loss:.6f}")
            if writer is not None:
                writer.add_scalar("loss/train", train_loss, epoch)
                writer.add_scalar("loss/val", val_loss, epoch)
                writer.add_scalar("optim/lr", policy.trainer.optimizer.param_groups[0]["lr"], epoch)
            if val_loss < best_val:
                best_val = val_loss
                policy.save(save_dir / "best_diffusion.pt")
            policy.save(save_dir / "last_diffusion.pt")
    else:
        policy = FlowMappingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            flow_cfg=FlowMatchingConfig(lr=args.lr, ode_steps=args.flow_steps),
        )
        if args.freeze_resnet and args.freeze_epochs > 0:
            policy.trainer.model.set_resnet_backbone_trainable(False)
            print(f"[Freeze] ResNet18 backbone frozen for first {args.freeze_epochs} epochs.")
        for epoch in range(1, args.epochs + 1):
            if args.freeze_resnet and epoch == args.freeze_epochs + 1:
                policy.trainer.model.set_resnet_backbone_trainable(True)
                print("[Freeze] ResNet18 backbone unfrozen. Start fine-tuning.")
            train_loss = policy.trainer.train_epoch(train_loader)
            val_loss = evaluate_flow(policy, val_loader)
            print(f"[Epoch {epoch:03d}] train={train_loss:.6f} val={val_loss:.6f}")
            if writer is not None:
                writer.add_scalar("loss/train", train_loss, epoch)
                writer.add_scalar("loss/val", val_loss, epoch)
                writer.add_scalar("optim/lr", policy.trainer.optimizer.param_groups[0]["lr"], epoch)
            if val_loss < best_val:
                best_val = val_loss
                policy.save(save_dir / "best_flow_mapping.pt")
            policy.save(save_dir / "last_flow_mapping.pt")

    if writer is not None:
        writer.add_scalar("loss/best_val", best_val, args.epochs)
        writer.close()

    print(f"[Done] best_val={best_val:.6f}, checkpoints in {save_dir}")


if __name__ == "__main__":
    main()
