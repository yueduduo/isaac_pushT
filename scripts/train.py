"""PushT 训练：仅使用 LeRobot DiffusionPolicy（1D 条件 U-Net + diffusers 调度）。"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LERO_SRC = PROJECT_ROOT / "lerobot" / "src"
if str(_LERO_SRC) not in sys.path:
    sys.path.insert(0, str(_LERO_SRC))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.lerobot_policy_stub import stub_lerobot_policies_packages

stub_lerobot_policies_packages(PROJECT_ROOT)

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

from utils.dataset import DatasetConfig, build_dataloaders
from utils.lerobot_push_diffusion import (
    PushTLerobotTrainer,
    build_push_diffusion_config,
    load_trainer_from_checkpoint,
    save_checkpoint,
    set_rgb_backbone_trainable,
)
from utils.normalization import (
    ckpt_norm_path,
    compute_state_action_stats,
    denormalize_action,
    load_norm_stats,
    save_norm_stats,
)
from utils.train_config import (
    RunBundle,
    load_run_bundle,
    train_arg_defaults,
    train_params_from_namespace,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None


def parse_args() -> tuple[argparse.Namespace, RunBundle]:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None, type=str)
    pre_args, remaining = pre.parse_known_args()
    file_bundle = load_run_bundle(pre_args.config)

    parser = argparse.ArgumentParser(
        description="Train PushT with LeRobot DiffusionPolicy.",
        epilog="JSON 见 scripts/configs/train_default.json：train / diffusion_trainer / lerobot_diffusion。",
    )
    parser.set_defaults(**train_arg_defaults(file_bundle))

    parser.add_argument("--repo-id", type=str, dest="repo_id")
    parser.add_argument("--root", type=str)
    parser.add_argument("--horizon", type=int)
    parser.add_argument(
        "--train-steps",
        type=int,
        dest="train_steps",
        help="本轮 optimizer.step 次数。",
    )
    parser.add_argument("--batch-size", type=int, dest="batch_size")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--num-workers", type=int, dest="num_workers")
    parser.add_argument("--device", type=str)
    parser.add_argument("--save-dir", type=str, dest="save_dir")
    parser.add_argument("--diffusion-steps", type=int, dest="diffusion_steps")
    parser.add_argument("--n-obs-steps", type=int, dest="n_obs_steps")
    parser.add_argument(
        "--n-action-steps",
        type=int,
        dest="n_action_steps",
        help="默认与 horizon 相同；须满足 ≤ horizon - n_obs_steps + 1。",
    )
    parser.add_argument("--freeze-resnet", action="store_true", dest="freeze_resnet")
    parser.add_argument("--freeze-steps", type=int, dest="freeze_steps")
    tb_grp = parser.add_mutually_exclusive_group()
    tb_grp.add_argument("--tensorboard", dest="tensorboard", action="store_true")
    tb_grp.add_argument("--no-tensorboard", dest="tensorboard", action="store_false")
    parser.add_argument("--tb-logdir", type=str, dest="tb_logdir")
    parser.add_argument("--tb-run-name", type=str, dest="tb_run_name")
    parser.add_argument("--action-mse", action="store_true", dest="action_mse")
    parser.add_argument("--action-mse-batches", type=int, dest="action_mse_batches")
    parser.add_argument("--random-window-split", action="store_true", dest="random_window_split")
    parser.add_argument("--val-split", action="store_true", dest="val_split")
    parser.add_argument("--train-ratio", type=float, dest="train_ratio")
    parser.add_argument("--split-seed", type=int, dest="split_seed")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--grad-clip-norm", type=float, dest="grad_clip_norm")
    parser.add_argument("--optimizer-weight-decay", type=float, dest="optimizer_weight_decay")

    args = parser.parse_args(remaining)
    train = train_params_from_namespace(args)
    bundle = RunBundle(train=train, diffusion_config_overrides=file_bundle.diffusion_config_overrides)
    return args, bundle


@torch.no_grad()
def evaluate_diffusion(trainer: PushTLerobotTrainer, dataloader: torch.utils.data.DataLoader) -> float:
    trainer.policy.eval()
    losses: list[float] = []
    pbar = tqdm(dataloader, desc="  Validating", leave=False, mininterval=1.0)
    for batch in pbar:
        loss_val = trainer.eval_loss(batch)
        losses.append(loss_val)
        pbar.set_postfix(loss=f"{loss_val:.4f}")
    return float(sum(losses) / max(len(losses), 1))


@torch.no_grad()
def evaluate_action_mse(
    trainer: PushTLerobotTrainer,
    dataloader: torch.utils.data.DataLoader,
    max_batches: int,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
) -> float:
    errors: list[float] = []
    for i, batch in enumerate(dataloader):
        if max_batches > 0 and i >= max_batches:
            break
        obs = batch["obs"]
        gt_action = batch["action"]
        if gt_action.dim() > 2:
            gt_action = gt_action.reshape(gt_action.shape[0], -1)
        pred_action = trainer.sample_action_chunk_flat(obs)
        pred_action_raw = denormalize_action(pred_action, action_mean, action_std)
        gt_action_raw = denormalize_action(gt_action, action_mean, action_std)
        errors.append(F.mse_loss(pred_action_raw, gt_action_raw).item())
    return float(sum(errors) / max(len(errors), 1))


def _assert_lerobot_ckpt(ckpt: dict) -> None:
    if ckpt.get("policy_type") != "lerobot_diffusion":
        raise ValueError(
            "Checkpoint 不是 lerobot_diffusion 格式，无法续训。"
            "旧版自研 algo 的 .pt 与本脚本不兼容，请重新训练。"
        )


def main() -> None:
    args, bundle = parse_args()
    if args.train_steps <= 0:
        raise ValueError("--train-steps 必须为正整数。")
    if args.freeze_steps < 0:
        raise ValueError("--freeze-steps 不能为负。")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset_cfg = DatasetConfig(repo_id=args.repo_id, root=args.root, horizon=args.horizon, preload_in_memory=True)
    train_loader, val_loader = build_dataloaders(
        dataset_cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        device=device,
        split_by_episode=not args.random_window_split,
        val_split=args.val_split,
        split_seed=args.split_seed,
    )
    use_val = val_loader is not None
    use_action_mse = args.action_mse
    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        raise ValueError("train_loader 为空，无法训练。")
    if not use_val:
        print("[Init] 未划分验证集：每 steps_per_epoch 步记录 train 指标，best 由 train loss 决定。")

    actual_ds = train_loader.dataset
    while hasattr(actual_ds, "dataset"):
        actual_ds = actual_ds.dataset

    state_dim = int(actual_ds.all_states.shape[-1])
    action_dim_per_step = int(actual_ds.all_actions.shape[-1])
    chw = tuple(int(x) for x in actual_ds.all_front_images.shape[1:4])

    resume_path: Path | None = None
    if args.resume:
        resume_path = Path(args.resume).resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Checkpoint 不存在: {resume_path}")
        ckpt_probe = torch.load(resume_path, map_location="cpu")
        _assert_lerobot_ckpt(ckpt_probe)

    import gc

    gc.collect()
    torch.cuda.empty_cache()

    print(f"[Init] state_dim={state_dim} action_dim_per_step={action_dim_per_step} horizon={args.horizon} image_chw={chw}")

    if resume_path is not None:
        norm_sidecar = ckpt_norm_path(resume_path)
        if norm_sidecar.is_file():
            state_mean, state_std, action_mean, action_std = load_norm_stats(norm_sidecar, device=None)
            print(f"[Init] Loaded normalization from {norm_sidecar}")
        else:
            state_mean, state_std, action_mean, action_std = compute_state_action_stats(
                actual_ds.all_states, actual_ds.all_actions
            )
            print("[Init] 续训 checkpoint 旁无 .norm.json，已用当前数据集计算归一化。")
    else:
        state_mean, state_std, action_mean, action_std = compute_state_action_stats(
            actual_ds.all_states, actual_ds.all_actions
        )
    actual_ds.set_normalization_stats(state_mean, state_std, action_mean, action_std)
    if use_action_mse:
        action_mean_dev = action_mean.to(device)
        action_std_dev = action_std.to(device)
    else:
        action_mean_dev = None
        action_std_dev = None
    print("[Init] Enabled state/action normalization for training data pipeline.")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    start_step = 0
    best_metric = float("inf")
    writer = None

    overrides = dict(bundle.diffusion_config_overrides)
    overrides["num_train_timesteps"] = args.diffusion_steps

    diff_cfg = build_push_diffusion_config(
        state_dim=state_dim,
        action_dim_per_step=action_dim_per_step,
        horizon=args.horizon,
        image_chw=chw,
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
        num_train_timesteps=args.diffusion_steps,
        overrides=overrides,
    )

    grad_clip = float(args.grad_clip_norm)
    wd = args.optimizer_weight_decay

    if resume_path is not None:
        trainer, start_step, best_metric = load_trainer_from_checkpoint(
            resume_path,
            device=device,
            lr=args.lr,
            grad_clip_norm=grad_clip,
        )
        for g in trainer.optimizer.param_groups:
            g["lr"] = args.lr
        if wd is not None:
            for g in trainer.optimizer.param_groups:
                g["weight_decay"] = wd
        print(f"[Resume] global_step={start_step} best_metric={best_metric}")
    else:
        policy = DiffusionPolicy(diff_cfg).to(device)
        trainer = PushTLerobotTrainer(
            policy,
            device=device,
            lr=args.lr,
            grad_clip_norm=grad_clip,
            weight_decay=wd,
        )

    if args.freeze_resnet and args.freeze_steps > 0 and start_step < args.freeze_steps:
        set_rgb_backbone_trainable(trainer.policy, False)
        print(f"[Freeze] RGB backbone frozen while global_step < {args.freeze_steps}; start_step={start_step}.")
    elif args.freeze_resnet and args.freeze_steps > 0:
        set_rgb_backbone_trainable(trainer.policy, True)

    train_windows = len(train_loader.dataset)
    equiv_epochs_run = args.train_steps / steps_per_epoch
    print(
        f"[Schedule] train_steps={args.train_steps} | steps_per_epoch={steps_per_epoch} "
        f"| batch_size={args.batch_size} | 训练窗口数={train_windows}"
    )
    print(f"[Schedule] 约合每轮 {equiv_epochs_run:.6f} 个 epoch（1 epoch = {steps_per_epoch} steps）。")

    print(f"[Init] Initializing on {device}...")
    torch.cuda.empty_cache()

    best_ckpt_path = save_dir / "best_diffusion.pt"
    last_ckpt_path = save_dir / "last_diffusion.pt"

    if args.tensorboard:
        if SummaryWriter is None:
            raise ImportError("请安装 tensorboard：pip install tensorboard")
        if args.tb_run_name is None:
            timestamp = datetime.now().strftime("%m%d_%H%M")
            run_name = (
                f"lerobot_dp_h{args.horizon}_st{args.diffusion_steps}_lr{args.lr}_bs{args.batch_size}_"
                f"ts{args.train_steps}_{timestamp}"
            )
        else:
            run_name = args.tb_run_name
        tb_dir = Path(args.tb_logdir) / run_name
        writer = SummaryWriter(log_dir=str(tb_dir))
        writer.add_text("config/device", str(device), 0)
        writer.add_text("config/repo_id", args.repo_id, 0)
        writer.add_text("config/root", args.root, 0)
        writer.add_text("config/horizon", str(args.horizon), 0)
        print(f"[TensorBoard] Logging to: {tb_dir}")

    import time

    target_step = start_step + args.train_steps
    completed = start_step
    segment_losses: list[float] = []
    seg_train_sec = 0.0
    pbar = tqdm(total=args.train_steps, desc="Train steps", unit="step", mininterval=1.0)

    while completed < target_step:
        for batch in train_loader:
            if completed >= target_step:
                break
            next_completed = completed + 1
            if args.freeze_resnet and args.freeze_steps > 0 and next_completed == args.freeze_steps + 1:
                set_rgb_backbone_trainable(trainer.policy, True)
                print("[Freeze] RGB backbone unfrozen.")

            if device.type == "cuda" and next_completed == start_step + 1:
                torch.cuda.synchronize()

            t0 = time.time()
            loss_val = trainer.train_step(batch)
            seg_train_sec += time.time() - t0
            segment_losses.append(loss_val)
            completed = next_completed
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss_val:.4f}")

            at_boundary = completed > 0 and completed % steps_per_epoch == 0
            at_end = completed >= target_step
            if not at_boundary and not at_end:
                continue

            train_loss = float(sum(segment_losses) / len(segment_losses))
            segment_losses.clear()
            t_ev0 = time.time()
            if use_val:
                val_loss = evaluate_diffusion(trainer, val_loader)
            else:
                val_loss = None
            if use_action_mse and action_mean_dev is not None and action_std_dev is not None:
                train_action_mse = evaluate_action_mse(
                    trainer, train_loader, args.action_mse_batches, action_mean_dev, action_std_dev
                )
                val_action_mse = (
                    evaluate_action_mse(
                        trainer, val_loader, args.action_mse_batches, action_mean_dev, action_std_dev
                    )
                    if use_val
                    else None
                )
            else:
                train_action_mse = None
                val_action_mse = None
            eval_sec = time.time() - t_ev0

            equiv_ep = completed / steps_per_epoch
            line = (
                f"  [Log] global_step={completed} (累计约合 {equiv_ep:.6f} epoch) "
                f"train_seg={train_loss:.4f} train_wall={seg_train_sec:.2f}s eval={eval_sec:.2f}s"
            )
            if use_val:
                line += f" val={val_loss:.4f}"
            tqdm.write(line)
            seg_train_sec = 0.0

            if writer is not None:
                writer.add_scalar("loss/train", train_loss, completed)
                if use_val:
                    writer.add_scalar("loss/val", val_loss, completed)
                if train_action_mse is not None:
                    writer.add_scalar("action_mse/train", train_action_mse, completed)
                    if val_action_mse is not None:
                        writer.add_scalar("action_mse/val", val_action_mse, completed)
                writer.add_scalar("optim/lr", trainer.optimizer.param_groups[0]["lr"], completed)

            metric_for_best = val_loss if use_val else train_loss
            if metric_for_best < best_metric:
                best_metric = metric_for_best
                save_checkpoint(
                    best_ckpt_path,
                    trainer,
                    step=completed,
                    best_metric=best_metric,
                )
                save_norm_stats(
                    ckpt_norm_path(best_ckpt_path),
                    state_mean=state_mean,
                    state_std=state_std,
                    action_mean=action_mean,
                    action_std=action_std,
                )
            save_checkpoint(last_ckpt_path, trainer, step=completed, best_metric=best_metric)
            save_norm_stats(
                ckpt_norm_path(last_ckpt_path),
                state_mean=state_mean,
                state_std=state_std,
                action_mean=action_mean,
                action_std=action_std,
            )

    pbar.close()

    if writer is not None:
        if use_val:
            writer.add_scalar("loss/best_val", best_metric, completed)
        else:
            writer.add_scalar("loss/best_train", best_metric, completed)
        writer.close()

    if use_val:
        print(f"[Done] best_val={best_metric:.6f}, checkpoints in {save_dir}")
    else:
        print(f"[Done] best_train_loss={best_metric:.6f}, checkpoints in {save_dir}")


if __name__ == "__main__":
    main()
