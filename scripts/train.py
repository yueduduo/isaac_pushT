"""Unified trainer for Transformer-based Diffusion / Flow Matching policies."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Make sure local extension package is importable even without editable install.
PROJECT_ROOT = Path(__file__).parent.parent
EXT_ROOT = PROJECT_ROOT 
if str(EXT_ROOT) not in sys.path:
    sys.path.append(str(EXT_ROOT))

from algo.diffusion.policy import DiffusionPolicy
from algo.flow_matching.policy import FlowMatchingPolicy
from utils.dataset import DatasetConfig, build_dataloaders
from utils.train_config import (
    RunBundle,
    load_run_bundle,
    make_diffusion_config,
    make_flow_matching_config,
    train_arg_defaults,
    train_params_from_namespace,
)
from utils.normalization import (
    ckpt_norm_path,
    compute_state_action_stats,
    denormalize_action,
    load_norm_stats,
    save_norm_stats,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - optional dependency guard
    SummaryWriter = None


def parse_args() -> tuple[argparse.Namespace, RunBundle]:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None, type=str)
    pre_args, remaining = pre.parse_known_args()
    file_bundle = load_run_bundle(pre_args.config)

    parser = argparse.ArgumentParser(
        description="Train PushT policy with Transformer backbone.",
        epilog="JSON 配置示例见 scripts/configs/train_default.json；--config 解析 train / diffusion_model / flow_matching_model / *_trainer 段。",
    )
    parser.set_defaults(**train_arg_defaults(file_bundle))

    parser.add_argument("--algo", type=str, choices=["diffusion", "flow_matching"])
    parser.add_argument("--repo-id", type=str, dest="repo_id")
    parser.add_argument("--root", type=str)
    parser.add_argument("--horizon", type=int)
    parser.add_argument(
        "--train-steps",
        type=int,
        dest="train_steps",
        help="本轮要执行的 optimizer.step 次数（与 batch_size、数据集大小共同决定约合多少个 epoch）。",
    )
    parser.add_argument("--batch-size", type=int, dest="batch_size")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--num-workers", type=int, dest="num_workers")
    parser.add_argument("--device", type=str)
    parser.add_argument("--save-dir", type=str, dest="save_dir")
    parser.add_argument("--diffusion-steps", type=int, dest="diffusion_steps")
    parser.add_argument("--flow-steps", type=int, dest="flow_steps")
    parser.add_argument("--freeze-resnet", action="store_true", dest="freeze_resnet")
    parser.add_argument(
        "--freeze-steps",
        type=int,
        dest="freeze_steps",
        help="若 --freeze-resnet：前多少个全局 optimizer step 冻结 ResNet backbone（按 step 计，非 epoch）。",
    )
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
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从 .pt 继续训练（与 --algo 一致；配置仍由 --config/命令行决定，权重与优化器从 checkpoint 恢复）。",
    )

    args = parser.parse_args(remaining)
    train = train_params_from_namespace(args)
    bundle = RunBundle(
        train=train,
        diffusion_model=file_bundle.diffusion_model,
        flow_matching_model=file_bundle.flow_matching_model,
        diffusion_trainer_overrides=file_bundle.diffusion_trainer_overrides,
        flow_matching_trainer_overrides=file_bundle.flow_matching_trainer_overrides,
    )
    return args, bundle


@torch.no_grad()
def evaluate_diffusion(policy: DiffusionPolicy, dataloader: torch.utils.data.DataLoader) -> float:
    model = policy.trainer.model
    trainer = policy.trainer
    model.eval()
    losses = []
    pbar = tqdm(dataloader, desc="  Validating", leave=False, mininterval=1.0)
    for batch in pbar:
        obs = batch["obs"]
        action = batch["action"]
        if action.dim() > 2:
            action = action.flatten(1)
        bsz = action.shape[0]
        t = torch.randint(0, trainer.config.num_diffusion_steps, (bsz,), device=policy.device)
        noise = torch.randn_like(action)
        alpha_bar_t = trainer.alpha_bars[t].unsqueeze(-1)
        noisy_action = torch.sqrt(alpha_bar_t) * action + torch.sqrt(1.0 - alpha_bar_t) * noise
        pred = model(noisy_action, t, obs)
        
        loss_val = F.mse_loss(pred, noise).item()
        losses.append(loss_val)
        pbar.set_postfix(loss=f"{loss_val:.4f}")
    return float(sum(losses) / max(len(losses), 1))


@torch.no_grad()
def evaluate_flow_matching(policy: FlowMatchingPolicy, dataloader: torch.utils.data.DataLoader) -> float:
    model = policy.trainer.model
    model.eval()
    losses = []
    pbar = tqdm(dataloader, desc="  Validating", leave=False, mininterval=1.0)
    for batch in pbar:
        obs = batch["obs"]
        action = batch["action"]
        if action.dim() > 2:
            action = action.flatten(1)
        x0 = torch.randn_like(action)
        t = torch.rand(action.shape[0], device=policy.device)
        x_t = (1.0 - t.unsqueeze(-1)) * x0 + t.unsqueeze(-1) * action
        target_v = action - x0
        pred_v = model(x_t, t, obs)
        
        loss_val = F.mse_loss(pred_v, target_v).item()
        losses.append(loss_val)
        pbar.set_postfix(loss=f"{loss_val:.4f}")
    return float(sum(losses) / max(len(losses), 1))


@torch.no_grad()
def evaluate_action_mse(
    policy: DiffusionPolicy | FlowMatchingPolicy,
    dataloader: torch.utils.data.DataLoader,
    max_batches: int,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
) -> float:
    errors = []
    for i, batch in enumerate(dataloader):
        if max_batches > 0 and i >= max_batches:
            break
        obs = batch["obs"]
        gt_action = batch["action"]
        if gt_action.dim() > 2:
            gt_action = gt_action.reshape(gt_action.shape[0], -1)
        if isinstance(policy, DiffusionPolicy):
            # 使用确定性采样记录更稳定的 action_mse 轨迹，减少训练日志抖动。
            pred_action = policy.trainer.sample_actions(obs, deterministic=True)
        else:
            pred_action = policy.trainer.sample_actions(obs)
        pred_action_raw = denormalize_action(pred_action, action_mean, action_std)
        gt_action_raw = denormalize_action(gt_action, action_mean, action_std)
        errors.append(F.mse_loss(pred_action_raw, gt_action_raw).item())
    return float(sum(errors) / max(len(errors), 1))


def _assert_algo_matches_ckpt(algo: str, ckpt: dict) -> None:
    cfg = ckpt.get("config")
    if not isinstance(cfg, dict):
        raise ValueError("Checkpoint missing valid config dict.")
    if algo == "diffusion":
        if "num_diffusion_steps" not in cfg:
            raise ValueError("Checkpoint config 不像 diffusion（缺少 num_diffusion_steps）。")
    else:
        if "ode_steps" not in cfg:
            raise ValueError("Checkpoint config 不像 flow matching（缺少 ode_steps；CLI 为 --algo flow_matching）。")


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

    # [优化] 直接从数据集对象获取维度，彻底移除 Subset 影响
    print(f"[Init] Extracting dimensions from dataset...")
    actual_ds = train_loader.dataset
    while hasattr(actual_ds, "dataset"):
        actual_ds = actual_ds.dataset
        
    state_dim = actual_ds.all_states.shape[-1]
    action_dim = actual_ds.all_actions.shape[-1] * args.horizon

    resume_path: Path | None = None
    if args.resume:
        resume_path = Path(args.resume).resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Checkpoint 不存在: {resume_path}")
        ckpt_probe = torch.load(resume_path, map_location="cpu")
        _assert_algo_matches_ckpt(args.algo, ckpt_probe)

    # 强制进行一次垃圾回收，确保初始化后的干净状态
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    print(f"[Init] Dimensions: state={state_dim}, action={action_dim}")
    if resume_path is not None:
        norm_sidecar = ckpt_norm_path(resume_path)
        if norm_sidecar.is_file():
            state_mean, state_std, action_mean, action_std = load_norm_stats(norm_sidecar, device=None)
            print(f"[Init] Loaded normalization from {norm_sidecar}")
        else:
            state_mean, state_std, action_mean, action_std = compute_state_action_stats(
                actual_ds.all_states, actual_ds.all_actions
            )
            print(
                "[Init] 续训 checkpoint 旁无 .norm.json，已用当前数据集计算归一化；"
                "若数据与初次训练不一致可能影响效果。"
            )
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

    train_windows = len(train_loader.dataset)
    equiv_epochs_run = args.train_steps / steps_per_epoch
    print(
        f"[Schedule] 本轮 train_steps={args.train_steps} | steps_per_epoch={steps_per_epoch} "
        f"| batch_size={args.batch_size} | 训练窗口数={train_windows}"
    )
    print(
        f"[Schedule] 在当前 batch 与数据集下，本轮约合完整遍历数据集 {equiv_epochs_run:.6f} 个 epoch "
        f"（epoch 仅作换算：1 epoch = {steps_per_epoch} 次 optimizer.step）。"
    )

    # [诊断] 初始化 CUDA，强制显存上升
    print(f"[Init] Initializing model on {device}...")
    torch.cuda.empty_cache()

    if args.algo == "diffusion":
        policy = DiffusionPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            diffusion_cfg=make_diffusion_config(bundle),
            model_kw=bundle.diffusion_model.to_model_kwargs(),
        )
    else:
        policy = FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            matching_cfg=make_flow_matching_config(bundle),
            model_kw=bundle.flow_matching_model.to_model_kwargs(),
        )

    if resume_path is not None:
        loaded_step, loaded_best = policy.load(resume_path)
        if loaded_step is not None:
            start_step = loaded_step
        else:
            legacy_ep = ckpt_probe.get("epoch")
            if legacy_ep is not None:
                start_step = int(legacy_ep) * steps_per_epoch
                print(
                    f"[Resume] 旧 checkpoint 仅含 epoch={int(legacy_ep)}，已按当前 steps_per_epoch={steps_per_epoch} "
                    f"折算为 completed global_step={start_step}（若 batch/数据与当初不同则仅为近似）。"
                )
            else:
                print(
                    "[Resume] checkpoint 无 step/epoch 元数据：global_step 从 0 计；"
                    "请用新版 train.py 保存的权重以记录 step。"
                )
        if loaded_best is not None:
            best_metric = loaded_best
        end_step = start_step + args.train_steps
        cum_epochs_end = end_step / steps_per_epoch
        print(
            f"[Resume] 已加载 {resume_path} | 起始 global_step={start_step} | 本轮再执行 {args.train_steps} steps "
            f"→ 结束于 step={end_step}，累计约合 {cum_epochs_end:.6f} 个 epoch | best_metric={best_metric}"
        )

    torch.cuda.synchronize()
    print(f"[Init] Model ready. GPU memory allocated.")

    if args.tensorboard:
        if SummaryWriter is None:
            raise ImportError("TensorBoard is not available. Please install it with: pip install tensorboard")
        
        # 自动根据参数生成详细的运行名称
        if args.tb_run_name is None:
            steps = args.diffusion_steps if args.algo == "diffusion" else args.flow_steps
            timestamp = datetime.now().strftime("%m%d_%H%M")
            run_name = (
                f"{args.algo}_h{args.horizon}_st{steps}_lr{args.lr}_bs{args.batch_size}_"
                f"ts{args.train_steps}_{timestamp}"
            )
        else:
            run_name = args.tb_run_name

        tb_dir = Path(args.tb_logdir) / run_name
        writer = SummaryWriter(log_dir=str(tb_dir))
        writer.add_text("config/algo", args.algo, 0)
        writer.add_text("config/device", str(device), 0)
        writer.add_text("config/repo_id", args.repo_id, 0)
        writer.add_text("config/root", args.root, 0)
        writer.add_text("config/horizon", str(args.horizon), 0)
        writer.add_text("config/val_split", str(use_val), 0)
        writer.add_text("config/action_mse", str(use_action_mse), 0)
        writer.add_text("config/train_steps", str(args.train_steps), 0)
        writer.add_text("config/steps_per_epoch", str(steps_per_epoch), 0)
        if resume_path is not None:
            writer.add_text("config/resume_from", str(resume_path), 0)
            writer.add_text("config/start_step", str(start_step), 0)
        print(f"[TensorBoard] Logging to: {tb_dir}")

    if args.algo == "diffusion":
        eval_val_fn = evaluate_diffusion
        best_ckpt_path = save_dir / "best_diffusion.pt"
        last_ckpt_path = save_dir / "last_diffusion.pt"
    else:
        eval_val_fn = evaluate_flow_matching
        best_ckpt_path = save_dir / "best_flow_matching.pt"
        last_ckpt_path = save_dir / "last_flow_matching.pt"

    if args.freeze_resnet and args.freeze_steps > 0 and start_step < args.freeze_steps:
        policy.trainer.model.set_resnet_backbone_trainable(False)
        print(
            f"[Freeze] ResNet18 frozen while global_step < {args.freeze_steps}; start_step={start_step}."
        )
    elif args.freeze_resnet and args.freeze_steps > 0:
        policy.trainer.model.set_resnet_backbone_trainable(True)

    import time

    target_step = start_step + args.train_steps
    completed = start_step
    segment_losses: list[float] = []
    seg_train_sec = 0.0
    pbar = tqdm(total=args.train_steps, desc="Train steps", unit="step", mininterval=1.0)
    last_log_step = start_step

    while completed < target_step:
        for batch in train_loader:
            if completed >= target_step:
                break
            next_completed = completed + 1
            if args.freeze_resnet and args.freeze_steps > 0 and next_completed == args.freeze_steps + 1:
                policy.trainer.model.set_resnet_backbone_trainable(True)
                print("[Freeze] ResNet18 backbone unfrozen.")

            if device.type == "cuda" and next_completed == start_step + 1:
                torch.cuda.synchronize()

            t0 = time.time()
            loss_val = policy.trainer.train_step(batch)
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
                val_loss = eval_val_fn(policy, val_loader)
            else:
                val_loss = None
            if use_action_mse:
                train_action_mse = evaluate_action_mse(
                    policy, train_loader, args.action_mse_batches, action_mean_dev, action_std_dev
                )
                val_action_mse = (
                    evaluate_action_mse(
                        policy, val_loader, args.action_mse_batches, action_mean_dev, action_std_dev
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

            if use_val and use_action_mse:
                pbar.set_postfix(
                    train=f"{train_loss:.4f}",
                    val=f"{val_loss:.4f}",
                    tr_mse=f"{train_action_mse:.4f}",
                    va_mse=f"{val_action_mse:.4f}",
                )
            elif use_val:
                pbar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}")
            elif use_action_mse:
                pbar.set_postfix(train=f"{train_loss:.4f}", tr_mse=f"{train_action_mse:.4f}")
            else:
                pbar.set_postfix(train=f"{train_loss:.4f}")

            if writer is not None:
                writer.add_scalar("loss/train", train_loss, completed)
                if use_val:
                    writer.add_scalar("loss/val", val_loss, completed)
                if use_action_mse:
                    writer.add_scalar("action_mse/train", train_action_mse, completed)
                    if use_val:
                        writer.add_scalar("action_mse/val", val_action_mse, completed)
                writer.add_scalar("optim/lr", policy.trainer.optimizer.param_groups[0]["lr"], completed)
            metric_for_best = val_loss if use_val else train_loss
            if metric_for_best < best_metric:
                best_metric = metric_for_best
                policy.save(
                    best_ckpt_path, extra_state={"step": completed, "best_metric": best_metric}
                )
                save_norm_stats(
                    ckpt_norm_path(best_ckpt_path),
                    state_mean=state_mean,
                    state_std=state_std,
                    action_mean=action_mean,
                    action_std=action_std,
                )
            policy.save(last_ckpt_path, extra_state={"step": completed, "best_metric": best_metric})
            save_norm_stats(
                ckpt_norm_path(last_ckpt_path),
                state_mean=state_mean,
                state_std=state_std,
                action_mean=action_mean,
                action_std=action_std,
            )
            last_log_step = completed

    pbar.close()

    if writer is not None:
        if use_val:
            writer.add_scalar("loss/best_val", best_metric, last_log_step)
        else:
            writer.add_scalar("loss/best_train", best_metric, last_log_step)
        writer.close()

    if use_val:
        print(f"[Done] best_val={best_metric:.6f}, checkpoints in {save_dir}")
    else:
        print(f"[Done] best_train_loss={best_metric:.6f}, checkpoints in {save_dir}")


if __name__ == "__main__":
    main()
