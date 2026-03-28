"""Test script for evaluating a trained Diffusion Policy in Isaac PushT environment."""

import argparse
import os
import sys
import torch
import numpy as np
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LERO_SRC = PROJECT_ROOT / "lerobot" / "src"
if str(_LERO_SRC) not in sys.path:
    sys.path.insert(0, str(_LERO_SRC))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# 1. 启动 Isaac Lab 环境 (必须在导入其他 isaaclab 组件前)
from isaaclab.app import AppLauncher

# ==========================================
# 参数解析
# ==========================================
parser = argparse.ArgumentParser(description="Evaluate a trained PushT model.")
parser.add_argument("--task", type=str, default="Isaac-Pusht-v0", help="Gym task name")
parser.add_argument("--checkpoint", type=str, default="checkpoints/best_diffusion.pt", help="Path to model checkpoint")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to evaluate")
parser.add_argument("--max_steps", type=int, default=400, help="Max steps per episode (timeout)")
parser.add_argument("--horizon", type=int, default=32, help="Temporal horizon used during training")
parser.add_argument("--action-steps", type=int, default=16, help="Number of action steps to execute per inference")

# 增加 AppLauncher 的参数并解析
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动 Isaac Sim
app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

# ==========================================
# 导入其他库 (在仿真器启动后导入)
# ==========================================
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import isaac_pusht.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from utils.lerobot_push_diffusion import PushTLerobotPolicyFacade, load_trainer_from_checkpoint
from utils.normalization import ckpt_norm_path, denormalize_action, load_norm_stats, normalize_state
from utils.tcp_trajectory_viz import visualize_tcp_chunk_trajectory


def main():
    device = torch.device(args_cli.device if torch.cuda.is_available() else "cpu")

    # 1. 创建环境配置
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    # 强制开启相机数据
    if hasattr(env_cfg.scene, "camera"):
        env_cfg.scene.camera.data_types = ["rgb"]
    
    # 禁用自动重置，由我们手动控制
    env_cfg.terminations.time_out = None
    
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # Debug draw interface for visualizing TCP future positions
    try:
        from isaacsim.util.debug_draw import _debug_draw

        debug_draw = _debug_draw.acquire_debug_draw_interface()
    except Exception:
        debug_draw = None
    
    # 2. 初始化 Policy（horizon * 8：单步动作为绝对 TCP 位姿 + gripper）
    action_dim_per_step = 8

    ckpt_path = Path(args_cli.checkpoint)
    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found at {ckpt_path}")
        simulation_app.close()
        return

    print(f"Loading checkpoint: {ckpt_path}")
    trainer, _, _ = load_trainer_from_checkpoint(
        ckpt_path,
        device=device,
        lr=1e-4,
        grad_clip_norm=1.0,
    )
    policy = PushTLerobotPolicyFacade(trainer)
    norm_path = ckpt_norm_path(ckpt_path)
    state_mean, state_std, action_mean, action_std = load_norm_stats(norm_path, device=device)
    
    success_count = 0
    total_reward = 0

    print("-" * 50)
    print(f"Starting Evaluation: {args_cli.num_episodes} episodes")
    print("-" * 50)

    try:
        for ep in range(args_cli.num_episodes):
            obs, info = env.reset()
            done = False
            step_idx = 0
            ep_reward = 0
            is_success = False
            
            print(f"Episode {ep+1} started...")
            
            while not done and step_idx < args_cli.max_steps:
                with torch.no_grad():
                    # 3. 提取当前观测值 (必须与 record_lerobot.py 逻辑完全一致)
                    
                    # 提取相机图像
                    front_rgb = env.scene["front_wrirst_camera"].data.output["rgb"][0].cpu().numpy()
                    front_img = torch.from_numpy(np.moveaxis(front_rgb, -1, 0).astype(np.float32) / 255.0).to(device)
                    
                    back_rgb = env.scene["back_wrirst_camera"].data.output["rgb"][0].cpu().numpy()
                    back_img = torch.from_numpy(np.moveaxis(back_rgb, -1, 0).astype(np.float32) / 255.0).to(device)
                    
                    # 提取状态 (TCP + Object + Goal)
                    current_tcp = env.scene["tfs"].data.target_pos_w[0, -1, :].cpu().numpy()
                    tcp_quat = env.scene["tfs"].data.target_quat_w[0, -1, :].cpu().numpy()
                    obj_pos = env.scene["t_block"].data.root_pos_w[0].cpu().numpy()
                    obj_quat = env.scene["t_block"].data.root_quat_w[0].cpu().numpy()
                    goal_pos = env.scene["goal_tee"].data.root_pos_w[0].cpu().numpy()
                    goal_quat = env.scene["goal_tee"].data.root_quat_w[0].cpu().numpy()
                    
                    state_vec = np.concatenate([current_tcp, tcp_quat, obj_pos, obj_quat, goal_pos, goal_quat])
                    state_tensor = torch.from_numpy(state_vec.astype(np.float32)).to(device)
                    state_tensor = normalize_state(state_tensor, state_mean, state_std)
                    
                    obs_dict = {
                        "observation.front_wrist_camera_image": front_img,
                        "observation.back_wrist_camera_image": back_img,
                        "observation.state": state_tensor,
                    }
                    
                    # 4. 模型推理
                    # 返回的是 (horizon * 8) 形状的张量
                    flat_action_seq = policy.act(obs_dict)

                    # 5. 执行一段动作序列 (Chunking Policy)
                    action_seq = flat_action_seq.view(args_cli.horizon, action_dim_per_step)
                    action_seq = denormalize_action(action_seq, action_mean, action_std)
                    if action_seq.shape[-1] != action_dim_per_step:
                        raise ValueError(
                            f"[Eval] 模型输出动作维度异常: got={action_seq.shape[-1]}, expected={action_dim_per_step}"
                        )

                    # 确定本次循环要执行的步数 (不能超过 horizon，且受限于 max_steps)
                    num_to_exec = min(args_cli.action_steps, args_cli.horizon)

                    # 5.1 可视化当前 chunk 内 TCP 在仿真中的未来轨迹（绝对位姿动作）
                    if debug_draw is not None:
                        visualize_tcp_chunk_trajectory(
                            current_tcp_pos=current_tcp,
                            action_seq=action_seq,
                            num_steps=num_to_exec,
                            debug_draw=debug_draw,
                            action_mode="absolute",
                        )

                    for i in range(num_to_exec):
                        # 如果环境已经结束或超时，提前跳出序列执行
                        if done or step_idx >= args_cli.max_steps:
                            break

                        # 取出序列中的当前动作
                        action = action_seq[i]

                        # 环境步进
                        obs, reward, terminated, truncated, info = env.step(action.unsqueeze(0).repeat(env.num_envs, 1))
                        
                        ep_reward += reward[0].item()
                        step_idx += 1
                        
                        # 6. 成功判定与终止检查
                        # 优先检查 info，其次使用距离阈值 (此处需要重新感知物体位置)
                        if isinstance(info, dict) and "is_success" in info and info["is_success"].any():
                            is_success = True
                            done = True
                            print(f"  [SUCCESS] Triggered by environment info['is_success']")
                        else:
                            # 重新从环境获取物理状态
                            obj_pos = env.scene["t_block"].data.root_pos_w[0].cpu().numpy()
                            goal_pos = env.scene["goal_tee"].data.root_pos_w[0].cpu().numpy()
                            dist = np.linalg.norm(obj_pos - goal_pos)
                            if dist < 0.01: # 5cm 阈值视为成功
                                is_success = True
                                done = True
                                print(f"  [SUCCESS] Triggered by distance: {dist:.4f} < 0.05m")
                        
                        if terminated.any() or truncated.any():
                            if not is_success:
                                print(f"  [TERMINATED] Environment signaled termination/truncation.")
                            done = True

            if is_success:
                print(f"  [SUCCESS] Episode {ep+1} completed in {step_idx} steps.")
                success_count += 1
            elif step_idx >= args_cli.max_steps:
                print(f"  [FAILURE] Episode {ep+1} timed out after {args_cli.max_steps} steps.")
            else:
                print(f"  [FAILURE] Episode {ep+1} terminated unexpectedly.")
            
            total_reward += ep_reward

    except Exception as e:
        print(f"\n[CRASH] Evaluation error: {e}")
        traceback.print_exc()
    finally:
        print("-" * 50)
        print(f"Evaluation Results ({args_cli.num_episodes} episodes):")
        print(f"  Success Rate: {success_count / args_cli.num_episodes * 100:.1f}%")
        print(f"  Average Reward: {total_reward / args_cli.num_episodes:.2f}")
        print("-" * 50)
        env.close()
        simulation_app.close()

if __name__ == "__main__":
    main()
