"""脚本：使用已录制的 LeRobot 数据，在 Isaac Lab 环境中回放 PushT 轨迹。

- 从 LeRobot 数据集加载 episode / frame
- 按照录制时存下来的 `action` 逐步驱动 Isaac 环境
- 仅用于回放和可视化，不再写回数据集
"""

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import cv2

from isaaclab.app import AppLauncher


# =========================
# 参数解析
# =========================
parser = argparse.ArgumentParser(description="Replay LeRobot dataset in Isaac Lab environment.")
parser.add_argument("--task", type=str, default="Isaac-Pusht-v0", help="Name of the task.")
parser.add_argument("--repo_id", type=str, default="isaac_pusht", help="Repo ID used when recording.")
parser.add_argument(
    "--root",
    type=str,
    default=None,
    help="Root directory of the LeRobot dataset. 默认：<project_root>/data/<repo_id>",
)
parser.add_argument("--num_episodes", type=int, default=None, help="Max number of episodes to replay.")
parser.add_argument("--start_episode", type=int, default=0, help="Episode index to start from.")

# 透传 Isaac Lab / AppLauncher 的通用参数
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


# =========================
# 启动 Isaac 应用
# =========================
app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

# 在仿真启动后再导入 Isaac / LeRobot 相关依赖
from isaaclab_tasks.utils import parse_env_cfg

import isaaclab_tasks  # noqa: F401
import isaac_pusht.tasks  # noqa: F401
from lerobot.datasets.lerobot_dataset import LeRobotDataset
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from utils.tcp_trajectory_viz import visualize_tcp_chunk_trajectory

try:
    from isaacsim.util.debug_draw import _debug_draw

    DEBUG_DRAW = _debug_draw.acquire_debug_draw_interface()
except Exception as e:
    DEBUG_DRAW = None
    print(f"[Replay] Debug draw unavailable: {e}")


def make_env():
    """构造与录制时一致的单环境 PushT 任务。"""
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    # 回放时也关闭 timeout，由我们控制 episode 生命周期
    env_cfg.terminations.time_out = None
    if hasattr(env_cfg.scene, "camera"):
        env_cfg.scene.camera.data_types = ["rgb"]
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    return env


def load_dataset() -> LeRobotDataset:
    """加载 LeRobot 数据集（只读）。"""
    project_root = Path(__file__).parent.parent
    root = Path(args_cli.root) if args_cli.root is not None else project_root / "data" / args_cli.repo_id
    if not root.exists():
        print(f"[Replay] 数据集根目录不存在: {root}")
        simulation_app.close()
        sys.exit(1)

    ds = LeRobotDataset(repo_id=args_cli.repo_id, root=root)
    print(f"当前数据集已包含 {ds.num_episodes} 条记录")
    print(f"[Replay] Loaded dataset from {root}, episodes={ds.num_episodes}")
    return ds


def replay_episode(env, dataset: LeRobotDataset, episode_idx: int):
    """使用数据集中给定 episode 的 action 序列在 Isaac 环境中回放一次。

    输入:
        env: Isaac 环境实例（单 env）。
        dataset: LeRobotDataset 实例。
        episode_idx: 要回放的 episode 索引。
    """
    # 取出该 episode 的所有帧索引
    episodes = dataset.meta.episodes
    ep_info = episodes[episode_idx]
    frame_indices = list(range(ep_info["dataset_from_index"], ep_info["dataset_to_index"]))

    print(f"[Replay] Episode {episode_idx} | frames={len(frame_indices)}")

    obs, info = env.reset()

    device = torch.device(args_cli.device if torch.cuda.is_available() else "cpu")

    # 在回放开始时，用 episode 第一帧的状态对齐一次方块和目标，其余帧不再强制覆盖
    
    first_frame = dataset[frame_indices[0]]
    state_np = first_frame["observation.state"]
    if not isinstance(state_np, np.ndarray):
        state_np = np.asarray(state_np, dtype=np.float32)
    state_np = state_np.astype(np.float32).reshape(-1)
    first_action = first_frame["action"]
    if not isinstance(first_action, np.ndarray):
        first_action = np.asarray(first_action, dtype=np.float32)
    first_action = first_action.astype(np.float32).reshape(-1)
    if first_action.size != 8:
        raise ValueError(f"[Replay] 仅支持8维绝对动作，当前数据集 action_dim={first_action.size}")

    # record_lerobot.py 中定义的 state 拼接顺序：
    # tcp_pos(3) + tcp_quat(4) + obj_pos(3) + obj_quat(4) + goal_pos(3) + goal_quat(4)
    obj_pos = state_np[7:10]
    obj_quat = state_np[10:14]
    goal_pos = state_np[14:17]
    goal_quat = state_np[17:21]

    obj_pos_t = torch.from_numpy(obj_pos).to(env.device).unsqueeze(0)
    obj_quat_t = torch.from_numpy(obj_quat).to(env.device).unsqueeze(0)
    goal_pos_t = torch.from_numpy(goal_pos).to(env.device).unsqueeze(0)
    goal_quat_t = torch.from_numpy(goal_quat).to(env.device).unsqueeze(0)

    # RigidObject / articulation 的接口可能不同，这里分别做适配
    t_block_entity = env.scene["t_block"]
    goal_entity = env.scene["goal_tee"]

    # 默认只回放单环境，因此 env_ids=[0]
    env_ids = torch.tensor([0], dtype=torch.long, device=env.device)

    # 方块
    if hasattr(t_block_entity, "set_world_poses"):
        t_block_entity.set_world_poses(obj_pos_t, obj_quat_t)
    elif hasattr(t_block_entity, "write_root_pose_to_sim"):
        # RigidObject.write_root_pose_to_sim(root_pose, env_ids=None)
        root_pose_obj = torch.cat([obj_pos_t, obj_quat_t], dim=-1)
        t_block_entity.write_root_pose_to_sim(root_pose_obj, env_ids=env_ids)

    # 目标
    if hasattr(goal_entity, "set_world_poses"):
        goal_entity.set_world_poses(goal_pos_t, goal_quat_t)
    elif hasattr(goal_entity, "write_root_pose_to_sim"):
        root_pose_goal = torch.cat([goal_pos_t, goal_quat_t], dim=-1)
        goal_entity.write_root_pose_to_sim(root_pose_goal, env_ids=env_ids)


    with torch.no_grad():
        for local_step, frame_idx in enumerate(frame_indices):
            frame = dataset[frame_idx]
            action_np = frame["action"]

            # 可视化采集时相机画面：前腕 & 后腕
            front_img = frame["observation.front_wrist_camera_image"].detach().cpu().numpy() * 255.0
            back_img = frame["observation.back_wrist_camera_image"].detach().cpu().numpy() * 255.0

            # 数据以 (C,H,W), uint8 RGB 存储，转换为 OpenCV 需要的 (H,W,C) BGR
            front_rgb = np.transpose(front_img.astype(np.uint8), (1, 2, 0))
            front_bgr = cv2.cvtColor(front_rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow("Front Wrist (Replay)", front_bgr)

            
            back_rgb = np.transpose(back_img.astype(np.uint8), (1, 2, 0))
            back_bgr = cv2.cvtColor(back_rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow("Back Wrist (Replay)", back_bgr)

            # 让窗口刷新，同时支持按 'q' 退出当前 episode
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print(f"[Replay]  ep={episode_idx} interrupted by 'q' key during step={local_step}")
                break


            # 准备当前帧的 action
            if not isinstance(action_np, np.ndarray):
                action_np = np.asarray(action_np, dtype=np.float32)
            action_np = action_np.astype(np.float32).reshape(-1)
            if action_np.size != 8:
                raise ValueError(f"[Replay] 帧 {frame_idx} 的 action_dim={action_np.size}，期望为8")

            action = torch.from_numpy(action_np).to(device)
            tcp_pos = env.scene["tfs"].data.target_pos_w[0, -1, :].cpu().numpy()

            if DEBUG_DRAW is not None:
                horizon = 16
                end_idx = min(local_step + horizon, len(frame_indices))
                seq_actions = []
                for k in range(local_step, end_idx):
                    f_k = dataset[frame_indices[k]]
                    a_k = f_k["action"]
                    if not isinstance(a_k, np.ndarray):
                        a_k = np.asarray(a_k, dtype=np.float32)
                    a_k = a_k.astype(np.float32).reshape(-1)
                    if a_k.size != 8:
                        raise ValueError(f"[Replay] 帧 {frame_indices[k]} 的 action_dim={a_k.size}，期望为8")
                    seq_actions.append(a_k)
                if len(seq_actions) > 0:
                    action_seq = torch.from_numpy(np.stack(seq_actions, axis=0)).to(device)
                    visualize_tcp_chunk_trajectory(
                        current_tcp_pos=tcp_pos,
                        action_seq=action_seq,
                        num_steps=action_seq.shape[0],
                        debug_draw=DEBUG_DRAW,
                        action_mode="absolute",
                        point_size=6,
                        clear_previous=True,
                    )

            # 应用当前帧的动作
            obs, reward, terminated, truncated, info = env.step(action.unsqueeze(0).repeat(env.num_envs, 1))

            # 若环境内部提前终止，则停止当前 episode 回放
            if terminated.any() or truncated.any():
                print(f"[Replay]  ep={episode_idx} terminated early at step={local_step}")
                break


def main():
    dataset = load_dataset()
    env = make_env()

    max_eps = dataset.num_episodes if args_cli.num_episodes is None else min(
        args_cli.num_episodes, dataset.num_episodes - args_cli.start_episode
    )
    start = max(0, args_cli.start_episode)

    print(f"[Replay] Replaying from episode {start} for {max_eps} episodes.")

    try:
        for ep_offset in range(max_eps):
            ep_idx = start + ep_offset
            if ep_idx >= dataset.num_episodes:
                break
            replay_episode(env, dataset, ep_idx)
    except KeyboardInterrupt:
        print("[Replay] Interrupted by user.")
    except Exception as e:
        print(f"[Replay] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        simulation_app.close()
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"[Replay] Failed to close OpenCV windows: {e}")


if __name__ == "__main__":
    main()

