"""
【单 episode 回放脚本】

本脚本一次只回放数据集中的一条 episode，使用采集时保存的 action 序列逐步驱动
Isaac Lab 仿真环境(动力学回放，而非运动学回放)，并在回放过程中：
  - 实时拼接并展示所有摄像头图像（左→右）
  - 每步执行后将当前 t_block / 机械臂 TCP 状态与数据集 next_state 比较，
    差异超过阈值时打印警告

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠  回放一致性 / 可复现性说明（尤其是接触任务）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 本脚本是“动力学回放”：同一 action 序列在不同仿真上下文中，可能因接触求解、
   浮点累积与执行顺序差异出现微小偏差；该偏差在长时序回放中可能被放大。
2. Isaac Lab / PhysX 在 reset 相关流程上存在已知限制：部分状态与传感器数据在
   reset 后不会立即完全刷新，通常需要至少一个仿真 step 才能进入稳定读数。
3. 对 pushT 这类强接触任务，第二条及以后 episode 在同一进程内连续回放时，t_block
   轨迹与数据集 next_state 出现渐进偏离属于常见现象，不代表脚本逻辑必然错误。
4. 如需高一致性的离线核对（逐帧比对 / 回归测试），建议采用“单 episode 单进程/
   新 scene”策略，并固定 seed、dt 与物理参数，避免跨 episode 共享历史上下文。
5. 因此，本脚本默认定位为“单 episode 诊断与可视化回放”；若做严格可复现实验，
   请优先使用独立进程重复运行并记录偏差统计。
"""

import argparse
import sys
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import torch

from isaaclab.app import AppLauncher

# ============================================================
# 命令行参数
# ============================================================
parser = argparse.ArgumentParser(
    description="Single-episode replay of a LeRobot dataset in Isaac Lab."
)
parser.add_argument("--task", type=str, default="Isaac-Pusht-v0", help="Isaac 任务名称。")
parser.add_argument("--repo_id", type=str, default="isaac_pusht", help="数据集 repo_id。")
parser.add_argument(
    "--root",
    type=str,
    default=None,
    help="LeRobot 数据集根目录（默认：<项目根>/data/<repo_id>）。",
)
parser.add_argument(
    "--episode_idx",
    type=int,
    default=0,
    help="要回放的 episode 索引（从 0 开始）。",
)
parser.add_argument(
    "--tcp_align_steps",
    type=int,
    default=12,
    help=(
        "回放开始前，用第一帧记录的 TCP 位姿连续驱动 IK 若干步，使末端执行器"
        "收敛至数据集初始 TCP 位置。设为 0 则跳过，仅写回默认关节位形。"
    ),
)
parser.add_argument(
    "--divergence_threshold",
    type=float,
    default=1e-3,
    help="env.step 后，当前状态与数据集 next_state 的 L2 距离超过该阈值时打印警告。",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# ============================================================
# 启动 Isaac Sim（必须在导入任何 Isaac / OmniVerse 模块之前完成）
# ============================================================
app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

# 延迟导入（仿真器启动后才可用）
import isaaclab_tasks  # noqa: F401
import isaac_pusht.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaac_pusht.tasks.manager_based.isaac_pusht.isaac_pusht_env_cfg import DEFAULT_JOINT_POSE
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 可选：调试绘制（无则跳过）
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from utils.tcp_trajectory_viz import visualize_tcp_chunk_trajectory

try:
    from isaacsim.util.debug_draw import _debug_draw
    DEBUG_DRAW = _debug_draw.acquire_debug_draw_interface()
except Exception:
    DEBUG_DRAW = None


# ============================================================
# 常量：状态向量各字段的切片（与 record_lerobot.py 保持一致）
# observation.state / next_state 是 21 维向量，拼接顺序如下：
#   tcp_pos(3) | tcp_quat(4) | obj_pos(3) | obj_quat(4) | goal_pos(3) | goal_quat(4)
# ============================================================
STATE_DIM     = 21
S_TCP_POS     = slice(0,  3)   # TCP 世界位置 (x, y, z)
S_TCP_QUAT    = slice(3,  7)   # TCP 朝向四元数 (w, x, y, z)
S_OBJ_POS     = slice(7,  10)  # t_block 位置
S_OBJ_QUAT    = slice(10, 14)  # t_block 朝向
S_GOAL_POS    = slice(14, 17)  # goal_tee 位置
S_GOAL_QUAT   = slice(17, 21)  # goal_tee 朝向

# action 维度：tcp_pos(3) + tcp_quat(4) + gripper(1) = 8
ACTION_DIM    = 8
A_TCP_POS     = slice(0, 3)
A_TCP_QUAT    = slice(3, 7)
A_GRIPPER     = slice(7, 8)

# 可视化：预览前向轨迹的帧数
VIZ_HORIZON   = 16

# OpenCV 窗口名称
CV_WINDOW     = "Camera Replay (Left→Right)"


# ============================================================
# 辅助函数
# ============================================================

def make_env() -> gym.Env:
    """构造与录制时一致的单环境 PushT 任务。"""
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    # 回放时关闭内置超时，由脚本自己控制生命周期
    env_cfg.terminations.time_out = None
    return gym.make(args_cli.task, cfg=env_cfg).unwrapped


def load_dataset() -> LeRobotDataset:
    """加载只读的 LeRobot 数据集。"""
    root = (
        Path(args_cli.root)
        if args_cli.root is not None
        else PROJECT_ROOT / "data" / args_cli.repo_id
    )
    if not root.exists():
        print(f"[Replay] 数据集根目录不存在：{root}")
        simulation_app.close()
        sys.exit(1)
    ds = LeRobotDataset(repo_id=args_cli.repo_id, root=root)
    print(f"[Replay] 数据集加载成功：{root}，共 {ds.num_episodes} 条 episode。")
    return ds


def read_state(env) -> np.ndarray:
    """从当前仿真状态读取 21 维观测向量（与采集时完全相同的拼接顺序）。"""
    tcp_pos  = env.scene["tfs"].data.target_pos_w[0, -1, :].cpu().numpy()
    tcp_quat = env.scene["tfs"].data.target_quat_w[0, -1, :].cpu().numpy()
    obj_pos  = env.scene["t_block"].data.root_pos_w[0].cpu().numpy()
    obj_quat = env.scene["t_block"].data.root_quat_w[0].cpu().numpy()
    goal_pos = env.scene["goal_tee"].data.root_pos_w[0].cpu().numpy()
    goal_quat= env.scene["goal_tee"].data.root_quat_w[0].cpu().numpy()
    return np.concatenate([tcp_pos, tcp_quat, obj_pos, obj_quat, goal_pos, goal_quat]).astype(np.float32)


def to_numpy_f32(x) -> np.ndarray:
    """将 Tensor 或 array-like 统一转换为 float32 ndarray。"""
    if not isinstance(x, np.ndarray):
        x = np.asarray(x, dtype=np.float32)
    return x.astype(np.float32).reshape(-1)


def reset_scene_to_state(env, state: np.ndarray, env_ids: torch.Tensor) -> None:
    """
    将仿真场景的 t_block 和 goal_tee 位姿写回到 state 中记录的值，
    并将机械臂关节重置为采集时的默认位形。

    注：TCP 位姿无法直接写回（只有关节位形可以直接写入），
        需要后续通过 tcp_align_steps 驱动 IK 收敛。
    """
    device = env.device

    # ---- t_block ----
    obj_pos  = torch.tensor(state[S_OBJ_POS],  dtype=torch.float32, device=device).unsqueeze(0)
    obj_quat = torch.tensor(state[S_OBJ_QUAT], dtype=torch.float32, device=device).unsqueeze(0)
    t_block = env.scene["t_block"]
    root_pose_obj = torch.cat([obj_pos, obj_quat], dim=-1)
    t_block.write_root_pose_to_sim(root_pose_obj, env_ids=env_ids)
    # 速度清零，避免上一次物理步的速度残留
    t_block.write_root_velocity_to_sim(
        torch.zeros(1, 6, device=device), env_ids=env_ids
    )

    # ---- goal_tee ----
    goal_pos  = torch.tensor(state[S_GOAL_POS],  dtype=torch.float32, device=device).unsqueeze(0)
    goal_quat = torch.tensor(state[S_GOAL_QUAT], dtype=torch.float32, device=device).unsqueeze(0)
    goal_tee = env.scene["goal_tee"]
    root_pose_goal = torch.cat([goal_pos, goal_quat], dim=-1)
    goal_tee.write_root_pose_to_sim(root_pose_goal, env_ids=env_ids)

    # ---- 机械臂关节（写回录制 reset 时的默认位形，速度清零）----
    robot = env.scene["robot"]
    joint_pos = torch.tensor(DEFAULT_JOINT_POSE, dtype=torch.float32, device=device).view(1, -1)
    joint_vel = torch.zeros_like(joint_pos)
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)
    robot.set_joint_velocity_target(joint_vel, env_ids=env_ids)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def align_tcp(env, state: np.ndarray, first_action: np.ndarray, steps: int) -> None:
    """
    用第一帧记录的 TCP 绝对位姿（位置 + 四元数 + 夹爪开合）连续驱动 IK 若干步，
    使末端执行器收敛至数据集的初始 TCP 状态。

    DifferentialIKController 每步从仿真关节位形出发做线性近似，
    多步迭代后 TCP 会收敛到目标位姿。
    """
    tcp_target = np.concatenate([state[S_TCP_POS], state[S_TCP_QUAT], first_action[A_GRIPPER]])
    action_t = (
        torch.from_numpy(tcp_target.astype(np.float32))
        .to(device=env.device)
        .unsqueeze(0)
        .repeat(env.num_envs, 1)
    )
    for _ in range(steps):
        env.step(action_t)
    print(f"[Replay] TCP 对齐完成，共驱动 {steps} 步。")


def show_cameras(frame: dict) -> bool:
    """
    将数据集中本帧的所有摄像头图像水平拼接后显示。
    返回 True 表示用户按下 'q' 请求退出。

    数据集中图像格式：Tensor (C, H, W)，值域 [0, 1]，RGB。
    """
    imgs_bgr = []
    for key in ("observation.front_wrist_camera_image", "observation.back_wrist_camera_image"):
        img = frame[key].detach().cpu().numpy()          # (C, H, W)，[0,1]
        img_u8 = (img * 255.0).astype(np.uint8)
        img_hwc = np.transpose(img_u8, (1, 2, 0))       # → (H, W, C) RGB
        imgs_bgr.append(cv2.cvtColor(img_hwc, cv2.COLOR_RGB2BGR))

    # 高度统一后水平拼接
    h_ref = imgs_bgr[0].shape[0]
    resized = []
    for img in imgs_bgr:
        if img.shape[0] != h_ref:
            scale = h_ref / img.shape[0]
            img = cv2.resize(img, (int(img.shape[1] * scale), h_ref))
        resized.append(img)
    combined = np.concatenate(resized, axis=1)
    cv2.imshow(CV_WINDOW, combined)

    return (cv2.waitKey(1) & 0xFF) == ord("q")


def check_divergence(
    env, next_state_gt: np.ndarray, step: int, ep_idx: int
) -> None:
    """
    将当前仿真状态与数据集记录的 next_state 比较。
    对 t_block（位置 + 朝向）和机械臂 TCP（位置 + 朝向）分别计算 L2，
    超过阈值时打印带组件分解的警告。
    """
    cur = read_state(env)
    diff = cur - next_state_gt
    total_l2 = float(np.linalg.norm(diff))

    if total_l2 <= args_cli.divergence_threshold:
        return

    # 按组件分解，方便定位发散来源
    tcp_pos_l2  = float(np.linalg.norm(diff[S_TCP_POS]))
    tcp_quat_l2 = float(np.linalg.norm(diff[S_TCP_QUAT]))
    obj_pos_l2  = float(np.linalg.norm(diff[S_OBJ_POS]))
    obj_quat_l2 = float(np.linalg.norm(diff[S_OBJ_QUAT]))
    goal_pos_l2 = float(np.linalg.norm(diff[S_GOAL_POS]))
    goal_quat_l2= float(np.linalg.norm(diff[S_GOAL_QUAT]))

    print(
        f"[⚠ Diverge] ep={ep_idx} step={step:4d} | "
        f"total_l2={total_l2:.5f}  "
        f"(阈值={args_cli.divergence_threshold})"
    )
    print(
        f"             TCP  pos={tcp_pos_l2:.5f}  quat={tcp_quat_l2:.5f}  |  "
        f"Block pos={obj_pos_l2:.5f}  quat={obj_quat_l2:.5f}  |  "
        f"Goal  pos={goal_pos_l2:.5f}  quat={goal_quat_l2:.5f}"
    )


# ============================================================
# 核心回放逻辑
# ============================================================

def replay_episode(env, dataset: LeRobotDataset, ep_idx: int) -> None:
    """
    回放数据集中指定 episode 的完整轨迹。

    流程：
      1. env.reset() 初始化环境
      2. 将 t_block / goal_tee 写回到数据集第一帧的位姿，机械臂写回默认关节
      3. 运行 tcp_align_steps 步使 TCP 收敛到初始位姿
      4. 逐帧执行 action → 显示相机 → 校验状态
    """
    episodes  = dataset.meta.episodes
    ep_info   = episodes[ep_idx]
    frame_ids = list(range(ep_info["dataset_from_index"], ep_info["dataset_to_index"]))
    n_frames  = len(frame_ids)
    print(f"\n[Replay] ▶ Episode {ep_idx}，共 {n_frames} 帧。")

    # 1. 重置环境（建立初始物理上下文）
    env.reset()
    env_ids = torch.tensor([0], dtype=torch.long, device=env.device)

    # 2. 读取第一帧，写回场景状态
    first_frame  = dataset[frame_ids[0]]
    init_state   = to_numpy_f32(first_frame["observation.state"])
    first_action = to_numpy_f32(first_frame["action"])

    if init_state.shape[0] != STATE_DIM:
        raise ValueError(f"state 维度 {init_state.shape[0]} ≠ {STATE_DIM}")
    if first_action.shape[0] != ACTION_DIM:
        raise ValueError(f"action 维度 {first_action.shape[0]} ≠ {ACTION_DIM}")

    reset_scene_to_state(env, init_state, env_ids)

    # 3. TCP 对齐（可选，设为 0 则跳过）
    if args_cli.tcp_align_steps > 0:
        align_tcp(env, init_state, first_action, args_cli.tcp_align_steps)

    # 4. 逐帧回放
    with torch.no_grad():
        for local_step, frame_id in enumerate(frame_ids):
            frame      = dataset[frame_id]
            action_np  = to_numpy_f32(frame["action"])
            next_state_gt = to_numpy_f32(frame["observation.next_state"])

            if action_np.shape[0] != ACTION_DIM:
                raise ValueError(f"帧 {frame_id} action 维度异常：{action_np.shape[0]}")

            # 显示摄像头画面（按 q 退出）
            if show_cameras(frame):
                print(f"[Replay] 用户按下 'q'，在 step={local_step} 中断。")
                break

            # 可选：调试绘制未来 TCP 轨迹
            if DEBUG_DRAW is not None:
                tcp_pos_cur = env.scene["tfs"].data.target_pos_w[0, -1, :].cpu().numpy()
                end_idx = min(local_step + VIZ_HORIZON, n_frames)
                future_actions = []
                for k in range(local_step, end_idx):
                    a = to_numpy_f32(dataset[frame_ids[k]]["action"])
                    future_actions.append(a)
                if future_actions:
                    action_seq = torch.from_numpy(
                        np.stack(future_actions, axis=0)
                    ).to(device=env.device, dtype=torch.float32)
                    visualize_tcp_chunk_trajectory(
                        current_tcp_pos=tcp_pos_cur,
                        action_seq=action_seq,
                        num_steps=action_seq.shape[0],
                        debug_draw=DEBUG_DRAW,
                        action_mode="absolute",
                        point_size=6,
                        clear_previous=True,
                    )

            # 执行动作
            action_t = (
                torch.from_numpy(action_np)
                .to(device=env.device, dtype=torch.float32)
                .unsqueeze(0)
                .repeat(env.num_envs, 1)
            )
            _, _, terminated, truncated, _ = env.step(action_t)

            # 校验：当前状态 vs 数据集 next_state
            check_divergence(env, next_state_gt, local_step, ep_idx)

            # 环境提前终止则退出（忽略 terminated，不停止；遇到 truncated 才退）
            if truncated.any():
                print(f"[Replay] 环境在 step={local_step} truncated，结束本 episode。")
                break

    print(f"[Replay] ✓ Episode {ep_idx} 回放完成。")


# ============================================================
# 入口
# ============================================================

def main() -> None:
    dataset = load_dataset()
    ep_idx  = args_cli.episode_idx

    if ep_idx < 0 or ep_idx >= dataset.num_episodes:
        print(
            f"[Replay] episode_idx={ep_idx} 超出范围 "
            f"[0, {dataset.num_episodes - 1}]，退出。"
        )
        simulation_app.close()
        sys.exit(1)

    env = make_env()
    try:
        replay_episode(env, dataset, ep_idx)
    except KeyboardInterrupt:
        print("[Replay] 被用户中断（Ctrl+C）。")
    finally:
        env.close()
        simulation_app.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
