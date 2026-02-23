import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.assets import Articulation, AssetBase
import isaaclab.utils.math as math_utils
import isaaclab.envs.mdp as mdp


# obs func 

def ee_frame_pos(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    from isaaclab.sensors import FrameTransformer
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos = ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins[:, 0:3]

    return ee_frame_pos
def ee_frame_quat(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    from isaaclab.sensors import FrameTransformer
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_quat = ee_frame.data.target_quat_w[:, 0, :]

    return ee_frame_quat


# init func
def set_default_joint_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    default_pose: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    # Set the default pose for robots in all envs
    asset = env.scene[asset_cfg.name]
    asset.data.default_joint_pos = torch.tensor(default_pose, device=env.device).repeat(env.num_envs, 1)

def randomize_joint_by_gaussian_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    mean: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: Articulation = env.scene[asset_cfg.name]

    # Add gaussian noise to joint states
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    joint_pos += math_utils.sample_gaussian(mean, std, joint_pos.shape, joint_pos.device)

    # Clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

    # Don't noise the gripper poses
    joint_pos[:, -2:] = asset.data.default_joint_pos[env_ids, -2:]

    # Set into the physics simulation
    asset.set_joint_position_target(joint_pos, env_ids=env_ids)
    asset.set_joint_velocity_target(joint_vel, env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

def tee_distance_reward(env: ManagerBasedRLEnv, tee_cfg: SceneEntityCfg, goal_cfg: SceneEntityCfg) -> torch.Tensor:
    tee_pos = env.scene[tee_cfg.name].data.root_pos_w
    goal_pos = env.scene[goal_cfg.name].data.root_pos_w
    dist = torch.linalg.norm(tee_pos[:, 0:2] - goal_pos[:, 0:2], dim=1)
    return ((1 - torch.tanh(5 * dist)) ** 2) / 2

def tcp_proximity_reward(env: ManagerBasedRLEnv, tcp_cfg: SceneEntityCfg, tee_cfg: SceneEntityCfg) -> torch.Tensor:
    tcp_pos = env.scene[tcp_cfg.name].data.root_pos_w
    tee_pos = env.scene[tee_cfg.name].data.root_pos_w
    dist = torch.linalg.norm(tee_pos - tcp_pos, dim=1)
    return ((1 - torch.tanh(5 * dist)).sqrt()) / 20



import torch

def quat_to_z_euler(quats: torch.Tensor) -> torch.Tensor:
    """
    从四元数中极速提取绕 Z 轴的旋转角 (Yaw)。
    
    利用了 PushT 任务仅在 2D 桌面 (XY平面) 旋转的物理先验。
    纯 Z 轴旋转的四元数定义为: q = [cos(alpha/2), 0, 0, sin(alpha/2)]
    其中 w = cos(alpha/2)，推导出角度 alpha = 2 * arccos(w)。
    """
    assert len(quats.shape) == 2 and quats.shape[-1] == 4
    
    # 提取 z 分量 (quats[:, 3]) 的符号，用于修复四元数的“双倍覆盖(Double Cover)”问题。
    # 因为在三维空间中，q 和 -q 代表完全相同的旋转姿态。
    # 统一利用 z 分量的符号来约束 w，可以保证算出的角度连续且方向正确。
    signs = torch.ones_like(quats[:, -1])
    signs[quats[:, -1] < 0] = -1.0
    
    # 取出 w 分量 (quats[:, 0])，并用 z 的符号进行修正
    qw = quats[:, 0] * signs  
    
    # 通过反余弦计算出实际的 Z 轴旋转角度 (弧度)
    z_euler = 2 * qw.acos()
    return z_euler

def quat_to_zrot(quats: torch.Tensor) -> torch.Tensor:
    """
    将四元数直接转换为标准的 Z 轴 3x3 旋转矩阵。
    
    构建的线性代数 Z 轴旋转矩阵 R_z(alpha) 如下：
    [ cos(alpha), -sin(alpha),  0 ]
    [ sin(alpha),  cos(alpha),  0 ]
    [          0,           0,  1 ]
    
    这种纯矩阵构造法避开了通用 3D 转换库(如 scipy)内部复杂的万向节死锁计算，
    专为 GPU 并行张量运算量身定制，速度极快。
    """
    # 1. 获取 Z 轴旋转角 (弧度)
    alphas = quat_to_z_euler(quats)
    
    # 2. 初始化一个全为 0 的 3x3 矩阵，形状为 (batch_size, 3, 3)
    rot_mats = torch.zeros(quats.shape[0], 3, 3, device=quats.device)
    
    # 3. 按照 Z 轴旋转矩阵的数学定义填充元素
    rot_mats[:, 2, 2] = 1.0            # 右下角，Z轴高度不发生变化
    rot_mats[:, 0, 0] = alphas.cos()   # 第一行第一列: cos(alpha)
    rot_mats[:, 1, 1] = alphas.cos()   # 第二行第二列: cos(alpha)
    rot_mats[:, 0, 1] = -alphas.sin()  # 第一行第二列: -sin(alpha)
    rot_mats[:, 1, 0] = alphas.sin()   # 第二行第一列: sin(alpha)
    
    return rot_mats

def success_termination(env: ManagerBasedRLEnv, tee_cfg: SceneEntityCfg, goal_cfg: SceneEntityCfg, intersection_thresh: float = 0.90) -> torch.Tensor:
    device = env.device
    b = env.num_envs

    # ---------------------------------------------------------
    # 1. 懒加载：仅初始化 T 块的基础网格模型（不含环境位置信息）
    # ---------------------------------------------------------
    if not hasattr(env, "_pusht_static_initialized"):
        res = 64
        uv_half_width = 0.15
        
        oned_grid = torch.arange(res, dtype=torch.float32, device=device).view(1, res).repeat(res, 1) - (res / 2)
        uv_grid = (torch.cat([oned_grid.unsqueeze(0), (-1 * oned_grid.T).unsqueeze(0)], dim=0) + 0.5) / ((res / 2) / uv_half_width)
        homo_uv = torch.cat([uv_grid, torch.ones_like(uv_grid[0]).unsqueeze(0)], dim=0)
        
        center_of_mass = (0, 0.0375)
        box1 = torch.tensor([[-0.1, 0.025], [0.1, 0.025], [-0.1, -0.025], [0.1, -0.025]], device=device)
        box2 = torch.tensor([[-0.025, 0.175], [0.025, 0.175], [-0.025, 0.025], [0.025, 0.025]], device=device)
        box1[:, 1] -= center_of_mass[1]
        box2[:, 1] -= center_of_mass[1]
        
        box1 = (box1 * ((res / 2) / uv_half_width) + res / 2).long()
        box2 = (box2 * ((res / 2) / uv_half_width) + res / 2).long()
        
        tee_render = torch.zeros(res, res, device=device)
        tee_render.T[box1[0, 0] : box1[1, 0], box1[2, 1] : box1[0, 1]] = 1
        tee_render.T[box2[0, 0] : box2[1, 0], box2[2, 1] : box2[0, 1]] = 1
        tee_render = tee_render.flip(0)
        
        env._pusht_homo_uv = homo_uv
        env._pusht_tee_render = tee_render
        env._pusht_res = res
        env._pusht_uv_half_width = uv_half_width
        env._pusht_static_initialized = True

    # ---------------------------------------------------------
    # 2. 动态读取并构建变换矩阵
    # ---------------------------------------------------------
    # 动态获取 T 块和目标 T 块的实时位姿
    tee_pos = env.scene[tee_cfg.name].data.root_pos_w
    tee_quat = env.scene[tee_cfg.name].data.root_quat_w
    goal_pos = env.scene[goal_cfg.name].data.root_pos_w
    goal_quat = env.scene[goal_cfg.name].data.root_quat_w
    
    # 构建 T 块到世界的变换矩阵 T_{T->W}
    tee_to_world_trans = quat_to_zrot(tee_quat)
    tee_to_world_trans[:, 0:2, 2] = tee_pos[:, :2]
    
    # 构建目标到世界的变换矩阵 T_{G->W}
    goal_to_world_trans = quat_to_zrot(goal_quat)
    goal_to_world_trans[:, 0:2, 2] = goal_pos[:, :2]
    
    # 实时对目标矩阵求逆，得到 T_{W->G}
    world_to_goal_trans = torch.linalg.inv(goal_to_world_trans)
    
    # 最终变换矩阵 T_{T->G} = T_{W->G} @ T_{T->W}
    tee_to_goal_trans = world_to_goal_trans @ tee_to_world_trans
    
    # ---------------------------------------------------------
    # 3. 执行伪渲染计算
    # ---------------------------------------------------------
    tees_in_goal_frame = (tee_to_goal_trans @ env._pusht_homo_uv.view(3, -1)).view(b, 3, env._pusht_res, env._pusht_res)
    tees_in_goal_frame = tees_in_goal_frame[:, 0:2, :, :] / tees_in_goal_frame[:, -1, :, :].unsqueeze(1)
    
    tee_coords = tees_in_goal_frame[:, :, env._pusht_tee_render == 1].view(b, 2, -1)
    tee_indices = ((tee_coords * ((env._pusht_res / 2) / env._pusht_uv_half_width) + (env._pusht_res / 2)).long().view(b, 2, -1))
    
    final_renders = torch.zeros(b, env._pusht_res, env._pusht_res, device=device)
    num_tee_pixels = tee_indices.shape[-1]
    batch_indices = torch.arange(b, device=device).view(-1, 1).repeat(1, num_tee_pixels)
    
    invalid_xs = (tee_indices[:, 0, :] < 0) | (tee_indices[:, 0, :] >= env._pusht_res)
    invalid_ys = (tee_indices[:, 1, :] < 0) | (tee_indices[:, 1, :] >= env._pusht_res)
    tee_indices[:, 0, :][invalid_xs] = 0
    tee_indices[:, 1, :][invalid_xs] = 0
    tee_indices[:, 0, :][invalid_ys] = 0
    tee_indices[:, 1, :][invalid_ys] = 0
    
    final_renders[batch_indices, tee_indices[:, 0, :], tee_indices[:, 1, :]] = 1
    final_renders = final_renders.permute(0, 2, 1).flip(1)
    
    intersection = (final_renders.bool() & env._pusht_tee_render.bool()).sum(dim=[-1, -2]).float()
    goal_area = env._pusht_tee_render.bool().sum().float()
    
    reward = intersection / goal_area
    
    return reward >= intersection_thresh