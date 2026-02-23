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

def success_termination(env: ManagerBasedRLEnv, tee_cfg: SceneEntityCfg, goal_cfg: SceneEntityCfg) -> torch.Tensor:
    # 这里接入你写好的伪渲染交并比张量运算，目前用距离阈值做占位演示
    tee_pos = env.scene[tee_cfg.name].data.root_pos_w
    goal_pos = env.scene[goal_cfg.name].data.root_pos_w
    dist = torch.linalg.norm(tee_pos[:, 0:2] - goal_pos[:, 0:2], dim=1)
    return dist < 0.05