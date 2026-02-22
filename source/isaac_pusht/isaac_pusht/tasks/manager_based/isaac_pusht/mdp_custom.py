import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

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