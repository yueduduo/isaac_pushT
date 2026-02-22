# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
import math
import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg, SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
import isaaclab.envs.mdp as mdp
import sys
import os
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
import mdp_custom 
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG


##
# Pre-defined configs
##



##
# Scene definition
##


@configclass
class IsaacPushtSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    t_block: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TBlock",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{BASE_DIR}/assets/t_block.usd",
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.05)),
    )
    
    goal_tee: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/GoalTee",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{BASE_DIR}/assets/goal_tee.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False)
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.244, -0.1, 0.001)),
    )


##
# MDP settings
##

@configclass
class ActionsCfg:
    """定义动作空间：由于是 PushT，使用末端执行器 (EE) 空间控制最合适"""
    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
    )

@configclass
class ObservationsCfg:
    """定义策略网络的观测输入"""
    @configclass
    class PolicyCfg(ObsGroup):
        # 机械臂末端位姿
        tcp_pose = ObsTerm(func=mdp.body_pose_w, params={"asset_cfg": SceneEntityCfg("robot", body_names="panda_hand")})
        # T 块位姿
        obj_pose = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("t_block")})
        # 目标位姿
        goal_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("goal_tee")})
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """回合初始化与重置时的随机化逻辑"""
    
    randomize_t_block = EventTerm(
        func=mdp.reset_root_state_uniform, # 换成这个正确的官方 API
        mode="reset", # 在每次环境 reset 时触发
        params={
            "asset_cfg": SceneEntityCfg("t_block"),
            
            # 1. 随机化位姿范围 (对应你的 spawnbox 逻辑)
            # 没写进来的坐标轴（比如 z, roll, pitch）将默认保持初始状态 (z=0.05悬空, 无倾斜)
            "pose_range": {
                "x": (-0.1, 0.1), 
                "y": (-0.1, 0.2), 
                "yaw": (0.0, 2 * math.pi) # Z轴随机旋转 [0, 2pi]
            },
            
            # 2. 随机化速度范围
            # 给空字典意味着在 reset 的时候，物体的线速度和角速度全部强制清零，防止 T 块乱飞
            "velocity_range": {}, 
        },
    )

@configclass
class RewardsCfg:
    """组合我们在 mdp_custom 中写的奖励函数"""
    dist_reward = RewTerm(
        func=mdp_custom.tee_distance_reward,
        weight=1.0,
        params={"tee_cfg": SceneEntityCfg("t_block"), "goal_cfg": SceneEntityCfg("goal_tee")}
    )
    
    tcp_prox_reward = RewTerm(
        func=mdp_custom.tcp_proximity_reward,
        weight=1.0,
        params={"tcp_cfg": SceneEntityCfg("robot", body_names="panda_hand"), "tee_cfg": SceneEntityCfg("t_block")}
    )

@configclass
class TerminationsCfg:
    """回合终止条件"""
    # 超时
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # 成功推入目标区域
    success = DoneTerm(
        func=mdp_custom.success_termination,
        params={
            "tee_cfg": SceneEntityCfg("t_block"), 
            "goal_cfg": SceneEntityCfg("goal_tee"),
        }
    )

##
# Environment configuration
##


@configclass
class IsaacPushtEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: IsaacPushtSceneCfg = IsaacPushtSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardTermCfg = RewardsCfg()
    terminations: TerminationTermCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation