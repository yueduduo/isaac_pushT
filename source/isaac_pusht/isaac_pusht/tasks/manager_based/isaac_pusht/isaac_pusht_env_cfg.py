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
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import RewardTermCfg, TerminationTermCfg, SceneEntityCfg
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
from mdp_custom import *

from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.devices.device_base import DeviceBase, DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.openxr.openxr_device import OpenXRDeviceCfg
from isaaclab.devices.openxr.retargeters.manipulator.gripper_retargeter import GripperRetargeterCfg
from isaaclab.devices.openxr.retargeters.manipulator.se3_rel_retargeter import Se3RelRetargeterCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip 
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

##
# Scene definition
##
@configclass
class IsaacPushtSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot") # use the high stiffness version of the panda config to make the pushing more stable

    push_stick = AssetBaseCfg(
        # 把圆柱体生成为机械臂末端连杆（panda_hand）的子节点，物理引擎会自动把它当成机械臂末端的一部分，带着它一起运动，并且一起计算碰撞！
        # 路径写在 panda_hand 之下，它就天然变成了手的一部分
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/stick_geometry",
        spawn=sim_utils.CylinderCfg(
            radius=0.009,       # 圆柱的半径 
            height=0.25,       # 圆柱的长度
            axis="Z",          # 圆柱朝向 Z 轴
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.9)), # 白色推杆
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True # 必须开启碰撞，否则推不动方块
            ),
            
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            # 注意：这是相对于 panda_hand 的【局部坐标偏移】
            # 因为夹爪本身有一定长度，我们需要顺着 Z 轴往下移一点，让它伸出来
            pos=(0.0, 0.0, 0.10), 
        ),
    )
    tcp_ball = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/stick_geometry/tcp_ball",
        spawn=sim_utils.SphereCfg(
            radius=0.015,       
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1)), 
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True 
            ),
            
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            # 因为圆柱本身有一定长度，我们需要顺着 Z 轴往下移一点，让它伸出来
            pos=(0.0, 0.0, 0.125), 
        ),
    )

    t_block: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TBlock",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{BASE_DIR}/assets/t_block.usd",
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.12, 0.05)),
    )
    
    goal_tee: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/GoalTee",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{BASE_DIR}/assets/goal_tee.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False)
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.42, -0.15, 0.001)),
    )


##
# MDP settings
##

@configclass
class ObservationsCfg:
    """定义策略网络的观测输入"""
    @configclass
    class PolicyCfg(ObsGroup):
        # T 块位姿
        obj_pose = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("t_block")})
        # 目标位姿
        goal_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("goal_tee")})

        # 机械臂tcp末端位姿 
        tcp_pose = ObsTerm(
            func=mdp_custom.tcp_pose_w, 
            params={"sensor_cfg": SceneEntityCfg("tfs")}
        )

        actions = ObsTerm(func=mdp.last_action)
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg:
    """定义动作空间：由于是 PushT，使用末端执行器 (EE) 空间控制最合适"""
    # Set actions for the specific robot franka 

    '''
    动作空间保持控制 panda_hand 即可。因为推杆是被死死固定在 panda_hand 上的。
    * 对于平移 (X, Y, Z)：当网络想要让 TCP 向左移动 1 厘米时，它只要输出让 panda_hand 向左移动 1 厘米的指令，TCP 就会完美地跟着向左移动 1 厘米（完全等效）。
    * 策略网络的自适应：只要 ObservationsCfg 里喂给神经网络的是 TCP 的真实坐标（而不是手腕坐标），PPO 算法会自然而然地把末端推杆当成自己的“手”，学习出完美的映射。这是强化学习最擅长解决的问题
    '''
    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=0.5,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
    )

    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.0}, # not 0.04
        close_command_expr={"panda_finger_.*": 0.0},
    )


@configclass
class RewardsCfg:
    """组合我们在 mdp_custom 中写的奖励函数"""
    tcp_prox_reward = RewTerm(
        func=mdp_custom.tcp_proximity_reward,
        weight=1.0,
        params={
            "tcp_cfg": SceneEntityCfg("tfs"), # 指向tf传感器！
            "tee_cfg": SceneEntityCfg("t_block")
        }
    )

    dist_reward = RewTerm(
        func=mdp_custom.tee_distance_reward,
        weight=1.0,
        params={"tee_cfg": SceneEntityCfg("t_block"), "goal_cfg": SceneEntityCfg("goal_tee")}
    )
     

@configclass
class EventCfg:
    """回合初始化与重置时的随机化逻辑"""

    init_franka_arm_pose = EventTerm(
        func=mdp_custom.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": [0.0444, -0.1894, -0.1107, -2.5148, 0.0044, 2.3775, 0.6952, 0., 0.],
        },
    )
    
    randomize_franka_joint_state = EventTerm(
        func=mdp_custom.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    
    randomize_t_block = EventTerm(
        func=mdp.reset_root_state_uniform, 
        mode="reset", 
        params={
            "asset_cfg": SceneEntityCfg("t_block"),
            "pose_range": {
                "x": (-0.1, 0.1), 
                "y": (-0.1, 0.2), 
                "yaw": (0.0, 2 * math.pi),
                # 其他坐标轴（比如 z, roll, pitch）将默认保持初始状态
            }, 
            "velocity_range": {
                # 空字典表示不随机化速度，直接清零
            }, 
        },
    )



@configclass
class TerminationsCfg:
    """回合终止条件"""
    # 超时
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    t_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("t_block")}
    )
    
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
    scene: IsaacPushtSceneCfg = IsaacPushtSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardTermCfg = RewardsCfg()
    terminations: TerminationTermCfg = TerminationsCfg()
    xr: XrCfg = XrCfg(
        anchor_pos=(-0.1, -0.5, -1.05),
        anchor_rot=(0.866, 0, 0, -0.5),
    )

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 100
        self.sim.render_interval = self.decimation

        # [vis] create adjusted frame transformer config for visualizing the end-effector pose in the scene
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.tfs = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                # FrameTransformerCfg.FrameCfg(
                #     prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                #     name="end_effector",
                #     offset=OffsetCfg(
                #         pos=[0.0, 0.0, 0.1034],
                #     ),
                # ),
                # FrameTransformerCfg.FrameCfg(
                #     prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                #     name="tool_rightfinger",
                #     offset=OffsetCfg(
                #         pos=(0.0, 0.0, 0.046),
                #     ),
                # ),
                # FrameTransformerCfg.FrameCfg(
                #     prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                #     name="tool_leftfinger",
                #     offset=OffsetCfg(
                #         pos=(0.0, 0.0, 0.046),
                #     ),
                # ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand", # 挂载在 panda_hand 上
                    name="tcp",
                    offset=OffsetCfg(
                        # 核心：根据你推杆的长度进行偏移！
                        # stick 中心在 Z=0.10，总长 0.25，所以尖端在 0.10 + 0.125 = 0.225
                        pos=(0.0, 0.0, 0.225), 
                    ),
                ),
            ],
        )
        
        # [teleop] set up teleoperation devices and retargeters
        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        Se3RelRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT,
                            zero_out_xy_rotation=True,
                            use_wrist_rotation=False,
                            use_wrist_position=True,
                            delta_pos_scale_factor=10.0,
                            delta_rot_scale_factor=10.0,
                            sim_device=self.sim.device,
                        ),
                        GripperRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT, sim_device=self.sim.device
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
            }
        )