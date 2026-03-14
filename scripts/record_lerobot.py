

"""脚本：使用 LeRobot 格式在 Isaac Lab 环境中收集数据。
优化：引入鼠标绝对位置映射遥操，Esc 退出，优化保存逻辑。
"""

import argparse
import os
import sys
import torch
import numpy as np
import gymnasium as gym
import traceback
from pathlib import Path
from collections.abc import Callable

# 1. 启动 Isaac Lab 环境
from isaaclab.app import AppLauncher

# 添加命令行参数
parser = argparse.ArgumentParser(description="Record LeRobot dataset for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--teleop_device", type=str, default="keyboard", help="Teleop device.")
parser.add_argument("--task", type=str, default="Isaac-Pusht-v0", help="Name of the task.")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to record.")
parser.add_argument("--output", type=str, default=None, help="Output directory.")
parser.add_argument("--repo_id", type=str, default="isaac_pusht", help="Repo ID for the dataset.")
parser.add_argument("--resume", action="store_true", default=False, help="Resume recording from existing dataset.")

# 增加 AppLauncher 的参数并解析
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动 Isaac Sim
app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app
import omni.ui as ui  
import carb
import omni.appwindow
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
from isaaclab_tasks.utils import parse_env_cfg
import isaaclab_tasks  # noqa: F401
import isaac_pusht.tasks  # noqa: F401

# 2. 导入 LeRobot 相关库 (在仿真器启动后导入)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import combine_feature_dicts


# ==========================================
# 0. 全局配置 (Constants & Dataset Format)
# ==========================================
FPS = 25                 # 数据采集频率 (Isaac Lab decimation=2, sim dt=1/50 => 10Hz)
ROBOT_TYPE = "franka"
RENDER_WIDTH = 224
RENDER_HEIGHT = 224
STATE_DIM = 21           # tcp_pos(3)+quat(4) + obj_pos(3)+quat(4) + goal_pos(3)+quat(4)
ACTION_DIM = 7           # pos_delta(3), rot_delta(3), gripper_action(1)
TASK_DESCRIPTION = "Push the T block into the target area."

DEFAULT_REPO_ID = args_cli.repo_id
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "data" / DEFAULT_REPO_ID

# 工作区物理尺寸映射 (Workspace: size=(0.4, 0.5), pos=(0.5, 0.0))
# X: [0.3, 0.7], Y: [-0.25, 0.25]

# ==========================================
# 1. 数据集特征定义
# ==========================================
OBS_FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (STATE_DIM,),
    },
    "observation.front_wrist_camera_image": {
        "dtype": "image",
        "shape": (3, RENDER_HEIGHT, RENDER_WIDTH),
        "names": ["color"],
    },
    "observation.back_wrist_camera_image": {
        "dtype": "image",
        "shape": (3, RENDER_HEIGHT, RENDER_WIDTH),
        "names": ["color"],
    },
}

ACTION_FEATURES = {
    "action": {
        "dtype": "float32",
        "shape": (ACTION_DIM,),
    }
}

DATASET_FEATURES = combine_feature_dicts(OBS_FEATURES, ACTION_FEATURES)

class IsaacUITeleop:
    """自适应 UI 遥操：支持拖拽机械臂末端投影，实时闭环反馈"""
    def __init__(self):
        self.is_recording = False
        self.should_reset = False
        self.should_quit = False
        self.is_dragging = False
        
        self.kb = Se3Keyboard(Se3KeyboardCfg())
        self.kb.reset()
        
        # 降低增益，增加稳定性
        self.ik_scale_inv = 5
        self.max_delta_m = 0.7
        
        self.curr_u = 0.5
        self.curr_v = 0.5
        self.mouse_u = 0.5
        self.mouse_v = 0.5
        self._drag_last_x = None
        self._drag_last_y = None
        
        # 创建 UI 窗口
        self._window = ui.Window("PushT Control Pad", width=400, height=500)
        self._build_ui()
        
        # 注册键盘监听
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input_interface = carb.input.acquire_input_interface()
        self._keyboard_sub = self._input_interface.subscribe_to_keyboard_events(
            self._appwindow.get_keyboard(), self._on_keyboard_event
        )

    def _build_ui(self):
        with self._window.frame:
            with ui.VStack(spacing=10, padding=10):
                ui.Label("PushT Teleop Console", style={"font_size": 18, "color": 0xFF00BFFF})
                
                with ui.HStack(height=25):
                    ui.Label("Recording:", width=80)
                    self.status_label = ui.Label("IDLE", style={"color": 0xFFAAAAAA})
                
                ui.Label("Workspace (Drag the RED dot)", style={"font_size": 12, "color": 0xFF888888})
                
                # 交互区域
                self.trackpad_frame = ui.Frame(height=300)
                with self.trackpad_frame:
                    with ui.ZStack():
                        # 1. 背景层 (最底层)
                        ui.Rectangle(style={
                            "background_color": 0xFF222222, 
                            "border_color": 0xFF444444, 
                            "border_width": 2,
                            "border_radius": 5
                        })
                        
                        # 2. 全局事件捕获层 (在背景上方，红点下方)
                        self.touch_area = ui.Rectangle(style={"background_color": 0x00FFFFFF})
                        self.touch_area.set_mouse_moved_fn(self._on_pad_moved)
                        self.touch_area.set_mouse_released_fn(self._on_pad_released)
                        
                        # 3. 拖拽指示环层 (黄色)
                        self.ring_placer = ui.Placer()
                        with self.ring_placer:
                            self.drag_ring = ui.Rectangle(width=40, height=40, style={
                                "background_color": 0x00000000,
                                "border_color": 0xFF00FFFF, # ABGR: Yellow
                                "border_width": 2,
                                "border_radius": 20
                            })
                            self.drag_ring.visible = False
                        
                        # 4. 机械臂红点层 (最顶层)
                        self.marker_placer = ui.Placer()
                        with self.marker_placer:
                            # 增大红点尺寸 (30x30)，并直接在其上监听按下
                            self.marker_rect = ui.Rectangle(width=30, height=30, style={
                                "background_color": 0xFF0000FF, # ABGR: Red
                                "border_radius": 15,
                                "border_color": 0xFFFFFFFF,
                                "border_width": 2
                            })
                            # 直接点红点开始拖拽
                            self.marker_rect.set_mouse_pressed_fn(self._on_marker_pressed)
                            # 拖拽期间优先保证红点本身也会持续上报移动/释放事件
                            self.marker_rect.set_mouse_moved_fn(self._on_pad_moved)
                            self.marker_rect.set_mouse_released_fn(self._on_pad_released)

                with ui.HStack(spacing=5, height=40):
                    ui.Button("Toggle Record (L)", clicked_fn=self._toggle_record)
                    ui.Button("Reset Env (R)", clicked_fn=self._on_reset_btn)
                
                ui.Label("Shortcuts: WASD/QE: Move | L: Record | R: Reset | Esc: Quit", 
                         style={"color": 0xFF666666, "font_size": 11})

    def _on_marker_pressed(self, x, y, button, modifier):
        """只有点中红点时才触发拖拽"""
        if button == 0:
            self.is_dragging = True
            self.drag_ring.visible = True
            # 重要：立即同步鼠标坐标到当前红点坐标，防止跳变
            self.mouse_u = self.curr_u
            self.mouse_v = self.curr_v
            # 延迟到第一帧移动事件再建立同坐标系锚点，避免 press/move 坐标系不同导致跳变
            self._drag_last_x = None
            self._drag_last_y = None
            print(f"[Teleop] DRAG STARTED at : {self.curr_u:.2f}, {self.curr_v:.2f}")

    def _on_pad_moved(self, x, y, button, modifier):
        """鼠标在整个灰色区域内移动时触发"""
        if self.is_dragging:
            w = max(1, self.touch_area.computed_width)
            h = max(1, self.touch_area.computed_height)

            # 首帧只对齐锚点，不更新位置，避免点击时瞬间跳点
            if self._drag_last_x is None or self._drag_last_y is None:
                self._drag_last_x = x
                self._drag_last_y = y
                return

            # 使用相对位移积分，规避不同回调坐标系导致的绝对位置跳变
            dx_pix = x - self._drag_last_x
            dy_pix = y - self._drag_last_y
            self._drag_last_x = x
            self._drag_last_y = y

            # 防止不同控件坐标系切换时出现超大跳变
            if abs(dx_pix) > 0.5 * w or abs(dy_pix) > 0.5 * h:
                return

            self.mouse_u = max(0.0, min(1.0, self.mouse_u + dx_pix / w))
            self.mouse_v = max(0.0, min(1.0, self.mouse_v + dy_pix / h))

    def _on_pad_released(self, x, y, button, modifier):
        if button == 0 and self.is_dragging:
            self.is_dragging = False
            self.drag_ring.visible = False
            self._drag_last_x = None
            self._drag_last_y = None
            print("[Teleop] DRAG STOPPED")

    def _on_keyboard_event(self, event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            key = event.input
            if key == carb.input.KeyboardInput.ESCAPE: self.should_quit = True
            elif key == carb.input.KeyboardInput.L: self._toggle_record()
            elif key == carb.input.KeyboardInput.R: self._on_reset_btn()

    def _toggle_record(self):
        self.is_recording = not self.is_recording
        self.status_label.text = "RECORDING" if self.is_recording else "IDLE"
        self.status_label.style = {"color": 0xFF0000FF if self.is_recording else 0xFFAAAAAA}
        # 切换录制状态时清空瞬时输入，避免历史按键影响下一帧
        self.clear_transient_input()

    def _on_reset_btn(self):
        self.should_reset = True

    def clear_transient_input(self):
        """清空键盘与拖拽瞬时状态，避免重置后残留输入。"""
        self.kb.reset()
        self.is_dragging = False
        self.drag_ring.visible = False
        self._drag_last_x = None
        self._drag_last_y = None
        # 同步目标到当前真实位置，避免下一帧出现突变
        self.mouse_u = self.curr_u
        self.mouse_v = self.curr_v

    def get_action(self, current_tcp_pos):
        delta_action = self.kb.advance().cpu().numpy().astype(np.float32)
        
        # 实时计算机器人位置
        curr_x, curr_y = current_tcp_pos[0], current_tcp_pos[1]
        self.curr_u = max(0.0, min(1.0, (0.25 - curr_y) / 0.5))
        self.curr_v = max(0.0, min(1.0, (0.7 - curr_x) / 0.4))
        
        # 拖拽时黄圈和红点使用同一显示坐标（重合移动，不分离）
        disp_u = self.mouse_u if self.is_dragging else self.curr_u
        disp_v = self.mouse_v if self.is_dragging else self.curr_v

        # 更新 UI
        w = max(1, self.touch_area.computed_width)
        h = max(1, self.touch_area.computed_height)
        self.marker_placer.offset_x = ui.Pixel(max(0, min(w - 30, disp_u * w - 15)))
        self.marker_placer.offset_y = ui.Pixel(max(0, min(h - 30, disp_v * h - 15)))
        if self.is_dragging:
            self.ring_placer.offset_x = ui.Pixel(max(0, min(w - 40, disp_u * w - 20)))
            self.ring_placer.offset_y = ui.Pixel(max(0, min(h - 40, disp_v * h - 20)))

        # 核心：计算 Delta Action
        if self.is_dragging:
            target_y = 0.25 - self.mouse_u * 0.5
            target_x = 0.7 - self.mouse_v * 0.4
            
            # 计算物理位移差 (meters)
            dx = target_x - curr_x
            dy = target_y - curr_y
            
            # 提高单步最大位移，减小拖拽跟随滞后
            dx = max(-self.max_delta_m, min(self.max_delta_m, dx))
            dy = max(-self.max_delta_m, min(self.max_delta_m, dy))
            
            delta_action[0] += dx * self.ik_scale_inv
            delta_action[1] += dy * self.ik_scale_inv

        return torch.from_numpy(delta_action)




def main():
    # 创建 LeRobot 数据集
    output_path = Path(args_cli.output) if args_cli.output else DEFAULT_OUTPUT_DIR
    if args_cli.resume:
        if not output_path.exists():
            print(f"错误：指定了 --resume 但路径不存在: {output_path}")
            sys.exit(1)
        print(f"加载现有数据集: {output_path}")
        dataset = LeRobotDataset(repo_id=DEFAULT_REPO_ID, root=output_path)
        print(f"当前数据集已包含 {dataset.num_episodes} 条记录")
    else:
        dataset = LeRobotDataset.create(
            repo_id=DEFAULT_REPO_ID,
            root=output_path,
            fps=FPS,
            robot_type=ROBOT_TYPE,
            features=DATASET_FEATURES,
            use_videos=False,
        )

    # 创建环境配置
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.terminations.time_out = None # 【核心】禁用自动超时重置，由人工控制录制时长
    if hasattr(env_cfg.scene, "camera"):
        env_cfg.scene.camera.data_types = ["rgb"] # 强制开启相机 RGB 数据
    
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # 初始化重构的遥操接口
    teleop = IsaacUITeleop()
    
    episode_idx = 0
    episode_buffer = []
    
    print("-" * 50)
    print(f"Dataset Path: {output_path}")
    print(f"Controls:")
    print("  Mouse Position: Directly maps to TCP X/Y in workspace")
    print("  L: Toggle Recording | R: Reset Environment")
    print("  Esc: Save & Exit")
    print("-" * 50)
    
    env.reset()
    teleop.clear_transient_input()

    try:
        while simulation_app.is_running() and episode_idx < args_cli.num_episodes:
            if teleop.should_quit:
                break

            # 先处理人工重置：不执行本帧动作，避免 R 触发后多走一步
            if teleop.should_reset:
                episode_buffer = []
                teleop.should_reset = False
                teleop.clear_transient_input()
                env.reset()
                teleop.clear_transient_input()
                continue
                
            # 不要对环境 step 使用 inference_mode：
            # 否则环境内部状态张量会变成 inference tensor，后续 reset 的 inplace 写入会报错。
            with torch.no_grad():
                # 获取当前 TCP 位置用于闭环映射
                current_tcp = env.scene["tfs"].data.target_pos_w[0, -1, :].cpu().numpy()
                
                # 1. 获取动作 (基于绝对映射计算的 Delta)
                delta_action = teleop.get_action(current_tcp).to(env.device)
                
                # 检查是否有输入
                has_input = torch.any(torch.abs(delta_action[:6]) > 0.01)
                
                # 2. 步进
                obs, reward, terminated, truncated, info = env.step(delta_action.repeat(env.num_envs, 1))
                
                # 3. 记录
                if teleop.is_recording and has_input:
                    try:
                        # 获取相机图像
                        # 转换格式 (H,W,C) -> (C,H,W), [0,255] -> [0,1]
                        front_rgb_image = env.scene["front_wrirst_camera"].data.output["rgb"][0].cpu().numpy()
                        front_img_tensor = np.moveaxis(front_rgb_image, -1, 0).astype(np.float32) / 255.0
                    
                    
                        back_rgb_image = env.scene["back_wrirst_camera"].data.output["rgb"][0].cpu().numpy()
                        back_img_tensor = np.moveaxis(back_rgb_image, -1, 0).astype(np.float32) / 255.0
                        
                        # 提取状态信息 (TCP + Object + Goal)
                        # 使用 -1 索引获取最后一个目标帧 (tcp)
                        tcp_pos = current_tcp
                        tcp_quat = env.scene["tfs"].data.target_quat_w[0, -1, :].cpu().numpy()
                        obj_pos = env.scene["t_block"].data.root_pos_w[0].cpu().numpy()
                        obj_quat = env.scene["t_block"].data.root_quat_w[0].cpu().numpy()
                        goal_pos = env.scene["goal_tee"].data.root_pos_w[0].cpu().numpy()
                        goal_quat = env.scene["goal_tee"].data.root_quat_w[0].cpu().numpy()
                        
                        state = np.concatenate([tcp_pos, tcp_quat, obj_pos, obj_quat, goal_pos, goal_quat])
                        
                        episode_buffer.append({
                            "observation.front_wrist_camera_image": front_img_tensor,
                            "observation.back_wrist_camera_image": back_img_tensor,
                            "observation.state": state.astype(np.float32),
                            "action": delta_action.cpu().numpy().astype(np.float32),
                            "task": TASK_DESCRIPTION,
                        })
                    except Exception as e:
                        print(f"Error during data collection: {e}")
                        traceback.print_exc()
                        break

                # 4. 终止后保存并重置
                if terminated.any():
                    if teleop.is_recording and len(episode_buffer) > 0:
                        print(f"\n[Episode {episode_idx+1}] Saving...")
                        [dataset.add_frame(f) for f in episode_buffer]
                        dataset.save_episode()
                        episode_idx += 1
                        print(f"Done. Frames: {len(episode_buffer)}")
                    
                    episode_buffer = []
                    # 重新初始化环境和输入设备
                    teleop.should_reset = False
                    teleop.clear_transient_input()
                    env.reset()
                    teleop.clear_transient_input()
                    if episode_idx >= args_cli.num_episodes:
                        break

    except Exception as e:
        print(f"\n[CRASH] Unexpected error: {e}")
        traceback.print_exc()
    finally:
        env.close()
        if hasattr(dataset, 'finalize'):
            dataset.finalize()
        print(f"\nFinalized. Total episodes: {episode_idx}")
        simulation_app.close()


if __name__ == "__main__":
    main()
