"""TCP 轨迹可视化工具函数。

封装 Isaac Sim Debug Drawing Extension 的调用，方便在不同脚本中复用：
- 输入当前 TCP 位置 + 一段 action 序列
- 预测 TCP 将要到达的若干点
- 使用 debug_draw 接口画出绿色、随步数衰减的点
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def visualize_tcp_chunk_trajectory(
    current_tcp_pos: np.ndarray,
    action_seq: torch.Tensor,
    num_steps: int,
    debug_draw,
    action_mode: str = "absolute",
    action_to_tcp_scale: Optional[float] = None,
    point_size: int = 10,
    clear_previous: bool = True,
) -> None:
    """可视化一段 action 序列对应的 TCP 未来轨迹。

    参数:
        current_tcp_pos: (3,) numpy 数组，当前 TCP 世界坐标位置。
        action_seq: (H, A) torch 张量，包含本次 chunk 的动作序列，
                    - absolute 模式: action_seq[i, :3] 为 TCP 绝对位置
                    - delta 模式: action_seq[i, :3] 为 TCP 位置增量
        num_steps: 实际要可视化的步数 (通常 <= H)。
        debug_draw: 通过 `isaacsim.util.debug_draw._debug_draw.acquire_debug_draw_interface()`
                    获得的绘图接口。
        action_mode: "absolute" 或 "delta"。
        action_to_tcp_scale: 可选，action[:3] 映射到真实 TCP 位移的缩放系数。
                             仅在 delta 模式生效；若为 None，则直接使用原始 delta。
        point_size: 绘制点的像素大小。
        clear_previous: 是否在绘制前清除之前画过的点，避免跨 chunk 累积。

    返回:
        无返回值，仅在仿真中绘制点。
    """
    if debug_draw is None or num_steps <= 0:
        return

    # 先清除上一次绘制的点，使每个 chunk 的可视化不累积
    if clear_previous:
        try:
            if hasattr(debug_draw, "clear_points"):
                debug_draw.clear_points()
            elif hasattr(debug_draw, "clear"):
                debug_draw.clear()
        except Exception:
            # 清理失败不影响后续绘制
            pass

    tcp_traj_points = []
    cur_pos = current_tcp_pos.astype(np.float32).copy()
    steps = min(num_steps, action_seq.shape[0])

    scale = 1.0 if action_to_tcp_scale is None else float(action_to_tcp_scale)

    for i in range(steps):
        action_pos = action_seq[i, :3].detach().cpu().numpy().astype(np.float32)
        if action_mode == "absolute":
            cur_pos = action_pos
        elif action_mode == "delta":
            cur_pos = cur_pos + scale * action_pos
        else:
            raise ValueError(f"Unsupported action_mode: {action_mode}. Expected 'absolute' or 'delta'.")
        tcp_traj_points.append((float(cur_pos[0]), float(cur_pos[1]), float(cur_pos[2])))

    if not tcp_traj_points:
        return

    # 颜色: 绿色为主色调，随时间步 alpha 衰减
    n_pts = len(tcp_traj_points)
    colors = []
    sizes = []
    for idx in range(n_pts):
        if n_pts > 1:
            t = idx / (n_pts - 1)
        else:
            t = 0.0
        alpha = max(0.2, 1.0 - 0.8 * t)
        colors.append((0.0, 1.0, 0.0, float(alpha)))
        sizes.append(point_size)

    try:
        debug_draw.draw_points(tcp_traj_points, colors, sizes)
    except Exception:
        # 绘制失败不应打断主逻辑
        pass

