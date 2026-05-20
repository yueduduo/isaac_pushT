"""
在独立进程 / 另一张 GPU 上加载 PushT 扩散策略，通过 WebSocket 提供 infer 服务。

仿真机仅跑 Isaac，本机或其它机器运行：

  python scripts/serve_push_diffusion_ws.py --checkpoint checkpoints/best_diffusion.pt --device cuda:0

仿真侧使用 eval_model.py / record_model_replay.py 的 --policy-host / --policy-port。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LERO_SRC = PROJECT_ROOT / "lerobot" / "src"
if str(_LERO_SRC) not in sys.path:
    sys.path.insert(0, str(_LERO_SRC))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.lerobot_push_diffusion import load_trainer_from_checkpoint
from utils.lerobot_processors import load_processor_bundle_for_checkpoint
from utils.remote_policy.push_diffusion_ws_policy import (
    PushDiffusionWebsocketServicer,
    build_server_metadata,
)
from utils.remote_policy.websocket_server import WebsocketPolicyServer


def main() -> None:
    parser = argparse.ArgumentParser(description="WebSocket 推理服务（LeRobot 扩散 PushT）。")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_diffusion.pt",
        help="与训练一致的 .pt 权重。",
    )
    parser.add_argument("--repo-id", type=str, default="isaac_pusht")
    parser.add_argument("--root", type=str, default="data/isaac_pusht")
    parser.add_argument("--horizon", type=int, default=32, help="与 checkpoint 内 horizon 一致。")
    parser.add_argument(
        "--action-dim-per-step",
        type=int,
        default=8,
        help="单步动作维度（TCP 绝对位姿 + gripper）。",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址。")
    parser.add_argument("--port", type=int, default=8765, help="监听端口。")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="推理设备，如 cuda、cuda:1、cpu。",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path
    if not ckpt_path.exists():
        raise SystemExit(f"未找到 checkpoint: {ckpt_path}")

    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            raise SystemExit("请求了 CUDA 设备但 torch.cuda.is_available() 为 False，请改用 --device cpu。")
        device = torch.device(args.device)

    logging.info("加载权重: %s device=%s", ckpt_path, device)
    trainer_probe, _, _ = load_trainer_from_checkpoint(
        ckpt_path,
        device=device,
        lr=1e-4,
        grad_clip_norm=1.0,
    )
    processors = load_processor_bundle_for_checkpoint(
        ckpt_path,
        trainer_probe.config,
        repo_id=args.repo_id,
        root=args.root,
        device=device,
    )
    trainer, _, _ = load_trainer_from_checkpoint(
        ckpt_path,
        device=device,
        lr=1e-4,
        grad_clip_norm=1.0,
        processors=processors,
    )

    servicer = PushDiffusionWebsocketServicer(
        trainer,
        device=device,
        horizon=args.horizon,
        action_dim_per_step=args.action_dim_per_step,
    )
    meta = build_server_metadata(
        horizon=args.horizon,
        action_dim_per_step=args.action_dim_per_step,
        checkpoint=ckpt_path,
    )
    server = WebsocketPolicyServer(servicer, host=args.host, port=args.port, metadata=meta)
    logging.info("策略服务已启动 ws://%s:%s （GET /healthz 健康检查）", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
