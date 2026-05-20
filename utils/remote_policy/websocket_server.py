"""WebSocket 策略服务（与根目录 websocket_policy_server.py / openpi 流程一致，使用本仓库 msgpack）。

依赖：websockets>=13（提供 websockets.asyncio.server；更早版本无 websockets.asyncio）。
"""

from __future__ import annotations

import asyncio
import http
import logging
import time
import traceback
from typing import Any, Callable, Protocol

import websockets.asyncio.server as _server
import websockets.frames

from utils.remote_policy.msgpack_numpy import Packer, unpackb

logger = logging.getLogger(__name__)


class ObsToActionPolicy(Protocol):
    """服务端持有的策略：obs / action 均为可 msgpack 的 dict（含 numpy 数组）。"""

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]: ...


class WebsocketPolicyServer:
    """启动后先推送 metadata，随后循环 recv(obs) -> infer -> send(action_dict)。"""

    def __init__(
        self,
        policy: ObsToActionPolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection) -> None:
        logger.info("Connection from %s opened", websocket.remote_address)
        packer = Packer()
        await websocket.send(packer.pack(self._metadata))

        prev_total_time: float | None = None
        while True:
            try:
                start_time = time.monotonic()
                obs = unpackb(await websocket.recv())

                infer_t0 = time.monotonic()
                action = self._policy.infer(obs)
                infer_ms = (time.monotonic() - infer_t0) * 1000

                action["server_timing"] = {
                    "infer_ms": infer_ms,
                }
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(
    connection: _server.ServerConnection, request: _server.Request
) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None
