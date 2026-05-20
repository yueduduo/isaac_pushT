"""与 websocket_server 配对的同步 WebSocket 策略客户端（参考 pi0_fast_deploy/utils/websocket_client_policy.py）。"""

import logging
import time
from typing import Dict, Optional, Tuple

import websockets.sync.client

from utils.remote_policy.msgpack_numpy import Packer, unpackb


class WebsocketClientPolicy:
    """通过 WebSocket 调用远端 `infer(obs)`，观测与返回值为 numpy 字典。"""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: Optional[int] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info("Waiting for policy server at %s...", self._uri)
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers
                )
                metadata = unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    def infer(self, obs: Dict) -> Dict:
        self._ws.send(self._packer.pack(obs))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return unpackb(response)

    def close(self) -> None:
        self._ws.close()
