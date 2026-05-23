"""WebSocket broadcaster — pub/sub for dashboard clients."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import WebSocket
from loguru import logger


class WebSocketBroadcaster:
    """Tracks connected dashboard WebSocket clients and broadcasts JSON messages."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)
        logger.debug(f"WS client connected ({len(self._clients)} total)")

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)
        logger.debug(f"WS client disconnected ({len(self._clients)} total)")

    async def broadcast(self, event: str, data: Any) -> None:
        payload = json.dumps({"event": event, "data": data})
        dead: list[WebSocket] = []
        async with self._lock:
            clients = list(self._clients)

        for ws in clients:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)

        if dead:
            async with self._lock:
                for ws in dead:
                    self._clients.discard(ws)

    @property
    def client_count(self) -> int:
        return len(self._clients)


# Module-level singleton — imported by api.py and injected into notifier
broadcaster = WebSocketBroadcaster()
