"""Live price and kline feeds via Binance WebSocket streams."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

from loguru import logger

try:
    import websockets
    from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
except ImportError:
    raise ImportError("websockets package required — pip install websockets") from None


# ─── Domain types ─────────────────────────────────────────────────────────────


@dataclass
class Kline:
    symbol: str
    timeframe: str       # "15m" | "1h"
    open_time: int       # ms epoch
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_closed: bool      # True when candle is finalised


@dataclass
class MarkPrice:
    symbol: str
    price: float
    timestamp: int       # ms epoch


KlineCallback = Callable[[Kline], Coroutine[Any, Any, None]]
MarkPriceCallback = Callable[[MarkPrice], Coroutine[Any, Any, None]]

# ─── Stream URL helpers ────────────────────────────────────────────────────────

_LIVE_BASE = "wss://fstream.binance.com/stream"
_TEST_BASE = "wss://fstream.binancefuture.com/stream"


def _stream_url(testnet: bool, streams: list[str]) -> str:
    base = _TEST_BASE if testnet else _LIVE_BASE
    combined = "/".join(streams)
    return f"{base}?streams={combined}"


def _kline_stream(symbol: str, interval: str) -> str:
    return f"{symbol.lower()}@kline_{interval}"


def _mark_price_stream(symbol: str) -> str:
    return f"{symbol.lower()}@markPrice@1s"


# ─── Feed manager ─────────────────────────────────────────────────────────────


class PriceFeed:
    """
    Manages Binance futures WebSocket streams for klines and mark prices.

    Subscribes to:
    - 15m klines for all symbols
    - 1h klines for all symbols
    - Mark price (1s) for all symbols

    Auto-reconnects with exponential backoff on disconnect.
    """

    _TIMEFRAMES = ("15m", "1h")
    _MAX_RECONNECT_DELAY = 60.0

    def __init__(self, symbols: list[str], testnet: bool = True) -> None:
        self._symbols = symbols
        self._testnet = testnet
        self._kline_callbacks: list[KlineCallback] = []
        self._mark_price_callbacks: list[MarkPriceCallback] = []
        self._task: asyncio.Task[None] | None = None
        self._running = False

    # ── Public API ─────────────────────────────────────────────────────────

    def on_kline(self, cb: KlineCallback) -> None:
        self._kline_callbacks.append(cb)

    def on_mark_price(self, cb: MarkPriceCallback) -> None:
        self._mark_price_callbacks.append(cb)

    def start(self) -> None:
        """Start the feed in a background asyncio task."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_with_reconnect(), name="price-feed")
        logger.info(f"PriceFeed started for {len(self._symbols)} symbols")

    async def stop(self) -> None:
        """Stop the feed gracefully."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        logger.info("PriceFeed stopped")

    # ── Internal ───────────────────────────────────────────────────────────

    def _build_streams(self) -> list[str]:
        streams: list[str] = []
        for symbol in self._symbols:
            for tf in self._TIMEFRAMES:
                streams.append(_kline_stream(symbol, tf))
            streams.append(_mark_price_stream(symbol))
        return streams

    async def _run_with_reconnect(self) -> None:
        delay = 1.0
        while self._running:
            try:
                url = _stream_url(self._testnet, self._build_streams())
                logger.debug(f"Connecting to Binance WS: {url[:80]}...")
                await self._listen(url)
                delay = 1.0  # reset on clean disconnect
            except (ConnectionClosedError, ConnectionClosedOK) as exc:
                logger.warning(f"WS disconnected: {exc} — reconnecting in {delay:.0f}s")
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception(f"WS error — reconnecting in {delay:.0f}s")
            if self._running:
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._MAX_RECONNECT_DELAY)

    async def _listen(self, url: str) -> None:
        async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:  # type: ignore[attr-defined]
            logger.info("WS connection established")
            async for raw in ws:
                if not self._running:
                    break
                try:
                    msg = json.loads(raw)
                    await self._dispatch(msg)
                except Exception:
                    logger.exception("Error processing WS message")

    async def _dispatch(self, msg: dict[str, Any]) -> None:
        data: dict[str, Any] = msg.get("data", msg)
        event = str(data.get("e", ""))

        if event == "kline":
            kline = _parse_kline(data)
            for cb in self._kline_callbacks:
                try:
                    await cb(kline)
                except Exception:
                    logger.exception("Error in kline callback")

        elif event == "markPriceUpdate":
            mp = _parse_mark_price(data)
            for cb in self._mark_price_callbacks:
                try:
                    await cb(mp)
                except Exception:
                    logger.exception("Error in mark-price callback")


# ─── Parsers ──────────────────────────────────────────────────────────────────


def _parse_kline(data: dict[str, Any]) -> Kline:
    k = data["k"]
    return Kline(
        symbol=str(data.get("s", k.get("s", ""))),
        timeframe=str(k.get("i", "")),
        open_time=int(k.get("t", 0)),
        open=float(k.get("o", 0)),
        high=float(k.get("h", 0)),
        low=float(k.get("l", 0)),
        close=float(k.get("c", 0)),
        volume=float(k.get("v", 0)),
        is_closed=bool(k.get("x", False)),
    )


def _parse_mark_price(data: dict[str, Any]) -> MarkPrice:
    return MarkPrice(
        symbol=str(data.get("s", "")),
        price=float(data.get("p", 0)),
        timestamp=int(data.get("T", 0)),
    )
