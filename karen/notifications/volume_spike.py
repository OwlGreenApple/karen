"""
Volume-spike monitor for Binance SPOT BTCUSDT (15m and 1h).

Runs independently of TRADING_ENABLED — it only reads public market data and
sends Telegram alerts.

Per timeframe:
  - Baseline = median volume of the previous N closed candles.
  - Spike if volume >= MULTIPLIER x baseline  OR  log-volume z-score >= Z.
  - Closed candle   → "konfirmasi" alert.
  - Running candle  → "early" alert as soon as it crosses (max 1x per candle).

Data: REST /api/v3/klines seeds history at start and backfills after every
reconnect; realtime updates come from the combined kline WebSocket stream.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import statistics
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
from loguru import logger

try:
    import websockets
    from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
except ImportError:
    raise ImportError("websockets package required — pip install websockets") from None

from karen.config import Settings
from karen.notifications.telegram import TelegramNotifier

SYMBOL = "BTCUSDT"
_REST_URL = "https://api.binance.com/api/v3/klines"
_WS_BASE = "wss://stream.binance.com:9443/stream"

_INTERVAL_MS = {"15m": 15 * 60_000, "1h": 60 * 60_000}

EARLY = "early"
CONFIRM = "konfirmasi"


# ─── Pure spike math ──────────────────────────────────────────────────────────


@dataclass
class SpikeStats:
    volume: float
    baseline: float     # median of previous N volumes
    ratio: float        # volume / baseline
    zscore: float       # (log(v) - mean(log)) / stdev(log)
    by_ratio: bool
    by_zscore: bool

    @property
    def is_spike(self) -> bool:
        return self.by_ratio or self.by_zscore


def _log(v: float) -> float:
    # log1p keeps zero-volume candles finite; at BTC volume scale it equals log()
    return math.log1p(max(v, 0.0))


def compute_spike(
    volume: float, history: list[float], multiplier: float, z_threshold: float
) -> SpikeStats | None:
    """Evaluate `volume` against `history` (previous N candle volumes, oldest first)."""
    if len(history) < 2:
        return None
    baseline = statistics.median(history)
    ratio = volume / baseline if baseline > 0 else math.inf if volume > 0 else 0.0

    logs = [_log(v) for v in history]
    mean = statistics.fmean(logs)
    std = statistics.pstdev(logs)
    zscore = (_log(volume) - mean) / std if std > 0 else 0.0

    return SpikeStats(
        volume=volume,
        baseline=baseline,
        ratio=ratio,
        zscore=zscore,
        by_ratio=baseline > 0 and ratio >= multiplier,
        by_zscore=std > 0 and zscore >= z_threshold,
    )


# ─── Per-timeframe state ──────────────────────────────────────────────────────


@dataclass
class Candle:
    open_time: int      # ms epoch
    open: float
    close: float
    volume: float


@dataclass
class _TFState:
    timeframe: str
    n: int
    closed: deque[Candle] = field(default_factory=deque)
    # (kind, open_time) pairs already alerted — enforces 1 alert per kind per candle
    alerted: set[tuple[str, int]] = field(default_factory=set)

    @property
    def last_closed_open_time(self) -> int | None:
        return self.closed[-1].open_time if self.closed else None

    def history(self) -> list[float]:
        return [c.volume for c in list(self.closed)[-self.n:]]

    def add_closed(self, c: Candle) -> bool:
        """Append a closed candle if newer than the last one. Returns True if added."""
        last = self.last_closed_open_time
        if last is not None and c.open_time <= last:
            return False
        self.closed.append(c)
        while len(self.closed) > self.n:
            self.closed.popleft()
        return True

    def prune_alerted(self) -> None:
        cutoff = (self.last_closed_open_time or 0) - self.n * _INTERVAL_MS[self.timeframe]
        self.alerted = {k for k in self.alerted if k[1] >= cutoff}


# ─── Monitor ──────────────────────────────────────────────────────────────────


class VolumeSpikeMonitor:
    _MAX_RECONNECT_DELAY = 60.0
    # BTCUSDT spot klines push every ~1-2s; this much silence means a dead socket
    _SILENCE_TIMEOUT = 90.0

    def __init__(
        self,
        settings_getter: Callable[[], Settings],
        notifier: TelegramNotifier,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings_getter
        self._notifier = notifier
        self._http = http
        s = settings_getter()
        self._state: dict[str, _TFState] = {
            "15m": _TFState("15m", s.vol_spike_15m_n),
            "1h": _TFState("1h", s.vol_spike_1h_n),
        }
        self._task: asyncio.Task[None] | None = None
        self._running = False

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run(), name="volume-spike")
        logger.info("Volume spike monitor started (spot BTCUSDT 15m/1h)")

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _run(self) -> None:
        own_http = self._http is None
        if own_http:
            self._http = httpx.AsyncClient(timeout=15.0)
        try:
            await self._run_with_reconnect()
        finally:
            if own_http and self._http is not None:
                await self._http.aclose()
                self._http = None

    async def _run_with_reconnect(self) -> None:
        delay = 1.0
        first = True
        while self._running:
            try:
                # Seed on first connect; backfill (and alert the latest missed
                # close) on every reconnect.
                await self._backfill(alert_latest=not first)
                first = False
                await self._listen()
                delay = 1.0
            except (ConnectionClosedError, ConnectionClosedOK) as exc:
                logger.warning(f"Volume WS disconnected: {exc} — reconnecting in {delay:.0f}s")
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(f"Volume monitor error — reconnecting in {delay:.0f}s")
            if self._running:
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._MAX_RECONNECT_DELAY)

    # ── REST ───────────────────────────────────────────────────────────────

    async def _fetch_closed(self, tf: str, limit: int) -> list[Candle]:
        assert self._http is not None
        resp = await self._http.get(
            _REST_URL, params={"symbol": SYMBOL, "interval": tf, "limit": limit}
        )
        resp.raise_for_status()
        now_ms = int(time.time() * 1000)
        out = []
        for row in resp.json():
            close_time = int(row[6])
            if close_time >= now_ms:  # still-running candle
                continue
            out.append(Candle(int(row[0]), float(row[1]), float(row[4]), float(row[5])))
        return out

    async def _backfill(self, alert_latest: bool) -> None:
        for tf, st in self._state.items():
            candles = await self._fetch_closed(tf, min(st.n + 2, 1000))
            added = [c for c in candles if st.add_closed(c)]
            if added:
                logger.debug(f"Volume monitor: {tf} backfilled {len(added)} candle(s)")
            if alert_latest and added:
                latest = added[-1]
                hist = [c.volume for c in candles if c.open_time < latest.open_time][-st.n:]
                await self._evaluate(st, latest, hist, CONFIRM)
            st.prune_alerted()

    # ── WebSocket ──────────────────────────────────────────────────────────

    def _ws_url(self) -> str:
        streams = "/".join(f"{SYMBOL.lower()}@kline_{tf}" for tf in self._state)
        return f"{_WS_BASE}?streams={streams}"

    async def _listen(self) -> None:
        async with websockets.connect(
            self._ws_url(), ping_interval=20, ping_timeout=10
        ) as ws:
            logger.info("Volume WS connected (spot)")
            while self._running:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=self._SILENCE_TIMEOUT)
                except TimeoutError:
                    logger.warning("Volume WS silent — reconnecting")
                    return
                try:
                    await self.on_message(json.loads(raw))
                except Exception:
                    logger.exception("Volume monitor: error processing message")

    async def on_message(self, msg: dict[str, Any]) -> None:
        data = msg.get("data", msg)
        if data.get("e") != "kline":
            return
        k = data["k"]
        st = self._state.get(str(k.get("i")))
        if st is None:
            return
        candle = Candle(int(k["t"]), float(k["o"]), float(k["c"]), float(k["v"]))

        # A gap means we missed at least one close — refill baseline via REST
        last = st.last_closed_open_time
        interval = _INTERVAL_MS[st.timeframe]
        if last is not None and candle.open_time - last > interval:
            logger.info(f"Volume monitor: {st.timeframe} gap detected — backfilling")
            await self._backfill(alert_latest=False)

        hist = st.history()
        if bool(k.get("x")):
            if st.add_closed(candle):
                await self._evaluate(st, candle, hist, CONFIRM)
                st.prune_alerted()
        elif st.last_closed_open_time is None or candle.open_time > st.last_closed_open_time:
            await self._evaluate(st, candle, hist, EARLY)

    # ── Alerting ───────────────────────────────────────────────────────────

    async def _evaluate(
        self, st: _TFState, candle: Candle, hist: list[float], kind: str
    ) -> None:
        key = (kind, candle.open_time)
        if key in st.alerted:
            return
        if len(hist) < max(2, st.n // 2):  # not enough baseline yet
            return
        s = self._settings()
        multiplier = s.vol_spike_15m_mult if st.timeframe == "15m" else s.vol_spike_1h_mult
        stats = compute_spike(candle.volume, hist, multiplier, s.vol_spike_zscore)
        if stats is None or not stats.is_spike:
            return
        st.alerted.add(key)
        if not s.volume_alert_enabled:
            return
        logger.info(
            f"Volume spike {st.timeframe} {kind}: vol={stats.volume:.2f} "
            f"base={stats.baseline:.2f} ratio={stats.ratio:.2f}x z={stats.zscore:.2f}"
        )
        await self._notifier.volume_spike(
            symbol=SYMBOL,
            timeframe=st.timeframe,
            kind=kind,
            open_time_ms=candle.open_time,
            interval_ms=_INTERVAL_MS[st.timeframe],
            volume=stats.volume,
            baseline=stats.baseline,
            ratio=stats.ratio,
            zscore=stats.zscore,
            open_price=candle.open,
            close_price=candle.close,
        )
