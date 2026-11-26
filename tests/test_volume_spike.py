"""Tests for the spot BTCUSDT volume-spike monitor."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from karen.config import Settings
from karen.notifications.telegram import TelegramNotifier
from karen.notifications.volume_spike import (
    CONFIRM,
    EARLY,
    Candle,
    VolumeSpikeMonitor,
    compute_spike,
)

M15 = 15 * 60_000


def _history(n: int = 96) -> list[float]:
    # Mildly noisy volumes around 100
    return [100.0 + (i % 7) * 3 - 9 for i in range(n)]


# ── compute_spike ─────────────────────────────────────────────────────────────


def test_baseline_is_median_not_mean() -> None:
    hist = [100.0] * 95 + [100_000.0]  # one old huge spike
    stats = compute_spike(150.0, hist, 3.0, 3.0)
    assert stats is not None
    assert stats.baseline == 100.0
    assert stats.ratio == pytest.approx(1.5)


def test_spike_by_ratio() -> None:
    stats = compute_spike(400.0, _history(), 3.0, 99.0)
    assert stats is not None and stats.by_ratio and stats.is_spike


def test_spike_by_zscore_only() -> None:
    stats = compute_spike(150.0, _history(), 3.0, 3.0)
    assert stats is not None
    assert not stats.by_ratio
    assert stats.zscore >= 3.0 and stats.by_zscore


def test_no_spike_on_normal_volume() -> None:
    stats = compute_spike(101.0, _history(), 3.0, 3.0)
    assert stats is not None and not stats.is_spike


def test_flat_history_has_zero_zscore() -> None:
    stats = compute_spike(100.0, [100.0] * 50, 3.0, 3.0)
    assert stats is not None and stats.zscore == 0.0 and not stats.is_spike


# ── Monitor ───────────────────────────────────────────────────────────────────


@pytest.fixture
def monitor(default_settings: Settings) -> VolumeSpikeMonitor:
    notifier = MagicMock(spec=TelegramNotifier)
    notifier.volume_spike = AsyncMock()
    mon = VolumeSpikeMonitor(lambda: default_settings, notifier, http=MagicMock())
    st = mon._state["15m"]
    for i, v in enumerate(_history(96)):
        st.add_closed(Candle(i * M15, 40_000.0, 40_010.0, v))
    return mon


def _msg(open_time: int, volume: float, closed: bool, tf: str = "15m") -> dict:
    return {
        "stream": f"btcusdt@kline_{tf}",
        "data": {
            "e": "kline",
            "s": "BTCUSDT",
            "k": {"t": open_time, "i": tf, "o": "40000", "c": "40400",
                  "v": str(volume), "x": closed},
        },
    }


async def test_early_alert_once_per_candle(monitor: VolumeSpikeMonitor) -> None:
    t = 96 * M15
    await monitor.on_message(_msg(t, 50.0, False))
    monitor._notifier.volume_spike.assert_not_awaited()
    await monitor.on_message(_msg(t, 500.0, False))
    await monitor.on_message(_msg(t, 600.0, False))
    assert monitor._notifier.volume_spike.await_count == 1
    kw = monitor._notifier.volume_spike.await_args.kwargs
    assert kw["kind"] == EARLY and kw["timeframe"] == "15m"


async def test_confirm_after_early_and_once(monitor: VolumeSpikeMonitor) -> None:
    t = 96 * M15
    await monitor.on_message(_msg(t, 500.0, False))
    await monitor.on_message(_msg(t, 700.0, True))
    await monitor.on_message(_msg(t, 700.0, True))  # duplicate close
    kinds = [c.kwargs["kind"] for c in monitor._notifier.volume_spike.await_args_list]
    assert kinds == [EARLY, CONFIRM]
    # Baseline for the confirm excludes the candle itself
    assert monitor._notifier.volume_spike.await_args.kwargs["baseline"] < 110


async def test_alerts_ignore_trading_enabled(monitor: VolumeSpikeMonitor) -> None:
    assert monitor._settings().trading_enabled is False
    await monitor.on_message(_msg(96 * M15, 700.0, True))
    monitor._notifier.volume_spike.assert_awaited_once()


async def test_volume_alert_disabled_suppresses(default_settings: Settings) -> None:
    s = default_settings.model_copy(update={"volume_alert_enabled": False})
    notifier = MagicMock(spec=TelegramNotifier)
    notifier.volume_spike = AsyncMock()
    mon = VolumeSpikeMonitor(lambda: s, notifier, http=MagicMock())
    for i, v in enumerate(_history(96)):
        mon._state["15m"].add_closed(Candle(i * M15, 1.0, 1.0, v))
    await mon.on_message(_msg(96 * M15, 700.0, True))
    notifier.volume_spike.assert_not_awaited()


async def test_gap_triggers_backfill(monitor: VolumeSpikeMonitor) -> None:
    monitor._backfill = AsyncMock()  # type: ignore[method-assign]
    await monitor.on_message(_msg(98 * M15, 10.0, False))  # skipped candle 96 & 97
    monitor._backfill.assert_awaited_once_with(alert_latest=False)


async def test_not_enough_history_no_alert(default_settings: Settings) -> None:
    notifier = MagicMock(spec=TelegramNotifier)
    notifier.volume_spike = AsyncMock()
    mon = VolumeSpikeMonitor(lambda: default_settings, notifier, http=MagicMock())
    for i in range(5):
        mon._state["15m"].add_closed(Candle(i * M15, 1.0, 1.0, 100.0))
    await mon.on_message(_msg(5 * M15, 10_000.0, True))
    notifier.volume_spike.assert_not_awaited()


async def test_notifier_volume_spike_message(default_settings: Settings) -> None:
    n = TelegramNotifier(default_settings)
    n._send = AsyncMock()  # type: ignore[method-assign]
    await n.volume_spike(
        symbol="BTCUSDT", timeframe="1h", kind=CONFIRM, open_time_ms=1_700_000_000_000,
        interval_ms=3_600_000, volume=4200.0, baseline=1000.0, ratio=4.2, zscore=3.5,
        open_price=40_000.0, close_price=39_600.0,
    )
    text = n._send.await_args.args[0]
    assert "KONFIRMASI" in text and "4.2x" in text and "3.50" in text
    assert "turun -1.00%" in text and "WIB" in text
