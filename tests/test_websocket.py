"""Tests for PriceFeed WebSocket handler."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from karen.exchange.websocket import (
    Kline,
    MarkPrice,
    PriceFeed,
    _kline_stream,
    _mark_price_stream,
    _parse_kline,
    _parse_mark_price,
    _stream_url,
)

# ─── Stream URL helpers ───────────────────────────────────────────────────────


def test_kline_stream_name() -> None:
    assert _kline_stream("BTCUSDT", "15m") == "btcusdt@kline_15m"
    assert _kline_stream("ETHUSDT", "1h") == "ethusdt@kline_1h"


def test_mark_price_stream_name() -> None:
    assert _mark_price_stream("BTCUSDT") == "btcusdt@markPrice@1s"


def test_stream_url_testnet() -> None:
    url = _stream_url(testnet=True, streams=["btcusdt@kline_15m", "ethusdt@markPrice@1s"])
    assert "fstream.binancefuture.com" in url
    assert "btcusdt@kline_15m" in url
    assert "ethusdt@markPrice@1s" in url


def test_stream_url_live() -> None:
    url = _stream_url(testnet=False, streams=["btcusdt@kline_15m"])
    assert "fstream.binance.com" in url


# ─── Message parsers ──────────────────────────────────────────────────────────


def test_parse_kline() -> None:
    data = {
        "e": "kline",
        "s": "BTCUSDT",
        "k": {
            "t": 1_700_000_000_000,
            "i": "15m",
            "o": "40000.0",
            "h": "40500.0",
            "l": "39800.0",
            "c": "40200.0",
            "v": "123.5",
            "x": True,
            "s": "BTCUSDT",
        },
    }
    kline = _parse_kline(data)
    assert kline.symbol == "BTCUSDT"
    assert kline.timeframe == "15m"
    assert kline.open == pytest.approx(40000.0)
    assert kline.close == pytest.approx(40200.0)
    assert kline.is_closed is True
    assert kline.open_time == 1_700_000_000_000


def test_parse_kline_not_closed() -> None:
    data = {
        "e": "kline",
        "s": "ETHUSDT",
        "k": {
            "t": 1_700_000_000_000,
            "i": "1h",
            "o": "2500.0",
            "h": "2520.0",
            "l": "2490.0",
            "c": "2510.0",
            "v": "50.0",
            "x": False,
            "s": "ETHUSDT",
        },
    }
    kline = _parse_kline(data)
    assert kline.is_closed is False
    assert kline.symbol == "ETHUSDT"


def test_parse_mark_price() -> None:
    data = {
        "e": "markPriceUpdate",
        "s": "BTCUSDT",
        "p": "40150.25",
        "T": 1_700_000_001_000,
    }
    mp = _parse_mark_price(data)
    assert mp.symbol == "BTCUSDT"
    assert mp.price == pytest.approx(40150.25)
    assert mp.timestamp == 1_700_000_001_000


# ─── PriceFeed dispatch ───────────────────────────────────────────────────────


@pytest.fixture
def feed() -> PriceFeed:
    return PriceFeed(symbols=["BTCUSDT", "ETHUSDT"], testnet=True)


@pytest.mark.asyncio
async def test_dispatch_kline_message(feed: PriceFeed) -> None:
    received: list[Kline] = []

    async def on_kline(k: Kline) -> None:
        received.append(k)

    feed.on_kline(on_kline)

    msg = {
        "data": {
            "e": "kline",
            "s": "BTCUSDT",
            "k": {
                "t": 1_700_000_000_000, "i": "15m",
                "o": "40000", "h": "40500", "l": "39800",
                "c": "40200", "v": "100", "x": True, "s": "BTCUSDT",
            },
        }
    }
    await feed._dispatch(msg)
    assert len(received) == 1
    assert received[0].symbol == "BTCUSDT"


@pytest.mark.asyncio
async def test_dispatch_mark_price_message(feed: PriceFeed) -> None:
    received: list[MarkPrice] = []

    async def on_mp(mp: MarkPrice) -> None:
        received.append(mp)

    feed.on_mark_price(on_mp)

    msg = {
        "data": {
            "e": "markPriceUpdate",
            "s": "ETHUSDT",
            "p": "2510.50",
            "T": 1_700_000_001_000,
        }
    }
    await feed._dispatch(msg)
    assert len(received) == 1
    assert received[0].symbol == "ETHUSDT"
    assert received[0].price == pytest.approx(2510.50)


@pytest.mark.asyncio
async def test_dispatch_unknown_event_ignored(feed: PriceFeed) -> None:
    kline_received: list[Kline] = []
    feed.on_kline(lambda k: kline_received.append(k))

    msg = {"data": {"e": "aggTrade", "s": "BTCUSDT", "p": "40000"}}
    await feed._dispatch(msg)  # should not raise
    assert len(kline_received) == 0


@pytest.mark.asyncio
async def test_multiple_callbacks_all_called(feed: PriceFeed) -> None:
    counts = [0, 0]

    async def cb1(k: Kline) -> None:
        counts[0] += 1

    async def cb2(k: Kline) -> None:
        counts[1] += 1

    feed.on_kline(cb1)
    feed.on_kline(cb2)

    msg = {
        "data": {
            "e": "kline",
            "s": "BTCUSDT",
            "k": {
                "t": 0, "i": "15m", "o": "1", "h": "1",
                "l": "1", "c": "1", "v": "1", "x": True, "s": "BTCUSDT",
            },
        }
    }
    await feed._dispatch(msg)
    assert counts == [1, 1]


@pytest.mark.asyncio
async def test_callback_exception_does_not_break_dispatch(feed: PriceFeed) -> None:
    async def bad_cb(k: Kline) -> None:
        raise ValueError("boom")

    good_received: list[Kline] = []

    async def good_cb(k: Kline) -> None:
        good_received.append(k)

    feed.on_kline(bad_cb)
    feed.on_kline(good_cb)

    msg = {
        "data": {
            "e": "kline",
            "s": "BTCUSDT",
            "k": {
                "t": 0, "i": "15m", "o": "1", "h": "1",
                "l": "1", "c": "1", "v": "1", "x": True, "s": "BTCUSDT",
            },
        }
    }
    await feed._dispatch(msg)
    # good_cb still ran despite bad_cb throwing
    assert len(good_received) == 1


# ─── PriceFeed stream building ───────────────────────────────────────────────


def test_build_streams_contains_all_expected(feed: PriceFeed) -> None:
    streams = feed._build_streams()
    # 2 symbols × 3 timeframes (1m, 15m, 1h) + 2 mark price = 8
    assert len(streams) == 8
    assert "btcusdt@kline_1m" in streams
    assert "btcusdt@kline_15m" in streams
    assert "btcusdt@kline_1h" in streams
    assert "ethusdt@kline_15m" in streams
    assert "ethusdt@kline_1h" in streams
    assert "btcusdt@markPrice@1s" in streams
    assert "ethusdt@markPrice@1s" in streams


# ─── PriceFeed lifecycle ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_feed_start_stop() -> None:
    feed = PriceFeed(symbols=["BTCUSDT"], testnet=True)

    async def fake_run_with_reconnect() -> None:
        await asyncio.sleep(999)

    with patch.object(feed, "_run_with_reconnect", fake_run_with_reconnect):
        feed.start()
        assert feed._running is True
        assert feed._task is not None
        await feed.stop()
        assert feed._running is False


@pytest.mark.asyncio
async def test_feed_start_idempotent() -> None:
    feed = PriceFeed(symbols=["BTCUSDT"], testnet=True)

    async def fake_run() -> None:
        await asyncio.sleep(999)

    with patch.object(feed, "_run_with_reconnect", fake_run):
        feed.start()
        first_task = feed._task
        feed.start()  # second call should be no-op
        assert feed._task is first_task
        await feed.stop()
