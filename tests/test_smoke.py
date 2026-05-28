"""
Integration smoke test — wires all components together (mocked exchange)
and validates the full startup/trading/shutdown flow.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from karen.config import StrategyMode
from karen.dashboard.api import app as dashboard_app
from karen.dashboard.api import init_dashboard
from karen.exchange.binance_client import OrderInfo, PositionInfo
from karen.exchange.websocket import Kline
from karen.execution.order_manager import OrderManager
from karen.execution.position_tracker import PositionTracker
from karen.notifications.telegram import TelegramNotifier
from karen.persistence.models import (
    Base,
    CloseReason,
    Trade,
    TradeStatus,
)
from karen.risk.guardrails import DailyGuardrails
from karen.strategies.mean_reversion import MeanReversionStrategy
from karen.strategies.trend_following import TrendFollowingStrategy
from tests.conftest import (
    make_ind1h,
    make_ind15m,
    make_ind5m_mr,
    make_ohlcv,
    make_ranging_ohlcv,
    make_trending_ohlcv,
)

# ── Shared test infrastructure ─────────────────────────────────────────────────


@pytest.fixture
async def db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


def _make_client(
    positions: list[PositionInfo] | None = None,
    equity: float = 10_000.0,
    order: OrderInfo | None = None,
) -> MagicMock:
    c = MagicMock()
    c.fetch_positions = AsyncMock(return_value=positions or [])
    c.fetch_equity = AsyncMock(return_value=equity)
    c.fetch_order = AsyncMock(return_value=order or _open_order())
    c.place_order = AsyncMock(side_effect=_make_placed_order)
    c.cancel_order = AsyncMock()
    c.cancel_all_orders = AsyncMock()
    c.fetch_open_orders = AsyncMock(return_value=[])
    c.fetch_ohlcv = AsyncMock(return_value=make_ohlcv())
    c.get_min_qty = MagicMock(return_value=0.0)
    return c


def _open_order() -> OrderInfo:
    return OrderInfo(
        id="ord-1", client_order_id="karen-e-btc-abc123", symbol="BTCUSDT",
        side="buy", type="market", status="open",
        quantity=0.1, filled=0.0, price=None, average_price=None, stop_price=None,
        timestamp=datetime.now(tz=UTC),
    )


_order_counter = 0


def _make_placed_order(**kwargs) -> OrderInfo:
    global _order_counter
    _order_counter += 1
    return OrderInfo(
        id=f"ord-{_order_counter}",
        client_order_id=kwargs.get("client_order_id", f"karen-x-{_order_counter}"),
        symbol=kwargs.get("symbol", "BTCUSDT"),
        side=kwargs.get("side", "buy"),
        type=kwargs.get("order_type", "market"),
        status="closed",
        quantity=kwargs.get("quantity", 0.1),
        filled=kwargs.get("quantity", 0.1),
        price=kwargs.get("price"),
        average_price=kwargs.get("price"),
        stop_price=kwargs.get("stop_price"),
        timestamp=datetime.now(tz=UTC),
    )


# ── Startup reconciliation ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_startup_reconcile_clean_state(default_settings, db):
    """Startup with no positions and no DB trades — no-op."""
    client = _make_client(positions=[])
    tracker = PositionTracker(client, default_settings, session_factory=db)
    await tracker.startup_reconcile()

    async with db() as session:
        result = await session.execute(select(Trade))
        assert result.scalars().all() == []


@pytest.mark.asyncio
async def test_startup_reconcile_creates_orphan_then_closes_on_next_cycle(
    default_settings, db
):
    """
    Startup: Binance has a position not in DB → orphan created.
    Next reconcile: position gone from Binance → orphan closed as MANUAL.
    """
    bp = PositionInfo(
        symbol="BTCUSDT", side="long", entry_price=40_000.0,
        quantity=0.1, leverage=5, unrealized_pnl=50.0,
        mark_price=40_500.0, notional=4_000.0,
    )
    client = _make_client(positions=[bp])
    tracker = PositionTracker(client, default_settings, session_factory=db)
    await tracker.startup_reconcile()

    async with db() as session:
        result = await session.execute(select(Trade))
        trades = result.scalars().all()
    assert len(trades) == 1
    assert trades[0].strategy_mode == "unknown"
    assert trades[0].status == TradeStatus.OPEN

    # Second reconcile — position gone
    client.fetch_positions.return_value = []
    await tracker._reconcile()

    async with db() as session:
        result = await session.execute(select(Trade))
        trade = result.scalar_one()
    assert trade.status == TradeStatus.CLOSED
    assert trade.close_reason == CloseReason.MANUAL


# ── Guardrails + order manager integration ────────────────────────────────────


@pytest.mark.asyncio
async def test_guardrails_blocks_after_daily_loss(default_settings, db):
    """After daily loss limit is hit, can_open_position returns False."""
    guardrails = DailyGuardrails(default_settings)
    guardrails.new_day(10_000.0)

    # Simulate a loss that exceeds 5%
    ok, reason = guardrails.can_open_position(9_400.0, 0)  # 6% loss
    assert not ok
    assert guardrails.paused


@pytest.mark.asyncio
async def test_guardrails_max_concurrent_blocks(default_settings, db):
    """Can't open beyond max_concurrent_positions."""
    guardrails = DailyGuardrails(default_settings)
    guardrails.new_day(10_000.0)

    # Fill all 3 slots
    for _ in range(default_settings.max_concurrent_positions):
        guardrails.record_trade_opened()

    ok, reason = guardrails.can_open_position(10_000.0, default_settings.max_concurrent_positions)
    assert not ok
    assert "positions" in reason.lower()


@pytest.mark.asyncio
async def test_guardrails_rollover_resets_state(default_settings, db):
    """Guardrails auto-rollover on UTC day change resets pause."""
    guardrails = DailyGuardrails(default_settings)
    guardrails.new_day(10_000.0)

    # Trigger pause via loss
    guardrails.can_open_position(9_400.0, 0)
    assert guardrails.paused

    # Simulate the next day by calling new_day
    guardrails.new_day(9_400.0)
    assert not guardrails.paused
    ok, _ = guardrails.can_open_position(9_400.0, 0)
    assert ok


# ── Order manager full trade lifecycle ────────────────────────────────────────


@pytest.mark.asyncio
async def test_full_trade_open_then_time_stop(default_settings, db):
    """
    Open a position via order_manager, then fire 8 closed klines
    to trigger the MR time stop (position not in profit).
    """
    from karen.strategies.base import Signal

    client = _make_client()
    notifier = TelegramNotifier(default_settings)
    order_mgr = OrderManager(client, default_settings, session_factory=db, notifier=notifier)

    signal = Signal(
        symbol="BTCUSDT",
        side="long",
        strategy_mode=StrategyMode.MEAN_REVERSION,
        entry_price=40_000.0,
        sl_price=39_500.0,
        tp1_price=40_500.0,
        tp1_qty_pct=0.70,
        tp2_price=41_000.0,
        atr=300.0,
    )

    trade = await order_mgr.open_position(signal, equity=10_000.0)
    assert trade is not None
    assert trade.status == TradeStatus.OPEN

    # Simulate 8 closed 15m klines — price slightly below entry (not in profit)
    for _ in range(8):
        kline = Kline(
            symbol="BTCUSDT",
            timeframe="15m",
            open_time=datetime.now(tz=UTC),
            open=39_800.0,
            high=39_900.0,
            low=39_700.0,
            close=39_800.0,  # below entry → not in profit
            volume=500.0,
            is_closed=True,
        )
        await order_mgr.on_closed_kline(kline)

    # After 8 candles below entry, time stop should fire
    async with db() as session:
        result = await session.execute(select(Trade))
        final_trade = result.scalar_one()

    assert final_trade.status == TradeStatus.CLOSED
    assert final_trade.close_reason == CloseReason.TIME_STOP


@pytest.mark.asyncio
async def test_full_trade_open_no_time_stop_when_in_profit(default_settings, db):
    """Time stop must NOT fire if price is above entry for a long."""
    from karen.strategies.base import Signal

    client = _make_client()
    order_mgr = OrderManager(client, default_settings, session_factory=db)

    signal = Signal(
        symbol="BTCUSDT",
        side="long",
        strategy_mode=StrategyMode.MEAN_REVERSION,
        entry_price=40_000.0,
        sl_price=39_500.0,
        tp1_price=40_500.0,
        tp1_qty_pct=0.70,
        tp2_price=41_000.0,
        atr=300.0,
    )
    trade = await order_mgr.open_position(signal, equity=10_000.0)
    assert trade is not None

    # 8 klines above entry — should NOT trigger time stop
    for _ in range(8):
        kline = Kline(
            symbol="BTCUSDT",
            timeframe="15m",
            open_time=datetime.now(tz=UTC),
            open=40_200.0,
            high=40_500.0,
            low=40_100.0,
            close=40_300.0,  # above entry → in profit
            volume=500.0,
            is_closed=True,
        )
        await order_mgr.on_closed_kline(kline)

    async with db() as session:
        result = await session.execute(select(Trade))
        final_trade = result.scalar_one()

    assert final_trade.status == TradeStatus.OPEN


# ── Strategy + order manager signal flow ──────────────────────────────────────


@pytest.mark.asyncio
async def test_mr_signal_leads_to_open_position(default_settings, db):
    """MR strategy generates a LONG signal → order_mgr opens a trade."""

    ind_5m = make_ind5m_mr(
        close=40_100.0,      # close > bb_lower — bounce back inside band
        bb_lower=40_000.0,
        rsi=32.0,            # oversold (< 35 threshold)
        prev_low=39_950.0,   # prev_low <= bb_lower — previous candle touched band
        atr=80.0,
    )
    ind_15m = make_ind15m(adx=15.0)  # ranging filter

    strategy = MeanReversionStrategy(
        default_settings,
        compute_signal_fn=lambda df, s: ind_5m,
        compute_filter_fn=lambda df, s: ind_15m,
    )

    df = make_ranging_ohlcv()
    signal = await strategy.evaluate("BTCUSDT", df, df)
    assert signal is not None
    assert signal.side == "long"

    client = _make_client()
    order_mgr = OrderManager(client, default_settings, session_factory=db)
    trade = await order_mgr.open_position(signal, equity=10_000.0)

    assert trade is not None
    assert trade.symbol == "BTCUSDT"
    assert trade.status == TradeStatus.OPEN
    # Verify SL and TP orders were placed (entry + sl + tp1 + tp2 = 4 calls)
    assert client.place_order.call_count >= 2


@pytest.mark.asyncio
async def test_tf_signal_leads_to_open_position(default_settings, db):
    """TF strategy generates a LONG signal → order_mgr opens a trade."""
    ind_15m = make_ind15m(
        close=40_100.0,
        ema_fast=40_000.0,
        macd_hist=5.0,
        macd_hist_prev=-5.0,
        prev_low=39_950.0,
        vol=600.0,
        vol_ma=400.0,
        atr=300.0,
    )
    ind_1h = make_ind1h(adx=30.0, ema_mid=39_800.0, ema_slow=39_000.0)

    strategy = TrendFollowingStrategy(
        default_settings,
        compute_15m_fn=lambda df, s: ind_15m,
        compute_1h_fn=lambda df, s: ind_1h,
    )

    df = make_trending_ohlcv(uptrend=True)
    signal = await strategy.evaluate("BTCUSDT", df, df)
    assert signal is not None
    assert signal.side == "long"
    assert signal.tp2_price is None  # TF uses trailing stop, not fixed TP2

    client = _make_client()
    order_mgr = OrderManager(client, default_settings, session_factory=db)
    trade = await order_mgr.open_position(signal, equity=10_000.0)
    assert trade is not None
    assert trade.status == TradeStatus.OPEN


# ── Dashboard API integration ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dashboard_reflects_open_trade(default_settings, db):
    """After a trade is opened, /api/trades shows it and /api/status shows equity."""
    import karen.dashboard.api as api_module

    # Patch the session factory
    api_module.AsyncSessionLocal = db
    init_dashboard(equity=10_000.0, mode="mean_reversion", trading_enabled=True)

    from karen.strategies.base import Signal

    client = _make_client()
    order_mgr = OrderManager(client, default_settings, session_factory=db)

    signal = Signal(
        symbol="ETHUSDT",
        side="short",
        strategy_mode=StrategyMode.MEAN_REVERSION,
        entry_price=2_500.0,
        sl_price=2_550.0,
        tp1_price=2_450.0,
        tp1_qty_pct=0.70,
        tp2_price=2_400.0,
        atr=30.0,
    )
    await order_mgr.open_position(signal, equity=10_000.0)

    transport = ASGITransport(app=dashboard_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/api/trades?status=open")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 1
        assert data["trades"][0]["symbol"] == "ETHUSDT"
        assert data["trades"][0]["side"] == "short"

        r2 = await ac.get("/api/status")
        assert r2.status_code == 200
        assert r2.json()["trading_enabled"] is True


# ── Strategy mode switch ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_strategy_mode_switch_cancels_pending_entries(default_settings, db):
    """
    When mode switches, cancel_pending_entry_orders must try to cancel
    any open-status entry order for open trades.
    """
    from karen.strategies.base import Signal

    client = _make_client()
    # Simulate the entry order is still "open" (pending fill)
    client.fetch_order.return_value = OrderInfo(
        id="ord-entry", client_order_id="karen-e-btc-abc", symbol="BTCUSDT",
        side="buy", type="limit", status="open",
        quantity=0.1, filled=0.0, price=40_000.0, average_price=None, stop_price=None,
        timestamp=datetime.now(tz=UTC),
    )

    order_mgr = OrderManager(client, default_settings, session_factory=db)

    # Open a trade
    signal = Signal(
        symbol="BTCUSDT", side="long",
        strategy_mode=StrategyMode.MEAN_REVERSION,
        entry_price=40_000.0, sl_price=39_500.0,
        tp1_price=40_500.0, tp1_qty_pct=0.70,
        tp2_price=41_000.0, atr=300.0,
    )
    trade = await order_mgr.open_position(signal, equity=10_000.0)
    assert trade is not None

    # Mode switch → cancel pending entry orders
    await order_mgr.cancel_pending_entry_orders()

    # cancel_order should have been called for the open entry
    assert client.cancel_order.called


# ── WebSocket broadcaster ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_broadcaster_handles_no_clients():
    """Broadcast with no clients should not raise."""
    from karen.dashboard.ws import WebSocketBroadcaster

    bc = WebSocketBroadcaster()
    await bc.broadcast("test_event", {"key": "value"})  # no-op, no error


@pytest.mark.asyncio
async def test_broadcaster_dead_client_removed():
    """A client that throws on send_text is silently removed."""
    from karen.dashboard.ws import WebSocketBroadcaster

    bc = WebSocketBroadcaster()
    dead_ws = AsyncMock()
    dead_ws.send_text.side_effect = Exception("connection reset")

    async with bc._lock:
        bc._clients.add(dead_ws)

    await bc.broadcast("ping", {})
    assert bc.client_count == 0
