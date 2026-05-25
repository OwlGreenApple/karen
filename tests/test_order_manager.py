"""Tests for OrderManager — in-memory SQLite, mocked BinanceClient."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from karen.config import StrategyMode
from karen.exchange.binance_client import OrderInfo
from karen.exchange.websocket import Kline
from karen.execution.order_manager import OrderManager, _coid, _compute_pnl
from karen.persistence.models import Base, CloseReason, Trade, TradeSide, TradeStatus
from karen.strategies.base import Signal

# ─── In-memory DB fixture ─────────────────────────────────────────────────────


@pytest.fixture
async def db_factory():
    """In-memory SQLite async session factory, fresh per test."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


# ─── Mock client fixture ──────────────────────────────────────────────────────


def _order(
    oid: str = "ord-1",
    coid: str = "karen-e-xxx",
    status: str = "closed",
    filled: float = 0.01,
    price: float = 40_000.0,
) -> OrderInfo:
    return OrderInfo(
        id=oid,
        client_order_id=coid,
        symbol="BTCUSDT",
        side="buy",
        type="market",
        status=status,
        quantity=0.01,
        filled=filled,
        price=price,
        stop_price=None,
        timestamp=datetime.now(tz=UTC),
    )


@pytest.fixture
def mock_client() -> MagicMock:
    c = MagicMock()
    c.place_order = AsyncMock(return_value=_order())
    c.cancel_order = AsyncMock()
    c.cancel_all_orders = AsyncMock()
    c.fetch_order = AsyncMock(return_value=_order(status="open"))
    c.fetch_open_orders = AsyncMock(return_value=[])
    return c


@pytest.fixture
async def order_mgr(default_settings, mock_client, db_factory):
    return OrderManager(mock_client, default_settings, session_factory=db_factory)


# ─── Signal helpers ───────────────────────────────────────────────────────────


def mr_long_signal(
    symbol: str = "BTCUSDT",
    entry: float = 40_000.0,
    sl: float = 39_500.0,
    tp1: float = 40_500.0,
    tp2: float = 41_000.0,
) -> Signal:
    return Signal(
        symbol=symbol,
        side="long",
        strategy_mode=StrategyMode.MEAN_REVERSION,
        entry_price=entry,
        sl_price=sl,
        tp1_price=tp1,
        tp1_qty_pct=0.70,
        tp2_price=tp2,
        atr=200.0,
    )


def tf_long_signal(
    symbol: str = "BTCUSDT",
    entry: float = 40_000.0,
    sl: float = 39_200.0,
    tp1: float = 40_800.0,
) -> Signal:
    return Signal(
        symbol=symbol,
        side="long",
        strategy_mode=StrategyMode.TREND_FOLLOWING,
        entry_price=entry,
        sl_price=sl,
        tp1_price=tp1,
        tp1_qty_pct=0.50,
        tp2_price=None,  # trailing stop
        atr=250.0,
    )


# ─── _coid ───────────────────────────────────────────────────────────────────


def test_coid_format():
    c = _coid("sl", "BTCUSDT")
    assert c.startswith("karen-sl-btc-")
    assert len(c) == len("karen-sl-btc-") + 8


def test_coid_is_unique():
    ids = {_coid("e", "BTCUSDT") for _ in range(50)}
    assert len(ids) == 50


# ─── _compute_pnl ─────────────────────────────────────────────────────────────


def test_compute_pnl_long_profit():
    trade = Trade(side=TradeSide.LONG, entry_price=40_000.0, quantity=0.1)
    assert _compute_pnl(trade, 41_000.0) == pytest.approx(100.0)


def test_compute_pnl_long_loss():
    trade = Trade(side=TradeSide.LONG, entry_price=40_000.0, quantity=0.1)
    assert _compute_pnl(trade, 39_000.0) == pytest.approx(-100.0)


def test_compute_pnl_short_profit():
    trade = Trade(side=TradeSide.SHORT, entry_price=40_000.0, quantity=0.1)
    assert _compute_pnl(trade, 39_000.0) == pytest.approx(100.0)


def test_compute_pnl_short_loss():
    trade = Trade(side=TradeSide.SHORT, entry_price=40_000.0, quantity=0.1)
    assert _compute_pnl(trade, 41_000.0) == pytest.approx(-100.0)


# ─── open_position ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_open_position_mr_places_four_orders(order_mgr, mock_client):
    """MR strategy: entry + SL + TP1 + TP2."""
    signal = mr_long_signal()
    trade = await order_mgr.open_position(signal, equity=10_000.0)

    assert trade is not None
    assert trade.status == TradeStatus.OPEN
    assert trade.symbol == "BTCUSDT"
    assert trade.side == TradeSide.LONG

    # entry + SL + TP1 + TP2 = 4 place_order calls
    assert mock_client.place_order.call_count == 4

    # Verify order types passed
    calls = [c[1]["order_type"] for c in mock_client.place_order.call_args_list]
    assert "market" in calls
    assert "stop_market" in calls
    calls_lower = [t.lower() for t in calls]
    assert sum(1 for c in calls_lower if "take_profit" in c) == 2


@pytest.mark.asyncio
async def test_open_position_tf_places_three_orders(order_mgr, mock_client):
    """TF strategy: entry + SL + TP1 (no TP2 — trailing stop handled separately)."""
    signal = tf_long_signal()
    trade = await order_mgr.open_position(signal, equity=10_000.0)

    assert trade is not None
    assert mock_client.place_order.call_count == 3

    calls = [c[1]["order_type"] for c in mock_client.place_order.call_args_list]
    assert calls[0] == "market"
    assert "stop_market" in calls
    assert sum(1 for c in calls if "take_profit" in c.lower()) == 1


@pytest.mark.asyncio
async def test_open_position_uses_reduce_only_for_sl_tp(order_mgr, mock_client):
    signal = mr_long_signal()
    await order_mgr.open_position(signal, equity=10_000.0)

    # All orders except the entry must be reduce_only
    for call in mock_client.place_order.call_args_list[1:]:
        assert call[1].get("reduce_only") is True


@pytest.mark.asyncio
async def test_open_position_sl_covers_full_qty(order_mgr, mock_client, default_settings):
    """SL order must cover 100% of quantity (reduce_only on Binance handles partial)."""
    signal = mr_long_signal()
    trade = await order_mgr.open_position(signal, equity=10_000.0)

    # Find the SL call (stop_market)
    sl_call = next(
        c for c in mock_client.place_order.call_args_list
        if c[1]["order_type"] == "stop_market"
    )
    assert sl_call[1]["quantity"] == pytest.approx(trade.quantity)


@pytest.mark.asyncio
async def test_open_position_tp1_covers_70pct(order_mgr, mock_client):
    signal = mr_long_signal(tp1=40_500.0, tp2=41_000.0)
    signal.tp1_qty_pct = 0.70
    trade = await order_mgr.open_position(signal, equity=10_000.0)

    tp_calls = [
        c for c in mock_client.place_order.call_args_list
        if "take_profit" in c[1]["order_type"].lower()
    ]
    tp1_call = tp_calls[0]  # first TP call
    assert tp1_call[1]["quantity"] == pytest.approx(trade.quantity * 0.70, rel=1e-3)


@pytest.mark.asyncio
async def test_open_position_uses_karen_client_order_ids(order_mgr, mock_client):
    signal = mr_long_signal()
    await order_mgr.open_position(signal, equity=10_000.0)

    for call in mock_client.place_order.call_args_list:
        coid = call[1]["client_order_id"]
        assert coid.startswith("karen-")


@pytest.mark.asyncio
async def test_open_position_saves_order_ids_to_db(order_mgr, mock_client, db_factory):
    # Return distinct IDs for each call
    mock_client.place_order.side_effect = [
        _order(oid="entry-1"),
        _order(oid="sl-1"),
        _order(oid="tp1-1"),
        _order(oid="tp2-1"),
    ]
    signal = mr_long_signal()
    trade = await order_mgr.open_position(signal, equity=10_000.0)

    from sqlalchemy import select as sa_select

    async with db_factory() as session:
        result = await session.execute(sa_select(Trade).where(Trade.id == trade.id))
        db_trade = result.scalar_one()

    assert db_trade.entry_order_id == "entry-1"
    assert db_trade.sl_order_id == "sl-1"
    assert db_trade.tp1_order_id == "tp1-1"
    assert db_trade.tp2_order_id == "tp2-1"


@pytest.mark.asyncio
async def test_open_position_returns_none_on_entry_failure(order_mgr, mock_client):
    mock_client.place_order.side_effect = Exception("network error")
    signal = mr_long_signal()
    trade = await order_mgr.open_position(signal, equity=10_000.0)
    assert trade is None


@pytest.mark.asyncio
async def test_open_position_continues_if_sl_fails(order_mgr, mock_client):
    """Entry succeeds but SL fails → trade still saved (unprotected but alive)."""
    mock_client.place_order.side_effect = [
        _order(oid="entry-1"),    # entry OK
        Exception("sl failed"),   # SL fails
        _order(oid="tp1-1"),      # TP1 OK
    ]
    # Remove TP2 by using tf_long_signal (no TP2)
    signal = tf_long_signal()
    trade = await order_mgr.open_position(signal, equity=10_000.0)
    assert trade is not None
    assert trade.entry_order_id == "entry-1"


@pytest.mark.asyncio
async def test_open_position_blocks_below_min_notional(order_mgr, mock_client, default_settings):
    """Position with notional < $5 should be skipped."""
    # Very tight SL → very large qty × very low entry price → tiny notional
    signal = mr_long_signal(entry=1.0, sl=0.999, tp1=1.001, tp2=1.002)
    # equity × 1% / sl_dist = 100 / 0.001 = 100,000 units → notional = 100,000 × $1 = $100k
    # Actually this won't be below min notional... let me use a different approach
    # Instead: mock compute_quantity to return a tiny qty
    with patch("karen.execution.order_manager.compute_quantity_by_margin_pct", return_value=0.000001):
        trade = await order_mgr.open_position(signal, equity=10_000.0)
    assert trade is None
    mock_client.place_order.assert_not_called()


# ─── close_position ───────────────────────────────────────────────────────────


@pytest.fixture
async def open_trade(db_factory):
    """A pre-existing open trade in the in-memory DB."""
    trade = Trade(
        symbol="BTCUSDT",
        side=TradeSide.LONG,
        strategy_mode="mean_reversion",
        status=TradeStatus.OPEN,
        entry_price=40_000.0,
        quantity=0.1,
        leverage=5,
        sl_price=39_500.0,
        tp1_price=40_500.0,
        tp2_price=41_000.0,
        entry_order_id="entry-open-1",
        sl_order_id="sl-1",
        tp1_order_id="tp1-1",
        tp2_order_id="tp2-1",
        opened_at=datetime.now(tz=UTC),
    )
    async with db_factory() as session:
        session.add(trade)
        await session.commit()
        await session.refresh(trade)
    return trade


@pytest.mark.asyncio
async def test_close_position_places_market_order(order_mgr, mock_client, open_trade):
    await order_mgr.close_position(open_trade, CloseReason.TIME_STOP, exit_price=40_200.0)

    # First place_order should be a market sell
    call = mock_client.place_order.call_args
    assert call[1]["order_type"] == "market"
    assert call[1]["side"] == "sell"
    assert call[1]["reduce_only"] is True


@pytest.mark.asyncio
async def test_close_position_cancels_sl_and_tp_orders(order_mgr, mock_client, open_trade):
    await order_mgr.close_position(open_trade, CloseReason.MANUAL, exit_price=40_000.0)

    cancelled_ids = {c.args[1] for c in mock_client.cancel_order.call_args_list}
    assert "sl-1" in cancelled_ids
    assert "tp1-1" in cancelled_ids
    assert "tp2-1" in cancelled_ids


@pytest.mark.asyncio
async def test_close_position_marks_trade_closed_in_db(
    order_mgr, mock_client, open_trade, db_factory
):
    await order_mgr.close_position(open_trade, CloseReason.STOP_LOSS, exit_price=39_500.0)

    from sqlalchemy import select as sa_select

    async with db_factory() as session:
        result = await session.execute(sa_select(Trade).where(Trade.id == open_trade.id))
        t = result.scalar_one()

    assert t.status == TradeStatus.CLOSED
    assert t.exit_price == pytest.approx(39_500.0)
    assert t.close_reason == CloseReason.STOP_LOSS
    assert t.closed_at is not None
    assert t.realized_pnl == pytest.approx(-50.0)  # (39500 - 40000) × 0.1


# ─── Time stop ────────────────────────────────────────────────────────────────


def _make_kline(symbol: str = "BTCUSDT", close: float = 40_000.0) -> Kline:
    return Kline(
        symbol=symbol, timeframe="15m", open_time=0,
        open=close, high=close + 50, low=close - 50, close=close, volume=100.0,
        is_closed=True,
    )


@pytest.mark.asyncio
async def test_time_stop_fires_after_8_candles_no_profit(
    order_mgr, mock_client, open_trade, db_factory
):
    """MR trade at 8 candles with price below entry → time stop."""
    # Pre-set candles_open to 8 (we call _check_time_stop directly, bypassing the increment)
    async with db_factory() as session:
        t = await session.get(Trade, open_trade.id)
        t.candles_open = 8
        t.strategy_mode = "mean_reversion"
        await session.commit()

    # Refresh detached object
    async with db_factory() as session:
        trade = await session.get(Trade, open_trade.id)

    kline = _make_kline(close=39_800.0)  # below entry (no profit)
    # Manually call the time stop check
    await order_mgr._check_time_stop(trade, kline, 8)

    # close_position should have been called (market order placed)
    mock_client.place_order.assert_called_once()
    assert mock_client.place_order.call_args[1]["order_type"] == "market"


@pytest.mark.asyncio
async def test_time_stop_does_not_fire_if_in_profit(
    order_mgr, mock_client, open_trade, db_factory
):
    """MR trade at 8 candles but price above entry → no time stop."""
    async with db_factory() as session:
        t = await session.get(Trade, open_trade.id)
        t.candles_open = 8
        t.strategy_mode = "mean_reversion"
        await session.commit()

    async with db_factory() as session:
        trade = await session.get(Trade, open_trade.id)

    kline = _make_kline(close=40_200.0)  # above entry (in profit)
    await order_mgr._check_time_stop(trade, kline, 8)
    mock_client.place_order.assert_not_called()


@pytest.mark.asyncio
async def test_time_stop_does_not_fire_before_8_candles(
    order_mgr, mock_client, open_trade, db_factory
):
    async with db_factory() as session:
        t = await session.get(Trade, open_trade.id)
        t.candles_open = 5
        await session.commit()

    async with db_factory() as session:
        trade = await session.get(Trade, open_trade.id)

    kline = _make_kline(close=39_500.0)
    await order_mgr._check_time_stop(trade, kline, 8)
    mock_client.place_order.assert_not_called()


# ─── cancel_pending_entry_orders ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cancel_pending_entries_cancels_open_entry_orders(
    order_mgr, mock_client, open_trade
):
    """Cancels entry order when it's still open (mode switch)."""
    mock_client.fetch_order.return_value = _order(status="open")

    await order_mgr.cancel_pending_entry_orders()

    mock_client.cancel_order.assert_called_once_with(
        open_trade.symbol, "entry-open-1"
    )


@pytest.mark.asyncio
async def test_cancel_pending_entries_skips_filled_orders(order_mgr, mock_client, open_trade):
    """Does not cancel entry order that is already filled."""
    mock_client.fetch_order.return_value = _order(status="closed")
    open_trade.entry_order_id = "already-filled"

    await order_mgr.cancel_pending_entry_orders()

    mock_client.cancel_order.assert_not_called()


# ─── get_open_symbol_set ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_open_symbol_set_returns_open_symbols(order_mgr, open_trade):
    symbols = await order_mgr.get_open_symbol_set()
    assert "BTCUSDT" in symbols


@pytest.mark.asyncio
async def test_get_open_symbol_set_excludes_closed_trades(
    order_mgr, open_trade, db_factory
):
    async with db_factory() as session:
        t = await session.get(Trade, open_trade.id)
        t.status = TradeStatus.CLOSED
        await session.commit()

    symbols = await order_mgr.get_open_symbol_set()
    assert "BTCUSDT" not in symbols


# ─── Trailing stop ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_trailing_stop_updates_sl_when_closer_for_long(
    order_mgr, mock_client, open_trade, db_factory
):
    """For a LONG position, SL moves up (closer to price) = update."""
    # Set up a low current SL
    async with db_factory() as session:
        t = await session.get(Trade, open_trade.id)
        t.sl_price = 39_000.0
        t.sl_order_id = "old-sl"
        t.strategy_mode = "trend_following"
        await session.commit()

    async with db_factory() as session:
        trade = await session.get(Trade, open_trade.id)

    # Narrow-range klines at a high price → small ATR → high chandelier SL
    # high=40200, low=40100 → range=100 → ATR≈100
    # chandelier = max_high(22) - 3×ATR = 40200 - 300 = 39900
    # 39900 > old_sl(39000) → update triggered
    klines = [
        Kline("BTCUSDT", "15m", i, 40_100, 40_200, 40_100, 40_150, 100.0, True)
        for i in range(30)
    ]
    order_mgr._kline_buffer["BTCUSDT"] = klines

    kline = _make_kline(close=40_400.0)
    await order_mgr._check_trailing_stop(trade, kline)

    # Should have cancelled old SL and placed new one
    mock_client.cancel_order.assert_called_with("BTCUSDT", "old-sl")
    assert mock_client.place_order.call_count == 1
    new_sl_call = mock_client.place_order.call_args
    assert new_sl_call[1]["order_type"] == "stop_market"
    new_sl = new_sl_call[1]["stop_price"]
    assert new_sl > 39_000.0  # SL moved up (closer to price for long)


@pytest.mark.asyncio
async def test_trailing_stop_no_update_when_further_for_long(
    order_mgr, mock_client, open_trade, db_factory
):
    """For a LONG, if new chandelier SL is LOWER than current SL → don't update."""
    async with db_factory() as session:
        t = await session.get(Trade, open_trade.id)
        t.sl_price = 40_100.0  # already very close to current price
        t.strategy_mode = "trend_following"
        await session.commit()

    async with db_factory() as session:
        trade = await session.get(Trade, open_trade.id)

    # Build a kline buffer where chandelier will be well below 40100
    klines = [
        Kline("BTCUSDT", "15m", i, 38000, 38500, 37500, 38200, 100.0, True)
        for i in range(30)
    ]
    order_mgr._kline_buffer["BTCUSDT"] = klines

    kline = _make_kline(close=38_200.0)
    await order_mgr._check_trailing_stop(trade, kline)

    mock_client.cancel_order.assert_not_called()
    mock_client.place_order.assert_not_called()
