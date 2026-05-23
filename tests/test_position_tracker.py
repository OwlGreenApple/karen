"""Tests for PositionTracker — in-memory SQLite, mocked BinanceClient."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from karen.exchange.binance_client import OrderInfo, PositionInfo
from karen.execution.position_tracker import PositionTracker
from karen.persistence.models import Base, CloseReason, Position, Trade, TradeSide, TradeStatus

# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
async def db_factory():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


def _mock_client(positions: list[PositionInfo] | None = None) -> MagicMock:
    c = MagicMock()
    c.fetch_positions = AsyncMock(return_value=positions or [])
    c.fetch_equity = AsyncMock(return_value=10_000.0)
    c.fetch_order = AsyncMock(return_value=_make_order(status="open"))
    return c


def _make_order(status: str = "open", price: float | None = None) -> OrderInfo:
    return OrderInfo(
        id="ord-1", client_order_id="k-1", symbol="BTCUSDT",
        side="sell", type="stop_market", status=status,
        quantity=0.1, filled=0.0 if status == "open" else 0.1,
        price=price, stop_price=39_500.0,
        timestamp=datetime.now(tz=UTC),
    )


def _make_binance_position(
    symbol: str = "BTCUSDT",
    side: str = "long",
    entry: float = 40_000.0,
    qty: float = 0.1,
    upnl: float = 50.0,
    mark: float = 40_500.0,
    leverage: int = 5,
) -> PositionInfo:
    return PositionInfo(
        symbol=symbol, side=side, entry_price=entry,
        quantity=qty, leverage=leverage,
        unrealized_pnl=upnl, mark_price=mark,
        notional=qty * entry,
    )


async def _insert_trade(
    db_factory,
    symbol: str = "BTCUSDT",
    side: TradeSide = TradeSide.LONG,
    status: TradeStatus = TradeStatus.OPEN,
    entry: float = 40_000.0,
    qty: float = 0.1,
    sl_order_id: str | None = "sl-1",
    tp1_order_id: str | None = "tp1-1",
) -> Trade:
    trade = Trade(
        symbol=symbol,
        side=side,
        strategy_mode="mean_reversion",
        status=status,
        entry_price=entry,
        quantity=qty,
        leverage=5,
        sl_price=39_500.0,
        tp1_price=40_500.0,
        sl_order_id=sl_order_id,
        tp1_order_id=tp1_order_id,
        opened_at=datetime.now(tz=UTC),
    )
    async with db_factory() as session:
        session.add(trade)
        await session.commit()
        await session.refresh(trade)
    return trade


@pytest.fixture
async def tracker(default_settings, db_factory):
    client = _mock_client()
    return PositionTracker(client, default_settings, session_factory=db_factory)


# ─── Reconcile: closed externally ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reconcile_detects_position_closed_by_sl(
    default_settings, db_factory
):
    """Position in DB but not on Binance → mark as closed, determine SL reason."""
    await _insert_trade(db_factory)

    # SL order is filled → reason = STOP_LOSS
    client = _mock_client(positions=[])  # no Binance positions
    client.fetch_order.return_value = _make_order(status="closed", price=39_500.0)

    tracker = PositionTracker(client, default_settings, session_factory=db_factory)
    await tracker._reconcile()

    async with db_factory() as session:
        result = await session.execute(select(Trade))
        trade = result.scalar_one()

    assert trade.status == TradeStatus.CLOSED
    assert trade.close_reason == CloseReason.STOP_LOSS
    assert trade.exit_price == pytest.approx(39_500.0)
    assert trade.realized_pnl == pytest.approx(-50.0)  # (39500 - 40000) × 0.1


@pytest.mark.asyncio
async def test_reconcile_detects_position_closed_by_tp(
    default_settings, db_factory
):
    """TP1 filled → close reason is TAKE_PROFIT."""
    await _insert_trade(db_factory)

    client = _mock_client(positions=[])
    # SL order still open, TP1 filled
    def order_side_effect(symbol, order_id):
        if order_id == "sl-1":
            return _make_order(status="open")
        return _make_order(status="closed", price=40_500.0)

    client.fetch_order.side_effect = order_side_effect

    tracker = PositionTracker(client, default_settings, session_factory=db_factory)
    await tracker._reconcile()

    async with db_factory() as session:
        result = await session.execute(select(Trade))
        trade = result.scalar_one()

    assert trade.status == TradeStatus.CLOSED
    assert trade.close_reason == CloseReason.TAKE_PROFIT


@pytest.mark.asyncio
async def test_reconcile_marks_manual_when_no_order_filled(
    default_settings, db_factory
):
    """Position gone, no filled orders → reason = MANUAL."""
    await _insert_trade(db_factory)

    client = _mock_client(positions=[])
    client.fetch_order.return_value = _make_order(status="open")

    tracker = PositionTracker(client, default_settings, session_factory=db_factory)
    await tracker._reconcile()

    async with db_factory() as session:
        result = await session.execute(select(Trade))
        trade = result.scalar_one()

    assert trade.close_reason == CloseReason.MANUAL


# ─── Reconcile: still open ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reconcile_leaves_open_trade_when_position_exists(
    default_settings, db_factory
):
    """Position still on Binance → trade stays open in DB."""
    await _insert_trade(db_factory)
    bp = _make_binance_position(upnl=100.0)
    client = _mock_client(positions=[bp])

    tracker = PositionTracker(client, default_settings, session_factory=db_factory)
    await tracker._reconcile()

    async with db_factory() as session:
        result = await session.execute(select(Trade))
        trade = result.scalar_one()

    assert trade.status == TradeStatus.OPEN


# ─── Reconcile: Position table sync ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_reconcile_creates_position_table_entry(default_settings, db_factory):
    bp = _make_binance_position(upnl=50.0, mark=40_500.0)
    client = _mock_client(positions=[bp])

    tracker = PositionTracker(client, default_settings, session_factory=db_factory)
    await tracker._reconcile()

    async with db_factory() as session:
        result = await session.execute(select(Position))
        positions = result.scalars().all()

    assert len(positions) == 1
    assert positions[0].symbol == "BTCUSDT"
    assert positions[0].unrealized_pnl == pytest.approx(50.0)
    assert positions[0].mark_price == pytest.approx(40_500.0)


@pytest.mark.asyncio
async def test_reconcile_removes_closed_position_from_table(
    default_settings, db_factory
):
    """When Binance position disappears, Position table entry is removed."""
    # Pre-insert a Position row
    async with db_factory() as session:
        pos = Position(
            symbol="BTCUSDT",
            side=TradeSide.LONG,
            entry_price=40_000.0,
            quantity=0.1,
            leverage=5,
            unrealized_pnl=50.0,
            mark_price=40_500.0,
        )
        session.add(pos)
        await session.commit()

    client = _mock_client(positions=[])  # Binance has no positions
    tracker = PositionTracker(client, default_settings, session_factory=db_factory)
    await tracker._reconcile()

    async with db_factory() as session:
        result = await session.execute(select(Position))
        positions = result.scalars().all()

    assert len(positions) == 0


# ─── Startup reconciliation ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_startup_reconcile_creates_orphan_trade(default_settings, db_factory):
    """Position on Binance but no DB trade → create orphan record."""
    bp = _make_binance_position()
    client = _mock_client(positions=[bp])

    tracker = PositionTracker(client, default_settings, session_factory=db_factory)
    await tracker.startup_reconcile()

    async with db_factory() as session:
        result = await session.execute(select(Trade))
        trades = result.scalars().all()

    assert len(trades) == 1
    assert trades[0].symbol == "BTCUSDT"
    assert trades[0].strategy_mode == "unknown"
    assert trades[0].status == TradeStatus.OPEN


@pytest.mark.asyncio
async def test_startup_reconcile_marks_closed_if_not_on_binance(
    default_settings, db_factory
):
    """Trade in DB but position not on Binance at startup → mark closed."""
    await _insert_trade(db_factory)
    client = _mock_client(positions=[])
    client.fetch_order.return_value = _make_order(status="closed", price=39_500.0)

    tracker = PositionTracker(client, default_settings, session_factory=db_factory)
    await tracker.startup_reconcile()

    async with db_factory() as session:
        result = await session.execute(select(Trade))
        trade = result.scalar_one()

    assert trade.status == TradeStatus.CLOSED


# ─── get_open_position_count ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_open_position_count(default_settings, db_factory):
    await _insert_trade(db_factory, symbol="BTCUSDT")
    await _insert_trade(db_factory, symbol="ETHUSDT")
    await _insert_trade(db_factory, symbol="DOTUSDT", status=TradeStatus.CLOSED)

    client = _mock_client()
    tracker = PositionTracker(client, default_settings, session_factory=db_factory)

    count = await tracker.get_open_position_count()
    assert count == 2


# ─── Equity snapshots ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_force_snapshot_saves_to_db(default_settings, db_factory):
    from karen.persistence.models import EquitySnapshot

    client = _mock_client()
    client.fetch_equity.return_value = 10_500.0
    tracker = PositionTracker(client, default_settings, session_factory=db_factory)

    await tracker.force_snapshot("hourly")

    async with db_factory() as session:
        result = await session.execute(select(EquitySnapshot))
        snaps = result.scalars().all()

    assert len(snaps) == 1
    assert snaps[0].equity_usdt == pytest.approx(10_500.0)
    assert snaps[0].interval == "hourly"
