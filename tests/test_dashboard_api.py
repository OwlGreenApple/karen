"""Tests for dashboard FastAPI endpoints — in-memory SQLite, httpx AsyncClient."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from karen.dashboard.api import app, init_dashboard
from karen.persistence.models import (
    Base,
    CloseReason,
    EquitySnapshot,
    Position,
    Trade,
    TradeSide,
    TradeStatus,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
async def _patch_session(monkeypatch):
    """Replace AsyncSessionLocal with an in-memory SQLite factory for each test."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    import karen.dashboard.api as api_module
    monkeypatch.setattr(api_module, "AsyncSessionLocal", factory)
    yield factory
    await engine.dispose()


@pytest.fixture
async def client():
    init_dashboard(equity=10_000.0, mode="mean_reversion", trading_enabled=False)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def _insert_trade(factory, **kwargs) -> Trade:
    defaults = dict(
        symbol="BTCUSDT",
        side=TradeSide.LONG,
        strategy_mode="mean_reversion",
        status=TradeStatus.OPEN,
        entry_price=40_000.0,
        quantity=0.1,
        leverage=5,
        sl_price=39_500.0,
        tp1_price=40_500.0,
        opened_at=datetime.now(tz=UTC),
    )
    defaults.update(kwargs)
    trade = Trade(**defaults)
    async with factory() as session:
        session.add(trade)
        await session.commit()
        await session.refresh(trade)
    return trade


async def _insert_position(factory, **kwargs) -> Position:
    defaults = dict(
        symbol="BTCUSDT",
        side=TradeSide.LONG,
        entry_price=40_000.0,
        quantity=0.1,
        leverage=5,
        unrealized_pnl=50.0,
        mark_price=40_500.0,
    )
    defaults.update(kwargs)
    pos = Position(**defaults)
    async with factory() as session:
        session.add(pos)
        await session.commit()
    return pos


async def _insert_snapshot(factory, equity=10_000.0, interval="hourly") -> EquitySnapshot:
    snap = EquitySnapshot(
        timestamp=datetime.now(tz=UTC),
        equity_usdt=equity,
        unrealized_pnl=0.0,
        interval=interval,
    )
    async with factory() as session:
        session.add(snap)
        await session.commit()
    return snap


# ── /api/status ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_status_returns_initialized_state(client):
    r = await client.get("/api/status")
    assert r.status_code == 200
    data = r.json()
    assert data["equity_usdt"] == pytest.approx(10_000.0)
    assert data["strategy_mode"] == "mean_reversion"
    assert data["trading_enabled"] is False


# ── /api/trades ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_trades_empty(client):
    r = await client.get("/api/trades")
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 0
    assert data["trades"] == []


@pytest.mark.asyncio
async def test_trades_returns_inserted_trade(_patch_session, client):
    await _insert_trade(_patch_session)
    r = await client.get("/api/trades")
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 1
    trade = data["trades"][0]
    assert trade["symbol"] == "BTCUSDT"
    assert trade["side"] == "long"
    assert trade["status"] == "open"


@pytest.mark.asyncio
async def test_trades_filter_open_only(_patch_session, client):
    await _insert_trade(_patch_session, symbol="BTCUSDT", status=TradeStatus.OPEN)
    await _insert_trade(
        _patch_session,
        symbol="ETHUSDT",
        status=TradeStatus.CLOSED,
        close_reason=CloseReason.STOP_LOSS,
        exit_price=39_500.0,
        realized_pnl=-50.0,
        realized_pnl_pct=-10.0,
    )
    r = await client.get("/api/trades?status=open")
    data = r.json()
    assert data["total"] == 1
    assert data["trades"][0]["symbol"] == "BTCUSDT"


@pytest.mark.asyncio
async def test_trades_filter_closed_only(_patch_session, client):
    await _insert_trade(_patch_session, symbol="BTCUSDT", status=TradeStatus.OPEN)
    await _insert_trade(
        _patch_session,
        symbol="ETHUSDT",
        status=TradeStatus.CLOSED,
        close_reason=CloseReason.TAKE_PROFIT,
        exit_price=40_500.0,
        realized_pnl=50.0,
        realized_pnl_pct=10.0,
    )
    r = await client.get("/api/trades?status=closed")
    data = r.json()
    assert data["total"] == 1
    assert data["trades"][0]["symbol"] == "ETHUSDT"


@pytest.mark.asyncio
async def test_trades_pagination(_patch_session, client):
    for i in range(5):
        await _insert_trade(_patch_session, symbol=f"SYM{i}USDT")
    r = await client.get("/api/trades?limit=2&offset=0")
    data = r.json()
    assert data["total"] == 5
    assert len(data["trades"]) == 2


# ── /api/positions ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_positions_empty(client):
    r = await client.get("/api/positions")
    assert r.status_code == 200
    assert r.json() == []


@pytest.mark.asyncio
async def test_positions_returns_position(_patch_session, client):
    await _insert_position(_patch_session)
    r = await client.get("/api/positions")
    data = r.json()
    assert len(data) == 1
    assert data[0]["symbol"] == "BTCUSDT"
    assert data[0]["unrealized_pnl"] == pytest.approx(50.0)
    assert data[0]["mark_price"] == pytest.approx(40_500.0)


# ── /api/equity ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_equity_empty(client):
    r = await client.get("/api/equity")
    assert r.status_code == 200
    assert r.json() == []


@pytest.mark.asyncio
async def test_equity_returns_snapshots(_patch_session, client):
    await _insert_snapshot(_patch_session, equity=10_000.0, interval="hourly")
    await _insert_snapshot(_patch_session, equity=10_100.0, interval="hourly")
    r = await client.get("/api/equity?interval=hourly")
    data = r.json()
    assert len(data) == 2
    # Returned in ascending order
    assert data[0]["equity_usdt"] <= data[1]["equity_usdt"]


@pytest.mark.asyncio
async def test_equity_interval_filter(_patch_session, client):
    await _insert_snapshot(_patch_session, interval="hourly")
    await _insert_snapshot(_patch_session, interval="daily")
    r = await client.get("/api/equity?interval=daily")
    data = r.json()
    assert len(data) == 1
