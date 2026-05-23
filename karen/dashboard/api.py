"""FastAPI dashboard backend — REST + WebSocket."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import desc, select

from karen.dashboard.ws import broadcaster
from karen.persistence.db import AsyncSessionLocal
from karen.persistence.models import EquitySnapshot, Position, Trade, TradeStatus

_STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Karen Dashboard", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Injected at startup by main.py
_state: dict[str, Any] = {
    "equity": 0.0,
    "mode": "unknown",
    "trading_enabled": False,
    "started_at": None,
    "paused": False,
}


def init_dashboard(equity: float, mode: str, trading_enabled: bool) -> None:
    _state["equity"] = equity
    _state["mode"] = mode
    _state["trading_enabled"] = trading_enabled
    _state["started_at"] = datetime.now(tz=UTC).isoformat()


def update_state(**kwargs: Any) -> None:
    _state.update(kwargs)


# ── REST endpoints ─────────────────────────────────────────────────────────────


@app.get("/api/status")
async def get_status() -> dict[str, Any]:
    return {
        "equity_usdt": _state["equity"],
        "strategy_mode": _state["mode"],
        "trading_enabled": _state["trading_enabled"],
        "paused": _state["paused"],
        "started_at": _state["started_at"],
        "ws_clients": broadcaster.client_count,
    }


@app.get("/api/trades")
async def get_trades(
    limit: int = 50,
    offset: int = 0,
    status: str | None = None,
) -> dict[str, Any]:
    async with AsyncSessionLocal() as session:
        query = select(Trade).order_by(desc(Trade.opened_at))
        if status == "open":
            query = query.where(Trade.status == TradeStatus.OPEN)
        elif status == "closed":
            query = query.where(Trade.status == TradeStatus.CLOSED)
        result = await session.execute(query.offset(offset).limit(limit))
        trades = result.scalars().all()

        count_q = select(Trade)
        if status == "open":
            count_q = count_q.where(Trade.status == TradeStatus.OPEN)
        elif status == "closed":
            count_q = count_q.where(Trade.status == TradeStatus.CLOSED)
        count_result = await session.execute(count_q)
        total = len(count_result.scalars().all())

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "trades": [_trade_to_dict(t) for t in trades],
    }


@app.get("/api/positions")
async def get_positions() -> list[dict[str, Any]]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Position))
        positions = result.scalars().all()
    return [_position_to_dict(p) for p in positions]


@app.get("/api/equity")
async def get_equity(interval: str = "hourly", limit: int = 168) -> list[dict[str, Any]]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(EquitySnapshot)
            .where(EquitySnapshot.interval == interval)
            .order_by(desc(EquitySnapshot.timestamp))
            .limit(limit)
        )
        snaps = result.scalars().all()
    # Return in ascending order for chart rendering
    return [
        {
            "timestamp": s.timestamp.isoformat(),
            "equity_usdt": s.equity_usdt,
            "unrealized_pnl": s.unrealized_pnl,
        }
        for s in reversed(snaps)
    ]


# ── WebSocket ──────────────────────────────────────────────────────────────────


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await broadcaster.connect(ws)
    try:
        while True:
            # Keep connection alive; client can send pings
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await broadcaster.disconnect(ws)


# ── SPA fallback (serve index.html for all non-API routes) ────────────────────

if _STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=_STATIC_DIR / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str) -> FileResponse:
        index = _STATIC_DIR / "index.html"
        return FileResponse(str(index))


# ── Helpers ────────────────────────────────────────────────────────────────────


def _trade_to_dict(t: Trade) -> dict[str, Any]:
    return {
        "id": t.id,
        "symbol": t.symbol,
        "side": t.side.value,
        "strategy_mode": t.strategy_mode,
        "status": t.status.value,
        "entry_price": t.entry_price,
        "exit_price": t.exit_price,
        "quantity": t.quantity,
        "leverage": t.leverage,
        "sl_price": t.sl_price,
        "tp1_price": t.tp1_price,
        "tp2_price": t.tp2_price,
        "realized_pnl": t.realized_pnl,
        "realized_pnl_pct": t.realized_pnl_pct,
        "close_reason": t.close_reason.value if t.close_reason else None,
        "opened_at": t.opened_at.isoformat() if t.opened_at else None,
        "closed_at": t.closed_at.isoformat() if t.closed_at else None,
        "candles_open": t.candles_open,
    }


def _position_to_dict(p: Position) -> dict[str, Any]:
    return {
        "symbol": p.symbol,
        "side": p.side.value,
        "entry_price": p.entry_price,
        "quantity": p.quantity,
        "leverage": p.leverage,
        "unrealized_pnl": p.unrealized_pnl,
        "mark_price": p.mark_price,
        "updated_at": p.updated_at.isoformat() if p.updated_at else None,
    }
