"""
Reconciles local DB state with live Binance positions every 30s.

Responsibilities:
- Detect positions that were closed externally (SL/TP hit)
- Update unrealized PnL and mark price for open positions
- Snapshot equity hourly/daily for the dashboard chart
- Startup reconciliation: sync DB with actual Binance state
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from karen.config import Settings
from karen.exchange.binance_client import BinanceClient, PositionInfo
from karen.persistence.db import AsyncSessionLocal
from karen.persistence.models import (
    CloseReason,
    EquitySnapshot,
    Position,
    Trade,
    TradeSide,
    TradeStatus,
)

_RECONCILE_INTERVAL = 30  # seconds
_HOURLY_SNAPSHOT_INTERVAL = 3600
_DAILY_SNAPSHOT_INTERVAL = 86400


class PositionTracker:
    """Keeps the local DB in sync with live Binance position state."""

    def __init__(
        self,
        client: BinanceClient,
        settings: Settings,
        session_factory: async_sessionmaker[AsyncSession] = AsyncSessionLocal,
        notifier: Any = None,
    ) -> None:
        self._client = client
        self._settings = settings
        self._session_factory = session_factory
        self._notifier = notifier
        self._last_hourly_snapshot: float = 0.0
        self._last_daily_snapshot: float = 0.0

    def update_settings(self, settings: Settings) -> None:
        self._settings = settings

    # ── Main loop ──────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Reconciliation loop — runs every 30 seconds."""
        logger.info("PositionTracker started")
        while True:
            try:
                await self._reconcile()
                await self._maybe_snapshot_equity()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("PositionTracker reconcile error")
            await asyncio.sleep(_RECONCILE_INTERVAL)

    # ── Startup reconciliation ─────────────────────────────────────────────

    async def startup_reconcile(self) -> None:
        """
        Called once on bot startup.

        Compares DB open trades against actual Binance positions.
        - Positions on Binance but not in DB → create orphaned position record
        - Trades in DB with no matching Binance position → mark as closed
        """
        logger.info("Running startup reconciliation...")
        try:
            binance_positions = await self._client.fetch_positions()
            await self._reconcile_inner(binance_positions, is_startup=True)
            logger.info("Startup reconciliation complete")
        except Exception:
            logger.exception("Startup reconciliation failed — continuing with DB state")

    # ── Core reconciliation ────────────────────────────────────────────────

    async def _reconcile(self) -> None:
        binance_positions = await self._client.fetch_positions()
        await self._reconcile_inner(binance_positions)

    async def _reconcile_inner(
        self, binance_positions: list[PositionInfo], is_startup: bool = False
    ) -> None:
        binance_map: dict[str, PositionInfo] = {p.symbol: p for p in binance_positions}

        async with self._session_factory() as session:
            result = await session.execute(
                select(Trade).where(Trade.status == TradeStatus.OPEN)
            )
            open_trades: list[Trade] = list(result.scalars().all())

        # ── Update open positions ─────────────────────────────────────────
        for trade in open_trades:
            bp = binance_map.get(trade.symbol)

            if bp is None:
                # Grace period for Karen-opened trades only (orphans have no entry_order_id)
                # Binance API can lag up to ~30s after a market order fills — avoid false closes
                if trade.entry_order_id is not None:
                    opened = trade.opened_at
                    if opened.tzinfo is None:
                        opened = opened.replace(tzinfo=UTC)
                    age_s = (datetime.now(tz=UTC) - opened).total_seconds()
                    if age_s < 60:
                        logger.debug(
                            f"Trade #{trade.id} not on Binance yet ({age_s:.0f}s old) — waiting"
                        )
                        continue

                # Position gone from Binance → closed externally
                reason = await self._determine_close_reason(trade)
                exit_price = await self._estimate_exit_price(trade)
                await self._mark_trade_closed(trade, reason, exit_price)
                logger.info(
                    f"Trade #{trade.id} detected as closed externally "
                    f"({reason.value}) exit≈{exit_price:.2f}"
                )
            else:
                # Update unrealized PnL and mark price
                await self._update_position_live_data(trade, bp)

        # ── Handle positions on Binance not in DB (orphans) ───────────────
        db_symbols = {t.symbol for t in open_trades}
        for symbol, bp in binance_map.items():
            if symbol not in db_symbols:
                logger.warning(
                    f"Binance position not in DB: {symbol} {bp.side} "
                    f"qty={bp.quantity:.6f} — registering as orphan"
                )
                await self._create_orphan_trade(bp)

        # ── Update Position table (live mirror) ───────────────────────────
        await self._sync_position_table(binance_map)

    async def _determine_close_reason(self, trade: Trade) -> CloseReason:
        """Check which order filled to infer close reason."""
        for order_id, reason in [
            (trade.sl_order_id, CloseReason.STOP_LOSS),
            (trade.tp2_order_id, CloseReason.TAKE_PROFIT),
            (trade.tp1_order_id, CloseReason.TAKE_PROFIT),
        ]:
            if order_id:
                try:
                    order = await self._client.fetch_order(trade.symbol, order_id)
                    if order.status == "closed":
                        return reason
                except Exception:
                    pass
        return CloseReason.MANUAL

    async def _estimate_exit_price(self, trade: Trade) -> float:
        """Best-effort exit price: filled orders → trade history → entry price."""
        for order_id in (trade.sl_order_id, trade.tp2_order_id, trade.tp1_order_id):
            if order_id:
                try:
                    order = await self._client.fetch_order(trade.symbol, order_id)
                    if order.status == "closed":
                        # stop_market/take_profit_market orders have no limit price;
                        # actual fill price is in average_price, fallback to stop_price
                        fill = order.average_price or order.price or order.stop_price
                        if fill:
                            return fill
                except Exception:
                    pass

        # Binance cancels unfilled reduce_only orders when a position closes,
        # so by the time reconciler runs, SL/TP orders may already be cancelled.
        # Use raw trade fill history as the reliable fallback.
        since_ms = int(trade.opened_at.timestamp() * 1000) if trade.opened_at else None
        price = await self._client.fetch_last_close_price(trade.symbol, since_ms)
        if price:
            logger.debug(
                f"Trade #{trade.id}: exit price {price} obtained from trade history"
            )
            return price

        logger.warning(
            f"Trade #{trade.id}: could not determine exit price — defaulting to entry price"
        )
        return trade.entry_price

    async def _mark_trade_closed(
        self, trade: Trade, reason: CloseReason, exit_price: float
    ) -> None:
        from karen.execution.order_manager import _compute_pnl

        pnl = _compute_pnl(trade, exit_price)
        margin = trade.entry_price * trade.quantity / trade.leverage
        pnl_pct = pnl / margin * 100 if margin else 0.0

        async with self._session_factory() as session:
            t = await session.get(Trade, trade.id)
            if t:
                t.status = TradeStatus.CLOSED
                t.exit_price = exit_price
                t.realized_pnl = pnl
                t.realized_pnl_pct = pnl_pct
                t.close_reason = reason
                t.closed_at = datetime.now(tz=UTC)
                await session.commit()

        if self._notifier:
            await self._notifier.trade_closed(
                symbol=trade.symbol,
                side=trade.side.value,
                entry_price=trade.entry_price,
                exit_price=exit_price,
                quantity=trade.quantity,
                pnl=pnl,
                pnl_pct=pnl_pct,
                reason=reason,
            )

    async def _update_position_live_data(self, trade: Trade, bp: PositionInfo) -> None:
        async with self._session_factory() as session:
            t = await session.get(Trade, trade.id)
            if t:
                # Update mark price tracking via Position table (handled separately)
                pass
            # Position table is the live-data home; Trade stores settled values

    async def _create_orphan_trade(self, bp: PositionInfo) -> None:
        """Create a minimal Trade record for a position found on Binance but not in DB."""
        trade = Trade(
            symbol=bp.symbol,
            side=TradeSide.LONG if bp.side == "long" else TradeSide.SHORT,
            strategy_mode="unknown",
            status=TradeStatus.OPEN,
            entry_price=bp.entry_price,
            quantity=bp.quantity,
            leverage=bp.leverage,
            sl_price=bp.entry_price * (0.98 if bp.side == "long" else 1.02),
            tp1_price=bp.entry_price * (1.02 if bp.side == "long" else 0.98),
            opened_at=datetime.now(tz=UTC),
        )
        async with self._session_factory() as session:
            session.add(trade)
            await session.commit()

    async def _sync_position_table(
        self, binance_map: dict[str, PositionInfo]
    ) -> None:
        """Keep Position table (live mirror) in sync with Binance."""
        async with self._session_factory() as session:
            # Remove positions that closed
            result = await session.execute(select(Position))
            db_positions: list[Position] = list(result.scalars().all())

            for pos in db_positions:
                if pos.symbol not in binance_map:
                    await session.delete(pos)

            # Upsert current positions
            for symbol, bp in binance_map.items():
                result2 = await session.execute(
                    select(Position).where(Position.symbol == symbol)
                )
                pos = result2.scalar_one_or_none()
                if pos is None:
                    pos = Position(symbol=symbol)
                    session.add(pos)
                pos.side = TradeSide.LONG if bp.side == "long" else TradeSide.SHORT
                pos.entry_price = bp.entry_price
                pos.quantity = bp.quantity
                pos.leverage = bp.leverage
                pos.unrealized_pnl = bp.unrealized_pnl
                pos.mark_price = bp.mark_price

            await session.commit()

    # ── Equity snapshots ───────────────────────────────────────────────────

    async def _maybe_snapshot_equity(self) -> None:
        import time

        now = time.monotonic()
        equity = await self._client.fetch_equity()
        unrealized = await self._total_unrealized_pnl()

        if now - self._last_hourly_snapshot >= _HOURLY_SNAPSHOT_INTERVAL:
            await self._save_snapshot(equity, unrealized, "hourly")
            self._last_hourly_snapshot = now

        if now - self._last_daily_snapshot >= _DAILY_SNAPSHOT_INTERVAL:
            await self._save_snapshot(equity, unrealized, "daily")
            self._last_daily_snapshot = now

    async def _save_snapshot(
        self, equity: float, unrealized_pnl: float, interval: str
    ) -> None:
        snap = EquitySnapshot(
            timestamp=datetime.now(tz=UTC),
            equity_usdt=equity,
            unrealized_pnl=unrealized_pnl,
            interval=interval,
        )
        async with self._session_factory() as session:
            session.add(snap)
            await session.commit()
        logger.debug(f"Equity snapshot ({interval}): ${equity:,.2f}")

    async def _total_unrealized_pnl(self) -> float:
        async with self._session_factory() as session:
            result = await session.execute(select(Position))
            positions: list[Position] = list(result.scalars().all())
        return sum(p.unrealized_pnl for p in positions)

    # ── Public helpers ─────────────────────────────────────────────────────

    async def get_open_position_count(self) -> int:
        async with self._session_factory() as session:
            result = await session.execute(
                select(Trade).where(Trade.status == TradeStatus.OPEN)
            )
            return len(result.scalars().all())

    async def force_snapshot(self, interval: str = "hourly") -> None:
        equity = await self._client.fetch_equity()
        unrealized = await self._total_unrealized_pnl()
        await self._save_snapshot(equity, unrealized, interval)
