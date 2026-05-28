"""Order lifecycle: open position, SL/TP placement, partial close, trailing stop."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from karen.config import Settings
from karen.exchange.binance_client import BinanceClient
from karen.exchange.websocket import Kline
from karen.persistence.db import AsyncSessionLocal
from karen.persistence.models import CloseReason, Trade, TradeSide, TradeStatus
from karen.risk.position_sizing import compute_dollar_risk, compute_quantity_by_margin_pct
from karen.strategies.base import Signal


class OrderManager:
    """
    Translates Signals into exchange orders and manages the full order lifecycle.

    Responsibilities:
    - Open positions: entry market + SL stop-market + TP limit orders
    - Close positions: market close + cancel remaining orders
    - Trailing stop: update chandelier SL on each closed 15m kline (TF strategy)
    - Time stop: close MR positions after 8 candles without profit
    - Mode switch: cancel all pending entry orders
    """

    _TIME_STOP_CANDLES = 8   # MR: 8 × 15m = 2 hours
    _SCALP_TIME_STOP = 10    # Scalp: 10 × 1m = 10 minutes

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
        # symbol → {15m OHLCV buffer} for trailing stop computation
        self._kline_buffer: dict[str, list[Kline]] = {}

    def update_settings(self, settings: Settings) -> None:
        self._settings = settings

    # ── Open position ──────────────────────────────────────────────────────

    async def open_position(self, signal: Signal, equity: float) -> Trade | None:
        """
        Full order flow for a new position:
          1. Compute quantity (ATR-based risk sizing)
          2. Place market entry order
          3. Save Trade to DB
          4. Place SL stop-market order (reduce_only, full quantity)
          5. Place TP1 take-profit order (reduce_only, tp1_qty_pct of quantity)
          6. Place TP2 order if signal has a fixed TP2 (MR runner)

        If entry fails → return None (nothing placed).
        If SL/TP placement fails → log error, continue (position is unprotected
        but alive; reconciliation will detect it).
        """
        s = self._settings

        try:
            qty = compute_quantity_by_margin_pct(
                equity_usdt=equity,
                margin_pct=s.position_size_pct,
                entry_price=signal.entry_price,
                leverage=s.leverage,
            )
        except ValueError as exc:
            logger.error(f"{signal.symbol}: position sizing failed — {exc}")
            return None

        # Ensure minimum sensible quantity (Binance min notional ~$5)
        min_notional = 5.0
        if qty * signal.entry_price < min_notional:
            logger.warning(
                f"{signal.symbol}: computed notional ${qty * signal.entry_price:.2f} "
                f"below minimum ${min_notional} — skipping"
            )
            return None

        # Check exchange minimum quantity (e.g. BTC min=0.001)
        min_qty = self._client.get_min_qty(signal.symbol)
        if min_qty > 0 and qty < min_qty:
            logger.warning(
                f"{signal.symbol}: computed qty={qty:.6f} below exchange minimum "
                f"{min_qty} (equity too small for this pair) — skipping"
            )
            return None

        entry_side = "buy" if signal.side == "long" else "sell"
        exit_side = "sell" if signal.side == "long" else "buy"

        # 1. Entry order (market)
        entry_coid = _coid("e", signal.symbol)
        try:
            entry_order = await self._client.place_order(
                symbol=signal.symbol,
                side=entry_side,
                order_type="market",
                quantity=qty,
                client_order_id=entry_coid,
            )
        except Exception as exc:
            logger.error(f"{signal.symbol}: entry order failed — {exc}")
            return None

        # 2. Persist trade immediately so reconciliation can find it
        trade = Trade(
            symbol=signal.symbol,
            side=TradeSide.LONG if signal.side == "long" else TradeSide.SHORT,
            strategy_mode=signal.strategy_mode.value,
            status=TradeStatus.OPEN,
            entry_price=signal.entry_price,
            quantity=qty,
            leverage=s.leverage,
            sl_price=signal.sl_price,
            tp1_price=signal.tp1_price,
            tp2_price=signal.tp2_price,
            entry_order_id=entry_order.id,
            opened_at=datetime.now(tz=UTC),
        )
        async with self._session_factory() as session:
            session.add(trade)
            await session.commit()
            await session.refresh(trade)

        logger.info(
            f"Trade #{trade.id} opened: {signal.symbol} {signal.side.upper()} "
            f"qty={qty:.6f} entry≈{signal.entry_price:.2f}"
        )

        # 3. SL (stop-market, reduce_only, full quantity)
        sl_coid = _coid("sl", signal.symbol)
        try:
            sl_order = await self._client.place_order(
                symbol=signal.symbol,
                side=exit_side,
                order_type="stop_market",
                quantity=qty,
                stop_price=signal.sl_price,
                reduce_only=True,
                client_order_id=sl_coid,
            )
            trade.sl_order_id = sl_order.id
        except Exception as exc:
            logger.error(f"Trade #{trade.id}: SL order failed — {exc}")
            if self._notifier:
                await self._notifier.error("sl_order_failed", f"SL failed {signal.symbol}: {exc}")

        # 4. TP1 (take-profit-market, reduce_only, partial quantity)
        tp1_qty = qty * signal.tp1_qty_pct
        tp1_coid = _coid("tp1", signal.symbol)
        try:
            tp1_order = await self._client.place_order(
                symbol=signal.symbol,
                side=exit_side,
                order_type="take_profit_market",
                quantity=tp1_qty,
                stop_price=signal.tp1_price,
                reduce_only=True,
                client_order_id=tp1_coid,
            )
            trade.tp1_order_id = tp1_order.id
        except Exception as exc:
            logger.error(f"Trade #{trade.id}: TP1 order failed — {exc}")

        # 5. TP2 (only for MR runner with fixed price)
        if signal.tp2_price is not None:
            tp2_qty = qty * (1.0 - signal.tp1_qty_pct)
            tp2_coid = _coid("tp2", signal.symbol)
            try:
                tp2_order = await self._client.place_order(
                    symbol=signal.symbol,
                    side=exit_side,
                    order_type="take_profit_market",
                    quantity=tp2_qty,
                    stop_price=signal.tp2_price,
                    reduce_only=True,
                    client_order_id=tp2_coid,
                )
                trade.tp2_order_id = tp2_order.id
            except Exception as exc:
                logger.error(f"Trade #{trade.id}: TP2 order failed — {exc}")

        # Persist all order IDs
        async with self._session_factory() as session:
            merged = await session.merge(trade)
            await session.commit()
            await session.refresh(merged)

        # Telegram notification
        if self._notifier:
            dollar_risk = compute_dollar_risk(qty, signal.entry_price, signal.sl_price)
            await self._notifier.trade_opened(
                symbol=signal.symbol,
                side=signal.side,
                mode=signal.strategy_mode,
                entry_price=signal.entry_price,
                quantity=qty,
                sl_price=signal.sl_price,
                tp1_price=signal.tp1_price,
                leverage=s.leverage,
                dollar_risk=dollar_risk,
            )

        return trade

    # ── Close position ─────────────────────────────────────────────────────

    async def close_position(
        self,
        trade: Trade,
        reason: CloseReason,
        exit_price: float | None = None,
    ) -> None:
        """
        Market-close a position and cancel all remaining open orders.
        Updates the trade record in DB.
        """
        side = "sell" if trade.side == TradeSide.LONG else "buy"
        coid = _coid("close", trade.symbol)

        try:
            await self._client.place_order(
                symbol=trade.symbol,
                side=side,
                order_type="market",
                quantity=trade.quantity,
                reduce_only=True,
                client_order_id=coid,
            )
        except Exception as exc:
            logger.error(f"Trade #{trade.id}: market close failed — {exc}")
            return

        await self._cancel_trade_orders(trade)

        now = datetime.now(tz=UTC)
        ep = exit_price or trade.entry_price
        pnl = _compute_pnl(trade, ep)
        pnl_pct = pnl / (trade.entry_price * trade.quantity / trade.leverage) * 100

        async with self._session_factory() as session:
            t = await session.get(Trade, trade.id)
            if t:
                t.status = TradeStatus.CLOSED
                t.exit_price = ep
                t.realized_pnl = pnl
                t.realized_pnl_pct = pnl_pct
                t.close_reason = reason
                t.closed_at = now
                await session.commit()

        logger.info(
            f"Trade #{trade.id} closed: {trade.symbol} reason={reason.value} "
            f"pnl=${pnl:+.2f} ({pnl_pct:+.2f}%)"
        )

        if self._notifier:
            await self._notifier.trade_closed(
                symbol=trade.symbol,
                side=trade.side.value,
                entry_price=trade.entry_price,
                exit_price=ep,
                quantity=trade.quantity,
                pnl=pnl,
                pnl_pct=pnl_pct,
                reason=reason,
            )

    # ── Closed kline handler ───────────────────────────────────────────────

    async def on_closed_kline(self, kline: Kline) -> None:
        """
        Called by the WebSocket feed on each finalized kline.

        Handles:
        - MR time stop: close if 8 × 15m candles passed without profit
        - Scalp time stop: close if 10 × 1m candles passed without profit
        - TF trailing stop: update chandelier SL on each 15m candle
        """
        if not kline.is_closed:
            return

        symbol = kline.symbol
        open_trades = await self._get_open_trades_for_symbol(symbol)

        for trade in open_trades:
            expected_tf = "1m" if trade.strategy_mode == "scalp_1m" else "15m"
            if kline.timeframe != expected_tf:
                continue

            self._update_kline_buffer(symbol, kline)
            trade.candles_open += 1
            await self._persist_candles_open(trade)

            if trade.strategy_mode == "mean_reversion":
                await self._check_time_stop(trade, kline, self._TIME_STOP_CANDLES)
            elif trade.strategy_mode == "scalp_1m":
                await self._check_time_stop(trade, kline, self._SCALP_TIME_STOP)
            else:
                await self._check_trailing_stop(trade, kline)

    async def _check_time_stop(self, trade: Trade, kline: Kline, max_candles: int) -> None:
        """Close trade if max_candles passed and not in profit."""
        if trade.candles_open < max_candles:
            return
        current_price = kline.close
        in_profit = (
            (trade.side == TradeSide.LONG and current_price > trade.entry_price)
            or (trade.side == TradeSide.SHORT and current_price < trade.entry_price)
        )
        if not in_profit:
            logger.info(
                f"Trade #{trade.id} time stop: {trade.symbol} "
                f"candles_open={trade.candles_open}"
            )
            await self.close_position(trade, CloseReason.TIME_STOP, exit_price=current_price)

    async def _check_trailing_stop(self, trade: Trade, kline: Kline) -> None:
        """Update chandelier exit SL if it has moved closer to price."""
        buf = self._kline_buffer.get(trade.symbol, [])
        if len(buf) < 22:
            return

        atr_approx = _approx_atr(buf)
        mult = 3.0  # chandelier multiplier

        if trade.side == TradeSide.LONG:
            highest_high = max(k.high for k in buf[-22:])
            new_sl = highest_high - mult * atr_approx
            if trade.sl_price is None or new_sl > trade.sl_price:
                await self._update_sl(trade, new_sl, kline.close)
        else:
            lowest_low = min(k.low for k in buf[-22:])
            new_sl = lowest_low + mult * atr_approx
            if trade.sl_price is None or new_sl < trade.sl_price:
                await self._update_sl(trade, new_sl, kline.close)

    async def _update_sl(self, trade: Trade, new_sl: float, current_price: float) -> None:
        """Cancel existing SL order and place a new one."""
        exit_side = "sell" if trade.side == TradeSide.LONG else "buy"

        if trade.sl_order_id:
            try:
                await self._client.cancel_order(trade.symbol, trade.sl_order_id)
            except Exception as exc:
                logger.warning(f"Trade #{trade.id}: cancel old SL failed — {exc}")

        # Remaining quantity (Binance reduce_only handles partial)
        coid = _coid("sl", trade.symbol)
        try:
            new_order = await self._client.place_order(
                symbol=trade.symbol,
                side=exit_side,
                order_type="stop_market",
                quantity=trade.quantity,
                stop_price=new_sl,
                reduce_only=True,
                client_order_id=coid,
            )
            old_sl = trade.sl_price
            trade.sl_price = new_sl
            trade.sl_order_id = new_order.id
            async with self._session_factory() as session:
                await session.merge(trade)
                await session.commit()
            logger.info(
                f"Trade #{trade.id} trailing SL updated: {old_sl:.2f} → {new_sl:.2f} "
                f"(price={current_price:.2f})"
            )
        except Exception as exc:
            logger.error(f"Trade #{trade.id}: update trailing SL failed — {exc}")

    # ── Cancel pending entries (mode switch) ───────────────────────────────

    async def cancel_pending_entry_orders(self) -> None:
        """
        Cancel all open entry orders.  Called when strategy mode changes.
        Existing open positions are intentionally left untouched.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(Trade).where(Trade.status == TradeStatus.OPEN)
            )
            open_trades = result.scalars().all()

        cancelled = 0
        for trade in open_trades:
            # Check entry order status — if not fully filled, cancel it
            if trade.entry_order_id:
                try:
                    order = await self._client.fetch_order(
                        trade.symbol, trade.entry_order_id
                    )
                    if order.status == "open":
                        await self._client.cancel_order(trade.symbol, trade.entry_order_id)
                        cancelled += 1
                except Exception as exc:
                    logger.warning(
                        f"Could not cancel entry order {trade.entry_order_id}: {exc}"
                    )

        logger.info(f"Cancelled {cancelled} pending entry orders on mode switch")

    # ── Helpers ────────────────────────────────────────────────────────────

    async def get_open_symbol_set(self) -> set[str]:
        """Return set of symbols with open trades."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(Trade.symbol).where(Trade.status == TradeStatus.OPEN)
            )
            return set(result.scalars().all())

    async def get_open_trade_count(self) -> int:
        async with self._session_factory() as session:
            result = await session.execute(
                select(Trade).where(Trade.status == TradeStatus.OPEN)
            )
            return len(result.scalars().all())

    async def _get_open_trades_for_symbol(self, symbol: str) -> list[Trade]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(Trade).where(
                    Trade.symbol == symbol,
                    Trade.status == TradeStatus.OPEN,
                )
            )
            return list(result.scalars().all())

    async def _cancel_trade_orders(self, trade: Trade) -> None:
        """Cancel all remaining SL/TP orders for a trade."""
        for oid in (trade.sl_order_id, trade.tp1_order_id, trade.tp2_order_id):
            if oid:
                try:
                    await self._client.cancel_order(trade.symbol, oid)
                except Exception as exc:
                    logger.debug(f"Cancel order {oid} failed (may already be filled): {exc}")

    async def _persist_candles_open(self, trade: Trade) -> None:
        async with self._session_factory() as session:
            t = await session.get(Trade, trade.id)
            if t:
                t.candles_open = trade.candles_open
                await session.commit()

    def _update_kline_buffer(self, symbol: str, kline: Kline) -> None:
        buf = self._kline_buffer.setdefault(symbol, [])
        buf.append(kline)
        if len(buf) > 50:
            self._kline_buffer[symbol] = buf[-50:]


# ─── Module-level helpers ─────────────────────────────────────────────────────


def _coid(tag: str, symbol: str) -> str:
    return f"karen-{tag}-{symbol[:3].lower()}-{uuid.uuid4().hex[:8]}"


def _compute_pnl(trade: Trade, exit_price: float) -> float:
    if trade.side == TradeSide.LONG:
        return (exit_price - trade.entry_price) * trade.quantity
    return (trade.entry_price - exit_price) * trade.quantity


def _approx_atr(klines: list[Kline], period: int = 14) -> float:
    """Approximate ATR using recent candle ranges (no smoothing for simplicity)."""
    ranges = [k.high - k.low for k in klines[-period:]]
    return sum(ranges) / len(ranges) if ranges else 1.0
