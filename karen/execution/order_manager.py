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
from karen.persistence.models import CloseReason, Trade, TradeSide, TradeStatus, is_orphan
from karen.risk.position_sizing import (
    MARGIN_FEE_BUFFER,
    MAX_LEVERAGE,
    MAX_SL_LOSS_PCT,
    cap_sl_to_max_loss,
    choose_leverage,
    compute_dollar_risk,
    compute_quantity_by_margin_pct,
    fit_leverage_to_step,
    leverage_from_confidence,
    max_safe_leverage,
)
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

        # Leverage 5x..20x from signal confidence; SL pulled in so a stop-out never
        # loses more than MAX_SL_LOSS_PCT of margin; final guard: SL before liquidation
        conf_lev = leverage_from_confidence(signal.confidence)
        capped_sl = cap_sl_to_max_loss(
            signal.side, signal.entry_price, signal.sl_price, conf_lev
        )
        if capped_sl != signal.sl_price:
            logger.info(
                f"{signal.symbol}: SL {signal.sl_price:.6g} → {capped_sl:.6g} "
                f"(max loss {MAX_SL_LOSS_PCT:.0f}% of margin at {conf_lev}x)"
            )
            signal.sl_price = capped_sl
        leverage = choose_leverage(signal.confidence, signal.entry_price, signal.sl_price)
        if leverage is None:
            logger.warning(
                f"{signal.symbol}: SL too far for safe leverage "
                f"(entry={signal.entry_price}, SL={signal.sl_price}) — skipping"
            )
            return None

        # Margin = POSITION_SIZE_PCT of free balance (never more than what's available)
        try:
            available = await self._client.fetch_available_balance()
        except Exception as exc:
            logger.error(f"{signal.symbol}: fetch available balance failed — {exc}")
            return None
        margin_base = min(equity, available) * MARGIN_FEE_BUFFER
        margin = margin_base * s.position_size_pct / 100

        # Lot-step rounding can drop a whole step on coarse pairs (BTC 0.001 ≈ $60):
        # raise leverage just enough to use ~all the margin, within safe limits,
        # then re-cap the SL so a stop-out still loses ≤ MAX_SL_LOSS_PCT of margin.
        max_lev = min(MAX_LEVERAGE, max_safe_leverage(signal.entry_price, signal.sl_price))
        fitted = fit_leverage_to_step(
            margin, signal.entry_price, self._client.get_qty_step(signal.symbol),
            leverage, max_lev,
        )
        if fitted != leverage:
            new_sl = cap_sl_to_max_loss(signal.side, signal.entry_price, signal.sl_price, fitted)
            logger.info(
                f"{signal.symbol}: leverage {leverage}x → {fitted}x to fit lot step "
                f"(SL {signal.sl_price:.6g} → {new_sl:.6g})"
            )
            leverage = fitted
            signal.sl_price = new_sl

        try:
            qty = compute_quantity_by_margin_pct(
                equity_usdt=margin_base,
                margin_pct=s.position_size_pct,
                entry_price=signal.entry_price,
                leverage=leverage,
            )
        except ValueError as exc:
            logger.error(f"{signal.symbol}: position sizing failed — {exc}")
            return None

        logger.info(
            f"{signal.symbol}: confidence={signal.confidence:.2f} → leverage {leverage}x "
            f"(safe max {max_safe_leverage(signal.entry_price, signal.sl_price)}x) | "
            f"margin=${margin:.2f} "
            f"notional=${qty * signal.entry_price:.2f}"
        )

        # What Binance will actually fill — keeps DB, SL and TP sizes consistent
        qty = self._client.floor_qty(signal.symbol, qty)

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

        try:
            await self._client.set_leverage(signal.symbol, leverage)
        except Exception as exc:
            logger.error(f"{signal.symbol}: set leverage {leverage}x failed — {exc}")
            return None

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
            leverage=leverage,
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
        tp1_qty, tp2_qty = _split_tp_qty(
            qty,
            signal.tp1_qty_pct,
            has_tp2=signal.tp2_price is not None,
            step=self._client.get_qty_step(signal.symbol),
            min_qty=min_qty,
        )
        if tp1_qty == qty and signal.tp1_qty_pct < 1.0:
            logger.info(
                f"{signal.symbol}: qty {qty} too small to split "
                f"{signal.tp1_qty_pct:.0%}/{1 - signal.tp1_qty_pct:.0%} — TP1 covers full position"
            )
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
        if signal.tp2_price is not None and tp2_qty > 0:
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
                leverage=leverage,
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
        - MR time stop: close if MR_TIME_STOP_CANDLES × 15m passed without profit
        - MR breakeven: SL → entry once TP1 fills (MR_BREAKEVEN_AFTER_TP1)
        - Scalp time stop: close if 10 × 1m candles passed without profit
        - TF trailing stop: update chandelier SL on each 15m candle
        """
        if not kline.is_closed:
            return

        symbol = kline.symbol
        open_trades = await self._get_open_trades_for_symbol(symbol)

        for trade in open_trades:
            # Manual positions are the user's to manage — never move their SL
            if is_orphan(trade.strategy_mode):
                continue

            # MR breakeven: react on every closed kline (1m included) so the SL
            # moves to entry within ~1 minute of TP1 filling
            if trade.strategy_mode == "mean_reversion":
                await self._check_breakeven(trade, kline)

            expected_tf = "1m" if trade.strategy_mode == "scalp_1m" else "15m"
            if kline.timeframe != expected_tf:
                continue

            self._update_kline_buffer(symbol, kline)
            trade.candles_open += 1
            await self._persist_candles_open(trade)

            if trade.strategy_mode == "mean_reversion":
                if self._settings.mr_time_stop_candles > 0:
                    await self._check_time_stop(
                        trade, kline, self._settings.mr_time_stop_candles
                    )
            elif trade.strategy_mode == "scalp_1m":
                await self._check_time_stop(trade, kline, self._SCALP_TIME_STOP)
            else:
                await self._check_trailing_stop(trade, kline)

    async def _check_breakeven(self, trade: Trade, kline: Kline) -> None:
        """
        MR_BREAKEVEN_AFTER_TP1: once TP1 has filled, move the SL to the entry price
        so the remaining position can no longer turn into a loss.

        Cheap pre-check on the kline range (TP1 price touched) before asking the
        exchange whether the TP1 order actually filled.
        """
        if not self._settings.mr_breakeven_after_tp1 or not trade.tp1_order_id:
            return
        if trade.tp1_price is None or trade.sl_price == trade.entry_price:
            return
        touched = (
            kline.low <= trade.tp1_price
            if trade.side == TradeSide.SHORT
            else kline.high >= trade.tp1_price
        )
        if not touched:
            return
        try:
            tp1 = await self._client.fetch_order(trade.symbol, trade.tp1_order_id)
        except Exception as exc:
            logger.warning(f"Trade #{trade.id}: TP1 status check failed — {exc}")
            return
        if tp1.status != "closed":
            return
        # a stop at entry is only valid while price is still on the profitable side
        in_profit = (
            kline.close < trade.entry_price
            if trade.side == TradeSide.SHORT
            else kline.close > trade.entry_price
        )
        if not in_profit:
            logger.info(f"Trade #{trade.id}: TP1 filled but price back at entry — SL kept")
            return
        logger.info(f"Trade #{trade.id}: TP1 filled — moving SL to entry {trade.entry_price}")
        await self._update_sl(trade, trade.entry_price, kline.close)

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
        """
        Replace the SL order: place the NEW stop first, cancel the old one only after
        it succeeded — so a rejected new stop never leaves the position unprotected.
        (Two reduce-only stops overlapping for a moment is harmless.)
        """
        exit_side = "sell" if trade.side == TradeSide.LONG else "buy"
        old_sl, old_id = trade.sl_price, trade.sl_order_id

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
        except Exception as exc:
            logger.error(
                f"Trade #{trade.id}: new SL {new_sl:.6g} rejected — keeping old SL "
                f"{old_sl} — {exc}"
            )
            return

        if old_id:
            try:
                await self._client.cancel_order(trade.symbol, old_id)
            except Exception as exc:
                logger.warning(f"Trade #{trade.id}: cancel old SL failed — {exc}")

        trade.sl_price = new_sl
        trade.sl_order_id = new_order.id
        async with self._session_factory() as session:
            await session.merge(trade)
            await session.commit()
        logger.info(
            f"Trade #{trade.id} SL updated: {old_sl} → {new_sl:.6g} (price={current_price:.6g})"
        )

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
        """Cancel every open order on the trade's symbol (SL, TPs, stale stops)."""
        await cancel_symbol_orders(self._client, trade)

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


def _split_tp_qty(
    qty: float, tp1_pct: float, *, has_tp2: bool, step: float, min_qty: float
) -> tuple[float, float]:
    """
    Split an already lot-rounded qty into (tp1_qty, tp2_qty) that Binance accepts.

    - TP1 gets tp1_pct of qty, rounded to the nearest lot step.
    - TP2 (if any) gets exactly the remainder, so nothing is left without a TP.
    - With no TP2 (TF runner), the remainder is the trailing-stop runner.
    - If either part would fall below the exchange minimum, TP1 takes the whole qty
      (better a full take-profit than an unprotected remainder).
    """
    raw = qty * tp1_pct
    tp1 = min(round(round(raw / step) * step, 12) if step > 0 else raw, qty)
    rest = round(qty - tp1, 12)
    floor = max(min_qty, step)
    if tp1 < floor or (rest > 0 and rest < floor):
        return qty, 0.0
    return tp1, (rest if has_tp2 else 0.0)


async def cancel_symbol_orders(client: BinanceClient, trade: Trade) -> None:
    """
    After a position is gone, cancel ALL open orders on its symbol — regular and
    algo — so no leftover TP/SL can hit a future position on the same coin.
    Falls back to cancelling the trade's known order IDs one by one.
    """
    try:
        await client.cancel_all_orders(trade.symbol)
        return
    except Exception as exc:
        logger.warning(f"Trade #{trade.id}: cancel-all on {trade.symbol} failed — {exc}")
    for oid in (trade.sl_order_id, trade.tp1_order_id, trade.tp2_order_id):
        if oid:
            try:
                await client.cancel_order(trade.symbol, oid)
            except Exception as exc:
                logger.debug(f"Cancel order {oid} failed (may already be filled): {exc}")
