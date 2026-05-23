"""Telegram notifier — sends key bot events via python-telegram-bot."""

from __future__ import annotations

import time

from loguru import logger

from karen.config import Settings, StrategyMode
from karen.persistence.models import CloseReason

_PLACEHOLDER_VALUES = {"your_bot_token_here", "your_chat_id_here", "", "0"}


class TelegramNotifier:
    """
    Sends Telegram messages for key bot events.

    No-ops gracefully when token/chat_id are not configured.
    Throttles error messages to one per error_key per 60 seconds.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._last_error_ts: dict[str, float] = {}
        self._enabled = self._check_configured()

    def _check_configured(self) -> bool:
        token = self._settings.telegram_bot_token or ""
        chat_id = self._settings.telegram_chat_id or ""
        if token.lower() in _PLACEHOLDER_VALUES or chat_id.lower() in _PLACEHOLDER_VALUES:
            logger.debug("Telegram not configured — notifications disabled")
            return False
        return True

    # ── Internal ───────────────────────────────────────────────────────────

    async def _send(self, text: str) -> None:
        if not self._enabled:
            logger.debug(f"[Telegram-noop] {text[:80]}")
            return
        try:
            from telegram import Bot
            from telegram.constants import ParseMode

            async with Bot(token=self._settings.telegram_bot_token) as bot:
                await bot.send_message(
                    chat_id=self._settings.telegram_chat_id,
                    text=text,
                    parse_mode=ParseMode.HTML,
                )
        except Exception as exc:
            logger.warning(f"Telegram send failed: {exc}")

    async def _send_error_throttled(self, error_key: str, text: str) -> None:
        now = time.monotonic()
        if now - self._last_error_ts.get(error_key, 0) < 60:
            return
        self._last_error_ts[error_key] = now
        await self._send(text)

    # ── Public events ──────────────────────────────────────────────────────

    async def bot_started(self, equity: float, mode: StrategyMode, trading_enabled: bool) -> None:
        status = "✅ ENABLED" if trading_enabled else "⏸ DISABLED"
        await self._send(
            f"🚀 <b>Karen started</b>\n"
            f"Mode: {mode.value}\n"
            f"Trading: {status}\n"
            f"Equity: ${equity:,.2f} USDT\n"
            f"Dashboard: {self._settings.dashboard_url}"
        )

    async def bot_stopped(self) -> None:
        await self._send("🛑 <b>Karen stopped</b>")

    async def mode_changed(self, old: StrategyMode, new: StrategyMode) -> None:
        await self._send(
            f"🔄 <b>Strategy mode changed</b>\n"
            f"{old.value} → {new.value}\n"
            f"Pending entry orders cancelled"
        )

    async def trading_enabled_changed(self, enabled: bool) -> None:
        icon = "✅" if enabled else "⏸"
        await self._send(f"{icon} Trading {'<b>enabled</b>' if enabled else '<b>disabled</b>'}")

    async def trade_opened(
        self,
        symbol: str,
        side: str,
        mode: StrategyMode,
        entry_price: float,
        quantity: float,
        sl_price: float,
        tp1_price: float,
        leverage: int,
        dollar_risk: float,
    ) -> None:
        icon = "🟢" if side == "long" else "🔴"
        sl_pct = abs(entry_price - sl_price) / entry_price * 100
        tp_pct = abs(tp1_price - entry_price) / entry_price * 100
        sl_sign = "-" if side == "long" else "+"
        tp_sign = "+" if side == "long" else "-"
        value = quantity * entry_price
        await self._send(
            f"{icon} <b>TRADE OPENED</b> — {mode.value.replace('_', ' ').title()}\n"
            f"📊 {symbol} {side.upper()} @ {entry_price:,.2f}\n"
            f"💰 Size: {quantity:.4f} (${value:,.2f})\n"
            f"🛑 SL: {sl_price:,.2f} ({sl_sign}{sl_pct:.2f}%)\n"
            f"🎯 TP1: {tp1_price:,.2f} ({tp_sign}{tp_pct:.2f}%)\n"
            f"⚙️ Leverage: {leverage}x | Risk: ${dollar_risk:.2f}\n"
            f"🔗 {self._settings.dashboard_url}"
        )

    async def trade_closed(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        pnl: float,
        pnl_pct: float,
        reason: CloseReason,
    ) -> None:
        icon = "✅" if pnl >= 0 else "❌"
        reason_str = reason.value.replace("_", " ").upper()
        await self._send(
            f"{icon} <b>TRADE CLOSED</b> ({reason_str})\n"
            f"📊 {symbol} {side.upper()}\n"
            f"Entry: {entry_price:,.2f} → Exit: {exit_price:,.2f}\n"
            f"PnL: ${pnl:+,.2f} ({pnl_pct:+.2f}%)"
        )

    async def partial_tp_hit(
        self, symbol: str, side: str, price: float, qty_closed: float, pnl: float
    ) -> None:
        await self._send(
            f"🎯 <b>PARTIAL TP</b> — {symbol} {side.upper()}\n"
            f"Closed {qty_closed:.4f} @ {price:,.2f} | PnL: ${pnl:+,.2f}"
        )

    async def daily_loss_limit_hit(self, loss_pct: float, equity: float) -> None:
        await self._send(
            f"⚠️ <b>DAILY LOSS LIMIT HIT</b>\n"
            f"Loss: {loss_pct:.2f}% | Equity: ${equity:,.2f}\n"
            f"Bot paused until next UTC day"
        )

    async def error(self, error_key: str, message: str) -> None:
        await self._send_error_throttled(error_key, f"🚨 <b>ERROR [{error_key}]</b>\n{message}")
