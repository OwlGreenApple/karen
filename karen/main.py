"""
Karen — entry point and async orchestrator.

Startup sequence:
  1. Logging + config watcher
  2. DB init + startup reconciliation
  3. Binance client connect + symbol setup
  4. WebSocket price feed
  5. 15m candle scan loop (blocking until shutdown)

All components run as asyncio tasks; SIGINT/SIGTERM triggers graceful shutdown:
  - Cancel pending orders
  - Flush DB
  - Close WebSocket + exchange connection
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
import sys
from collections.abc import Callable
from datetime import UTC, datetime

import uvicorn
from loguru import logger

from karen.config import ConfigWatcher, Settings, StrategyMode, init_config_watcher
from karen.dashboard.api import app as dashboard_app
from karen.dashboard.api import init_dashboard, update_state
from karen.exchange.binance_client import BinanceClient
from karen.exchange.websocket import PriceFeed
from karen.execution.order_manager import OrderManager
from karen.execution.position_tracker import PositionTracker
from karen.logging_setup import setup_logging
from karen.notifications.telegram import TelegramNotifier
from karen.persistence.db import init_db
from karen.risk.guardrails import DailyGuardrails
from karen.strategies.base import AbstractStrategy
from karen.strategies.mean_reversion import MeanReversionStrategy
from karen.strategies.trend_following import TrendFollowingStrategy

# ─── Strategy factory ─────────────────────────────────────────────────────────


def make_strategy(settings: Settings) -> AbstractStrategy:
    if settings.strategy_mode == StrategyMode.MEAN_REVERSION:
        return MeanReversionStrategy(settings)
    return TrendFollowingStrategy(settings)


# ─── Timing helpers ───────────────────────────────────────────────────────────


async def _wait_for_15m_close() -> None:
    """Sleep until 2 seconds after the next 15m candle close."""
    now = datetime.now(tz=UTC)
    seconds_in_day = now.hour * 3600 + now.minute * 60 + now.second + now.microsecond / 1e6
    period = 15 * 60
    seconds_into_period = seconds_in_day % period
    sleep_for = period - seconds_into_period + 2.0  # +2s exchange latency buffer
    if sleep_for <= 2.0:
        sleep_for += period
    logger.debug(f"Next 15m candle in {sleep_for:.0f}s")
    await asyncio.sleep(sleep_for)


# ─── Candle scan loop ─────────────────────────────────────────────────────────


async def candle_loop(
    client: BinanceClient,
    strategy_getter: Callable[[], AbstractStrategy],
    order_mgr: OrderManager,
    guardrails: DailyGuardrails,
    config_watcher: ConfigWatcher,
    notifier: TelegramNotifier,
) -> None:
    """
    Wakes on each 15m candle close, fetches OHLCV for all symbols,
    runs the active strategy, and places orders if conditions are met.
    """
    logger.info("Candle loop started")
    while True:
        await _wait_for_15m_close()
        settings = config_watcher.settings

        try:
            equity = await client.fetch_equity()
        except Exception:
            logger.exception("Failed to fetch equity — skipping this candle")
            continue

        guardrails.maybe_rollover(equity)

        if guardrails.paused:
            pct = guardrails.current_loss_pct(equity)
            logger.warning(f"Bot paused (daily loss {pct:.2f}%) — waiting for UTC day rollover")
            await notifier.daily_loss_limit_hit(pct, equity)
            continue

        strategy = strategy_getter()
        open_symbols = await order_mgr.get_open_symbol_set()
        open_count = len(open_symbols)

        for symbol in settings.symbol_list:
            if symbol in open_symbols:
                logger.debug(f"{symbol}: already in position — skipping")
                continue

            try:
                df_15m = await client.fetch_ohlcv(symbol, "15m", limit=250)
                df_1h = await client.fetch_ohlcv(symbol, "1h", limit=250)
                signal = await strategy.evaluate(symbol, df_15m, df_1h)
            except Exception:
                logger.exception(f"{symbol}: error during evaluation")
                continue

            if signal is None:
                continue

            if not settings.trading_enabled:
                logger.info(
                    f"Signal {symbol} {signal.side.upper()} — "
                    f"TRADING_ENABLED=false, logged only"
                )
                continue

            ok, reason = guardrails.can_open_position(equity, open_count)
            if not ok:
                logger.info(f"{symbol}: blocked — {reason}")
                continue

            trade = await order_mgr.open_position(signal, equity)
            if trade:
                guardrails.record_trade_opened()
                open_count += 1
                open_symbols.add(symbol)


# ─── Main entry point ─────────────────────────────────────────────────────────


async def run() -> None:
    setup_logging()
    logger.info("Karen starting up...")

    loop = asyncio.get_event_loop()
    config_watcher = init_config_watcher(loop)
    config_watcher.start()
    settings = config_watcher.settings

    await init_db()

    notifier = TelegramNotifier(settings)
    guardrails = DailyGuardrails(settings)

    async with BinanceClient(settings) as client:
        # Setup leverage/margin for all symbols
        for symbol in settings.symbol_list:
            try:
                await client.setup_symbol(symbol)
            except Exception:
                logger.exception(f"Failed to setup {symbol} — continuing")

        order_mgr = OrderManager(client, settings, notifier=notifier)
        pos_tracker = PositionTracker(client, settings, notifier=notifier)

        # Startup reconciliation
        await pos_tracker.startup_reconcile()

        # Seed daily guardrails
        equity = await client.fetch_equity()
        guardrails.new_day(equity)

        # Init dashboard state
        init_dashboard(equity, settings.strategy_mode.value, settings.trading_enabled)

        # Active strategy (mutable reference via closure)
        _strategy: list[AbstractStrategy] = [make_strategy(settings)]

        def strategy_getter() -> AbstractStrategy:
            return _strategy[0]

        # WebSocket price feed
        price_feed = PriceFeed(settings.symbol_list, settings.binance_testnet)
        price_feed.on_kline(
            lambda k: asyncio.ensure_future(order_mgr.on_closed_kline(k))
        )
        price_feed.start()

        # Hot-reload config change handler
        async def on_config_change(old: Settings, new: Settings) -> None:
            order_mgr.update_settings(new)
            pos_tracker.update_settings(new)
            guardrails._settings = new  # guardrails reads settings on each check

            if old.strategy_mode != new.strategy_mode:
                logger.info(
                    f"Strategy mode changed: {old.strategy_mode.value} → {new.strategy_mode.value}"
                )
                await order_mgr.cancel_pending_entry_orders()
                _strategy[0] = make_strategy(new)
                await notifier.mode_changed(old.strategy_mode, new.strategy_mode)

            if old.trading_enabled != new.trading_enabled:
                logger.info(f"Trading enabled: {old.trading_enabled} → {new.trading_enabled}")
                await notifier.trading_enabled_changed(new.trading_enabled)

            update_state(
                mode=new.strategy_mode.value,
                trading_enabled=new.trading_enabled,
            )

        config_watcher.add_callback(on_config_change)

        await notifier.bot_started(equity, settings.strategy_mode, settings.trading_enabled)

        # Graceful shutdown
        shutdown_event = asyncio.Event()

        def _handle_signal() -> None:
            logger.info("Shutdown signal received")
            shutdown_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _handle_signal)

        scan_task = asyncio.create_task(
            candle_loop(client, strategy_getter, order_mgr, guardrails, config_watcher, notifier),
            name="candle-loop",
        )
        track_task = asyncio.create_task(pos_tracker.run(), name="pos-tracker")

        # Dashboard server
        uvicorn_config = uvicorn.Config(
            dashboard_app,
            host=settings.dashboard_host,
            port=settings.dashboard_port,
            log_level="warning",
        )
        uvicorn_server = uvicorn.Server(uvicorn_config)
        dash_task = asyncio.create_task(uvicorn_server.serve(), name="dashboard")
        logger.info(f"Dashboard at http://{settings.dashboard_host}:{settings.dashboard_port}")

        try:
            await asyncio.wait(
                [
                    scan_task,
                    track_task,
                    dash_task,
                    asyncio.create_task(shutdown_event.wait()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            logger.info("Shutting down...")
            scan_task.cancel()
            track_task.cancel()
            uvicorn_server.should_exit = True
            await asyncio.gather(scan_task, track_task, dash_task, return_exceptions=True)
            await price_feed.stop()
            config_watcher.stop()
            await notifier.bot_stopped()
            logger.info("Karen stopped cleanly")


def main() -> None:
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run())
    sys.exit(0)


if __name__ == "__main__":
    main()
