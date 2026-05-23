"""Tests for TelegramNotifier — verifies no-op when unconfigured, and message formatting."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from karen.config import Settings, StrategyMode
from karen.notifications.telegram import TelegramNotifier
from karen.persistence.models import CloseReason

# ── Helpers ────────────────────────────────────────────────────────────────────

def _notifier(settings: Settings) -> TelegramNotifier:
    return TelegramNotifier(settings)


def _configured_settings(default_settings: Settings) -> Settings:
    return default_settings.model_copy(
        update={
            "telegram_bot_token": "1234567890:ABCDEFghijklmNOPQRSTuvwxyz",
            "telegram_chat_id": "987654321",
        }
    )


# ── No-op when unconfigured ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_notifier_noop_when_no_token(default_settings):
    notifier = _notifier(default_settings)
    assert not notifier._enabled
    # Should not raise
    await notifier.bot_started(10_000.0, StrategyMode.MEAN_REVERSION, False)


@pytest.mark.asyncio
async def test_notifier_noop_when_placeholder_token(default_settings):
    s = default_settings.model_copy(
        update={
            "telegram_bot_token": "your_bot_token_here",
            "telegram_chat_id": "your_chat_id_here",
        }
    )
    notifier = _notifier(s)
    assert not notifier._enabled


# ── Enabled when properly configured ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_notifier_enabled_when_configured(default_settings):
    notifier = _notifier(_configured_settings(default_settings))
    assert notifier._enabled


# ── Actual send invokes Bot.send_message ──────────────────────────────────────

@pytest.mark.asyncio
async def test_send_calls_telegram_api(default_settings):
    notifier = _notifier(_configured_settings(default_settings))

    mock_bot = AsyncMock()
    mock_bot.__aenter__ = AsyncMock(return_value=mock_bot)
    mock_bot.__aexit__ = AsyncMock(return_value=False)

    with patch("telegram.Bot", return_value=mock_bot):
        await notifier._send("hello")

    mock_bot.send_message.assert_awaited_once()
    call_kwargs = mock_bot.send_message.call_args.kwargs
    assert call_kwargs["text"] == "hello"
    assert call_kwargs["chat_id"] == "987654321"


@pytest.mark.asyncio
async def test_send_swallows_telegram_exception(default_settings):
    notifier = _notifier(_configured_settings(default_settings))

    mock_bot = AsyncMock()
    mock_bot.__aenter__ = AsyncMock(return_value=mock_bot)
    mock_bot.__aexit__ = AsyncMock(return_value=False)
    mock_bot.send_message.side_effect = Exception("network error")

    with patch("telegram.Bot", return_value=mock_bot):
        # Should not raise
        await notifier._send("hello")


# ── Throttling ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_error_throttle_suppresses_duplicate(default_settings):
    notifier = _notifier(_configured_settings(default_settings))
    sent: list[str] = []

    async def fake_send(text: str) -> None:
        sent.append(text)

    notifier._send = fake_send

    await notifier.error("key1", "first")
    await notifier.error("key1", "second")  # within 60s window — throttled
    assert len(sent) == 1


@pytest.mark.asyncio
async def test_error_throttle_allows_different_keys(default_settings):
    notifier = _notifier(_configured_settings(default_settings))
    sent: list[str] = []

    async def fake_send(text: str) -> None:
        sent.append(text)

    notifier._send = fake_send

    await notifier.error("key1", "first")
    await notifier.error("key2", "second")  # different key — allowed
    assert len(sent) == 2


# ── Message content ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_trade_opened_message_contains_symbol(default_settings):
    notifier = _notifier(default_settings)
    sent: list[str] = []

    async def fake_send(text: str) -> None:
        sent.append(text)

    notifier._send = fake_send

    await notifier.trade_opened(
        symbol="BTCUSDT",
        side="long",
        mode=StrategyMode.MEAN_REVERSION,
        entry_price=40_000.0,
        quantity=0.1,
        sl_price=39_500.0,
        tp1_price=40_500.0,
        leverage=5,
        dollar_risk=50.0,
    )
    assert len(sent) == 1
    assert "BTCUSDT" in sent[0]
    assert "LONG" in sent[0]


@pytest.mark.asyncio
async def test_trade_closed_pnl_in_message(default_settings):
    notifier = _notifier(default_settings)
    sent: list[str] = []

    async def fake_send(text: str) -> None:
        sent.append(text)

    notifier._send = fake_send

    await notifier.trade_closed(
        symbol="BTCUSDT",
        side="long",
        entry_price=40_000.0,
        exit_price=40_500.0,
        quantity=0.1,
        pnl=50.0,
        pnl_pct=10.0,
        reason=CloseReason.TAKE_PROFIT,
    )
    assert "$+50.00" in sent[0]
    assert "TAKE PROFIT" in sent[0]
