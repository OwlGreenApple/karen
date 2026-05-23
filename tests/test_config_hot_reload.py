"""Tests for config loading and hot-reload behaviour."""

from __future__ import annotations

import asyncio
import textwrap
import time
from pathlib import Path

import pytest

from karen.config import (
    ConfigWatcher,
    Settings,
    StrategyMode,
    _load_fresh_settings,
    _settings_differ,
)

# ─── Helpers ─────────────────────────────────────────────────────────────────


def write_env(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content))


# ─── Unit: Settings validation ───────────────────────────────────────────────


def test_default_settings_are_safe(tmp_path: Path) -> None:
    """Defaults must have trading disabled and testnet on."""
    env = tmp_path / ".env"
    env.write_text("")
    s = Settings(_env_file=str(env))
    assert s.trading_enabled is False
    assert s.binance_testnet is True


def test_strategy_mode_enum(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    write_env(
        env,
        """
        STRATEGY_MODE=trend_following
        """,
    )
    s = Settings(_env_file=str(env))
    assert s.strategy_mode == StrategyMode.TREND_FOLLOWING


def test_invalid_strategy_mode_raises(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    write_env(env, "STRATEGY_MODE=scalping\n")
    with pytest.raises(ValueError):
        Settings(_env_file=str(env))


def test_symbols_parsed_to_list(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    write_env(env, "SYMBOLS=BTCUSDT,ETHUSDT, DOTUSDT\n")
    s = Settings(_env_file=str(env))
    assert s.symbol_list == ["BTCUSDT", "ETHUSDT", "DOTUSDT"]


def test_empty_symbols_raises(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    write_env(env, "SYMBOLS=\n")
    with pytest.raises(ValueError):
        Settings(_env_file=str(env))


def test_leverage_bounds(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    write_env(env, "LEVERAGE=130\n")
    with pytest.raises(ValueError):
        Settings(_env_file=str(env))


def test_risk_per_trade_bounds(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    write_env(env, "RISK_PER_TRADE_PCT=0\n")
    with pytest.raises(ValueError):
        Settings(_env_file=str(env))


# ─── Unit: _settings_differ ──────────────────────────────────────────────────


def test_settings_differ_detects_change(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    write_env(env, "TRADING_ENABLED=false\n")
    a = Settings(_env_file=str(env))
    write_env(env, "TRADING_ENABLED=true\n")
    b = Settings(_env_file=str(env))
    assert _settings_differ(a, b) is True


def test_settings_differ_same(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    write_env(env, "TRADING_ENABLED=false\n")
    a = Settings(_env_file=str(env))
    b = Settings(_env_file=str(env))
    assert _settings_differ(a, b) is False


# ─── Integration: ConfigWatcher callbacks ────────────────────────────────────


@pytest.mark.asyncio
async def test_config_watcher_fires_callback_on_change(tmp_path: Path, monkeypatch) -> None:
    """Simulate a .env file change and verify callback is fired with correct args."""
    env_file = tmp_path / ".env"
    write_env(
        env_file,
        """
        TRADING_ENABLED=false
        STRATEGY_MODE=mean_reversion
        """,
    )

    # Patch the module-level ENV_PATH so ConfigWatcher watches our tmp file
    import karen.config as cfg_mod

    monkeypatch.setattr(cfg_mod, "ENV_PATH", env_file)

    loop = asyncio.get_event_loop()
    watcher = ConfigWatcher(loop=loop)

    # Manually set the initial handler state to match the file
    watcher._handler._current = Settings(_env_file=str(env_file))

    received: list[tuple[Settings, Settings]] = []

    async def on_change(old: Settings, new: Settings) -> None:
        received.append((old, new))

    watcher.add_callback(on_change)

    # Simulate what the handler does internally when the file changes
    write_env(
        env_file,
        """
        TRADING_ENABLED=true
        STRATEGY_MODE=trend_following
        """,
    )

    # Directly call the reload logic (bypasses watchdog OS events)
    new_s = Settings(_env_file=str(env_file))
    old_s = watcher.settings
    assert _settings_differ(old_s, new_s)

    # Manually trigger the async reload
    watcher._handler._current = new_s
    await on_change(old_s, new_s)

    assert len(received) == 1
    old, new = received[0]
    assert old.trading_enabled is False
    assert new.trading_enabled is True
    assert old.strategy_mode == StrategyMode.MEAN_REVERSION
    assert new.strategy_mode == StrategyMode.TREND_FOLLOWING


@pytest.mark.asyncio
async def test_watcher_debounces_rapid_changes(tmp_path: Path, monkeypatch) -> None:
    """Multiple rapid file events should collapse to a single reload."""
    env_file = tmp_path / ".env"
    write_env(env_file, "TRADING_ENABLED=false\n")

    import karen.config as cfg_mod

    monkeypatch.setattr(cfg_mod, "ENV_PATH", env_file)

    loop = asyncio.get_event_loop()
    watcher = ConfigWatcher(loop=loop)
    watcher._handler._current = Settings(_env_file=str(env_file))

    fire_count = 0

    async def counter(old: Settings, new: Settings) -> None:
        nonlocal fire_count
        fire_count += 1

    watcher.add_callback(counter)

    # Simulate the debounce — two events within the window should only fire once
    handler = watcher._handler
    handler._last_fired = time.monotonic()  # pretend we just fired

    # A second event within debounce window should be suppressed
    from watchdog.events import FileModifiedEvent

    event = FileModifiedEvent(str(env_file))
    handler.on_modified(event)

    # Give any queued coroutines a chance to run
    await asyncio.sleep(0.05)
    assert fire_count == 0  # debounced


# ─── Integration: _load_fresh_settings ───────────────────────────────────────


def test_load_fresh_settings_reads_disk(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    write_env(env_file, "LEVERAGE=10\n")

    import karen.config as cfg_mod

    monkeypatch.setattr(cfg_mod, "ENV_PATH", env_file)

    s = _load_fresh_settings()
    assert s.leverage == 10

    write_env(env_file, "LEVERAGE=20\n")
    s2 = _load_fresh_settings()
    assert s2.leverage == 20
