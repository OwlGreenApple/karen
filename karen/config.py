"""Configuration with pydantic-settings and watchdog-based hot-reload."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from watchdog.events import FileModifiedEvent, FileSystemEventHandler
from watchdog.observers import Observer

ENV_PATH = Path(__file__).parent.parent / ".env"


class StrategyMode(StrEnum):
    MEAN_REVERSION = "mean_reversion"
    TREND_FOLLOWING = "trend_following"
    SCALP_1M = "scalp_1m"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Binance
    binance_api_key: str = Field(default="")
    binance_api_secret: str = Field(default="")
    binance_testnet: bool = Field(default=True)

    # Master switches
    trading_enabled: bool = Field(default=False)
    strategy_mode: StrategyMode = Field(default=StrategyMode.MEAN_REVERSION)

    # Trading pairs
    symbols: str = Field(
        default="BTCUSDT,ETHUSDT,DOTUSDT,HYPEUSDT,AAVEUSDT,DOGEUSDT,XRPUSDT,BNBUSDT,ALGOUSDT"
    )

    # Risk params
    leverage: int = Field(default=20, ge=1, le=125)
    position_size_pct: float = Field(default=10.0, gt=0, le=100)
    risk_per_trade_pct: float = Field(default=1.0, gt=0, le=10)
    max_concurrent_positions: int = Field(default=3, ge=1, le=20)
    max_daily_loss_pct: float = Field(default=5.0, gt=0, le=50)
    max_trades_per_day: int = Field(default=25, ge=1)

    # Mean Reversion params
    mr_bb_period: int = Field(default=20, ge=5)
    mr_bb_std: float = Field(default=2.0, gt=0)
    mr_rsi_period: int = Field(default=14, ge=2)
    mr_rsi_oversold: float = Field(default=30.0, gt=0, lt=50)
    mr_rsi_overbought: float = Field(default=70.0, gt=50, lt=100)
    mr_adx_max: float = Field(default=20.0, gt=0)

    # Trend Following params
    tf_ema_fast: int = Field(default=21, ge=5)
    tf_ema_mid: int = Field(default=50, ge=10)
    tf_ema_slow: int = Field(default=200, ge=50)
    tf_adx_min: float = Field(default=25.0, gt=0)
    tf_volume_multiplier: float = Field(default=1.2, gt=0)

    # Scalp (1m signal, 5m filter) params
    scalp_ema_fast: int = Field(default=9, ge=2)
    scalp_ema_slow: int = Field(default=21, ge=5)
    scalp_rsi_period: int = Field(default=7, ge=2)
    scalp_volume_multiplier: float = Field(default=1.3, gt=0)
    scalp_atr_period: int = Field(default=7, ge=2)
    scalp_tp_atr_mult: float = Field(default=1.5, gt=0)
    scalp_sl_atr_mult: float = Field(default=1.0, gt=0)
    scalp_filter_ema: int = Field(default=20, ge=5)
    scalp_5m_adx_min: float = Field(default=15.0, gt=0)

    # Telegram
    telegram_bot_token: str = Field(default="")
    telegram_chat_id: str = Field(default="")

    # Dashboard
    dashboard_host: str = Field(default="0.0.0.0")
    dashboard_port: int = Field(default=8000, ge=1024, le=65535)
    dashboard_url: str = Field(default="http://localhost:8000")

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, v: str) -> str:
        parts = [s.strip() for s in v.split(",") if s.strip()]
        if not parts:
            raise ValueError("SYMBOLS must contain at least one trading pair")
        return ",".join(parts)

    @property
    def symbol_list(self) -> list[str]:
        return [s.strip() for s in self.symbols.split(",") if s.strip()]


# ─── Hot-reload machinery ─────────────────────────────────────────────────────

ChangeCallback = Callable[[Settings, Settings], Any]


class _EnvFileHandler(FileSystemEventHandler):
    """Watchdog handler that debounces .env changes and fires callbacks."""

    _DEBOUNCE_SECONDS = 0.5

    def __init__(self, env_path: Path, loop: asyncio.AbstractEventLoop) -> None:
        super().__init__()
        self._env_path = env_path.resolve()
        self._loop = loop
        self._last_fired: float = 0.0
        self._callbacks: list[ChangeCallback] = []
        self._current: Settings = Settings()

    def add_callback(self, cb: ChangeCallback) -> None:
        self._callbacks.append(cb)

    def on_modified(self, event: FileModifiedEvent) -> None:  # type: ignore[override]
        if not isinstance(event, FileModifiedEvent):
            return
        changed = Path(str(event.src_path)).resolve()
        if changed != self._env_path:
            return
        now = time.monotonic()
        if now - self._last_fired < self._DEBOUNCE_SECONDS:
            return
        self._last_fired = now
        self._loop.call_soon_threadsafe(self._schedule_reload)

    def _schedule_reload(self) -> None:
        asyncio.ensure_future(self._reload(), loop=self._loop)

    async def _reload(self) -> None:
        try:
            # Force re-read from disk by clearing cached env vars that pydantic may have cached
            new_settings = _load_fresh_settings()
            old_settings = self._current
            if _settings_differ(old_settings, new_settings):
                logger.info("Config reloaded from .env")
                self._current = new_settings
                for cb in self._callbacks:
                    try:
                        result = cb(old_settings, new_settings)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception:
                        logger.exception("Error in config change callback")
        except Exception:
            logger.exception("Failed to reload .env")


def _load_fresh_settings() -> Settings:
    """Load a fresh Settings instance, bypassing any os-level caching."""
    # Re-read the .env file manually so pydantic-settings picks up the latest values.
    # We temporarily clear any env vars that might shadow the file values.
    env_vars: dict[str, str] = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            env_vars[key.upper()] = val

    saved: dict[str, str | None] = {}
    for key, val in env_vars.items():
        saved[key] = os.environ.get(key)
        os.environ[key] = val

    try:
        return Settings()
    finally:
        for key, original in saved.items():
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original


def _settings_differ(a: Settings, b: Settings) -> bool:
    return a.model_dump() != b.model_dump()


class ConfigWatcher:
    """Starts watchdog observer and manages hot-reload lifecycle."""

    def __init__(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._loop = loop or asyncio.get_event_loop()
        self._handler = _EnvFileHandler(ENV_PATH, self._loop)
        self._observer: Observer | None = None
        # Initialise current settings
        self._handler._current = _load_fresh_settings()

    @property
    def settings(self) -> Settings:
        return self._handler._current

    def add_callback(self, cb: ChangeCallback) -> None:
        self._handler.add_callback(cb)

    def start(self) -> None:
        self._observer = Observer()
        watch_dir = str(ENV_PATH.parent)
        self._observer.schedule(self._handler, watch_dir, recursive=False)
        self._observer.start()
        logger.info(f"Config watcher started — watching {ENV_PATH}")

    def stop(self) -> None:
        if self._observer:
            self._observer.stop()
            self._observer.join()
            logger.info("Config watcher stopped")


# Module-level singleton — replaced during tests
_watcher: ConfigWatcher | None = None


def get_settings() -> Settings:
    """Return the current (possibly hot-reloaded) settings."""
    if _watcher is not None:
        return _watcher.settings
    return Settings()


def init_config_watcher(loop: asyncio.AbstractEventLoop | None = None) -> ConfigWatcher:
    global _watcher
    _watcher = ConfigWatcher(loop=loop)
    return _watcher
