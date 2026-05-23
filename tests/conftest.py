"""Shared test fixtures and DataFrame builders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from karen.config import Settings, StrategyMode
from karen.indicators.ta import Indicators1h, Indicators15m

# ─── Settings fixture ─────────────────────────────────────────────────────────


@pytest.fixture
def default_settings() -> Settings:
    return Settings(
        binance_api_key="test",
        binance_api_secret="test",
        binance_testnet=True,
        trading_enabled=False,
        strategy_mode=StrategyMode.MEAN_REVERSION,
        leverage=5,
        risk_per_trade_pct=1.0,
        max_concurrent_positions=3,
        max_daily_loss_pct=5.0,
        max_trades_per_day=25,
        mr_bb_period=20,
        mr_bb_std=2.0,
        mr_rsi_period=14,
        mr_rsi_oversold=30.0,
        mr_rsi_overbought=70.0,
        mr_adx_max=20.0,
        tf_ema_fast=21,
        tf_ema_mid=50,
        tf_ema_slow=200,
        tf_adx_min=25.0,
        tf_volume_multiplier=1.2,
    )


# ─── OHLCV DataFrame builders ─────────────────────────────────────────────────


def make_ohlcv(
    n: int = 250,
    base_price: float = 40_000.0,
    drift: float = 0.0,
    vol: float = 0.005,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame with n rows."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(drift, vol, n)
    close = base_price * np.cumprod(1 + returns)

    candle_range = close * rng.uniform(0.001, 0.008, n)
    high = close + candle_range * rng.uniform(0.3, 1.0, n)
    low = close - candle_range * rng.uniform(0.3, 1.0, n)
    open_ = np.roll(close, 1)
    open_[0] = base_price
    volume = rng.uniform(200, 2000, n)

    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def make_ranging_ohlcv(n: int = 250, base_price: float = 40_000.0) -> pd.DataFrame:
    """Sideways price action — suitable for MR strategy testing."""
    return make_ohlcv(n=n, base_price=base_price, drift=0.0, vol=0.003, seed=1)


def make_trending_ohlcv(
    n: int = 250,
    base_price: float = 40_000.0,
    uptrend: bool = True,
) -> pd.DataFrame:
    """Strong trend — suitable for TF strategy testing."""
    drift = 0.002 if uptrend else -0.002
    return make_ohlcv(n=n, base_price=base_price, drift=drift, vol=0.004, seed=2)


# ─── Mock indicator factories ─────────────────────────────────────────────────


def make_ind15m(
    n: int = 50,
    close: float = 40_000.0,
    bb_upper: float = 41_000.0,
    bb_middle: float = 40_500.0,
    bb_lower: float = 40_000.0,
    rsi: float = 50.0,
    atr: float = 300.0,
    ema_fast: float = 40_300.0,
    macd_hist: float = 0.0,
    macd_hist_prev: float = 0.0,
    vol: float = 500.0,
    vol_ma: float = 400.0,
    prev_low: float | None = None,
    prev_high: float | None = None,
) -> Indicators15m:
    """
    Build a minimal Indicators15m with controlled last-two values.
    Used by strategy unit tests to test each signal condition independently.
    """
    _close = _const_series(close, n)
    _high = _const_series(close + 200, n)
    _low = _const_series(close - 200, n)
    _volume = _const_series(vol, n)

    # Override second-to-last row for prev_low / prev_high
    if prev_low is not None:
        _low.iloc[-2] = prev_low
    if prev_high is not None:
        _high.iloc[-2] = prev_high

    # MACD hist: prev then curr
    _macd_hist = _const_series(macd_hist, n)
    _macd_hist.iloc[-2] = macd_hist_prev

    return Indicators15m(
        bb_upper=_const_series(bb_upper, n),
        bb_middle=_const_series(bb_middle, n),
        bb_lower=_const_series(bb_lower, n),
        rsi=_const_series(rsi, n),
        atr=_const_series(atr, n),
        ema_fast=_const_series(ema_fast, n),
        macd_line=_const_series(0.0, n),
        macd_hist=_macd_hist,
        macd_signal=_const_series(0.0, n),
        vol_ma=_const_series(vol_ma, n),
        close=_close,
        high=_high,
        low=_low,
        volume=_volume,
    )


def make_ind1h(
    n: int = 50,
    adx: float = 15.0,
    ema_mid: float = 40_200.0,
    ema_slow: float = 39_800.0,
    atr: float = 400.0,
    close: float = 40_000.0,
) -> Indicators1h:
    return Indicators1h(
        adx=_const_series(adx, n),
        ema_mid=_const_series(ema_mid, n),
        ema_slow=_const_series(ema_slow, n),
        atr=_const_series(atr, n),
        close=_const_series(close, n),
    )


def _const_series(val: float, n: int) -> pd.Series:
    return pd.Series([val] * n, dtype=float)
