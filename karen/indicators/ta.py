"""Indicator computations via pandas-ta."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pandas_ta as ta

from karen.config import Settings

# ─── Result dataclasses ───────────────────────────────────────────────────────


@dataclass
class Indicators15m:
    """All indicators computed from the 15-minute OHLCV DataFrame."""

    # Bollinger Bands
    bb_upper: pd.Series
    bb_middle: pd.Series
    bb_lower: pd.Series
    # RSI
    rsi: pd.Series
    # ATR (absolute, price units)
    atr: pd.Series
    # EMA
    ema_fast: pd.Series    # default EMA21 for trend following
    # MACD
    macd_line: pd.Series
    macd_hist: pd.Series
    macd_signal: pd.Series
    # Volume SMA-20
    vol_ma: pd.Series
    # Raw OHLCV pass-through
    close: pd.Series
    high: pd.Series
    low: pd.Series
    volume: pd.Series


@dataclass
class Indicators1m:
    """All indicators computed from the 1-minute OHLCV DataFrame."""

    ema_fast: pd.Series    # EMA9 — fast crossover line
    ema_slow: pd.Series    # EMA21 — slow crossover line
    rsi: pd.Series         # RSI(7) — short momentum
    atr: pd.Series         # ATR(7) — for SL/TP sizing
    vol_ma: pd.Series      # SMA-20 of volume
    close: pd.Series
    high: pd.Series
    low: pd.Series
    volume: pd.Series


@dataclass
class Indicators5m:
    """All indicators computed from the 5-minute OHLCV DataFrame (regime filter)."""

    ema_filter: pd.Series  # EMA20 — bias direction
    adx: pd.Series         # ADX(14) — trend strength
    close: pd.Series


@dataclass
class Indicators1h:
    """All indicators computed from the 1-hour OHLCV DataFrame."""

    # ADX
    adx: pd.Series
    # EMAs for trend detection
    ema_mid: pd.Series     # EMA50
    ema_slow: pd.Series    # EMA200
    # ATR
    atr: pd.Series
    # Raw close
    close: pd.Series


# ─── Public compute functions ─────────────────────────────────────────────────


def compute_15m(df: pd.DataFrame, settings: Settings) -> Indicators15m:
    """Compute all 15m indicators needed by both strategies."""
    _validate(df, min_rows=30)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # Bollinger Bands — use prefix search because pandas-ta encodes std twice
    # e.g., BBL_20_2.0_2.0 in some versions
    bb_df = ta.bbands(close, length=settings.mr_bb_period, std=settings.mr_bb_std)
    bb_upper = _pick(bb_df, "BBU_")
    bb_middle = _pick(bb_df, "BBM_")
    bb_lower = _pick(bb_df, "BBL_")

    # RSI
    rsi = _coerce(ta.rsi(close, length=settings.mr_rsi_period))

    # ATR — named ATRr_14; values ARE in price units (Wilder's RMA smoothing)
    atr = _coerce(ta.atr(high, low, close, length=14))

    # EMA fast
    ema_fast = _coerce(ta.ema(close, length=settings.tf_ema_fast))

    # MACD (12/26/9)
    macd_df = ta.macd(close)
    macd_line = _pick(macd_df, "MACD_")
    macd_hist = _pick(macd_df, "MACDh_")
    macd_signal = _pick(macd_df, "MACDs_")

    # Volume SMA
    vol_ma = _coerce(ta.sma(volume, length=20))

    return Indicators15m(
        bb_upper=bb_upper,
        bb_middle=bb_middle,
        bb_lower=bb_lower,
        rsi=rsi,
        atr=atr,
        ema_fast=ema_fast,
        macd_line=macd_line,
        macd_hist=macd_hist,
        macd_signal=macd_signal,
        vol_ma=vol_ma,
        close=close,
        high=high,
        low=low,
        volume=volume,
    )


def compute_1m(df: pd.DataFrame, settings: Settings) -> Indicators1m:
    """Compute all 1m indicators needed by ScalpStrategy."""
    _validate(df, min_rows=30)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    ema_fast = _coerce(ta.ema(close, length=settings.scalp_ema_fast))
    ema_slow = _coerce(ta.ema(close, length=settings.scalp_ema_slow))
    rsi = _coerce(ta.rsi(close, length=settings.scalp_rsi_period))
    atr = _coerce(ta.atr(high, low, close, length=settings.scalp_atr_period))
    vol_ma = _coerce(ta.sma(volume, length=20))

    return Indicators1m(
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        rsi=rsi,
        atr=atr,
        vol_ma=vol_ma,
        close=close,
        high=high,
        low=low,
        volume=volume,
    )


def compute_5m(df: pd.DataFrame, settings: Settings) -> Indicators5m:
    """Compute 5m regime indicators needed by ScalpStrategy."""
    _validate(df, min_rows=30)

    close = df["close"]
    high = df["high"]
    low = df["low"]

    ema_filter = _coerce(ta.ema(close, length=settings.scalp_filter_ema))
    adx_df = ta.adx(high, low, close, length=14)
    adx = _pick(adx_df, "ADX_")

    return Indicators5m(
        ema_filter=ema_filter,
        adx=adx,
        close=close,
    )


def compute_1h(df: pd.DataFrame, settings: Settings) -> Indicators1h:
    """Compute all 1h indicators needed by both strategies."""
    _validate(df, min_rows=30)

    close = df["close"]
    high = df["high"]
    low = df["low"]

    # ADX
    adx_df = ta.adx(high, low, close, length=14)
    adx = _pick(adx_df, "ADX_")

    # EMAs
    ema_mid = _coerce(ta.ema(close, length=settings.tf_ema_mid))
    ema_slow = _coerce(ta.ema(close, length=settings.tf_ema_slow))

    # ATR
    atr = _coerce(ta.atr(high, low, close, length=14))

    return Indicators1h(
        adx=adx,
        ema_mid=ema_mid,
        ema_slow=ema_slow,
        atr=atr,
        close=close,
    )


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _validate(df: pd.DataFrame, min_rows: int = 30) -> None:
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")
    if len(df) < min_rows:
        raise ValueError(f"Need at least {min_rows} rows, got {len(df)}")


def _pick(df: pd.DataFrame | None, prefix: str) -> pd.Series:
    """Return the first column whose name starts with `prefix`."""
    if df is None:
        return pd.Series(dtype=float)
    matches = [c for c in df.columns if c.startswith(prefix)]
    if not matches:
        raise KeyError(f"No column with prefix '{prefix}' in {list(df.columns)}")
    return df[matches[0]]


def _coerce(s: pd.Series | None) -> pd.Series:
    return s if s is not None else pd.Series(dtype=float)
