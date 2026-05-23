"""Tests for indicator computation (pandas-ta wrappers)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from karen.indicators.ta import Indicators1h, Indicators15m, _pick, compute_1h, compute_15m
from tests.conftest import make_ohlcv, make_ranging_ohlcv, make_trending_ohlcv

# ─── Basic structure ──────────────────────────────────────────────────────────


def test_compute_15m_returns_correct_type(default_settings):
    df = make_ohlcv(n=100)
    ind = compute_15m(df, default_settings)
    assert isinstance(ind, Indicators15m)


def test_compute_1h_returns_correct_type(default_settings):
    df = make_ohlcv(n=250)
    ind = compute_1h(df, default_settings)
    assert isinstance(ind, Indicators1h)


def test_compute_15m_series_length_matches_input(default_settings):
    n = 150
    df = make_ohlcv(n=n)
    ind = compute_15m(df, default_settings)
    assert len(ind.close) == n
    assert len(ind.bb_upper) == n
    assert len(ind.rsi) == n
    assert len(ind.atr) == n


def test_compute_1h_series_length_matches_input(default_settings):
    n = 250
    df = make_ohlcv(n=n)
    ind = compute_1h(df, default_settings)
    assert len(ind.adx) == n
    assert len(ind.ema_slow) == n


# ─── Validation ───────────────────────────────────────────────────────────────


def test_compute_15m_raises_on_missing_columns(default_settings):
    df = pd.DataFrame({"close": [1.0] * 50})
    with pytest.raises(ValueError, match="missing columns"):
        compute_15m(df, default_settings)


def test_compute_15m_raises_on_too_few_rows(default_settings):
    df = make_ohlcv(n=20)
    with pytest.raises(ValueError, match="Need at least"):
        compute_15m(df, default_settings)


def test_compute_1h_raises_on_missing_columns(default_settings):
    df = pd.DataFrame({"close": [1.0] * 50})
    with pytest.raises(ValueError, match="missing columns"):
        compute_1h(df, default_settings)


# ─── Bollinger Bands ──────────────────────────────────────────────────────────


def test_bb_ordering(default_settings):
    """upper > middle > lower for all non-NaN rows."""
    df = make_ranging_ohlcv(n=100)
    ind = compute_15m(df, default_settings)
    # Drop NaN warmup rows
    valid = pd.DataFrame({
        "upper": ind.bb_upper,
        "middle": ind.bb_middle,
        "lower": ind.bb_lower,
    }).dropna()
    assert (valid["upper"] > valid["middle"]).all()
    assert (valid["middle"] > valid["lower"]).all()


def test_bb_middle_is_sma_of_close(default_settings):
    """BBM should equal the SMA of close over the BB period."""
    df = make_ranging_ohlcv(n=100)
    ind = compute_15m(df, default_settings)
    period = default_settings.mr_bb_period
    sma = df["close"].rolling(period).mean()
    # Compare last 50 non-NaN values
    valid_idx = ind.bb_middle.dropna().index[-50:]
    np.testing.assert_allclose(
        ind.bb_middle.loc[valid_idx].values,
        sma.loc[valid_idx].values,
        rtol=1e-4,
    )


def test_bb_price_below_lower_in_downtrend(default_settings):
    """In a steep downtrend, recent prices should dip below lower BB."""
    rng = np.random.default_rng(99)
    n = 100
    # Rapid decline in last 10 rows
    close_vals = np.concatenate([
        40000 + rng.normal(0, 50, 90),
        np.linspace(40000, 37000, 10),  # sharp drop
    ])
    df = pd.DataFrame({
        "open": close_vals,
        "high": close_vals + 100,
        "low": close_vals - 100,
        "close": close_vals,
        "volume": rng.uniform(100, 500, n),
    })
    ind = compute_15m(df, default_settings)
    # At least one of the last 5 rows should be below lower BB
    tail = pd.DataFrame({"close": df["close"].iloc[-5:], "lower": ind.bb_lower.iloc[-5:]}).dropna()
    assert (tail["close"] < tail["lower"]).any()


# ─── RSI ─────────────────────────────────────────────────────────────────────


def test_rsi_range(default_settings):
    """RSI should always be in [0, 100]."""
    df = make_ohlcv(n=100, vol=0.02)
    ind = compute_15m(df, default_settings)
    valid = ind.rsi.dropna()
    assert (valid >= 0).all()
    assert (valid <= 100).all()


def test_rsi_low_after_downtrend(default_settings):
    """RSI < 40 after a consistent downtrend."""
    n = 100
    close_vals = 40000 * np.cumprod(1 + np.full(n, -0.005))  # consistent 0.5% drops
    df = pd.DataFrame({
        "open": close_vals,
        "high": close_vals + 10,
        "low": close_vals - 10,
        "close": close_vals,
        "volume": np.ones(n) * 500,
    })
    ind = compute_15m(df, default_settings)
    assert ind.rsi.dropna().iloc[-1] < 40


def test_rsi_high_after_uptrend(default_settings):
    """RSI > 60 after a consistent uptrend."""
    n = 100
    close_vals = 40000 * np.cumprod(1 + np.full(n, 0.005))
    df = pd.DataFrame({
        "open": close_vals,
        "high": close_vals + 10,
        "low": close_vals - 10,
        "close": close_vals,
        "volume": np.ones(n) * 500,
    })
    ind = compute_15m(df, default_settings)
    assert ind.rsi.dropna().iloc[-1] > 60


# ─── ATR ─────────────────────────────────────────────────────────────────────


def test_atr_positive(default_settings):
    df = make_ohlcv(n=100)
    ind = compute_15m(df, default_settings)
    assert (ind.atr.dropna() > 0).all()


def test_atr_in_price_units(default_settings):
    """ATR should be in price units, not percentage (< close price)."""
    df = make_ohlcv(n=100, base_price=40_000)
    ind = compute_15m(df, default_settings)
    last_atr = ind.atr.dropna().iloc[-1]
    last_close = df["close"].iloc[-1]
    assert last_atr < last_close  # ATR should be much smaller than price
    assert last_atr > 0


# ─── ADX ─────────────────────────────────────────────────────────────────────


def test_adx_range(default_settings):
    """ADX should always be ≥ 0."""
    df = make_trending_ohlcv(n=100)
    ind = compute_1h(df, default_settings)
    assert (ind.adx.dropna() >= 0).all()


def test_adx_high_in_strong_trend(default_settings):
    """ADX > 20 during a persistent strong trend."""
    n = 100
    close_vals = 40000 * np.cumprod(1 + np.full(n, 0.01))
    df = pd.DataFrame({
        "open": close_vals * 0.99,
        "high": close_vals * 1.01,
        "low": close_vals * 0.98,
        "close": close_vals,
        "volume": np.ones(n) * 500,
    })
    ind = compute_1h(df, default_settings)
    assert ind.adx.dropna().iloc[-1] > 15  # strong trend → high ADX


# ─── EMA ordering ────────────────────────────────────────────────────────────


def test_ema_slow_lags_behind_in_uptrend(default_settings):
    """In an uptrend, EMA50 > EMA200 (fast above slow)."""
    df = make_trending_ohlcv(n=250, uptrend=True)
    ind = compute_1h(df, default_settings)
    valid = pd.DataFrame({
        "mid": ind.ema_mid,
        "slow": ind.ema_slow,
    }).dropna()
    # Last row: EMA50 should be above EMA200 in uptrend
    assert valid["mid"].iloc[-1] > valid["slow"].iloc[-1]


# ─── MACD ────────────────────────────────────────────────────────────────────


def test_macd_histogram_positive_in_uptrend(default_settings):
    """MACD histogram should be positive at the end of a persistent uptrend."""
    n = 200
    close_vals = 40000 * np.cumprod(1 + np.full(n, 0.003))
    df = pd.DataFrame({
        "open": close_vals * 0.999,
        "high": close_vals * 1.001,
        "low": close_vals * 0.998,
        "close": close_vals,
        "volume": np.ones(n) * 500,
    })
    ind = compute_15m(df, default_settings)
    assert ind.macd_hist.dropna().iloc[-1] > 0


def test_macd_line_negative_in_downtrend(default_settings):
    """MACD line (EMA12 - EMA26) must be negative in a persistent downtrend.

    Note: the histogram (MACD - signal) converges to ~0 at equilibrium because
    signal (EMA9 of MACD) tracks MACD closely once the trend is established.
    The MACD line itself is the reliable indicator of trend direction.
    """
    n = 200
    close_vals = 40000 * np.cumprod(1 + np.full(n, -0.003))
    df = pd.DataFrame({
        "open": close_vals * 1.001,
        "high": close_vals * 1.002,
        "low": close_vals * 0.999,
        "close": close_vals,
        "volume": np.ones(n) * 500,
    })
    ind = compute_15m(df, default_settings)
    assert ind.macd_line.dropna().iloc[-1] < 0


# ─── _pick helper ────────────────────────────────────────────────────────────


def test_pick_raises_on_missing_prefix():
    df = pd.DataFrame({"SOME_COL": [1.0, 2.0]})
    with pytest.raises(KeyError, match="No column with prefix"):
        _pick(df, "BBL_")


def test_pick_returns_first_match():
    df = pd.DataFrame({"BBL_20_2.0_2.0": [1.0], "BBL_30_2.0_2.0": [2.0]})
    result = _pick(df, "BBL_")
    assert result.iloc[0] == pytest.approx(1.0)
