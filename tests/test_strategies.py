"""Tests for MeanReversion and TrendFollowing strategies."""

from __future__ import annotations

import pytest

from karen.config import StrategyMode
from karen.strategies.base import Signal
from karen.strategies.mean_reversion import MeanReversionStrategy
from karen.strategies.trend_following import TrendFollowingStrategy, _swing_sl
from tests.conftest import (
    make_ind1h,
    make_ind15m,
    make_ohlcv,
    make_ranging_ohlcv,
    make_trending_ohlcv,
)

# ─── Helpers ──────────────────────────────────────────────────────────────────


def make_mr(settings, ind_15m=None, ind_1h=None):
    """Build MeanReversionStrategy with injected indicator functions."""
    return MeanReversionStrategy(
        settings,
        compute_15m_fn=lambda df, s: ind_15m if ind_15m is not None else make_ind15m(),
        compute_1h_fn=lambda df, s: ind_1h if ind_1h is not None else make_ind1h(),
    )


def make_tf(settings, ind_15m=None, ind_1h=None):
    """Build TrendFollowingStrategy with injected indicator functions."""
    return TrendFollowingStrategy(
        settings,
        compute_15m_fn=lambda df, s: ind_15m if ind_15m is not None else make_ind15m(),
        compute_1h_fn=lambda df, s: ind_1h if ind_1h is not None else make_ind1h(),
    )


def dummy_df(n=100):
    return make_ohlcv(n=n)


# ─── Strategy mode property ───────────────────────────────────────────────────


def test_mr_strategy_mode(default_settings):
    s = MeanReversionStrategy(default_settings)
    assert s.mode == StrategyMode.MEAN_REVERSION


def test_tf_strategy_mode(default_settings):
    s = TrendFollowingStrategy(default_settings)
    assert s.mode == StrategyMode.TREND_FOLLOWING


# ─── MeanReversion: insufficient data ────────────────────────────────────────


@pytest.mark.asyncio
async def test_mr_returns_none_if_too_few_15m_rows(default_settings):
    strat = MeanReversionStrategy(default_settings)
    signal = await strat.evaluate("BTCUSDT", make_ohlcv(n=20), make_ohlcv(n=100))
    assert signal is None


@pytest.mark.asyncio
async def test_mr_returns_none_if_too_few_1h_rows(default_settings):
    strat = MeanReversionStrategy(default_settings)
    signal = await strat.evaluate("BTCUSDT", make_ohlcv(n=100), make_ohlcv(n=10))
    assert signal is None


# ─── MeanReversion: ADX filter ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_mr_blocks_when_adx_too_high(default_settings):
    """ADX ≥ mr_adx_max (trending) → no signal."""
    ind_1h = make_ind1h(adx=25.0)  # above the 20.0 threshold
    strat = make_mr(default_settings, ind_1h=ind_1h)
    signal = await strat.evaluate("BTCUSDT", dummy_df(), dummy_df())
    assert signal is None


@pytest.mark.asyncio
async def test_mr_passes_when_adx_is_low(default_settings):
    """ADX < mr_adx_max and other conditions met → signal allowed."""
    close = 40_000.0
    lower_bb = 39_500.0
    ind_1h = make_ind1h(adx=10.0, atr=300.0, close=close)  # ranging
    ind_15m = make_ind15m(
        close=close + 100,            # current close just inside lower BB
        bb_lower=lower_bb,
        bb_middle=40_500.0,
        bb_upper=41_000.0,
        rsi=25.0,                      # oversold
        atr=300.0,
        prev_low=lower_bb - 50,        # prev candle touched below lower BB
    )
    strat = make_mr(default_settings, ind_15m=ind_15m, ind_1h=ind_1h)
    signal = await strat.evaluate("BTCUSDT", dummy_df(), dummy_df())
    assert signal is not None
    assert signal.side == "long"


# ─── MeanReversion: ATR/price filter ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_mr_blocks_when_atr_too_low(default_settings):
    """ATR/price < 0.3% → no signal."""
    close = 40_000.0
    # ATR = 100 → 100/40000 = 0.25%, below 0.3%
    ind_1h = make_ind1h(adx=10.0, atr=100.0, close=close)
    ind_15m = make_ind15m(close=close, rsi=25.0, atr=100.0, prev_low=39_400.0, bb_lower=39_500.0)
    strat = make_mr(default_settings, ind_15m=ind_15m, ind_1h=ind_1h)
    signal = await strat.evaluate("BTCUSDT", dummy_df(), dummy_df())
    assert signal is None


# ─── MeanReversion: LONG signal conditions ───────────────────────────────────


@pytest.mark.asyncio
async def test_mr_long_signal_all_conditions_met(default_settings):
    close = 40_100.0  # current close is ABOVE lower BB (39_500)
    lower_bb = 39_500.0
    middle_bb = 40_500.0
    upper_bb = 41_500.0
    atr = 300.0

    ind_1h = make_ind1h(adx=10.0, atr=400.0, close=close)
    ind_15m = make_ind15m(
        close=close,
        bb_lower=lower_bb,
        bb_middle=middle_bb,
        bb_upper=upper_bb,
        rsi=25.0,          # oversold
        atr=atr,
        prev_low=lower_bb - 100,  # prev candle low was below lower BB
    )
    strat = make_mr(default_settings, ind_15m=ind_15m, ind_1h=ind_1h)
    signal = await strat.evaluate("BTCUSDT", dummy_df(), dummy_df())

    assert signal is not None
    assert signal.side == "long"
    assert signal.symbol == "BTCUSDT"
    assert signal.strategy_mode == StrategyMode.MEAN_REVERSION
    assert signal.entry_price == pytest.approx(close)
    assert signal.sl_price == pytest.approx(close - 1.5 * atr)
    assert signal.tp1_price == pytest.approx(middle_bb)
    assert signal.tp2_price == pytest.approx(upper_bb)
    assert signal.tp1_qty_pct == pytest.approx(0.70)


@pytest.mark.asyncio
async def test_mr_no_long_if_rsi_not_oversold(default_settings):
    close = 40_100.0
    lower_bb = 39_500.0
    ind_1h = make_ind1h(adx=10.0, atr=400.0, close=close)
    ind_15m = make_ind15m(
        close=close, bb_lower=lower_bb, rsi=45.0,  # RSI not oversold
        atr=300.0, prev_low=lower_bb - 100,
    )
    strat = make_mr(default_settings, ind_15m=ind_15m, ind_1h=ind_1h)
    signal = await strat.evaluate("BTCUSDT", dummy_df(), dummy_df())
    assert signal is None


@pytest.mark.asyncio
async def test_mr_no_long_if_prev_low_not_below_lower_bb(default_settings):
    close = 40_100.0
    lower_bb = 39_500.0
    ind_1h = make_ind1h(adx=10.0, atr=400.0, close=close)
    ind_15m = make_ind15m(
        close=close, bb_lower=lower_bb, rsi=25.0, atr=300.0,
        prev_low=39_600.0,  # prev low was ABOVE lower BB — no touch
    )
    strat = make_mr(default_settings, ind_15m=ind_15m, ind_1h=ind_1h)
    signal = await strat.evaluate("BTCUSDT", dummy_df(), dummy_df())
    assert signal is None


@pytest.mark.asyncio
async def test_mr_no_long_if_current_close_still_below_lower_bb(default_settings):
    """Current close must be ABOVE lower BB (closed back inside)."""
    lower_bb = 39_500.0
    close = 39_400.0  # still below lower BB
    ind_1h = make_ind1h(adx=10.0, atr=400.0, close=close)
    ind_15m = make_ind15m(
        close=close, bb_lower=lower_bb, rsi=25.0, atr=300.0,
        prev_low=lower_bb - 100,
    )
    strat = make_mr(default_settings, ind_15m=ind_15m, ind_1h=ind_1h)
    signal = await strat.evaluate("BTCUSDT", dummy_df(), dummy_df())
    assert signal is None


# ─── MeanReversion: SHORT signal conditions ───────────────────────────────────


@pytest.mark.asyncio
async def test_mr_short_signal_all_conditions_met(default_settings):
    close = 40_900.0  # current close is BELOW upper BB (41_500)
    upper_bb = 41_500.0
    middle_bb = 40_500.0
    lower_bb = 39_500.0
    atr = 300.0

    ind_1h = make_ind1h(adx=10.0, atr=400.0, close=close)
    ind_15m = make_ind15m(
        close=close,
        bb_lower=lower_bb,
        bb_middle=middle_bb,
        bb_upper=upper_bb,
        rsi=75.0,          # overbought
        atr=atr,
        prev_high=upper_bb + 100,  # prev candle high was above upper BB
    )
    strat = make_mr(default_settings, ind_15m=ind_15m, ind_1h=ind_1h)
    signal = await strat.evaluate("BTCUSDT", dummy_df(), dummy_df())

    assert signal is not None
    assert signal.side == "short"
    assert signal.sl_price == pytest.approx(close + 1.5 * atr)
    assert signal.tp1_price == pytest.approx(middle_bb)
    assert signal.tp2_price == pytest.approx(lower_bb)


@pytest.mark.asyncio
async def test_mr_no_short_if_rsi_not_overbought(default_settings):
    close = 40_900.0
    upper_bb = 41_500.0
    ind_1h = make_ind1h(adx=10.0, atr=400.0, close=close)
    ind_15m = make_ind15m(
        close=close, bb_upper=upper_bb, rsi=60.0, atr=300.0,  # RSI not overbought
        prev_high=upper_bb + 100,
    )
    strat = make_mr(default_settings, ind_15m=ind_15m, ind_1h=ind_1h)
    signal = await strat.evaluate("BTCUSDT", dummy_df(), dummy_df())
    assert signal is None


# ─── TrendFollowing: ADX filter ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tf_blocks_when_adx_too_low(default_settings):
    """ADX ≤ tf_adx_min (ranging) → no signal."""
    ind_1h = make_ind1h(adx=20.0)  # below 25 threshold
    strat = make_tf(default_settings, ind_1h=ind_1h)
    signal = await strat.evaluate("BTCUSDT", dummy_df(), dummy_df())
    assert signal is None


# ─── TrendFollowing: LONG signal conditions ───────────────────────────────────


@pytest.mark.asyncio
async def test_tf_long_signal_all_conditions_met(default_settings):
    close = 40_500.0
    ema_fast = 40_300.0  # EMA21 below current close

    # Uptrend on 1h
    ind_1h = make_ind1h(adx=30.0, ema_mid=40_200.0, ema_slow=39_000.0)
    ind_15m = make_ind15m(
        close=close,
        ema_fast=ema_fast,
        macd_hist=0.05,       # current: positive
        macd_hist_prev=-0.02, # prev: negative → turned positive
        vol=600.0,
        vol_ma=400.0,         # vol > 1.2 × vol_ma
        atr=300.0,
        prev_low=ema_fast - 50,  # prev candle touched EMA21
    )
    strat = make_tf(default_settings, ind_15m=ind_15m, ind_1h=ind_1h)
    signal = await strat.evaluate("BTCUSDT", dummy_df(), dummy_df())

    assert signal is not None
    assert signal.side == "long"
    assert signal.strategy_mode == StrategyMode.TREND_FOLLOWING
    assert signal.tp2_price is None   # trailing stop, no fixed TP2
    assert signal.tp1_qty_pct == pytest.approx(0.50)
    assert signal.tp1_price == pytest.approx(close + 2.5 * 300.0)


@pytest.mark.asyncio
async def test_tf_no_long_if_macd_hist_already_positive(default_settings):
    """MACD histogram must TURN positive, not already be positive."""
    close = 40_500.0
    ema_fast = 40_300.0
    ind_1h = make_ind1h(adx=30.0, ema_mid=40_200.0, ema_slow=39_000.0)
    ind_15m = make_ind15m(
        close=close, ema_fast=ema_fast,
        macd_hist=0.05, macd_hist_prev=0.03,  # both positive — not a crossover
        vol=600.0, vol_ma=400.0, atr=300.0,
        prev_low=ema_fast - 50,
    )
    strat = make_tf(default_settings, ind_15m=ind_15m, ind_1h=ind_1h)
    signal = await strat.evaluate("BTCUSDT", dummy_df(), dummy_df())
    assert signal is None


@pytest.mark.asyncio
async def test_tf_no_long_if_volume_insufficient(default_settings):
    close = 40_500.0
    ema_fast = 40_300.0
    ind_1h = make_ind1h(adx=30.0, ema_mid=40_200.0, ema_slow=39_000.0)
    ind_15m = make_ind15m(
        close=close, ema_fast=ema_fast,
        macd_hist=0.05, macd_hist_prev=-0.02,
        vol=400.0, vol_ma=400.0,   # 400 < 1.2 × 400 = 480 → blocked
        atr=300.0, prev_low=ema_fast - 50,
    )
    strat = make_tf(default_settings, ind_15m=ind_15m, ind_1h=ind_1h)
    signal = await strat.evaluate("BTCUSDT", dummy_df(), dummy_df())
    assert signal is None


@pytest.mark.asyncio
async def test_tf_no_long_if_no_pullback_to_ema(default_settings):
    close = 40_500.0
    ema_fast = 40_300.0
    ind_1h = make_ind1h(adx=30.0, ema_mid=40_200.0, ema_slow=39_000.0)
    ind_15m = make_ind15m(
        close=close, ema_fast=ema_fast,
        macd_hist=0.05, macd_hist_prev=-0.02,
        vol=600.0, vol_ma=400.0, atr=300.0,
        prev_low=ema_fast + 100,  # prev low was ABOVE EMA21 — no pullback
    )
    strat = make_tf(default_settings, ind_15m=ind_15m, ind_1h=ind_1h)
    signal = await strat.evaluate("BTCUSDT", dummy_df(), dummy_df())
    assert signal is None


@pytest.mark.asyncio
async def test_tf_no_long_if_downtrend_on_1h(default_settings):
    """Uptrend signal requires EMA50 > EMA200 on 1h."""
    close = 40_500.0
    ema_fast = 40_300.0
    # Downtrend on 1h (EMA50 < EMA200)
    ind_1h = make_ind1h(adx=30.0, ema_mid=39_000.0, ema_slow=40_500.0)
    ind_15m = make_ind15m(
        close=close, ema_fast=ema_fast,
        macd_hist=0.05, macd_hist_prev=-0.02,
        vol=600.0, vol_ma=400.0, atr=300.0,
        prev_low=ema_fast - 50,
    )
    strat = make_tf(default_settings, ind_15m=ind_15m, ind_1h=ind_1h)
    signal = await strat.evaluate("BTCUSDT", dummy_df(), dummy_df())
    assert signal is None  # no long in downtrend


# ─── TrendFollowing: SHORT signal conditions ──────────────────────────────────


@pytest.mark.asyncio
async def test_tf_short_signal_all_conditions_met(default_settings):
    close = 39_500.0
    ema_fast = 39_700.0  # EMA21 above current close

    # Downtrend on 1h
    ind_1h = make_ind1h(adx=30.0, ema_mid=39_000.0, ema_slow=40_500.0)
    ind_15m = make_ind15m(
        close=close,
        ema_fast=ema_fast,
        macd_hist=-0.05,       # current: negative
        macd_hist_prev=0.02,   # prev: positive → turned negative
        vol=600.0,
        vol_ma=400.0,
        atr=300.0,
        prev_high=ema_fast + 50,  # prev candle touched EMA21 from below
    )
    strat = make_tf(default_settings, ind_15m=ind_15m, ind_1h=ind_1h)
    signal = await strat.evaluate("BTCUSDT", dummy_df(), dummy_df())

    assert signal is not None
    assert signal.side == "short"
    assert signal.tp1_price == pytest.approx(close - 2.5 * 300.0)
    assert signal.tp2_price is None


@pytest.mark.asyncio
async def test_tf_no_short_if_macd_hist_already_negative(default_settings):
    close = 39_500.0
    ema_fast = 39_700.0
    ind_1h = make_ind1h(adx=30.0, ema_mid=39_000.0, ema_slow=40_500.0)
    ind_15m = make_ind15m(
        close=close, ema_fast=ema_fast,
        macd_hist=-0.05, macd_hist_prev=-0.03,  # both negative
        vol=600.0, vol_ma=400.0, atr=300.0,
        prev_high=ema_fast + 50,
    )
    strat = make_tf(default_settings, ind_15m=ind_15m, ind_1h=ind_1h)
    signal = await strat.evaluate("BTCUSDT", dummy_df(), dummy_df())
    assert signal is None


# ─── Swing SL helper ─────────────────────────────────────────────────────────


def test_swing_sl_long_uses_recent_low(default_settings):
    df = make_ohlcv(n=50)
    atr = 300.0
    sl = _swing_sl(df, "long", atr, lookback=5)
    close = float(df["close"].iloc[-1])
    max_sl_dist = close - 2 * atr
    swing_low = float(df["low"].iloc[-5:].min())
    # cap = max(swing_low, close - 2*atr) — use the value closer to price
    expected = max(swing_low, max_sl_dist)
    assert sl == pytest.approx(expected)


def test_swing_sl_short_uses_recent_high(default_settings):
    df = make_ohlcv(n=50)
    atr = 300.0
    sl = _swing_sl(df, "short", atr, lookback=5)
    close = float(df["close"].iloc[-1])
    max_sl_dist = close + 2 * atr
    swing_high = float(df["high"].iloc[-5:].max())
    # cap = min(swing_high, close + 2*atr) — use the value closer to price
    expected = min(swing_high, max_sl_dist)
    assert sl == pytest.approx(expected)


def test_swing_sl_long_caps_at_2x_atr(default_settings):
    """If swing low is very far below, SL is capped at 2×ATR from close."""
    df = make_ohlcv(n=50)
    close = float(df["close"].iloc[-1])
    atr = 100.0
    # Force swing low to be very far below close (more than 2×ATR)
    df["low"] = close - 10_000
    sl = _swing_sl(df, "long", atr, lookback=5)
    # max(close - 10000, close - 200) = close - 200
    assert sl == pytest.approx(close - 2 * atr)


# ─── Integration: real indicators on synthetic data ───────────────────────────


@pytest.mark.asyncio
async def test_mr_real_indicators_ranging_market_produces_valid_signal_or_none(default_settings):
    """End-to-end: MR strategy with real indicator computation on ranging data."""
    df_15m = make_ranging_ohlcv(n=100)
    df_1h = make_ranging_ohlcv(n=100)
    strat = MeanReversionStrategy(default_settings)
    # Should not raise regardless of whether signal is produced
    signal = await strat.evaluate("BTCUSDT", df_15m, df_1h)
    assert signal is None or isinstance(signal, Signal)


@pytest.mark.asyncio
async def test_tf_real_indicators_trending_market_produces_valid_signal_or_none(default_settings):
    """End-to-end: TF strategy with real indicator computation on trending data."""
    df_15m = make_trending_ohlcv(n=250, uptrend=True)
    df_1h = make_trending_ohlcv(n=250, uptrend=True)
    strat = TrendFollowingStrategy(default_settings)
    signal = await strat.evaluate("BTCUSDT", df_15m, df_1h)
    assert signal is None or isinstance(signal, Signal)
