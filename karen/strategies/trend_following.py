"""Trend Following strategy — trades pullbacks in the direction of the trend."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
from loguru import logger

from karen.config import Settings, StrategyMode
from karen.indicators.ta import Indicators1h, Indicators15m, compute_1h, compute_15m
from karen.strategies.base import AbstractStrategy, Signal
from karen.strategies.mean_reversion import _last, _nth_last


class TrendFollowingStrategy(AbstractStrategy):
    """
    Entry conditions (ALL must pass):

    1h filter:
      - ADX(14) > TF_ADX_MIN (trending regime)
      - EMA50 > EMA200 → uptrend; EMA50 < EMA200 → downtrend

    15m signal — LONG (uptrend on 1h):
      - Price pulls back: low ≤ EMA21
      - Bounce: current close > EMA21
      - MACD histogram turns positive (prev ≤ 0, curr > 0)
      - Volume > TF_VOLUME_MULTIPLIER × 20-period avg

    15m signal — SHORT (downtrend on 1h):
      - Price rallies to EMA21: high ≥ EMA21
      - Rejection: current close < EMA21
      - MACD histogram turns negative (prev ≥ 0, curr < 0)
      - Volume > TF_VOLUME_MULTIPLIER × 20-period avg

    Exit:
      - TP1 (50%): 2.5 × ATR from entry
      - TP2 (50%): chandelier exit trailing stop (ATR × 3, 22-period high/low)
      - SL:        below swing low / above swing high, capped at 2 × ATR
      - No time stop — let the trend run
    """

    _SWING_LOOKBACK = 5   # candles to look back for swing low/high

    def __init__(
        self,
        settings: Settings,
        *,
        compute_15m_fn: Callable[..., Indicators15m] = compute_15m,
        compute_1h_fn: Callable[..., Indicators1h] = compute_1h,
    ) -> None:
        super().__init__(settings)
        self._compute_15m = compute_15m_fn
        self._compute_1h = compute_1h_fn

    @property
    def mode(self) -> StrategyMode:
        return StrategyMode.TREND_FOLLOWING

    async def evaluate(
        self,
        symbol: str,
        df_lower: pd.DataFrame,
        df_upper: pd.DataFrame,
    ) -> Signal | None:
        s = self._settings

        if not self._has_enough_data(df_lower, 50) or not self._has_enough_data(df_upper, 30):
            logger.debug(f"{symbol} TF: insufficient data")
            return None

        try:
            ind_1h = self._compute_1h(df_upper, s)
            ind_15m = self._compute_15m(df_lower, s)
        except Exception:
            logger.exception(f"{symbol} TF: indicator error")
            return None

        # ── 1h regime filter ──────────────────────────────────────────────

        adx_1h = _last(ind_1h.adx)
        if adx_1h is None or adx_1h <= s.tf_adx_min:
            logger.debug(f"{symbol} TF: ADX={adx_1h} ≤ {s.tf_adx_min} (not trending)")
            return None

        ema_mid = _last(ind_1h.ema_mid)
        ema_slow = _last(ind_1h.ema_slow)
        if ema_mid is None or ema_slow is None:
            return None

        uptrend = ema_mid > ema_slow
        downtrend = ema_mid < ema_slow

        if not uptrend and not downtrend:
            return None

        # ── 15m indicators ────────────────────────────────────────────────

        ema_fast = _last(ind_15m.ema_fast)
        macd_hist_curr = _last(ind_15m.macd_hist)
        macd_hist_prev = _nth_last(ind_15m.macd_hist, 2)
        vol_ma = _last(ind_15m.vol_ma)
        vol_curr = _last(ind_15m.volume)
        atr = _last(ind_15m.atr)
        curr_close = _last(ind_15m.close)
        prev_low = _nth_last(ind_15m.low, 2)
        prev_high = _nth_last(ind_15m.high, 2)

        if any(v is None for v in [ema_fast, macd_hist_curr, macd_hist_prev,
                                    vol_ma, vol_curr, atr, curr_close]):
            return None

        # Volume filter
        if vol_ma == 0 or vol_curr < s.tf_volume_multiplier * vol_ma:  # type: ignore[operator]
            logger.debug(
                f"{symbol} TF: volume={vol_curr:.0f} < {s.tf_volume_multiplier}×MA={vol_ma:.0f}"
            )
            return None

        # ── LONG signal ───────────────────────────────────────────────────

        if uptrend:
            macd_turned_positive = (
                macd_hist_prev <= 0 and macd_hist_curr > 0  # type: ignore[operator]
            )
            pullback_to_ema = (
                prev_low is not None and prev_low <= ema_fast  # type: ignore[operator]
                and curr_close > ema_fast  # type: ignore[operator]
            )

            if pullback_to_ema and macd_turned_positive:
                sl = _swing_sl(df_lower, "long", atr, self._SWING_LOOKBACK)  # type: ignore[arg-type]
                tp1 = curr_close + 2.5 * atr  # type: ignore[operator]
                conf = _confidence(
                    adx_1h, s.tf_adx_min, vol_curr, vol_ma, s.tf_volume_multiplier,  # type: ignore[arg-type]
                    ema_mid, ema_slow, macd_hist_curr, atr, curr_close, sl,  # type: ignore[arg-type]
                )
                logger.info(
                    f"{symbol} TF LONG signal | close={curr_close:.2f} "
                    f"EMA21={ema_fast:.2f} MACD_h={macd_hist_curr:.4f} SL={sl:.2f} TP1={tp1:.2f} "
                    f"conf={conf:.2f}"
                )
                return Signal(
                    symbol=symbol,
                    side="long",
                    strategy_mode=StrategyMode.TREND_FOLLOWING,
                    entry_price=curr_close,
                    sl_price=sl,
                    tp1_price=tp1,
                    tp1_qty_pct=0.50,
                    tp2_price=None,  # trailing stop (chandelier exit)
                    atr=atr,
                    trail_atr_mult=3.0,
                    confidence=conf,
                    reason="TF_LONG: 1h uptrend + EMA21 pullback + MACD histogram flipped + volume",
                )

        # ── SHORT signal ──────────────────────────────────────────────────

        if downtrend:
            macd_turned_negative = (
                macd_hist_prev >= 0 and macd_hist_curr < 0  # type: ignore[operator]
            )
            rally_to_ema = (
                prev_high is not None and prev_high >= ema_fast  # type: ignore[operator]
                and curr_close < ema_fast  # type: ignore[operator]
            )

            if rally_to_ema and macd_turned_negative:
                sl = _swing_sl(df_lower, "short", atr, self._SWING_LOOKBACK)  # type: ignore[arg-type]
                tp1 = curr_close - 2.5 * atr  # type: ignore[operator]
                conf = _confidence(
                    adx_1h, s.tf_adx_min, vol_curr, vol_ma, s.tf_volume_multiplier,  # type: ignore[arg-type]
                    ema_mid, ema_slow, macd_hist_curr, atr, curr_close, sl,  # type: ignore[arg-type]
                )
                logger.info(
                    f"{symbol} TF SHORT signal | close={curr_close:.2f} "
                    f"EMA21={ema_fast:.2f} MACD_h={macd_hist_curr:.4f} SL={sl:.2f} TP1={tp1:.2f} "
                    f"conf={conf:.2f}"
                )
                return Signal(
                    symbol=symbol,
                    side="short",
                    strategy_mode=StrategyMode.TREND_FOLLOWING,
                    entry_price=curr_close,
                    sl_price=sl,
                    tp1_price=tp1,
                    tp1_qty_pct=0.50,
                    tp2_price=None,
                    atr=atr,
                    trail_atr_mult=3.0,
                    confidence=conf,
                    reason="TF_SHORT: downtrend + EMA21 rejection + MACD flipped + volume",
                )

        return None


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _clamp01(x: float) -> float:
    return min(1.0, max(0.0, x))


def _confidence(
    adx_1h: float,
    adx_min: float,
    vol_curr: float,
    vol_ma: float,
    vol_mult: float,
    ema_mid: float,
    ema_slow: float,
    macd_hist: float,
    atr: float,
    close: float,
    sl: float,
) -> float:
    """
    Score a TF signal 0..1 (weighted sum of normalised components):

      30%  1h trend strength   ADX from ADX_MIN (→0) to 50 (→1)
      20%  1h trend separation |EMA50−EMA200|/EMA200 from 0 to 2%
      20%  volume surge        vol/MA from the filter multiplier to 3×
      10%  MACD momentum       |hist|/ATR from 0 to 0.2
      20%  reward/risk         TP1 (2.5 ATR) / SL distance from 1.25 to 2.5
    """
    adx_s = _clamp01((adx_1h - adx_min) / max(50.0 - adx_min, 1e-9))
    sep_s = _clamp01(abs(ema_mid - ema_slow) / ema_slow / 0.02)
    vol_s = _clamp01((vol_curr / vol_ma - vol_mult) / max(3.0 - vol_mult, 1e-9))
    macd_s = _clamp01(abs(macd_hist) / atr / 0.2) if atr > 0 else 0.0
    sl_dist = abs(close - sl)
    rr = (2.5 * atr / sl_dist) if sl_dist > 0 else 0.0
    rr_s = _clamp01((rr - 1.25) / 1.25)
    return 0.30 * adx_s + 0.20 * sep_s + 0.20 * vol_s + 0.10 * macd_s + 0.20 * rr_s


def _swing_sl(
    df_15m: pd.DataFrame,
    side: str,
    atr: float,
    lookback: int,
) -> float:
    """
    SL = swing low/high over the last `lookback` candles, capped at 2×ATR from close.
    """
    close = float(df_15m["close"].iloc[-1])
    max_sl_dist = 2.0 * atr

    if side == "long":
        swing_low = float(df_15m["low"].iloc[-lookback:].min())
        # Cap: SL must not be more than 2×ATR below close (use the value closer to price)
        sl = max(swing_low, close - max_sl_dist)
        return sl

    # short
    swing_high = float(df_15m["high"].iloc[-lookback:].max())
    # Cap: SL must not be more than 2×ATR above close (use the value closer to price)
    sl = min(swing_high, close + max_sl_dist)
    return sl
