"""Scalp strategy — 1m signal with 5m regime filter."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
from loguru import logger

from karen.config import Settings, StrategyMode
from karen.indicators.ta import Indicators1m, Indicators5m, compute_1m, compute_5m
from karen.strategies.base import AbstractStrategy, Signal
from karen.strategies.mean_reversion import _last, _nth_last


class ScalpStrategy(AbstractStrategy):
    """
    Entry conditions (ALL must pass):

    5m regime filter:
      - ADX(14) > SCALP_5M_ADX_MIN (some directional momentum, avoids dead flat)
      - Price above 5m EMA20 → long bias; price below → short bias

    1m signal — LONG (5m long bias):
      - EMA9 crosses above EMA21 (prev: EMA9 ≤ EMA21, curr: EMA9 > EMA21)
      - RSI(7) > 45 (momentum is bullish, not yet overbought)
      - Volume > SCALP_VOLUME_MULTIPLIER × 20-period vol MA

    1m signal — SHORT (5m short bias):
      - EMA9 crosses below EMA21 (prev: EMA9 ≥ EMA21, curr: EMA9 < EMA21)
      - RSI(7) < 55 (momentum is bearish)
      - Volume > SCALP_VOLUME_MULTIPLIER × 20-period vol MA

    Exit:
      - TP1 (70%): SCALP_TP_ATR_MULT × ATR(7) from entry
      - TP2 (30%): trailing chandelier exit (2.5 × ATR)
      - SL:        SCALP_SL_ATR_MULT × ATR(7) from entry
      - Time stop: 10 candles (10 minutes) — enforced by order_manager
    """

    def __init__(
        self,
        settings: Settings,
        *,
        compute_1m_fn: Callable[..., Indicators1m] = compute_1m,
        compute_5m_fn: Callable[..., Indicators5m] = compute_5m,
    ) -> None:
        super().__init__(settings)
        self._compute_1m = compute_1m_fn
        self._compute_5m = compute_5m_fn

    @property
    def mode(self) -> StrategyMode:
        return StrategyMode.SCALP_1M

    @property
    def timeframes(self) -> tuple[str, str, int]:
        return ("1m", "5m", 300)

    @property
    def candle_interval_minutes(self) -> int:
        return 1

    async def evaluate(
        self,
        symbol: str,
        df_lower: pd.DataFrame,
        df_upper: pd.DataFrame,
    ) -> Signal | None:
        s = self._settings

        if not self._has_enough_data(df_lower, 50) or not self._has_enough_data(df_upper, 30):
            logger.debug(f"{symbol} SCALP: insufficient data")
            return None

        try:
            ind_5m = self._compute_5m(df_upper, s)
            ind_1m = self._compute_1m(df_lower, s)
        except Exception:
            logger.exception(f"{symbol} SCALP: indicator error")
            return None

        # ── 5m regime filter ──────────────────────────────────────────────

        adx_5m = _last(ind_5m.adx)
        if adx_5m is None or adx_5m <= s.scalp_5m_adx_min:
            logger.debug(f"{symbol} SCALP: 5m ADX={adx_5m} ≤ {s.scalp_5m_adx_min} (flat)")
            return None

        close_5m = _last(ind_5m.close)
        ema_filter = _last(ind_5m.ema_filter)
        if close_5m is None or ema_filter is None:
            return None

        long_bias = close_5m > ema_filter
        short_bias = close_5m < ema_filter

        if not long_bias and not short_bias:
            return None

        # ── 1m indicators ─────────────────────────────────────────────────

        ema_fast_curr = _last(ind_1m.ema_fast)
        ema_fast_prev = _nth_last(ind_1m.ema_fast, 2)
        ema_slow_curr = _last(ind_1m.ema_slow)
        ema_slow_prev = _nth_last(ind_1m.ema_slow, 2)
        rsi = _last(ind_1m.rsi)
        atr = _last(ind_1m.atr)
        vol_curr = _last(ind_1m.volume)
        vol_ma = _last(ind_1m.vol_ma)
        curr_close = _last(ind_1m.close)

        if any(v is None for v in [
            ema_fast_curr, ema_fast_prev, ema_slow_curr, ema_slow_prev,
            rsi, atr, vol_curr, vol_ma, curr_close,
        ]):
            return None

        # Volume filter
        if vol_ma == 0 or vol_curr < s.scalp_volume_multiplier * vol_ma:  # type: ignore[operator]
            logger.debug(
                f"{symbol} SCALP: vol={vol_curr:.0f} < {s.scalp_volume_multiplier}×MA={vol_ma:.0f}"
            )
            return None

        # ── LONG signal ───────────────────────────────────────────────────

        if long_bias:
            bullish_cross = (
                ema_fast_prev <= ema_slow_prev  # type: ignore[operator]
                and ema_fast_curr > ema_slow_curr  # type: ignore[operator]
            )
            if bullish_cross and rsi > 45:  # type: ignore[operator]
                sl = curr_close - s.scalp_sl_atr_mult * atr  # type: ignore[operator]
                tp1 = curr_close + s.scalp_tp_atr_mult * atr  # type: ignore[operator]
                logger.info(
                    f"{symbol} SCALP LONG | close={curr_close:.4f} "
                    f"EMA9={ema_fast_curr:.4f} EMA21={ema_slow_curr:.4f} "
                    f"RSI={rsi:.1f} SL={sl:.4f} TP1={tp1:.4f}"
                )
                return Signal(
                    symbol=symbol,
                    side="long",
                    strategy_mode=StrategyMode.SCALP_1M,
                    entry_price=curr_close,  # type: ignore[arg-type]
                    sl_price=sl,
                    tp1_price=tp1,
                    tp1_qty_pct=0.70,
                    tp2_price=None,
                    atr=atr,  # type: ignore[arg-type]
                    trail_atr_mult=2.5,
                    reason="SCALP_LONG: 5m upbias + 1m EMA9/EMA21 cross + RSI + volume",
                )

        # ── SHORT signal ──────────────────────────────────────────────────

        if short_bias:
            bearish_cross = (
                ema_fast_prev >= ema_slow_prev  # type: ignore[operator]
                and ema_fast_curr < ema_slow_curr  # type: ignore[operator]
            )
            if bearish_cross and rsi < 55:  # type: ignore[operator]
                sl = curr_close + s.scalp_sl_atr_mult * atr  # type: ignore[operator]
                tp1 = curr_close - s.scalp_tp_atr_mult * atr  # type: ignore[operator]
                logger.info(
                    f"{symbol} SCALP SHORT | close={curr_close:.4f} "
                    f"EMA9={ema_fast_curr:.4f} EMA21={ema_slow_curr:.4f} "
                    f"RSI={rsi:.1f} SL={sl:.4f} TP1={tp1:.4f}"
                )
                return Signal(
                    symbol=symbol,
                    side="short",
                    strategy_mode=StrategyMode.SCALP_1M,
                    entry_price=curr_close,  # type: ignore[arg-type]
                    sl_price=sl,
                    tp1_price=tp1,
                    tp1_qty_pct=0.70,
                    tp2_price=None,
                    atr=atr,  # type: ignore[arg-type]
                    trail_atr_mult=2.5,
                    reason="SCALP_SHORT: 5m downbias + 1m EMA9/EMA21 cross + RSI + volume",
                )

        return None
