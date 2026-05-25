"""Mean Reversion strategy — trades price extremes back to the mean."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
from loguru import logger

from karen.config import Settings, StrategyMode
from karen.indicators.ta import Indicators1h, Indicators15m, compute_1h, compute_15m
from karen.strategies.base import AbstractStrategy, Signal


class MeanReversionStrategy(AbstractStrategy):
    """
    Entry conditions (ALL must pass):

    1h filter:
      - ADX(14) < MR_ADX_MAX (ranging regime)
      - ATR/price > 0.3% (enough volatility)

    15m signal — LONG:
      - Previous candle low ≤ lower BB  (touched/crossed below)
      - Current candle closes > lower BB (reversal confirmation)
      - RSI(14) < MR_RSI_OVERSOLD

    15m signal — SHORT:
      - Previous candle high ≥ upper BB
      - Current candle closes < upper BB
      - RSI(14) > MR_RSI_OVERBOUGHT

    Exit:
      - TP1 (70%): middle BB (SMA20)
      - TP2 (30%): opposite band
      - SL:        1.5 × ATR(14) beyond entry
      - Time stop: 8 candles — enforced by order_manager, not here
    """

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
        return StrategyMode.MEAN_REVERSION

    async def evaluate(
        self,
        symbol: str,
        df_lower: pd.DataFrame,
        df_upper: pd.DataFrame,
    ) -> Signal | None:
        s = self._settings

        # Need enough history for all indicators (BB 20, EMA 200 headroom)
        if not self._has_enough_data(df_lower, 50) or not self._has_enough_data(df_upper, 30):
            logger.debug(f"{symbol} MR: insufficient data")
            return None

        try:
            ind_1h = self._compute_1h(df_upper, s)
            ind_15m = self._compute_15m(df_lower, s)
        except Exception:
            logger.exception(f"{symbol} MR: indicator error")
            return None

        # ── 1h regime filter ──────────────────────────────────────────────

        adx_1h = _last(ind_1h.adx)
        if adx_1h is None or adx_1h >= s.mr_adx_max:
            logger.debug(f"{symbol} MR: ADX={adx_1h:.1f} ≥ {s.mr_adx_max} (not ranging)")
            return None

        atr_1h = _last(ind_1h.atr)
        price_1h = _last(ind_1h.close)
        if atr_1h is None or price_1h is None or price_1h == 0:
            return None
        if atr_1h / price_1h < 0.003:
            logger.debug(f"{symbol} MR: ATR/price={atr_1h/price_1h:.4f} < 0.003 (low vol)")
            return None

        # ── 15m signal check ──────────────────────────────────────────────

        atr = _last(ind_15m.atr)
        rsi = _last(ind_15m.rsi)
        upper = _last(ind_15m.bb_upper)
        middle = _last(ind_15m.bb_middle)
        lower = _last(ind_15m.bb_lower)

        prev_high = _nth_last(ind_15m.high, 2)
        prev_low = _nth_last(ind_15m.low, 2)
        curr_close = _last(ind_15m.close)

        if any(v is None for v in [atr, rsi, upper, middle, lower,
                                    prev_high, prev_low, curr_close]):
            return None

        # LONG: prev candle touched/crossed below lower BB, current closes back inside
        if (
            prev_low <= lower  # type: ignore[operator]
            and curr_close > lower  # type: ignore[operator]
            and rsi < s.mr_rsi_oversold  # type: ignore[operator]
        ):
            sl = curr_close - 1.5 * atr  # type: ignore[operator]
            tp1 = middle
            tp2 = upper
            logger.info(
                f"{symbol} MR LONG signal | close={curr_close:.2f} "
                f"lower_bb={lower:.2f} RSI={rsi:.1f} SL={sl:.2f} TP1={tp1:.2f}"
            )
            return Signal(
                symbol=symbol,
                side="long",
                strategy_mode=StrategyMode.MEAN_REVERSION,
                entry_price=curr_close,
                sl_price=sl,
                tp1_price=tp1,
                tp1_qty_pct=0.70,
                tp2_price=tp2,
                atr=atr,
                reason="MR_LONG: price below lower BB + RSI oversold + close back inside",
            )

        # SHORT: prev candle touched/crossed above upper BB, current closes back inside
        if (
            prev_high >= upper  # type: ignore[operator]
            and curr_close < upper  # type: ignore[operator]
            and rsi > s.mr_rsi_overbought  # type: ignore[operator]
        ):
            sl = curr_close + 1.5 * atr  # type: ignore[operator]
            tp1 = middle
            tp2 = lower
            logger.info(
                f"{symbol} MR SHORT signal | close={curr_close:.2f} "
                f"upper_bb={upper:.2f} RSI={rsi:.1f} SL={sl:.2f} TP1={tp1:.2f}"
            )
            return Signal(
                symbol=symbol,
                side="short",
                strategy_mode=StrategyMode.MEAN_REVERSION,
                entry_price=curr_close,
                sl_price=sl,
                tp1_price=tp1,
                tp1_qty_pct=0.70,
                tp2_price=tp2,
                atr=atr,
                reason="MR_SHORT: price above upper BB + RSI overbought + close back inside",
            )

        return None


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _last(s: pd.Series) -> float | None:
    """Return the last non-NaN value, or None."""
    if s is None or len(s) == 0:
        return None
    val = s.iloc[-1]
    return None if pd.isna(val) else float(val)


def _nth_last(s: pd.Series, n: int) -> float | None:
    """Return the nth-from-last value (n=1 → last, n=2 → second-to-last)."""
    idx = -n
    if abs(idx) > len(s):
        return None
    val = s.iloc[idx]
    return None if pd.isna(val) else float(val)
