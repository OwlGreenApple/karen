"""Mean Reversion strategy — 5m signal, 15m filter, aggressive settings."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
from loguru import logger

from karen.config import Settings, StrategyMode
from karen.indicators.ta import Indicators5m_MR, Indicators15m, compute_5m_mr, compute_15m
from karen.strategies.base import AbstractStrategy, Signal


class MeanReversionStrategy(AbstractStrategy):
    """
    Entry conditions (ALL must pass):

    15m filter:
      - ADX(14) < MR_ADX_MAX (ranging regime)

    5m signal — LONG:
      - Previous candle low ≤ lower BB  (touched/crossed below)
      - Current candle closes > lower BB (reversal confirmation)
      - RSI(14) < MR_RSI_OVERSOLD

    5m signal — SHORT:
      - Previous candle high ≥ upper BB
      - Current candle closes < upper BB
      - RSI(14) > MR_RSI_OVERBOUGHT

    Exit:
      - TP1 (70%): middle BB (SMA20)
      - TP2 (30%): opposite band
      - SL:        MR_SL_ATR × ATR(14) beyond entry (capped by the 18% max-loss rule)
      - Time stop: MR_TIME_STOP_CANDLES × 15m — enforced by order_manager
      - Optional:  SL → entry after TP1 (MR_BREAKEVEN_AFTER_TP1) — order_manager
    """

    def __init__(
        self,
        settings: Settings,
        *,
        compute_signal_fn: Callable[..., Indicators5m_MR] = compute_5m_mr,
        compute_filter_fn: Callable[..., Indicators15m] = compute_15m,
    ) -> None:
        super().__init__(settings)
        self._compute_signal = compute_signal_fn
        self._compute_filter = compute_filter_fn

    @property
    def mode(self) -> StrategyMode:
        return StrategyMode.MEAN_REVERSION

    @property
    def timeframes(self) -> tuple[str, str, int]:
        return ("5m", "15m", 300)

    @property
    def candle_interval_minutes(self) -> int:
        return 5

    async def evaluate(
        self,
        symbol: str,
        df_lower: pd.DataFrame,   # 5m — signal
        df_upper: pd.DataFrame,   # 15m — regime filter
    ) -> Signal | None:
        s = self._settings

        if not self._has_enough_data(df_lower, 50) or not self._has_enough_data(df_upper, 30):
            logger.debug(f"{symbol} MR: insufficient data")
            return None

        try:
            ind_15m = self._compute_filter(df_upper, s)
            ind_5m = self._compute_signal(df_lower, s)
        except Exception:
            logger.exception(f"{symbol} MR: indicator error")
            return None

        # ── 15m regime filter ──────────────────────────────────────────────

        adx_15m = _last(ind_15m.adx)
        if adx_15m is None or adx_15m >= s.mr_adx_max:
            logger.debug(f"{symbol} MR: ADX={adx_15m:.1f} ≥ {s.mr_adx_max} (not ranging)")
            return None

        # ── 5m signal check ───────────────────────────────────────────────

        atr = _last(ind_5m.atr)
        rsi = _last(ind_5m.rsi)
        upper = _last(ind_5m.bb_upper)
        middle = _last(ind_5m.bb_middle)
        lower = _last(ind_5m.bb_lower)
        curr_close = _last(ind_5m.close)
        prev_high = _nth_last(ind_5m.high, 2)
        prev_low = _nth_last(ind_5m.low, 2)

        if any(
            v is None
            for v in [atr, rsi, upper, middle, lower, curr_close, prev_high, prev_low]
        ):
            return None

        # LONG: prev candle touched/crossed below lower BB, current closes back inside
        if (
            prev_low <= lower  # type: ignore[operator]
            and curr_close > lower  # type: ignore[operator]
            and rsi < s.mr_rsi_oversold  # type: ignore[operator]
        ):
            sl = curr_close - s.mr_sl_atr * atr  # type: ignore[operator]
            tp1 = middle
            tp2 = upper
            conf = _mr_confidence(
                "long", adx_15m, s.mr_adx_max, rsi, s.mr_rsi_oversold,  # type: ignore[arg-type]
                prev_low, lower, middle, upper, curr_close, atr,  # type: ignore[arg-type]
            )
            logger.info(
                f"{symbol} MR LONG signal | close={curr_close:.2f} "
                f"lower_bb={lower:.2f} RSI={rsi:.1f} SL={sl:.2f} TP1={tp1:.2f} conf={conf:.2f}"
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
                confidence=conf,
                reason="MR_LONG: price below lower BB + RSI oversold + close back inside",
            )

        # SHORT: prev candle touched/crossed above upper BB, current closes back inside
        if (
            prev_high >= upper  # type: ignore[operator]
            and curr_close < upper  # type: ignore[operator]
            and rsi > s.mr_rsi_overbought  # type: ignore[operator]
        ):
            sl = curr_close + s.mr_sl_atr * atr  # type: ignore[operator]
            tp1 = middle
            tp2 = lower
            conf = _mr_confidence(
                "short", adx_15m, s.mr_adx_max, rsi, s.mr_rsi_overbought,  # type: ignore[arg-type]
                prev_high, upper, middle, lower, curr_close, atr,  # type: ignore[arg-type]
            )
            logger.info(
                f"{symbol} MR SHORT signal | close={curr_close:.2f} "
                f"upper_bb={upper:.2f} RSI={rsi:.1f} SL={sl:.2f} TP1={tp1:.2f} conf={conf:.2f}"
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
                confidence=conf,
                reason="MR_SHORT: price above upper BB + RSI overbought + close back inside",
            )

        return None


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _clamp01(x: float) -> float:
    return min(1.0, max(0.0, x))


def _mr_confidence(
    side: str,
    adx_15m: float,
    adx_max: float,
    rsi: float,
    rsi_threshold: float,
    prev_extreme: float,
    band: float,
    middle: float,
    opposite_band: float,
    close: float,
    atr: float,
) -> float:
    """
    Score an MR signal 0..1 (weighted sum of normalised components):

      25%  ranging regime   15m ADX from ADX_MAX (→0) down to 0 (→1)
      25%  RSI extremity    beyond the threshold, full at 20 points past it
                            (e.g. short: RSI 70→0, 90→1; long: RSI 30→0, 10→1)
      15%  band overshoot   how far the previous candle pierced the band, 0..1 ATR
      25%  reward/risk      distance to TP1 (middle BB) / SL distance (1.5 ATR),
                            from 0.5 (→0) to 1.5 (→1)
      10%  band width       (upper−lower)/middle from 0.5% (→0) to 3% (→1) —
                            narrow bands can't pay for fees
    """
    if atr <= 0 or middle <= 0:
        return 0.0
    adx_s = _clamp01((adx_max - adx_15m) / adx_max) if adx_max > 0 else 0.0
    if side == "long":
        rsi_s = _clamp01((rsi_threshold - rsi) / 20.0)
        overshoot = band - prev_extreme
        reward = middle - close
    else:
        rsi_s = _clamp01((rsi - rsi_threshold) / 20.0)
        overshoot = prev_extreme - band
        reward = close - middle
    over_s = _clamp01(overshoot / atr)
    rr_s = _clamp01((reward / (1.5 * atr) - 0.5) / 1.0)
    width_s = _clamp01((abs(band - opposite_band) / middle - 0.005) / 0.025)
    return 0.25 * adx_s + 0.25 * rsi_s + 0.15 * over_s + 0.25 * rr_s + 0.10 * width_s


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
