"""Abstract strategy interface and shared Signal type."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from karen.config import Settings, StrategyMode


@dataclass
class Signal:
    """A trade signal produced by a strategy."""

    symbol: str
    side: str             # "long" | "short"
    strategy_mode: StrategyMode
    entry_price: float    # estimated (last close); actual fill may differ
    sl_price: float
    tp1_price: float
    tp1_qty_pct: float    # fraction of position to close at TP1 (0.0–1.0)
    tp2_price: float | None  # None → use trailing stop for the runner
    atr: float            # ATR at signal time (used for trailing stop sizing)
    trail_atr_mult: float = 3.0  # chandelier exit multiplier
    reason: str = ""


class AbstractStrategy(ABC):
    """
    Common interface for all strategies.

    Each strategy receives raw OHLCV DataFrames and returns a Signal (or None).
    Indicator computation is internal to the strategy, making the interface
    data-source agnostic and easy to test via dependency injection.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    @abstractmethod
    def mode(self) -> StrategyMode:
        """Return the StrategyMode this strategy implements."""
        ...

    @abstractmethod
    async def evaluate(
        self,
        symbol: str,
        df_15m: pd.DataFrame,
        df_1h: pd.DataFrame,
    ) -> Signal | None:
        """
        Evaluate market conditions and return a Signal, or None if no entry.

        Args:
            symbol:  Trading pair, e.g. "BTCUSDT"
            df_15m:  15-minute OHLCV DataFrame (index: UTC DatetimeTZDtype)
            df_1h:   1-hour OHLCV DataFrame (same index type)

        Returns:
            Signal if all entry conditions are met, else None.
        """
        ...

    def _has_enough_data(self, df: pd.DataFrame, min_rows: int) -> bool:
        return len(df) >= min_rows
