"""Daily guardrails: max loss, max trades, max concurrent positions."""

from __future__ import annotations

from datetime import UTC, date, datetime

from loguru import logger

from karen.config import Settings


class DailyGuardrails:
    """
    Tracks per-UTC-day limits:
    - Max daily loss (% of starting equity) → auto-pause until next UTC day
    - Max trades per day (hard cap)

    Concurrent-position check lives here too (stateless, just checks a count).
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._start_equity: float | None = None
        self._trade_count: int = 0
        self._paused: bool = False
        self._day: date | None = None

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def new_day(self, current_equity: float) -> None:
        """Call once at bot start and at each UTC midnight rollover."""
        self._start_equity = current_equity
        self._trade_count = 0
        self._paused = False
        self._day = datetime.now(tz=UTC).date()
        logger.info(
            f"Daily guardrails reset | equity=${current_equity:,.2f} "
            f"max_loss={self._settings.max_daily_loss_pct}% "
            f"max_trades={self._settings.max_trades_per_day}"
        )

    def maybe_rollover(self, current_equity: float) -> bool:
        """Auto-call new_day if UTC date has changed. Returns True if rolled over."""
        today = datetime.now(tz=UTC).date()
        if self._day is None or today != self._day:
            self.new_day(current_equity)
            return True
        return False

    # ── Checks (return True = OK to trade, False = blocked) ───────────────

    @property
    def paused(self) -> bool:
        return self._paused

    def can_open_position(
        self,
        current_equity: float,
        open_positions: int,
    ) -> tuple[bool, str]:
        """
        Check all guards before opening a new position.

        Returns (allowed: bool, reason: str).
        """
        if self._paused:
            return False, "daily loss limit reached — bot paused until next UTC day"

        if self._start_equity is None:
            return False, "guardrails not initialised (call new_day first)"

        # Daily loss check
        loss_pct = _daily_loss_pct(self._start_equity, current_equity)
        if loss_pct >= self._settings.max_daily_loss_pct:
            self._paused = True
            logger.warning(
                f"Daily loss limit hit: {loss_pct:.2f}% ≥ {self._settings.max_daily_loss_pct}% "
                f"— pausing until next UTC day"
            )
            return False, f"daily loss {loss_pct:.2f}% exceeds limit"

        # Trade count check
        if self._trade_count >= self._settings.max_trades_per_day:
            n = self._settings.max_trades_per_day
            return False, f"max daily trades reached ({self._trade_count}/{n})"

        # Concurrent positions check
        if open_positions >= self._settings.max_concurrent_positions:
            return False, (
                f"max concurrent positions reached "
                f"({open_positions}/{self._settings.max_concurrent_positions})"
            )

        return True, ""

    def record_trade_opened(self) -> None:
        """Call whenever a new position is opened."""
        self._trade_count += 1
        logger.debug(
            f"Trade count today: {self._trade_count}/{self._settings.max_trades_per_day}"
        )

    # ── Read-only state ────────────────────────────────────────────────────

    @property
    def trade_count(self) -> int:
        return self._trade_count

    @property
    def start_equity(self) -> float | None:
        return self._start_equity

    def current_loss_pct(self, current_equity: float) -> float:
        """Return the current daily loss as a positive percentage (0 if gaining)."""
        if self._start_equity is None:
            return 0.0
        return max(0.0, _daily_loss_pct(self._start_equity, current_equity))


def _daily_loss_pct(start: float, current: float) -> float:
    """Return % loss from start (positive = loss, negative = gain)."""
    if start <= 0:
        return 0.0
    return (start - current) / start * 100.0
