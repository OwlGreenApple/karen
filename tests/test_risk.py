"""Tests for position sizing and daily guardrails."""

from __future__ import annotations

import pytest

from karen.risk.guardrails import DailyGuardrails
from karen.risk.position_sizing import (
    compute_dollar_risk,
    compute_margin_required,
    compute_quantity,
)

# ─── Position sizing ─────────────────────────────────────────────────────────


class TestComputeQuantity:
    def test_basic_calculation(self):
        """$10k equity, 1% risk, $40k entry, $500 SL distance → 0.2 BTC."""
        qty = compute_quantity(
            equity_usdt=10_000,
            risk_pct=1.0,
            entry_price=40_000,
            sl_price=39_500,
        )
        # risk_amount = 100, sl_distance = 500 → qty = 0.2
        assert qty == pytest.approx(0.2)

    def test_loss_equals_risk_amount(self):
        """Verify: quantity × sl_distance = risk_amount exactly."""
        equity = 5_000
        risk_pct = 2.0
        entry = 50_000
        sl = 49_000  # $1000 distance

        qty = compute_quantity(equity, risk_pct, entry, sl)
        dollar_risk = compute_dollar_risk(qty, entry, sl)
        expected_risk = equity * risk_pct / 100  # $100
        assert dollar_risk == pytest.approx(expected_risk, rel=1e-6)

    def test_short_side_symmetric(self):
        """SL above entry (short) should produce the same quantity as same distance below."""
        qty_long = compute_quantity(10_000, 1.0, 40_000, 39_500)  # sl 500 below
        qty_short = compute_quantity(10_000, 1.0, 40_000, 40_500)  # sl 500 above
        assert qty_long == pytest.approx(qty_short)

    def test_larger_equity_scales_quantity(self):
        qty_small = compute_quantity(10_000, 1.0, 40_000, 39_500)
        qty_large = compute_quantity(100_000, 1.0, 40_000, 39_500)
        assert qty_large == pytest.approx(10 * qty_small)

    def test_smaller_risk_pct_scales_quantity(self):
        qty_1pct = compute_quantity(10_000, 1.0, 40_000, 39_500)
        qty_05pct = compute_quantity(10_000, 0.5, 40_000, 39_500)
        assert qty_05pct == pytest.approx(0.5 * qty_1pct)

    def test_tighter_sl_gives_larger_quantity(self):
        """Tighter SL → smaller distance → larger position for same $ risk."""
        qty_wide_sl = compute_quantity(10_000, 1.0, 40_000, 39_000)   # $1000 SL
        qty_tight_sl = compute_quantity(10_000, 1.0, 40_000, 39_800)  # $200 SL
        assert qty_tight_sl > qty_wide_sl

    def test_raises_on_zero_equity(self):
        with pytest.raises(ValueError, match="equity_usdt"):
            compute_quantity(0, 1.0, 40_000, 39_500)

    def test_raises_on_negative_equity(self):
        with pytest.raises(ValueError, match="equity_usdt"):
            compute_quantity(-100, 1.0, 40_000, 39_500)

    def test_raises_on_zero_entry_price(self):
        with pytest.raises(ValueError, match="entry_price"):
            compute_quantity(10_000, 1.0, 0, -500)

    def test_raises_when_sl_equals_entry(self):
        with pytest.raises(ValueError, match="SL price equals entry"):
            compute_quantity(10_000, 1.0, 40_000, 40_000)


class TestComputeMarginRequired:
    def test_basic_5x_leverage(self):
        """0.2 BTC at $40k with 5x leverage → $40k notional / 5 = $1600 margin."""
        margin = compute_margin_required(quantity=0.2, entry_price=40_000, leverage=5)
        assert margin == pytest.approx(1_600.0)

    def test_higher_leverage_reduces_margin(self):
        m5 = compute_margin_required(0.1, 40_000, 5)
        m10 = compute_margin_required(0.1, 40_000, 10)
        assert m10 == pytest.approx(m5 / 2)

    def test_raises_on_zero_leverage(self):
        with pytest.raises(ValueError, match="leverage"):
            compute_margin_required(0.1, 40_000, 0)


class TestComputeDollarRisk:
    def test_long_risk(self):
        risk = compute_dollar_risk(quantity=0.2, entry_price=40_000, sl_price=39_500)
        assert risk == pytest.approx(100.0)

    def test_short_risk_same(self):
        risk = compute_dollar_risk(quantity=0.2, entry_price=40_000, sl_price=40_500)
        assert risk == pytest.approx(100.0)


# ─── Daily guardrails ────────────────────────────────────────────────────────


class TestDailyGuardrails:
    @pytest.fixture
    def guards(self, default_settings):
        g = DailyGuardrails(default_settings)
        g.new_day(current_equity=10_000)
        return g

    def test_can_open_position_initially(self, guards):
        ok, reason = guards.can_open_position(current_equity=10_000, open_positions=0)
        assert ok is True
        assert reason == ""

    def test_blocks_before_new_day_called(self, default_settings):
        g = DailyGuardrails(default_settings)  # no new_day called
        ok, reason = g.can_open_position(10_000, 0)
        assert ok is False
        assert "not initialised" in reason

    # ── Max concurrent positions ──────────────────────────────────────────

    def test_blocks_at_max_concurrent_positions(self, guards):
        ok, reason = guards.can_open_position(10_000, open_positions=3)
        assert ok is False
        assert "concurrent" in reason

    def test_allows_below_max_concurrent(self, guards):
        ok, _ = guards.can_open_position(10_000, open_positions=2)
        assert ok is True

    # ── Daily trade count ─────────────────────────────────────────────────

    def test_blocks_at_max_trades(self, guards, default_settings):
        for _ in range(default_settings.max_trades_per_day):
            guards.record_trade_opened()
        ok, reason = guards.can_open_position(10_000, open_positions=0)
        assert ok is False
        assert "trades" in reason

    def test_trade_count_increments(self, guards):
        assert guards.trade_count == 0
        guards.record_trade_opened()
        guards.record_trade_opened()
        assert guards.trade_count == 2

    # ── Daily loss limit ──────────────────────────────────────────────────

    def test_blocks_at_max_daily_loss(self, guards):
        # 5% loss from $10k start = equity drops to $9,500
        ok, reason = guards.can_open_position(current_equity=9_490, open_positions=0)
        assert ok is False
        assert "loss" in reason

    def test_pauses_bot_when_loss_limit_hit(self, guards):
        assert guards.paused is False
        guards.can_open_position(current_equity=9_000, open_positions=0)  # 10% loss
        assert guards.paused is True

    def test_paused_bot_stays_blocked(self, guards):
        guards.can_open_position(current_equity=9_000, open_positions=0)  # triggers pause
        # Even if equity recovers, still paused
        ok, reason = guards.can_open_position(current_equity=11_000, open_positions=0)
        assert ok is False
        assert "paused" in reason

    def test_gain_does_not_trigger_pause(self, guards):
        ok, _ = guards.can_open_position(current_equity=11_000, open_positions=0)
        assert ok is True
        assert guards.paused is False

    def test_allows_just_below_loss_limit(self, guards):
        # 4.9% loss → below the 5% limit
        ok, _ = guards.can_open_position(current_equity=9_510, open_positions=0)
        assert ok is True

    # ── New day resets ────────────────────────────────────────────────────

    def test_new_day_resets_trade_count(self, guards):
        guards.record_trade_opened()
        guards.record_trade_opened()
        guards.new_day(current_equity=9_500)
        assert guards.trade_count == 0

    def test_new_day_unpauses_bot(self, guards):
        guards.can_open_position(current_equity=9_000, open_positions=0)  # pause
        guards.new_day(current_equity=9_000)
        assert guards.paused is False

    def test_new_day_updates_start_equity(self, guards):
        guards.new_day(current_equity=9_500)
        assert guards.start_equity == pytest.approx(9_500)

    # ── maybe_rollover ────────────────────────────────────────────────────

    def test_maybe_rollover_calls_new_day_on_date_change(self, guards):
        guards._day = None  # force rollover
        rolled = guards.maybe_rollover(9_800)
        assert rolled is True
        assert guards.start_equity == pytest.approx(9_800)

    def test_maybe_rollover_noop_same_day(self, guards):
        original_equity = guards.start_equity
        rolled = guards.maybe_rollover(9_800)
        assert rolled is False
        assert guards.start_equity == pytest.approx(original_equity)

    # ── current_loss_pct ──────────────────────────────────────────────────

    def test_current_loss_pct_positive_on_loss(self, guards):
        pct = guards.current_loss_pct(9_500)
        assert pct == pytest.approx(5.0, rel=1e-3)

    def test_current_loss_pct_zero_on_gain(self, guards):
        pct = guards.current_loss_pct(10_500)
        assert pct == pytest.approx(0.0)

    def test_current_loss_pct_zero_before_new_day(self, default_settings):
        g = DailyGuardrails(default_settings)
        assert g.current_loss_pct(9_000) == pytest.approx(0.0)
