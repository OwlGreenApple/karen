"""ATR-based position sizing: risk a fixed % of equity per trade."""

from __future__ import annotations


def compute_quantity(
    equity_usdt: float,
    risk_pct: float,
    entry_price: float,
    sl_price: float,
) -> float:
    """
    Return the quantity (base currency) that risks exactly `risk_pct`% of equity.

    Formula:
        risk_amount  = equity × risk_pct / 100
        sl_distance  = |entry - sl|          (USDT per unit of base currency)
        quantity     = risk_amount / sl_distance

    Leverage does not change this formula — it only determines margin reserved.
    If price hits the SL exactly, the loss equals risk_amount.

    Raises:
        ValueError: if equity or entry_price ≤ 0, or SL distance is zero.
    """
    if equity_usdt <= 0:
        raise ValueError(f"equity_usdt must be positive, got {equity_usdt}")
    if entry_price <= 0:
        raise ValueError(f"entry_price must be positive, got {entry_price}")
    sl_distance = abs(entry_price - sl_price)
    if sl_distance == 0:
        raise ValueError("SL price equals entry price — SL distance is zero")

    risk_amount = equity_usdt * (risk_pct / 100.0)
    return risk_amount / sl_distance


def compute_quantity_by_margin_pct(
    equity_usdt: float,
    margin_pct: float,
    entry_price: float,
    leverage: int,
) -> float:
    """
    Return quantity where margin = equity × margin_pct%.

    margin   = equity × margin_pct / 100
    notional = margin × leverage
    quantity = notional / entry_price
    """
    if equity_usdt <= 0:
        raise ValueError(f"equity_usdt must be positive, got {equity_usdt}")
    if entry_price <= 0:
        raise ValueError(f"entry_price must be positive, got {entry_price}")
    if leverage <= 0:
        raise ValueError(f"leverage must be positive, got {leverage}")
    margin = equity_usdt * (margin_pct / 100.0)
    notional = margin * leverage
    return notional / entry_price


def compute_margin_required(
    quantity: float,
    entry_price: float,
    leverage: int,
) -> float:
    """Return the margin (USDT) required to open a position at the given leverage."""
    if leverage <= 0:
        raise ValueError(f"leverage must be positive, got {leverage}")
    notional = quantity * entry_price
    return notional / leverage


def compute_dollar_risk(
    quantity: float,
    entry_price: float,
    sl_price: float,
) -> float:
    """Return the dollar loss if SL is hit exactly."""
    return quantity * abs(entry_price - sl_price)
