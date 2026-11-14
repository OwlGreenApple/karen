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


# ─── Dynamic leverage ─────────────────────────────────────────────────────────

MIN_LEVERAGE = 5
MAX_LEVERAGE = 20
# Binance maintenance margin rate for the lowest notional tier (~0.4–0.5% on majors)
MAINTENANCE_MARGIN_RATE = 0.005
# SL must sit inside this fraction of the liquidation distance (20% safety buffer)
LIQ_SAFETY = 0.8
# Keep a little balance free so Binance accepts the order (entry fee is charged on top)
MARGIN_FEE_BUFFER = 0.97


def leverage_from_confidence(confidence: float) -> int:
    """Map a 0..1 confidence score linearly onto MIN_LEVERAGE..MAX_LEVERAGE."""
    c = min(1.0, max(0.0, confidence))
    return int(round(MIN_LEVERAGE + c * (MAX_LEVERAGE - MIN_LEVERAGE)))


def max_safe_leverage(entry_price: float, sl_price: float) -> int:
    """
    Highest leverage at which the isolated-margin liquidation price is still
    comfortably beyond the stop loss.

    Isolated liquidation distance ≈ 1/leverage − maintenance_margin_rate.
    We require:  sl_distance_pct ≤ LIQ_SAFETY × (1/lev − mmr)
              →  lev ≤ 1 / (sl_distance_pct / LIQ_SAFETY + mmr)
    """
    if entry_price <= 0:
        raise ValueError(f"entry_price must be positive, got {entry_price}")
    sl_pct = abs(entry_price - sl_price) / entry_price
    return int(1.0 / (sl_pct / LIQ_SAFETY + MAINTENANCE_MARGIN_RATE))


def choose_leverage(confidence: float, entry_price: float, sl_price: float) -> int | None:
    """
    Leverage for a trade: confidence-based, capped so the SL triggers before
    liquidation. Returns None if even MIN_LEVERAGE would liquidate before the SL.
    """
    lev = min(leverage_from_confidence(confidence), max_safe_leverage(entry_price, sl_price))
    lev = min(lev, MAX_LEVERAGE)
    return lev if lev >= MIN_LEVERAGE else None


# ─── Max loss per position ────────────────────────────────────────────────────

# A stop-loss hit may lose at most this fraction of the position's margin
# (Binance ROE −18%). With 100% margin per entry that is ~18% of the balance.
MAX_SL_LOSS_PCT = 18.0


def cap_sl_to_max_loss(side: str, entry_price: float, sl_price: float, leverage: int) -> float:
    """
    Pull the SL closer to entry if hitting it would lose more than MAX_SL_LOSS_PCT
    of margin.  Loss on margin = leverage × |entry − sl| / entry.
    """
    if entry_price <= 0 or leverage <= 0:
        raise ValueError("entry_price and leverage must be positive")
    max_dist = entry_price * (MAX_SL_LOSS_PCT / 100.0) / leverage
    if side == "long":
        return max(sl_price, entry_price - max_dist)
    return min(sl_price, entry_price + max_dist)
