"""Async Binance USDT-M Futures client via ccxt."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import ccxt.async_support as ccxt
import pandas as pd
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from karen.config import Settings

# ─── Domain types ─────────────────────────────────────────────────────────────


@dataclass
class PositionInfo:
    symbol: str
    side: str          # "long" | "short"
    entry_price: float
    quantity: float    # in base currency
    leverage: int
    unrealized_pnl: float
    mark_price: float
    notional: float    # position value in USDT


@dataclass
class OrderInfo:
    id: str
    client_order_id: str
    symbol: str
    side: str           # "buy" | "sell"
    type: str           # "market" | "limit" | "stop_market" | "take_profit_market"
    status: str         # "open" | "closed" | "canceled"
    quantity: float
    filled: float
    price: float | None
    stop_price: float | None
    timestamp: datetime


# ─── Retry policy ─────────────────────────────────────────────────────────────

_RETRYABLE = (
    ccxt.NetworkError,
    ccxt.RequestTimeout,
    ccxt.ExchangeNotAvailable,
    ccxt.RateLimitExceeded,
)


def _make_retry(label: str) -> Any:
    return retry(
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        before_sleep=lambda rs: logger.warning(
            f"[{label}] attempt {rs.attempt_number} failed — retrying in {rs.next_action.sleep:.1f}s"  # type: ignore[union-attr]  # noqa: E501
        ),
        reraise=True,
    )


# ─── Client ───────────────────────────────────────────────────────────────────


class BinanceClient:
    """Async wrapper around ccxt binanceusdm (USDT-M perpetual futures)."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._exchange: ccxt.binanceusdm | None = None

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Initialise the ccxt exchange and verify connectivity."""
        self._exchange = ccxt.binanceusdm(
            {
                "apiKey": self._settings.binance_api_key,
                "secret": self._settings.binance_api_secret,
                "enableRateLimit": True,
                "options": {
                    "defaultType": "future",
                    "adjustForTimeDifference": True,
                },
            }
        )
        if self._settings.binance_testnet:
            self._exchange.set_sandbox_mode(True)
            logger.info("BinanceClient connected (TESTNET)")
        else:
            logger.info("BinanceClient connected (LIVE)")

    async def close(self) -> None:
        if self._exchange is not None:
            await self._exchange.close()
            self._exchange = None
            logger.info("BinanceClient closed")

    async def __aenter__(self) -> BinanceClient:
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    @property
    def exchange(self) -> ccxt.binanceusdm:
        if self._exchange is None:
            raise RuntimeError("BinanceClient not connected — call connect() first")
        return self._exchange

    # ── Symbol setup ───────────────────────────────────────────────────────

    async def setup_symbol(self, symbol: str) -> None:
        """Set isolated margin mode and leverage for a symbol."""
        try:
            await _set_margin_mode(self.exchange, symbol)
        except ccxt.ExchangeError as exc:
            # Binance returns an error if margin mode is already set correctly
            if "No need to change margin type" in str(exc):
                logger.debug(f"{symbol}: margin mode already isolated")
            else:
                raise

        try:
            await _set_leverage(self.exchange, symbol, self._settings.leverage)
            logger.info(f"{symbol}: leverage set to {self._settings.leverage}x isolated")
        except ccxt.ExchangeError as exc:
            if "leverage not changed" in str(exc).lower():
                logger.debug(f"{symbol}: leverage already at {self._settings.leverage}x")
            else:
                raise

    # ── Market data ────────────────────────────────────────────────────────

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 250,
    ) -> pd.DataFrame:
        """Return OHLCV DataFrame with UTC-aware timestamps."""
        raw = await _fetch_ohlcv_with_retry(self.exchange, symbol, timeframe, limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").astype(float)
        return df

    # ── Account ────────────────────────────────────────────────────────────

    async def fetch_equity(self) -> float:
        """Return total wallet balance (equity) in USDT."""
        balance = await _fetch_balance_with_retry(self.exchange)
        # binanceusdm reports margin balance under 'USDT'
        usdt = balance.get("USDT", {})
        # 'total' = wallet balance + unrealized PnL
        return float(usdt.get("total", 0.0))

    # ── Positions ──────────────────────────────────────────────────────────

    async def fetch_positions(self) -> list[PositionInfo]:
        """Return all open positions (non-zero notional)."""
        raw: list[dict[str, Any]] = await _fetch_positions_with_retry(self.exchange)
        result: list[PositionInfo] = []
        for p in raw:
            contracts = float(p.get("contracts") or 0)
            if contracts == 0:
                continue
            side_raw = str(p.get("side") or "").lower()
            if side_raw not in ("long", "short"):
                continue
            result.append(
                PositionInfo(
                    symbol=str(p["symbol"]),
                    side=side_raw,
                    entry_price=float(p.get("entryPrice") or 0),
                    quantity=abs(contracts),
                    leverage=int(float(p.get("leverage") or self._settings.leverage)),
                    unrealized_pnl=float(p.get("unrealizedPnl") or 0),
                    mark_price=float(p.get("markPrice") or 0),
                    notional=abs(float(p.get("notional") or 0)),
                )
            )
        return result

    # ── Orders ─────────────────────────────────────────────────────────────

    async def place_order(
        self,
        symbol: str,
        side: str,          # "buy" | "sell"
        order_type: str,    # "market" | "limit" | "stop_market" | "take_profit_market"
        quantity: float,
        price: float | None = None,
        stop_price: float | None = None,
        reduce_only: bool = False,
        client_order_id: str | None = None,
    ) -> OrderInfo:
        """Place an order and return a typed OrderInfo."""
        coid = client_order_id or _make_client_order_id()
        params: dict[str, Any] = {
            "newClientOrderId": coid,
        }
        if reduce_only:
            params["reduceOnly"] = True
        if stop_price is not None:
            params["stopPrice"] = stop_price

        raw = await _create_order_with_retry(
            self.exchange,
            symbol=symbol,
            order_type=order_type.upper(),
            side=side,
            amount=quantity,
            price=price,
            params=params,
        )
        logger.info(
            f"Order placed: {symbol} {side.upper()} {order_type} qty={quantity} "
            f"coid={coid} → id={raw['id']}"
        )
        return _parse_order(raw)

    async def cancel_order(self, symbol: str, order_id: str) -> None:
        """Cancel a single order by exchange ID."""
        await _cancel_order_with_retry(self.exchange, order_id, symbol)
        logger.info(f"Order cancelled: {symbol} id={order_id}")

    async def cancel_all_orders(self, symbol: str) -> None:
        """Cancel all open orders for a symbol."""
        await _cancel_all_orders_with_retry(self.exchange, symbol)
        logger.info(f"All orders cancelled: {symbol}")

    async def fetch_order(self, symbol: str, order_id: str) -> OrderInfo:
        """Fetch order status by exchange ID."""
        raw = await _fetch_order_with_retry(self.exchange, order_id, symbol)
        return _parse_order(raw)

    async def fetch_open_orders(self, symbol: str) -> list[OrderInfo]:
        """Return all open orders for a symbol."""
        raw_list = await _fetch_open_orders_with_retry(self.exchange, symbol)
        return [_parse_order(o) for o in raw_list]


# ─── Tenacity-wrapped helpers (module-level so they can be patched in tests) ──


@_make_retry("set_margin_mode")
async def _set_margin_mode(exchange: ccxt.binanceusdm, symbol: str) -> None:
    await exchange.set_margin_mode("isolated", symbol)


@_make_retry("set_leverage")
async def _set_leverage(exchange: ccxt.binanceusdm, symbol: str, leverage: int) -> None:
    await exchange.set_leverage(leverage, symbol)


@_make_retry("fetch_ohlcv")
async def _fetch_ohlcv_with_retry(
    exchange: ccxt.binanceusdm, symbol: str, timeframe: str, limit: int
) -> list[list[Any]]:
    return await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)  # type: ignore[return-value]


@_make_retry("fetch_balance")
async def _fetch_balance_with_retry(exchange: ccxt.binanceusdm) -> dict[str, Any]:
    return await exchange.fetch_balance()  # type: ignore[return-value]


@_make_retry("fetch_positions")
async def _fetch_positions_with_retry(
    exchange: ccxt.binanceusdm,
) -> list[dict[str, Any]]:
    return await exchange.fetch_positions()  # type: ignore[return-value]


@_make_retry("create_order")
async def _create_order_with_retry(
    exchange: ccxt.binanceusdm,
    symbol: str,
    order_type: str,
    side: str,
    amount: float,
    price: float | None,
    params: dict[str, Any],
) -> dict[str, Any]:
    return await exchange.create_order(  # type: ignore[return-value]
        symbol, order_type, side, amount, price, params
    )


@_make_retry("cancel_order")
async def _cancel_order_with_retry(
    exchange: ccxt.binanceusdm, order_id: str, symbol: str
) -> None:
    await exchange.cancel_order(order_id, symbol)


@_make_retry("cancel_all_orders")
async def _cancel_all_orders_with_retry(exchange: ccxt.binanceusdm, symbol: str) -> None:
    await exchange.cancel_all_orders(symbol)


@_make_retry("fetch_order")
async def _fetch_order_with_retry(
    exchange: ccxt.binanceusdm, order_id: str, symbol: str
) -> dict[str, Any]:
    return await exchange.fetch_order(order_id, symbol)  # type: ignore[return-value]


@_make_retry("fetch_open_orders")
async def _fetch_open_orders_with_retry(
    exchange: ccxt.binanceusdm, symbol: str
) -> list[dict[str, Any]]:
    return await exchange.fetch_open_orders(symbol)  # type: ignore[return-value]


# ─── Utilities ────────────────────────────────────────────────────────────────


def _make_client_order_id() -> str:
    """Generate a unique client order ID identifiable as Karen's."""
    return f"karen-{uuid.uuid4().hex[:12]}"


def _parse_order(raw: dict[str, Any]) -> OrderInfo:
    ts_ms = raw.get("timestamp") or 0
    ts = datetime.fromtimestamp(ts_ms / 1000, tz=UTC) if ts_ms else datetime.now(tz=UTC)
    status_map = {  # noqa: E501
        "open": "open", "closed": "closed", "canceled": "canceled", "cancelled": "canceled"
    }
    raw_status = str(raw.get("status") or "open").lower()
    return OrderInfo(
        id=str(raw.get("id") or ""),
        client_order_id=str(raw.get("clientOrderId") or ""),
        symbol=str(raw.get("symbol") or ""),
        side=str(raw.get("side") or "").lower(),
        type=str(raw.get("type") or "").lower(),
        status=status_map.get(raw_status, raw_status),
        quantity=float(raw.get("amount") or 0),
        filled=float(raw.get("filled") or 0),
        price=float(raw["price"]) if raw.get("price") else None,
        stop_price=float(raw["stopPrice"]) if raw.get("stopPrice") else None,
        timestamp=ts,
    )
