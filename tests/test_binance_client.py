"""Tests for BinanceClient — fully mocked, no real API keys required."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from karen.config import Settings, StrategyMode
from karen.exchange.binance_client import (
    BinanceClient,
    OrderInfo,
    _make_client_order_id,
    _parse_order,
)

# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def settings() -> Settings:
    return Settings(
        binance_api_key="test_key",
        binance_api_secret="test_secret",
        binance_testnet=True,
        trading_enabled=False,
        strategy_mode=StrategyMode.MEAN_REVERSION,
        leverage=5,
    )


@pytest.fixture
def mock_exchange() -> MagicMock:
    """A MagicMock that looks like a ccxt binanceusdm exchange."""
    ex = MagicMock()
    ex.set_sandbox_mode = MagicMock()
    ex.set_margin_mode = AsyncMock()
    ex.set_leverage = AsyncMock()
    ex.fetch_ohlcv = AsyncMock()
    ex.fetch_balance = AsyncMock()
    ex.fetch_positions = AsyncMock()
    ex.create_order = AsyncMock()
    ex.cancel_order = AsyncMock()
    ex.cancel_all_orders = AsyncMock()
    ex.fetch_order = AsyncMock()
    ex.fetch_open_orders = AsyncMock()
    ex.close = AsyncMock()
    return ex


@pytest.fixture
def client(settings: Settings, mock_exchange: MagicMock) -> BinanceClient:
    """A BinanceClient with a pre-injected mock exchange."""
    c = BinanceClient(settings)
    c._exchange = mock_exchange
    return c


# ─── _make_client_order_id ────────────────────────────────────────────────────


def test_client_order_id_format() -> None:
    coid = _make_client_order_id()
    assert coid.startswith("karen-")
    assert len(coid) == len("karen-") + 12


def test_client_order_ids_are_unique() -> None:
    ids = {_make_client_order_id() for _ in range(100)}
    assert len(ids) == 100


# ─── connect / close ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_connect_sets_sandbox_on_testnet(settings: Settings) -> None:
    with patch("karen.exchange.binance_client.ccxt") as mock_ccxt:
        mock_ex = MagicMock()
        mock_ex.set_sandbox_mode = MagicMock()
        mock_ex.close = AsyncMock()
        mock_ccxt.binanceusdm.return_value = mock_ex

        c = BinanceClient(settings)
        await c.connect()
        mock_ex.set_sandbox_mode.assert_called_once_with(True)
        await c.close()


@pytest.mark.asyncio
async def test_connect_no_sandbox_on_live() -> None:
    s = Settings(
        binance_api_key="k",
        binance_api_secret="s",
        binance_testnet=False,
        trading_enabled=False,
        strategy_mode=StrategyMode.MEAN_REVERSION,
    )
    with patch("karen.exchange.binance_client.ccxt") as mock_ccxt:
        mock_ex = MagicMock()
        mock_ex.set_sandbox_mode = MagicMock()
        mock_ex.close = AsyncMock()
        mock_ccxt.binanceusdm.return_value = mock_ex

        c = BinanceClient(s)
        await c.connect()
        mock_ex.set_sandbox_mode.assert_not_called()
        await c.close()


@pytest.mark.asyncio
async def test_exchange_raises_when_not_connected(settings: Settings) -> None:
    c = BinanceClient(settings)
    with pytest.raises(RuntimeError, match="not connected"):
        _ = c.exchange


# ─── setup_symbol ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_setup_symbol_sets_isolated_and_leverage(
    client: BinanceClient, mock_exchange: MagicMock
) -> None:
    await client.setup_symbol("BTCUSDT")
    mock_exchange.set_margin_mode.assert_called_once_with("isolated", "BTCUSDT")
    mock_exchange.set_leverage.assert_called_once_with(5, "BTCUSDT")


@pytest.mark.asyncio
async def test_setup_symbol_ignores_already_set_margin_mode(
    client: BinanceClient, mock_exchange: MagicMock
) -> None:
    import ccxt

    mock_exchange.set_margin_mode.side_effect = ccxt.ExchangeError(
        "No need to change margin type"
    )
    # Should not raise
    await client.setup_symbol("BTCUSDT")
    mock_exchange.set_leverage.assert_called_once()


@pytest.mark.asyncio
async def test_setup_symbol_propagates_unexpected_margin_error(
    client: BinanceClient, mock_exchange: MagicMock
) -> None:
    import ccxt

    mock_exchange.set_margin_mode.side_effect = ccxt.ExchangeError("Unknown error")
    with pytest.raises(ccxt.ExchangeError):
        await client.setup_symbol("BTCUSDT")


# ─── fetch_ohlcv ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_ohlcv_returns_dataframe(
    client: BinanceClient, mock_exchange: MagicMock
) -> None:
    import pandas as pd

    raw = [
        [1_700_000_000_000, 40000.0, 40500.0, 39800.0, 40200.0, 123.5],
        [1_700_000_900_000, 40200.0, 40300.0, 40100.0, 40250.0, 98.2],
    ]
    mock_exchange.fetch_ohlcv.return_value = raw

    df = await client.fetch_ohlcv("BTCUSDT", "15m", limit=2)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 2
    assert df.index.tz is not None  # UTC-aware
    assert df["close"].iloc[0] == pytest.approx(40200.0)


@pytest.mark.asyncio
async def test_fetch_ohlcv_calls_correct_params(
    client: BinanceClient, mock_exchange: MagicMock
) -> None:
    mock_exchange.fetch_ohlcv.return_value = []
    await client.fetch_ohlcv("ETHUSDT", "1h", limit=100)
    mock_exchange.fetch_ohlcv.assert_called_once_with("ETHUSDT", "1h", limit=100)


# ─── fetch_equity ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_equity_returns_usdt_total(
    client: BinanceClient, mock_exchange: MagicMock
) -> None:
    mock_exchange.fetch_balance.return_value = {
        "USDT": {"free": 900.0, "used": 100.0, "total": 1000.0}
    }
    equity = await client.fetch_equity()
    assert equity == pytest.approx(1000.0)


@pytest.mark.asyncio
async def test_fetch_equity_missing_usdt_returns_zero(
    client: BinanceClient, mock_exchange: MagicMock
) -> None:
    mock_exchange.fetch_balance.return_value = {}
    equity = await client.fetch_equity()
    assert equity == pytest.approx(0.0)


# ─── fetch_positions ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_positions_filters_zero_size(
    client: BinanceClient, mock_exchange: MagicMock
) -> None:
    mock_exchange.fetch_positions.return_value = [
        {
            "symbol": "BTCUSDT",
            "side": "long",
            "contracts": 0.0,
            "entryPrice": 40000.0,
            "unrealizedPnl": 0.0,
            "markPrice": 40100.0,
            "leverage": "5",
            "notional": "0",
        },
        {
            "symbol": "ETHUSDT",
            "side": "short",
            "contracts": 1.5,
            "entryPrice": 2500.0,
            "unrealizedPnl": -50.0,
            "markPrice": 2533.3,
            "leverage": "5",
            "notional": "3800",
        },
    ]
    positions = await client.fetch_positions()
    assert len(positions) == 1
    pos = positions[0]
    assert pos.symbol == "ETHUSDT"
    assert pos.side == "short"
    assert pos.quantity == pytest.approx(1.5)
    assert pos.unrealized_pnl == pytest.approx(-50.0)
    assert pos.leverage == 5


@pytest.mark.asyncio
async def test_fetch_positions_empty(
    client: BinanceClient, mock_exchange: MagicMock
) -> None:
    mock_exchange.fetch_positions.return_value = []
    positions = await client.fetch_positions()
    assert positions == []


# ─── place_order ──────────────────────────────────────────────────────────────


def _order_response(
    oid: str = "123456",
    coid: str = "karen-abc",
    symbol: str = "BTCUSDT",
    side: str = "buy",
    otype: str = "market",
    status: str = "closed",
    amount: float = 0.01,
    filled: float = 0.01,
    price: float | None = None,
    stop_price: float | None = None,
) -> dict:
    return {
        "id": oid,
        "clientOrderId": coid,
        "symbol": symbol,
        "side": side,
        "type": otype,
        "status": status,
        "amount": amount,
        "filled": filled,
        "price": price,
        "stopPrice": stop_price,
        "timestamp": 1_700_000_000_000,
    }


@pytest.mark.asyncio
async def test_place_market_order(
    client: BinanceClient, mock_exchange: MagicMock
) -> None:
    mock_exchange.create_order.return_value = _order_response(oid="111", side="buy")
    order = await client.place_order("BTCUSDT", "buy", "market", 0.01)
    assert isinstance(order, OrderInfo)
    assert order.id == "111"
    assert order.side == "buy"
    # Verify ccxt was called with correct order type (uppercased)
    call_args = mock_exchange.create_order.call_args
    assert call_args[0][1] == "MARKET"


@pytest.mark.asyncio
async def test_place_stop_market_order(
    client: BinanceClient, mock_exchange: MagicMock
) -> None:
    mock_exchange.create_order.return_value = _order_response(
        oid="222", otype="stop_market", stop_price=39000.0
    )
    await client.place_order(
        "BTCUSDT", "sell", "stop_market", 0.01,
        stop_price=39000.0, reduce_only=True
    )
    params = mock_exchange.create_order.call_args[0][5]
    assert params["stopPrice"] == 39000.0
    assert params["reduceOnly"] is True


@pytest.mark.asyncio
async def test_place_order_uses_provided_client_order_id(
    client: BinanceClient, mock_exchange: MagicMock
) -> None:
    mock_exchange.create_order.return_value = _order_response(coid="my-id")
    await client.place_order("BTCUSDT", "buy", "market", 0.01, client_order_id="my-id")
    params = mock_exchange.create_order.call_args[0][5]
    assert params["newClientOrderId"] == "my-id"


@pytest.mark.asyncio
async def test_place_order_generates_client_order_id_if_none(
    client: BinanceClient, mock_exchange: MagicMock
) -> None:
    mock_exchange.create_order.return_value = _order_response()
    await client.place_order("BTCUSDT", "buy", "market", 0.01)
    params = mock_exchange.create_order.call_args[0][5]
    assert params["newClientOrderId"].startswith("karen-")


# ─── cancel_order / cancel_all_orders ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_cancel_order(client: BinanceClient, mock_exchange: MagicMock) -> None:
    await client.cancel_order("BTCUSDT", "999")
    mock_exchange.cancel_order.assert_called_once_with("999", "BTCUSDT")


@pytest.mark.asyncio
async def test_cancel_all_orders(client: BinanceClient, mock_exchange: MagicMock) -> None:
    await client.cancel_all_orders("ETHUSDT")
    mock_exchange.cancel_all_orders.assert_called_once_with("ETHUSDT")


# ─── fetch_order ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_order_returns_order_info(
    client: BinanceClient, mock_exchange: MagicMock
) -> None:
    mock_exchange.fetch_order.return_value = _order_response(oid="777", status="open")
    order = await client.fetch_order("BTCUSDT", "777")
    assert order.id == "777"
    assert order.status == "open"


# ─── Retry behaviour ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_equity_retries_on_network_error(
    client: BinanceClient, mock_exchange: MagicMock
) -> None:
    import ccxt

    # Fail twice then succeed
    mock_exchange.fetch_balance.side_effect = [
        ccxt.NetworkError("timeout"),
        ccxt.NetworkError("timeout"),
        {"USDT": {"total": 500.0}},
    ]
    equity = await client.fetch_equity()
    assert equity == pytest.approx(500.0)
    assert mock_exchange.fetch_balance.call_count == 3


@pytest.mark.asyncio
async def test_fetch_equity_raises_after_max_retries(
    client: BinanceClient, mock_exchange: MagicMock
) -> None:
    import ccxt

    mock_exchange.fetch_balance.side_effect = ccxt.NetworkError("timeout")
    with pytest.raises(ccxt.NetworkError):
        await client.fetch_equity()
    assert mock_exchange.fetch_balance.call_count == 3


@pytest.mark.asyncio
async def test_auth_error_not_retried(
    client: BinanceClient, mock_exchange: MagicMock
) -> None:
    import ccxt

    mock_exchange.fetch_balance.side_effect = ccxt.AuthenticationError("bad key")
    with pytest.raises(ccxt.AuthenticationError):
        await client.fetch_equity()
    assert mock_exchange.fetch_balance.call_count == 1  # no retry


# ─── _parse_order edge cases ──────────────────────────────────────────────────


def test_parse_order_handles_missing_price() -> None:
    raw = {
        "id": "1",
        "clientOrderId": "k",
        "symbol": "BTCUSDT",
        "side": "buy",
        "type": "market",
        "status": "closed",
        "amount": 0.01,
        "filled": 0.01,
        "price": None,
        "stopPrice": None,
        "timestamp": 1_700_000_000_000,
    }
    order = _parse_order(raw)
    assert order.price is None
    assert order.stop_price is None


def test_parse_order_normalises_cancelled_spelling() -> None:
    raw = {
        "id": "2", "clientOrderId": "k", "symbol": "BTCUSDT",
        "side": "sell", "type": "limit", "status": "cancelled",
        "amount": 0.01, "filled": 0.0,
        "price": 40000.0, "stopPrice": None, "timestamp": 0,
    }
    order = _parse_order(raw)
    assert order.status == "canceled"
