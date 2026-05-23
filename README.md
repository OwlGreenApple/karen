# Karen — AI Trading Bot

> ⚠️ **Disclaimer:** This software is for educational purposes only. Trading cryptocurrency futures with leverage carries substantial risk of loss and can result in losses exceeding your initial deposit. Never trade with money you cannot afford to lose.

Karen is a production-grade automated trading bot for Binance USDT-M Futures with two switchable strategies, real-time Telegram notifications, and a local live dashboard.

---

## Features

- **Two strategies** — Mean Reversion (ranging markets) and Trend Following (trending markets)
- **Hot-reload** — change `STRATEGY_MODE` or `TRADING_ENABLED` in `.env` and Karen reacts within 5 seconds, no restart
- **Risk management** — position sizing by ATR, max concurrent positions, daily loss limit
- **Telegram notifications** — every trade open/close, mode change, errors
- **Live dashboard** — React + FastAPI, real-time via WebSocket
- **Testnet-first** — defaults to Binance testnet; you must consciously enable live trading

---

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+ (for dashboard frontend)
- Binance account + API keys (testnet recommended to start)
- Telegram bot token (optional but recommended)

### Setup

```bash
git clone <repo>
cd karen

# Python environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env — add your Binance testnet keys and Telegram token

# Run everything
chmod +x run.sh
./run.sh
```

Open `http://localhost:8000` to see the dashboard.

---

## Configuration (`.env`)

All settings are hot-reloadable (except `DASHBOARD_PORT` which requires restart).

| Key | Default | Description |
|-----|---------|-------------|
| `TRADING_ENABLED` | `false` | Global kill switch. `false` = log signals only, no orders placed |
| `STRATEGY_MODE` | `mean_reversion` | `mean_reversion` \| `trend_following` |
| `BINANCE_TESTNET` | `true` | Use testnet. Set to `false` for live trading |
| `LEVERAGE` | `5` | Isolated margin leverage per position |
| `RISK_PER_TRADE_PCT` | `1.0` | % of equity risked per trade (SL-based sizing) |
| `MAX_CONCURRENT_POSITIONS` | `3` | Hard limit on open positions |
| `MAX_DAILY_LOSS_PCT` | `5.0` | Auto-pause threshold (% of starting equity) |
| `MAX_TRADES_PER_DAY` | `25` | Hard cap on daily trades |

See `.env.example` for all strategy-specific parameters.

---

## Strategies

### Mean Reversion
Best in **ranging** markets (ADX < 20). Trades price extremes back to the mean.
- Entry: Price outside Bollinger Band + RSI extreme + close back inside band
- TP: Middle BB (SMA20), partial 70% / runner 30%
- SL: 1.5 × ATR beyond entry
- Time stop: 8 candles (2 hours)

### Trend Following
Best in **trending** markets (ADX > 25). Trades pullbacks in the direction of the trend.
- Entry: 1h trend confirmed, 15m pullback to EMA21 + MACD + volume
- TP: 2.5 × ATR (partial 50%), then Chandelier Exit trailing
- SL: Below swing low/high, max 2 × ATR

---

## Toggling Strategy Mode

Edit `.env`:
```env
STRATEGY_MODE=trend_following
```
Karen detects the change within 5 seconds, cancels pending entry orders from the old mode, and starts scanning with the new mode. Existing open positions are untouched.

---

## Dashboard

Visit `http://localhost:8000` after starting Karen.

- **Status bar** — current mode, trading enabled/disabled, equity, today's PnL
- **Equity chart** — toggle daily (30d) / hourly (7d) view
- **Positions panel** — live unrealized PnL via WebSocket
- **Trades table** — paginated history with entry/exit/PnL/reason

---

## Running Tests

```bash
pytest -v
# 193 tests — strategies, order manager, position tracker, guardrails, dashboard API
```

## Building the Dashboard Frontend

```bash
cd karen/dashboard/frontend
npm install
npm run build
# Built assets are output to karen/dashboard/static/
# The FastAPI server serves them automatically
```

The `run.sh` script does this automatically if `node_modules/` is present.

---

## Project Structure

```
karen/
├── karen/
│   ├── config.py              # settings + hot-reload (watchdog)
│   ├── exchange/              # ccxt wrapper + WebSocket feeds
│   ├── strategies/            # MeanReversion + TrendFollowing
│   ├── indicators/            # pandas-ta wrappers
│   ├── risk/                  # position sizing + guardrails
│   ├── execution/             # order management + position tracking
│   ├── notifications/         # Telegram
│   ├── persistence/           # SQLite (SQLAlchemy async)
│   └── dashboard/             # FastAPI + React frontend
└── tests/
```
