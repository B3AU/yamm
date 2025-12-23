# Trading Bot for Interactive Brokers

Short-only trading bot that uses ML model predictions to identify underperforming stocks.

## Prerequisites

1. **Interactive Brokers Account** - Paper or live trading account
2. **TWS or IB Gateway** - Running with API enabled
3. **FMP API Key** - For market data (set as `FMP_API_KEY` environment variable)
4. **Trained Model** - `data/model_final.pt` from the training notebooks

## Installation

```bash
pip install -r requirements.txt
```

## IB Gateway/TWS Setup

1. Open TWS or IB Gateway
2. Go to **Edit → Global Configuration → API → Settings**
3. Enable **"Enable ActiveX and Socket Clients"**
4. Set **Socket port**: 7497 (paper) or 7496 (live)
5. Uncheck **"Read-Only API"** to allow trading
6. Add **127.0.0.1** to trusted IPs

## Configuration

Edit `trading/config.py` or set environment variables:

```bash
export FMP_API_KEY="your_api_key"
```

### Strategy Parameters (in `config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k_short` | 10 | Number of stocks to short |
| `hold_days` | 3 | Days between rebalances |
| `max_position_pct` | 0.15 | Max 15% per position |
| `max_portfolio_short` | 1.0 | Max 100% short exposure |
| `min_market_cap` | $500M | Minimum market cap filter |
| `min_price` | $5 | Minimum stock price |
| `stop_loss_pct` | 0.15 | 15% stop-loss per position |

## Usage

### Check Status

```bash
python trading/main.py --status
```

### Dry Run (see what would happen)

```bash
python trading/main.py --rebalance --dry-run
```

### Execute Rebalance (Paper Trading)

```bash
python trading/main.py --rebalance
```

### Close All Positions

```bash
python trading/main.py --close-all
```

### Live Trading

```bash
# Requires typing "CONFIRM" to proceed
python trading/main.py --live --rebalance
```

## Components

| File | Description |
|------|-------------|
| `config.py` | Configuration dataclasses |
| `data_pipeline.py` | FMP API client, feature computation |
| `model.py` | Load model, score and rank stocks |
| `ib_client.py` | IB connection, orders, positions |
| `strategy.py` | Position sizing, rebalance logic |
| `risk.py` | Limits, stop-losses, circuit breaker |
| `main.py` | CLI entry point |

## Risk Management

- **Position Limits**: Max 15% per position, 100% total short exposure
- **Stop-Loss**: 15% per position (configurable)
- **Circuit Breaker**: Halts trading on 5% daily loss or 10% drawdown
- **Filters**: $500M market cap, $5 price, 500K volume minimums

## Logging

Logs are written to `trading/logs/trading.log` and stdout.

Set log level with `--log-level`:

```bash
python trading/main.py --status --log-level DEBUG
```

## Paper vs Live Trading

Paper trading uses port **7497**, live uses **7496**. The code is identical - only the connection port differs.

**Important**: Paper trading does NOT simulate short availability. In live trading, some stocks may be hard-to-borrow or unavailable.

## Typical Workflow

1. Start TWS/Gateway and ensure API is enabled
2. Check status: `python trading/main.py --status`
3. Dry run: `python trading/main.py --rebalance --dry-run`
4. If looks good, execute: `python trading/main.py --rebalance`
5. Monitor positions in TWS

## Troubleshooting

### "Failed to connect to IB"

- Ensure TWS/Gateway is running
- Check API settings are enabled
- Verify port number (7497 for paper)
- Check no other client is using the same client ID

### "No candidates" or empty results

- Check FMP API key is set
- Verify `data/model_final.pt` exists
- Check universe file or dataset exists

### Orders not filling

- Market may be closed
- Check order status in TWS
- Verify account has sufficient margin for shorts
