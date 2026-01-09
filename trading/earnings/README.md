# Earnings Straddle Trading System

Automated system for trading volatility around earnings announcements. Uses ML to predict post-earnings moves and trades straddles when implied volatility misprices the distribution.

## Quick Start

```bash
# Start the daemon (runs on schedule)
python3 -m trading.earnings.daemon

# View dashboard
python3 -m trading.earnings.dashboard --watch --live

# Test screening pipeline
python3 -m trading.earnings.test_screening
```

## Components

| File | Purpose |
|------|---------|
| `daemon.py` | Main scheduler - screens candidates, places orders, manages exits |
| `screener.py` | Fetches earnings calendar, screens option chains for liquidity |
| `ml_predictor.py` | LightGBM quantile regression for move prediction |
| `llm_sanity_check.py` | LLM + web search pre-trade validation |
| `executor.py` | IBKR order placement and fill monitoring |
| `logging.py` | SQLite trade/non-trade logging |
| `dashboard.py` | CLI dashboard for monitoring |
| `live_news.py` | News fetching and embedding for ML features |
| `test_screening.py` | Standalone pipeline testing |

## Test Screening Script

Run the ML and LLM screening pipeline outside of the daemon:

```bash
# Full pipeline with IBKR connection
python3 -m trading.earnings.test_screening

# Specific ticker(s) with manual earnings date
python3 -m trading.earnings.test_screening --ticker AAPL --earnings-date 2026-01-15 --timing AMC

# Skip IBKR screening (ML + LLM only, uses mock candidate data)
python3 -m trading.earnings.test_screening --no-ibkr --ticker AAPL

# Skip LLM check
python3 -m trading.earnings.test_screening --no-llm

# Custom thresholds
python3 -m trading.earnings.test_screening --spread-threshold 12 --edge-threshold 0.03

# All options
python3 -m trading.earnings.test_screening --help
```

## Dashboard

```bash
# One-shot view
python3 -m trading.earnings.dashboard

# Watch mode with live IBKR prices
python3 -m trading.earnings.dashboard --watch --live

# Include completed trades
python3 -m trading.earnings.dashboard --all
```

Interactive commands in watch mode:
- `c` - Close a position manually
- `r` - Refresh
- `q` - Quit

## Daemon Schedule (ET)

| Time | Task |
|------|------|
| 09:25 | IBKR connection check |
| 14:00 | Screen candidates + place orders |
| 14:45 | Exit existing positions |
| 15:55 | Check fills, cancel unfilled orders |

## Environment Variables

```bash
# Required
FMP_API_KEY=xxx              # Financial Modeling Prep API

# IBKR
IB_PORT=4002                 # Gateway port (4002=paper, 7497=live)
IB_CLIENT_ID=1               # Client ID

# LLM Sanity Check (optional)
OPENROUTER_API_KEY=xxx       # For LLM calls
TAVILY_API_KEY=xxx           # For web search
LLM_SANITY_MODEL=anthropic/claude-3.5-sonnet

# LLM threshold: PASS, WARN, NO_TRADE, DISABLED
LLM_SANITY_THRESHOLD=WARN              # Live default
PAPER_LLM_SANITY_THRESHOLD=NO_TRADE    # Paper default

# Trading config
PAPER_MODE=true              # Paper trading mode
SPREAD_THRESHOLD=15.0        # Max spread %
EDGE_THRESHOLD=0.05          # Min edge (5%)
MAX_DAILY_TRADES=5           # Daily trade limit
TARGET_POSITION_DOLLARS=500  # Position size target
```

## Model Files

Required in `models/` directory:
- `earnings_q50.txt`, `earnings_q75.txt`, `earnings_q90.txt`, `earnings_q95.txt` - LightGBM models
- `feature_config.json` - Feature configuration
- `news_pca.joblib` - PCA model for news embeddings

## Database

SQLite database at `data/earnings_trades.db`:

| Table | Purpose |
|-------|---------|
| `trades` | Trade lifecycle (entry, fills, exit, P&L) |
| `non_trades` | Rejected candidates with reasons |
| `llm_checks` | LLM sanity check results |
| `price_snapshots` | Intraday position price tracking |

## Pipeline Flow

```
1. fetch_upcoming_earnings()     # Nasdaq API
         |
         v
2. screen_all_candidates()       # IBKR option chains, liquidity gates
         |
         v
3. predictor.predict()           # ML edge calculation
         |
         v
4. check_with_llm()              # LLM + web search validation
         |
         v
5. executor.place_straddle()     # IBKR combo order
```

## Key Thresholds

| Metric | Default | Description |
|--------|---------|-------------|
| Spread % | ≤ 15% | Combined straddle spread |
| Edge | ≥ 5% | predicted_q75 - implied_move |
| Open Interest | ≥ 50 | ATM strike (currently disabled) |
