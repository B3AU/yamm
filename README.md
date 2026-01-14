# YAMM - Earnings Volatility Options Strategy

Exploit volatility mispricing around earnings in US equities using options straddles.

**Core edge:** Vol + tails, not direction.

## Strategy Overview

- **Events:** Earnings announcements only
- **Instruments:** Listed equity options (straddles/strangles)
- **Holding period:** T-1 close → T+1 close (overnight hold through earnings)
- **Position sizing:** Fixed risk per trade, 0.25% NAV max loss
- **Risk posture:** Long volatility, defined risk

See `CLAUDE.md` for the full V1 plan.

---

## Quick Start

### Prerequisites

1. **IB Gateway** running (paper trading on port 4002)
2. **IBKR Market Data Subscription** (~$15/mo for stocks + options)
3. **FMP API Key** for earnings calendar (set in `.env`)

### Environment Setup

```bash
# Install dependencies
pip install ib_insync apscheduler pandas numpy python-dotenv pytz

# Create .env file
cat > .env << EOF
FMP_API_KEY=your_key_here
IB_HOST=127.0.0.1
IB_PORT=4002
DRY_RUN=false
EOF
```

### Running the Trading Daemon

**Option 1: Manual (for testing)**
```bash
# Dry run - logs but doesn't place orders
DRY_RUN=true python3 -m trading.earnings.daemon

# Paper trading - places real orders
python3 -m trading.earnings.daemon
```

### Test Screening Pipeline

Run the ML and LLM screening pipeline outside of the daemon:

```bash
# Full pipeline with IBKR connection
python3 -m trading.earnings.test_screening

# Test specific ticker(s)
python3 -m trading.earnings.test_screening --ticker AAPL --earnings-date 2026-01-30 --timing AMC

# Skip IBKR screening (ML + LLM only)
python3 -m trading.earnings.test_screening --no-ibkr --ticker AAPL

# Skip LLM sanity check
python3 -m trading.earnings.test_screening --no-llm

# Custom thresholds
python3 -m trading.earnings.test_screening --spread-threshold 12 --edge-threshold 0.03
```

**Option 2: Systemd (for production)**
```bash
# Install service
sudo cp trading/earnings/yamm-trading.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable yamm-trading
sudo systemctl start yamm-trading

# Check status
sudo systemctl status yamm-trading
journalctl -u yamm-trading -f
```

### Schedule (all times ET, Mon-Fri)

| Time | Task |
|------|------|
| 09:25 | Connect to IB Gateway |
| 09:30-16:00 | Position snapshots (every 5 min) |
| 14:00 | Exit previous day's positions |
| 14:15 | Screen + place orders |
| 14:25, 14:35, 14:45, 14:55 | Price improvement loop |
| 15:55 | Final fill check |
| 15:58 | Cancel unfilled entries + force exit remaining |
| 16:05 | Disconnect |
| 16:30 | Counterfactual backfill |

---

## Configuration

Environment variables (set in `.env` or export):

| Variable | Default | Description |
|----------|---------|-------------|
| `FMP_API_KEY` | (required) | FMP API key for earnings calendar |
| `IB_HOST` | 127.0.0.1 | IB Gateway host |
| `IB_PORT` | 4002 | IB Gateway port (4002=paper, 7497=TWS live, 4001=gateway live) |
| `IB_CLIENT_ID` | 1 | IB client ID |
| `SPREAD_THRESHOLD` | 15.0 | Max spread % to trade |
| `MAX_CONTRACTS` | 1 | Contracts per leg |
| `MAX_DAILY_TRADES` | 5 | Max trades per day |
| `MAX_CANDIDATES_TO_SCREEN` | 50 | Candidates to screen |
| `LIMIT_AGGRESSION` | 0.3 | How far above mid to place limit |
| `DRY_RUN` | false | Log but don't place orders |

### Paper Trading Settings (Recommended)

For faster data collection during paper trading:
```bash
MAX_DAILY_TRADES=10
SPREAD_THRESHOLD=20
MAX_CANDIDATES_TO_SCREEN=100
```

---

## Project Structure

```
yamm/
├── CLAUDE.md                 # V1 strategy plan
├── README.md                 # This file
├── .env                      # Configuration (not in git)
│
├── trading/
│   └── earnings/
│       ├── __init__.py
│       ├── daemon.py         # Main trading daemon
│       ├── screener.py       # Earnings screening
│       ├── executor.py       # Order execution
│       ├── logging.py        # Trade/non-trade logging
│       ├── ib_options.py     # IBKR options client
│       ├── run_daemon.sh     # Manual run script
│       └── yamm-trading.service  # Systemd service
│
├── notebooks/
│   ├── 0.1 options_data_validation.ipynb
│   ├── 0.2 historical_earnings_moves.ipynb
│   ├── 0.3 ibkr_connection_test.ipynb
│   └── 0.4 phase0_execution.ipynb
│
├── data/
│   ├── earnings_trades.db    # SQLite trade log
│   └── earnings/             # Historical data
│
└── logs/
    └── daemon.log            # Daemon logs
```

---

## Phase 0: Execution Validation

Current phase focuses on validating execution before building ML models.

**Target metrics:**
- [ ] 50+ paper trades logged
- [ ] System runs reliably for 1 week
- [ ] Strategy P&L positive (before slippage)
- [ ] Identify optimal spread threshold

**After Phase 0:**
- Analyze paper trading results
- Switch to live with 1 contract, strict 15% spread gate
- Collect 30+ real fills for execution model
- Then proceed to ML model development

---

## Monitoring

### View Logs
```bash
tail -f logs/daemon.log
```

### Check Trade Database
```python
from trading.earnings import TradeLogger

logger = TradeLogger('data/earnings_trades.db')
stats = logger.get_summary_stats()
print(f"Trades: {stats['total_trades']}, P&L: ${stats['total_pnl']:.2f}")
```

### Manual Screening (via notebook)
Open `notebooks/0.4 phase0_execution.ipynb` for interactive screening and order placement.

---

## Troubleshooting

**"Connection refused" on IB Gateway**
- Check IB Gateway is running
- Verify port (4002 for paper, 4001 for live)
- Enable API connections in IB Gateway settings

**"Market data requires subscription"**
- Subscribe to US Securities Snapshot Bundle (~$10/mo)
- Subscribe to US Equity and Options Add-On (~$4.50/mo)

**No candidates passing screening**
- Check if market is open (screening needs live quotes)
- Loosen `SPREAD_THRESHOLD` temporarily
- Verify FMP API key is valid

---

## Archive

The previous news-ranking strategy was concluded as non-viable. Files are in:
- `archive/news_ranking/` - notebooks
- `trading/news_ranking/` - code
- `PROJECT_REPORT_NEWS_RANKING.md` - analysis
