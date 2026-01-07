# Claude rules
- python command is "python3"
- don't mention claude in git commits


---

# V1 PLAN — Earnings-Driven Options Volatility Strategy

## Executive Summary

Exploit volatility mispricing around earnings in semi-illiquid US equities. Use ML to predict the distribution of post-earnings moves, then trade defined-risk option structures when implied volatility misprices that distribution.

**Core edge:** Vol + tails, not direction.

**V1 in one sentence:** Earnings-only, T-1 close entry, T+1 close exit, strict option liquidity gates, straddles/strangles only, fixed position sizing, mechanical risk controls, execution-first validation.

---

## Implementation Status

### What's Built

#### Trading Daemon (`trading/earnings/daemon.py`)
- APScheduler-based daemon running on configurable schedule
- **Schedule (all times ET):**
  - 09:30 - Morning IBKR connection check
  - 14:45 - Exit existing positions
  - 15:00 - Screen candidates + place new orders (combined for more fill time)
  - 15:50 - Check fills, cancel unfilled orders near close
- Order recovery on restart (persists IBKR order IDs to DB)
- Graceful shutdown handling

#### Screener (`trading/earnings/screener.py`)
- Fetches earnings calendar from FMP API
- Gets live option chains from IBKR
- Applies liquidity gates (spread %, OI, etc.)
- Computes implied move from ATM straddle

#### ML Predictor (`trading/earnings/ml_predictor.py`)
- LightGBM quantile regression (q50, q75, q90, q95)
- **55 features** including:
  - Historical earnings moves (mean, std, max, trend, etc.)
  - Price/volatility features (realized vol, momentum, gaps)
  - Earnings surprises (beat rate, streak)
  - Fundamentals (16 metrics: P/E, margins, growth, etc.)
  - News embeddings (PCA-reduced to 10 components)
  - Timing features (day of week, earnings season)
- **Live-first data sourcing** with parquet fallback:
  | Data | Live Source | Fallback |
  |------|-------------|----------|
  | Historical earnings | FMP `/stable/earnings` + prices | `data/earnings/historical_earnings_moves.parquet` |
  | Prices | FMP `/stable/historical-price-eod/dividend-adjusted` | `data/prices.pqt` |
  | Fundamentals | FMP `/stable/key-metrics`, `/stable/ratios`, `/stable/financial-growth` | defaults to 0 |
  | Surprises | FMP `/stable/earnings` | defaults |
  | News | FMP `/stable/news/stock` + live embedding | `data/news_ranking/news_embeddings.pqt` |

#### Executor (`trading/earnings/executor.py`)
- **Combo (BAG) orders** - Both legs execute atomically, no orphan risk
- Places straddle as single combo order via IBKR
- Limit pricing: `mid + aggression * spread` (combined straddle price)
- Fill monitoring (simplified - combo fills or doesn't)
- Order cancellation for unfilled orders
- Order recovery after daemon restart

#### Trade Logging (`trading/earnings/logging.py`)
- SQLite database (`data/earnings_trades.db`)
- Full trade lifecycle: entry quotes, fills, slippage, exit, P&L
- Non-trade logging (rejections with reasons)
- Execution metrics (fill rate, slippage stats)
- IBKR order ID persistence for recovery

#### Dashboard (`trading/earnings/dashboard.py`)
- CLI dashboard with ANSI colors
- Shows open positions, completed trades, summary stats
- **Recent warnings/errors from daemon log** (fallback warnings, connection errors)
- Live IBKR prices with `--live` flag
- Watch mode with auto-refresh (`--watch`)
- **Interactive commands:** c=close position, r=refresh, q=quit
- Manual position closing (for partial fills or emergencies)

#### Live News (`trading/earnings/live_news.py`)
- Fetches news from FMP API
- Computes embeddings using sentence-transformers (BAAI/bge-base-en-v1.5)
- Anonymizes company names for cleaner embeddings
- PCA projection to match training features

### Changes from Original Plan

1. **Combined screen + place orders** - Originally separate (15:00 screen, 15:30 place). Now combined at 15:00 for more fill time before close.

2. **Exit before new orders** - Exit moved to 14:45 ET (before 15:00 new orders) to avoid position conflicts.

3. **Live-first data** - Originally planned to use static parquet files. Now fetches live from FMP API with parquet as fallback only.

4. **Order recovery** - Added IBKR order ID persistence and recovery on daemon restart.

5. **Combo orders** - Using IBKR BAG orders instead of separate call/put orders. Eliminates orphan leg risk entirely.

6. **Interactive dashboard** - Added keyboard commands for manual position management.

7. **Exit order monitoring** - Track exit fills and calculate P&L automatically.

8. **Counterfactual logging** - After market close, backfill realized moves for non-traded candidates to measure what we missed.

### Current Limitations / TODO

#### High Priority
- [ ] No position sizing logic (fixed 1 contract currently)

#### Medium Priority
- [ ] Strangle structure not implemented (straddles only)
- [ ] No IV rank/percentile features
- [ ] Model retraining pipeline not automated

#### Low Priority
- [ ] Early exit logic (profit taking, loss cutting)

#### Completed
- [x] Combo (BAG) orders - eliminates orphan leg risk
- [x] Exit order monitoring with P&L calculation
- [x] Max daily trades limit enforcement
- [x] Dashboard warnings/errors display
- [x] Counterfactual logging for non-trades

### File Structure

```
trading/earnings/
├── daemon.py          # Main scheduler daemon
├── screener.py        # Earnings + options screening
├── ml_predictor.py    # ML model inference
├── executor.py        # Order placement + management
├── logging.py         # Trade/non-trade logging
├── counterfactual.py  # Counterfactual backfill for non-trades
├── dashboard.py       # CLI dashboard
└── live_news.py       # Live news embeddings

models/
├── earnings_q50.txt   # LightGBM model files
├── earnings_q75.txt
├── earnings_q90.txt
├── earnings_q95.txt
├── feature_config.json
└── news_pca.joblib    # PCA model for news

data/
├── earnings_trades.db # SQLite trade log
├── prices.pqt         # Historical prices (fallback)
├── earnings/
│   └── historical_earnings_moves.parquet
└── news_ranking/
    ├── news_embeddings.pqt
    └── all_the_news_anon.pqt
```

### Running the System

```bash
# Start daemon
python -m trading.earnings.daemon

# Dashboard (one-shot)
python -m trading.earnings.dashboard

# Dashboard (watch mode with live prices)
python -m trading.earnings.dashboard --watch --live

# Dashboard (show completed trades)
python -m trading.earnings.dashboard --all
```

### Environment Variables

```bash
FMP_API_KEY=xxx          # Financial Modeling Prep API
IB_CLIENT_ID=1           # IBKR client ID (default: 1)
```

---

## Research Notes

### Historical Options Data Sources (Jan 2025)

| Source | Price | Coverage | Best For |
|--------|-------|----------|----------|
| **ORATS** | $99/mo | 2007-present | Best fit - EOD bid/ask, greeks, IV via API |
| **Polygon/Massive** | $29-199/mo | 2-20 years | Stocks cheaper, options needs higher tier |
| **CBOE DataShop** | Quote-based | 2012-present | Official OPRA data, pay-per-symbol |
| **Theta Data** | ~$30-100/mo | Unknown | Popular retail option |

**Current approach:** Synthetic backtest using stock prices + estimated IV (free). ORATS at $99/mo recommended if real options data needed.

### Training Data

- **Period:** 2024-03-30 to 2025-12-18 (~21 months)
- **Samples:** 3,350 earnings events, 1,419 symbols
- **Out-of-sample for backtest:** Re-train on earlier period to get true OOS

---

## Phase 0: Execution Validation (FIRST PRIORITY)

**Status: IN PROGRESS**

Currently collecting execution data. Need 30+ fills to build fill model.

### Objective
Validate that you can actually execute trades profitably before building models.

### Metrics to Collect
For every order:
- Quoted bid/ask at decision time
- Your limit price
- Time to fill (or non-fill)
- Partial fill behavior
- Final fill price
- Post-fill markout (price 1min, 5min, 30min after fill)

### Outputs
- Fill probability as function of (spread, OI, time-of-day, order size)
- Slippage model: realized fill vs mid, vs your limit
- Minimum viable liquidity thresholds (empirical, not theoretical)

### Gate to Proceed
Do NOT proceed to Phase 1 until:
- Fill probability model exists and is validated
- Slippage estimates are stable
- You understand which liquidity buckets are tradeable

**If fills are consistently worse than expected, nothing else matters.**

---

## Scope (V1 Constraints)

### In Scope
- **Events:** Earnings announcements only
- **Markets:** US equities
- **Instruments:** Listed equity options
- **Holding period:** T-1 close → T+1 close (single overnight hold through earnings)
- **Structures:** Straddles and strangles only
- **Risk posture:** Long volatility, defined risk
- **Position sizing:** Fixed risk per trade, binary trade/no-trade decision

### Explicitly Out of Scope (V1)
- Directional bets / debit spreads (requires calibrated directional model)
- Short volatility / selling premium
- Naked options / undefined risk
- Intraday delta hedging
- News-only events (non-earnings)
- Variable position sizing / Kelly criterion
- Multiple entry points (only T-1 close)
- Complex Greeks management
- IV surface arbitrage

These are V2+ features.

---

## Core Hypothesis

For small/semi-illiquid names:
1. Options have wide spreads and defensive pricing
2. Market makers use coarse heuristics for earnings risk
3. ML can forecast earnings move distributions better than implied move alone
4. Mispricing is largest in **tails**, not means
5. Wide spreads are the cost of the edge, not a reason to avoid

The edge exists precisely because execution is hard. If it were easy, it would be arbitraged.

---

## Universe Construction

### Stock-Level Filters
Daily filter for stocks with earnings in next 1-10 trading days:
- US-listed common stock (no ADRs initially)
- Share price > $5 (avoid penny stock dynamics)
- Average daily dollar volume > $10M (enough liquidity for underlying)
- Not on manual blacklist (meme stocks, frequent halts, binary events)

### Option-Level Liquidity Gates (HARD REQUIREMENTS)

These are strict. Better to have 10 tradeable opportunities than 50 paper opportunities.

| Metric | Threshold | Notes |
|--------|-----------|-------|
| ATM spread (% of mid) | ≤ 12-15% | Primary filter |
| ATM spread (absolute) | ≤ $0.20 | Catches cheap options where % is misleading |
| Open interest (ATM) | ≥ 50 | Start here, tighten to 100 if fills degrade |
| Expiration availability | Weekly required | Monthly-only = too few opportunities |
| Strikes available | ≥ 3 near ATM | Need flexibility for structure selection |

### Expiration Selection
- Must be first expiration **after** earnings date
- Verify programmatically that earnings falls within the expiration window
- If earnings date changes and no longer spans, auto-close or re-evaluate

### Universe Size Target
- 10-50 tradeable candidates per day during earnings season
- Accept that many days will have 0-5 candidates
- Quality over quantity

---

## ML Model (V1)

### Objective
Predict the **distribution of post-earnings absolute returns**, focusing on tail probabilities.

### Target Variable
Primary: `|Close_T → Close_T+1|` (matches your exit timing)

### Model Outputs
Quantiles of |return|:
- **q50** (median move)
- **q75** (upper quartile)
- **q90** (tail threshold)
- **q95** (extreme tail)

The key comparison: `predicted_q75` vs `implied_move`

### Features (55 total)

**Historical Earnings (10 features)**
- hist_move_mean, hist_move_median, hist_move_std
- hist_move_max, hist_move_min, hist_move_cv
- recent_move_mean, move_trend
- gap_continuation_ratio, n_past_earnings

**Price/Volatility (10 features)**
- rvol_5d, rvol_10d, rvol_20d
- ret_5d, ret_10d, ret_20d
- dist_from_high_20d, dist_from_low_20d
- gap_frequency, volume_ratio

**Earnings Surprises (3 features)**
- surprise_pct_mean, beat_rate, surprise_streak

**Timing (5 features)**
- day_of_week, month, quarter
- is_earnings_season, timing_encoded

**Fundamentals (16 features)**
- evToEBITDA, freeCashFlowYield, earningsYield
- returnOnEquity, returnOnAssets, currentRatio
- priceToEarningsRatio, priceToBookRatio, priceToSalesRatio
- grossProfitMargin, operatingProfitMargin, netProfitMargin
- debtToEquityRatio, revenueGrowth, netIncomeGrowth, epsgrowth

**News (11 features)**
- pre_earnings_news_count
- news_pca_0 through news_pca_9

---

## Edge Calculation

For each candidate:

```
edge_q75 = predicted_q75 - implied_move
edge_q90 = predicted_q90 - implied_move
```

### Trade Decision (Binary)

**Trade if:**
```
predicted_q75 > implied_move + cost_buffer
```

Where `cost_buffer` includes:
- Half the bid-ask spread (your execution cost)
- Slippage estimate (from Phase 0)
- Model uncertainty margin (wider early, tighter as calibration improves)

---

## Risk Management

### Position-Level Limits
- Max loss per position: 0.25-1% NAV (phase-dependent)
- All positions defined risk (max loss known at entry)
- No averaging down

### Portfolio-Level Limits
| Constraint | Limit |
|------------|-------|
| Max positions per day | 5 (initially) |
| Max positions per sector per day | 2 |
| Max total earnings exposure | 3% NAV aggregate max loss |
| Max correlated positions | 3 (same sector earnings same week) |

### Mechanical Kill Switches (TODO)

These should trigger automatically, not discretionally.

**1. Calibration Drift Monitor** - Not implemented
**2. Drawdown Throttle** - Not implemented
**3. Execution Degradation** - Not implemented

---

## Success Criteria

### Phase 0 Success (Execution Validation)
- [ ] 30+ fills logged with full execution details
- [ ] Fill model: can predict fill probability within 10%
- [ ] Slippage model: realized within 1% of predicted
- [ ] Identified minimum viable liquidity thresholds

### Phase 1 Success (Model Validation)
- [ ] 50+ live trades with model predictions
- [ ] Calibration: predicted q90 exceedance within 3% of 10%
- [ ] Positive P&L after costs (any positive)
- [ ] No kill switches triggered

### V1 Success (Ready to Scale)
- [ ] 200+ trades
- [ ] Stable calibration across 2+ earnings seasons
- [ ] Sharpe > 0.5 after all costs
- [ ] Slippage model validated
- [ ] Drawdown < 10% max
- [ ] Clear edge attribution (know why you're making money)
