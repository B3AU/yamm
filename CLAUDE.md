# Claude rules
- python command is "python3"
- don't mention claude in git commits


---

# V1 PLAN — Earnings-Driven Options Volatility Strategy

## Executive Summary

Exploit volatility mispricing around earnings in semi-illiquid US equities. Use ML to predict the distribution of post-earnings moves, then trade defined-risk option structures when implied volatility misprices that distribution.

**Core edge:** Vol + tails, not direction.

**V1 in one sentence:** Earnings-only, T-1 afternoon entry, T+0 afternoon exit (~24h hold), strict option liquidity gates, straddles/strangles only, fixed position sizing, mechanical risk controls, execution-first validation.

---

## Implementation Status

### What's Built

#### Trading Daemon (`trading/earnings/daemon.py`)
- Async APScheduler-based daemon with asyncio architecture
- **Schedule (all times ET):**
  - **09:25** - Connect to IB Gateway, load positions to exit
  - **09:30-16:00** - Position snapshots every 5 minutes (intraday P&L tracking)
  - **14:00** - Exit positions from previous day (free up capital first)
  - **14:00-16:00** - Monitor fills every minute
  - **14:15** - Screen upcoming earnings + place new orders
  - **14:25, 14:35, 14:45, 14:55** - Price improvement loop (aggression: 0.4→0.5→0.6→0.7)
  - **15:55** - Final fill check, reprice unfilled exits to bid
  - **15:58** - Cancel unfilled entry orders, force exit remaining positions with market orders
  - **16:05** - Disconnect from IB Gateway
  - **16:30** - Backfill counterfactuals for non-trades
- Order recovery on restart (persists IBKR order IDs to DB)
- Exit order recovery for orphaned positions
- **Position reconciliation** - Detects DB/IBKR state mismatches on startup
- **Force exit with market orders** - Handles positions stuck without quotes
- Startup catch-up: runs screening immediately if started between 14:00-21:00 ET
- Graceful shutdown handling

#### Screener (`trading/earnings/screener.py`)
- Fetches earnings calendar from **Nasdaq API** (FMP was unreliable for calendar)
- Gets live option chains from IBKR
- Applies liquidity gates (mostly spread %)
  - *Open interest check currently disabled/commented out*
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
- **Market order fallback** - `close_position_market()` for force exits without quotes

#### Trade Logging (`trading/earnings/logging.py`)
- SQLite database (`data/earnings_trades.db`)
- Full trade lifecycle: entry quotes, fills, slippage, exit, P&L
- Non-trade logging (rejections with reasons)
- Execution metrics (fill rate, slippage stats)
- IBKR order ID persistence for recovery
- **Database tables:**
  - `trades` - Main trade log with fill/exit/P&L data
  - `non_trades` - Rejected candidates with reasons
  - `price_snapshots` - Intraday position tracking (every 5 min)
  - `order_events` - IB order status changes and fills
  - `llm_checks` - LLM sanity check decisions and reasoning
  - `earnings_calendar` - Multi-source calendar (Nasdaq + FMP)
- **New fields:** `decision_latency_ms`, `fill_latency_seconds`, `spread_at_fill`, `markout_1min/5min/30min`

#### Dashboard (`trading/earnings/dashboard.py`)
- CLI dashboard with ANSI colors
- Shows open positions, completed trades, summary stats
- **Position age indicator** - Shows "4.8d", "12h" for each position
- **BMO vs AMC breakdown** - Performance split by earnings timing
- **Upcoming candidates preview** - Shows Today AMC / Tomorrow BMO symbols
- **Recent warnings/errors from daemon log** (fallback warnings, connection errors)
- Live IBKR prices with `--live` flag
- Watch mode with auto-refresh (`--watch`)
- **Interactive commands:** c=close position, l=llm details, r=refresh, q=quit
- Manual position closing (for partial fills or emergencies)

#### Live News (`trading/earnings/live_news.py`)
- Fetches news from FMP API
- Computes embeddings using sentence-transformers (BAAI/bge-base-en-v1.5)
- Anonymizes company names for cleaner embeddings
- PCA projection to match training features
- Returns raw headlines for LLM sanity check

#### LLM Sanity Check (`trading/earnings/llm_sanity_check.py`)
- Pre-trade validation using LLM + web search
- **Tavily API** for web search (earnings releases, halts, offerings, acquisitions)
- **OpenRouter API** for LLM reasoning (Claude 3.5 Sonnet default)
- Returns decision: `PASS`, `WARN`, or `NO_TRADE`
- Configurable threshold via environment variable
- All results logged to `llm_checks` table for analysis

#### Test Screening (`trading/earnings/test_screening.py`)
- Standalone script to test ML + LLM pipeline outside daemon
- Supports specific tickers with manual earnings dates
- Can skip IBKR or LLM for targeted testing
- Useful for debugging and pre-market validation

#### IB Options Client (`trading/earnings/ib_options.py`)
- Wrapper around `ib_insync` for option chains and market data
- Handles connection maintenance and error recovery

### Changes from Original Plan

1. **Combined screen + place orders** - Originally separate (15:00 screen, 15:30 place). Now combined at **14:00 ET** for maximum fill time before close.

2. **Exit time** - Exit moved to **14:45 ET** to avoid position conflicts with new orders.

3. **Data Sources** - Earnings calendar uses **Nasdaq API** instead of FMP (better accuracy).

4. **Live-first data** - Originally planned to use static parquet files. Now fetches live from FMP API with parquet as fallback only.

5. **Order recovery** - Added IBKR order ID persistence and recovery on daemon restart.

6. **Combo orders** - Using IBKR BAG orders instead of separate call/put orders. Eliminates orphan leg risk entirely.

7. **Interactive dashboard** - Added keyboard commands for manual position management.

8. **Exit order monitoring** - Track exit fills and calculate P&L automatically.

9. **Counterfactual logging** - After market close, backfill realized moves for non-traded candidates to measure what we missed.

10. **Price improvement loop** - 4-step progressive repricing (14:25-14:55) with increasing aggression (0.4→0.7) to maximize fills.

11. **Intraday position snapshots** - Every 5 minutes during market hours for P&L tracking and exit timing analysis.

12. **Post-fill markouts** - Automatic 1min, 5min, 30min price recordings after each fill for slippage analysis.

13. **Exit-first scheduling** - Exits now run at 14:00, before screening (14:15), to free up capital first.

14. **Startup catch-up** - If daemon starts during 14:00-21:00 ET, immediately runs screening to catch missed window.

15. **Force exit at close** - 15:58 job forces remaining exits to market orders to avoid overnight positions.

### Current Limitations / TODO

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
- [x] Dashboard UX improved (Portugal time, visual timeline, compact view)
- [x] Daemon duplicate trade prevention (restart safety)
- [x] Proportional position sizing (target dollar entry amount)
- [x] News count tracking (no stale parquet fallback)
- [x] Realized move tracking (spot_at_exit, realized_move_pct)
- [x] LLM sanity check before order placement (Tavily + OpenRouter)
- [x] Intraday price snapshots for exit timing analysis
- [x] Orphan leg retry logic for failed exits
- [x] Price improvement loop (progressive repricing 14:25-14:55)
- [x] Post-fill markouts (1min, 5min, 30min price recordings)
- [x] Exit order recovery on daemon restart
- [x] Startup catch-up screening (missed window detection)
- [x] Force exit to market at 15:58 (avoid overnight positions)
- [x] Async daemon architecture (asyncio + AsyncIOScheduler)
- [x] Order event logging (IB status changes)
- [x] Decision/fill latency tracking
- [x] Dry run mode for testing
- [x] Market order fallback for orphaned exit positions
- [x] Position reconciliation on daemon startup
- [x] Dashboard: position age indicator
- [x] Dashboard: BMO vs AMC timing breakdown
- [x] Dashboard: upcoming candidates preview

---

### Analysis Queries (When More Data Available)

#### Edge Hit Rate
```sql
-- Did actual moves exceed implied move?
SELECT
    COUNT(*) as total_trades,
    SUM(CASE WHEN realized_move_pct > implied_move THEN 1 ELSE 0 END) as edge_hits,
    ROUND(100.0 * SUM(CASE WHEN realized_move_pct > implied_move THEN 1 ELSE 0 END) / COUNT(*), 1) as hit_rate_pct
FROM trades WHERE status = 'exited' AND realized_move_pct IS NOT NULL;
```

#### Implied vs Realized Analysis
```sql
-- Compare what we paid for vs what happened
SELECT
    ticker,
    ROUND(implied_move * 100, 1) as implied_pct,
    ROUND(realized_move_pct * 100, 1) as actual_pct,
    ROUND((realized_move_pct - implied_move) * 100, 1) as edge_realized,
    ROUND(exit_pnl, 0) as pnl,
    news_count
FROM trades
WHERE status = 'exited'
ORDER BY earnings_date DESC;
```

#### Edge vs P&L Correlation
```sql
-- Does positive edge correlate with positive P&L?
SELECT
    CASE WHEN realized_move_pct > implied_move THEN 'Edge Hit' ELSE 'Edge Miss' END as category,
    COUNT(*) as trades,
    ROUND(AVG(exit_pnl), 0) as avg_pnl,
    SUM(exit_pnl) as total_pnl
FROM trades WHERE status = 'exited' AND realized_move_pct IS NOT NULL
GROUP BY category;
```

#### News Impact
```sql
-- Does having news improve predictions?
SELECT
    CASE WHEN news_count > 0 THEN 'With News' ELSE 'No News' END as category,
    COUNT(*) as trades,
    ROUND(AVG(exit_pnl_pct * 100), 1) as avg_return_pct,
    ROUND(AVG(realized_move_pct - implied_move) * 100, 1) as avg_edge_realized
FROM trades WHERE status = 'exited'
GROUP BY category;
```

#### Position Sizing Impact
```sql
-- Did sizing help normalize risk?
SELECT
    contracts,
    COUNT(*) as trades,
    ROUND(AVG(premium_paid), 0) as avg_entry,
    ROUND(AVG(exit_pnl), 0) as avg_pnl
FROM trades WHERE status = 'exited'
GROUP BY contracts ORDER BY contracts;
```

### File Structure

```
trading/earnings/
├── daemon.py            # Main scheduler daemon
├── screener.py          # Earnings + options screening
├── ml_predictor.py      # ML model inference
├── executor.py          # Order placement + management
├── logging.py           # Trade/non-trade logging
├── counterfactual.py    # Counterfactual backfill for non-trades
├── dashboard.py         # CLI dashboard
├── live_news.py         # Live news embeddings
├── llm_sanity_check.py  # LLM + web search pre-trade validation
├── test_screening.py    # Standalone pipeline testing
├── ib_options.py        # IBKR options client wrapper
└── README.md            # Component documentation

models/
├── earnings_q50.txt   # LightGBM model files
├── earnings_q75.txt
├── earnings_q90.txt
├── earnings_q95.txt
├── feature_config.json
└── news_pca.joblib    # PCA model for news (768-dim → 10)

data/
├── earnings_trades.db # SQLite trade log
├── prices.pqt         # Historical prices (fallback)
├── earnings/
│   ├── historical_earnings_moves.parquet  # With timing + corrected moves
│   └── ml_features.parquet                # Training dataset (69k rows)
└── news_ranking/
    ├── news_embeddings.pqt   # 1.7M embeddings (768-dim)
    └── all_the_news_anon.pqt # Anonymized news articles

notebooks/
├── 0.2 historical_earnings_moves.ipynb   # Fetch earnings + compute moves
├── 0.2c_infer_earnings_timing.ipynb      # BMO/AMC timing inference fix
├── 1.0 feature_engineering.ipynb         # Build ML features (55 total)
├── 1.1 model_training.ipynb              # Train quantile models
├── 1.2 calibration_analysis.ipynb        # Model calibration study
└── 3.0_edge_analysis.ipynb               # Edge/P&L analysis
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

# Test screening pipeline (without daemon)
python -m trading.earnings.test_screening

# Test specific ticker
python -m trading.earnings.test_screening --ticker AAPL --earnings-date 2026-01-30 --timing AMC

# Test ML only (no IBKR)
python -m trading.earnings.test_screening --no-ibkr --ticker AAPL
```

### Environment Variables

```bash
# Required
FMP_API_KEY=xxx              # Financial Modeling Prep API

# IBKR
IB_HOST=127.0.0.1            # Gateway host
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
DRY_RUN=false                # Dry run (no actual orders)
SPREAD_THRESHOLD=15.0        # Max spread %
EDGE_THRESHOLD=0.05          # Min edge (5%)
MAX_DAILY_TRADES=5           # Daily trade limit
TARGET_ENTRY_AMOUNT=2000     # Target $ per trade
MIN_CONTRACTS=1              # Minimum contract size
MAX_CONTRACTS=5              # Maximum contract size (safety cap)

# Paper mode overrides (optional, use PAPER_ prefix)
PAPER_SPREAD_THRESHOLD=20.0
PAPER_EDGE_THRESHOLD=0.03
PAPER_MAX_DAILY_TRADES=10
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

- **Period:** 2021-02-16 to 2025-12-18 (~4 years)
- **Samples:** 69,783 earnings events, 4,228 symbols (after filtering)
- **Walk-forward validation:** 5 time-based folds, expanding window

### BMO/AMC Timing Alignment Fix (Critical)

**Problem discovered:** Historical Nasdaq data has ~0% BMO/AMC timing coverage (Nasdaq only provides timing for *upcoming* earnings, not historical). Without timing, ~50% of training data had misaligned price moves.

**The bug:** For AMC (After Market Close) earnings, the original code computed:
- `Close_T-1 → Close_T` which captures the move **before** earnings (wrong)

**Correct calculation by timing:**
- **BMO:** Reaction is `Close_T-1 → Open_T → Close_T` (original was correct)
- **AMC:** Reaction is `Close_T → Open_T+1 → Close_T+1` (needed correction)

**Solution (notebook `0.2c_infer_earnings_timing.ipynb`):**
1. Infer timing from overnight gap magnitudes:
   - `gap_T` = |Close_T-1 → Open_T| (gap on earnings day)
   - `gap_T+1` = |Close_T → Open_T+1| (gap on day after)
   - `gap_ratio = gap_T / gap_T+1`
2. Classification thresholds:
   - `gap_ratio > 2.0` → **BMO** (large gap on T)
   - `gap_ratio < 0.5` → **AMC** (large gap on T+1)
   - Otherwise → unknown
3. Coverage: ~60% of historical data now has inferred timing

**Data changes:**
- Added `timing` column to `historical_earnings_moves.parquet` (BMO/AMC/unknown)
- Added `corrected_gap`, `corrected_full` columns with timing-aligned moves
- Feature engineering (`1.0 feature_engineering.ipynb`) uses `corrected_full_abs` as target
- Model retrained on timing-corrected data (`1.1 model_training.ipynb`)

**Why "overnight_move" still works:** The `Close_T-1 → Close_T+1` move captures the full window for both BMO and AMC, making it robust to timing uncertainty. However, the gap/full moves needed correction for proper feature engineering.

### Known Backtest/Model Issues (Pre-Live Checklist)

These issues should be addressed or validated before deploying with significant capital:

#### 1. Backtest P&L Likely Overstated (30-50%)
**Location:** `notebooks/1.2 calibration_analysis.ipynb`

The strategy simulation assumes straddles cost exactly the historical move (`implied_move = hist_move_mean * 1.0`). In reality, options market prices in a volatility premium - implied moves are typically **1.3-1.5x realized moves**.

**Impact:** The reported +2.39% mean P&L at 7% edge threshold is likely break-even or slightly negative in reality.

**Fix needed:** Use realistic implied move multiplier (1.3-1.5x) and validate with Phase 0 paper trading data.

#### 2. No Transaction Costs in Backtest
**Location:** `notebooks/1.2 calibration_analysis.ipynb`

Simulation includes spread cost but NOT:
- Commissions ($1-2 per leg x 2 = $2-4 per straddle)
- Assumes 100% fill rate (unrealistic for semi-illiquid names)

**Impact:** True P&L likely 1-2% lower than backtest shows.

**Fix needed:** Add realistic commission model ($1.30/contract typical for IBKR) to simulation.

#### 3. Edge Threshold Test-Set Overfitting
**Location:** `notebooks/1.2 calibration_analysis.ipynb`

The 7% edge threshold was optimized on the same out-of-sample data used for model evaluation.

**Impact:** Reported Sharpe ~3.0 is overstated; likely 2.0-2.5 in practice.

**Fix needed:** Hold out final 20% of test data for threshold selection.

#### 4. Gap Ratio Threshold Data Snooping
**Location:** `notebooks/0.2c_infer_earnings_timing.ipynb`

BMO/AMC timing thresholds (gap_ratio > 2.0 for BMO, < 0.5 for AMC) were tuned on the same data used for training, with no held-out validation.

**Impact:** Timing labels may be partially arbitrary rather than representing true BMO/AMC signal.

**Fix needed:** Hold out last 10% of data for threshold tuning, or validate against external timing source.

#### 5. News Feature Defaults to Zero
**Location:** `trading/earnings/ml_predictor.py:481-483`

When news is unavailable, all 10 PCA components default to exactly 0.0 instead of training median.

**Impact:** Creates discontinuity in feature space; model may systematically bias predictions for low-news stocks.

**Fix needed:** Store training-set medians per `news_pca_*` component and use those as defaults.

#### Recommended Validation Before Live

1. Re-run backtest with 1.3x implied move multiplier and commissions
2. Compare paper trading P&L to revised backtest predictions
3. Verify calibration: q75 exceedance should be ~25% (currently untested live)
4. Track actual fill rates vs 100% assumption

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
- **Holding period:** T-1 afternoon (14:15 ET) → T+0 afternoon (14:00 ET), ~24h hold through earnings
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