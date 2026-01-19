# Research Notes

This document contains detailed research topics and investigations for future improvements to the earnings volatility strategy. For the main project documentation, see [CLAUDE.md](CLAUDE.md).

---

## Future Research: Optimal Exit Timing with ORATS 1-Minute Data

**Priority:** Medium (V2) — Waiting for more live data to validate pattern

**Status:** Initial investigation complete (Jan 2026). Need 20+ more trades to confirm.

### Problem Statement: Stock Move ≠ Straddle P&L

The current model predicts **stock price movement** (`|Close_T → Close_T+1|`), but we actually care about **straddle P&L**. These are fundamentally different:

```
Stock Move:  Can continue increasing throughout the day
Straddle P&L: Intrinsic value + Extrinsic value
              - Intrinsic grows with stock move
              - Extrinsic decays via theta (accelerates post vol-crush)
```

**Key insight:** After IV crush at open, theta decay often exceeds intrinsic gains from continued stock movement. This creates a **peak straddle value** that occurs well before market close.

### Live Data Evidence (Jan 2026, 5 BMO Trades)

Analysis of `price_snapshots` table shows clear pattern of P&L decay after mid-morning peak:

**Average P&L by Time of Day (Exit Day, BMO earnings):**

| Time ET | Minutes After Open | Avg P&L % | Notes |
|---------|-------------------|-----------|-------|
| 09:30 | 0 | +9.1% | Market open, post-gap |
| 10:00 | 30 | +29.8% | Ramping up |
| 10:30 | 60 | +28.4% | |
| 11:00 | 90 | +30.0% | |
| 11:30 | 120 | +34.7% | |
| **12:00** | **150** | **+46.2%** | **← PEAK** |
| 12:30 | 180 | +43.3% | Decay begins |
| 13:00 | 210 | +37.6% | |
| 13:30 | 240 | +32.2% | |
| 14:00 | 270 | +26.2% | ← Previous exit time |
| 14:30 | 300 | +25.0% | |
| 15:00 | 330 | +23.1% | |
| 15:30 | 360 | +21.9% | |

**Per-Trade Peak Analysis:**

| Ticker | Peak P&L | Peak Time | Final P&L | Left on Table |
|--------|----------|-----------|-----------|---------------|
| TSM | +85.4% | 160 min (12:10) | +36.0% | 49.4 ppts |
| BAC | +59.6% | 165 min (12:15) | +20.1% | 39.5 ppts |
| WFC | +20.3% | 165 min (12:15) | -4.2% | 24.4 ppts |
| C | +16.9% | 170 min (12:20) | -9.8% | 26.6 ppts |
| MS | +82.7% | 285 min (14:15) | +66.5% | 16.2 ppts |

**Aggregate Statistics:**
- **Average peak P&L:** 53.0%
- **Average final P&L:** 21.7%
- **Average left on table:** 31.2 percentage points
- **Average peak time:** 189 minutes (3.2 hours) after open = ~12:40 ET

**Implication:** Exiting at 12:00-12:30 ET instead of 14:00 ET could improve returns by ~20-30 percentage points on average.

### Root Cause Analysis

**Why historical data shows gap < full move, but straddle peaks early:**

```python
# Historical stock data (53,856 samples with timing):
Gap (overnight only):     5.35% average
Full (close-to-close):    6.65% average
Gap > Full (reversion):   39.5% of events
Gap < Full (continuation): 57.5% of events  # Stock continues moving!
```

**The paradox:** Stocks continue moving after the gap (6.65% > 5.35%), yet straddle P&L peaks early. Why?

**Answer: Theta decay post vol-crush**

1. **Pre-earnings:** High IV inflates option prices (extrinsic value)
2. **At open:** IV crushes immediately → extrinsic drops sharply
3. **Post-crush:** Theta decay accelerates on near-expiry options
4. **Through the day:**
   - Stock may move another 1-2% (intrinsic grows slowly)
   - But theta eats 3-5% of remaining extrinsic
   - Net effect: straddle value declines despite stock movement

**Mathematical intuition:**
```
Straddle_value = Intrinsic + Extrinsic
d(Straddle)/dt = d(Intrinsic)/dt + d(Extrinsic)/dt
              = (delta effect from stock move) + (theta decay)
              
Post vol-crush: theta decay >> delta gains from continued movement
```

### Current Model Mismatch

| Aspect | Current State | Optimal State |
|--------|---------------|---------------|
| **Training target** | `corrected_full_abs` (stock move at close) | Straddle P&L at optimal exit |
| **Exit timing** | 14:00 ET (now 12:00 ET via env var) | Dynamic based on P&L trajectory |
| **Optimization** | Predict when stock moves big | Predict when straddle is most valuable |

**This is NOT "model underestimating gains"** — the model correctly predicts stock moves. The issue is:
1. We're predicting the wrong thing (stock move vs option value)
2. We're exiting at the wrong time (close vs peak)

### ORATS 1-Minute Data: The Solution

**ORATS 1-Minute API** provides historical intraday option prices going back to August 2020.

**Relevant Endpoints:**

| Endpoint | Description | Coverage | Use Case |
|----------|-------------|----------|----------|
| `hist/one-minute/strikes/chain` | Full option chain at any minute | Aug 2020+ | Get ATM straddle prices |
| `hist/one-minute/strikes/option` | Specific strike by OPRA symbol | Jan 2022+ | Track specific contracts |
| `hist/one-minute/summaries` | IV surface summaries | Aug 2020+ | IV metrics at any time |

**API Details:**
- **Format:** CSV
- **Rate limit:** 1000 requests/minute
- **Max range:** 40 trading days per request
- **Auth:** Token-based ($99/month premium service)
- **Docs:** https://docs.orats.io/one-minute-api-guide/

**Example Request:**
```bash
# Get AAPL option chain on 2022-08-10 at 11:30am ET
curl -L "https://api.orats.io/datav2/hist/one-minute/strikes/chain?token=YOUR_TOKEN&ticker=AAPL&tradeDate=202208101130"
```

**Response includes:**
- `callBidPrice`, `callAskPrice`, `putBidPrice`, `putAskPrice`
- `callValue`, `putValue` (theoretical values)
- `smvVol` (smoothed implied volatility)
- `delta`, `gamma`, `theta`, `vega`
- `stockPrice`, `strike`, `expirDate`

### Research Implementation Plan

**Phase 1: Validate Pattern with Live Data (Current)**
- Continue collecting `price_snapshots` for all trades
- Wait for 20+ exited trades
- Confirm peak timing is consistently 2-4 hours post-open
- Monitor BMO vs AMC differences
- **Exit time already moved to 12:00 ET via `EXIT_TIME_ET` env var**

**Phase 2: Historical Backtest with ORATS ($99/month)**

Goal: Find optimal exit time across 5+ years of data

```python
# Pseudocode for ORATS backtest
for earnings_event in historical_earnings:  # ~50,000 events
    # Get straddle prices at multiple times on exit day
    timestamps = ['0930', '1000', '1030', '1100', '1130', 
                  '1200', '1230', '1300', '1330', '1400']
    
    for ts in timestamps:
        chain = orats.get_chain(ticker, exit_date, ts)
        atm_strike = find_atm(chain, spot_price)
        straddle_price = chain[atm_strike].call_mid + chain[atm_strike].put_mid
        record(earnings_event, ts, straddle_price)
    
    # Compute P&L curve and find peak
    entry_price = straddle_at_close_t_minus_1
    pnl_curve = [(ts, (price - entry_price) / entry_price) for ts, price in prices]
    optimal_exit = max(pnl_curve, key=lambda x: x[1])
```

**Data volume estimate:**
- ~50,000 historical earnings events
- 10 timestamps per event = 500,000 API calls
- At 1000/min rate limit = ~8 hours of API calls
- Can be batched over several days

**Phase 3: Retrain Model on Straddle P&L**

Once we have ORATS data, retrain the model with a better target:

| Current | Proposed |
|---------|----------|
| `target = \|Close_T+1 / Close_T - 1\|` | `target = (straddle_at_optimal_exit - straddle_at_entry) / straddle_at_entry` |

This directly optimizes for what we care about: **option P&L**, not stock movement.

**Alternative targets to test:**
1. Straddle P&L at fixed time (e.g., 12:00 ET)
2. Straddle P&L at optimal exit (varies per trade)
3. Maximum straddle P&L achieved during exit day

### Cost-Benefit Analysis

**ORATS Cost:** $99/month

**Potential Improvement:**
- Current avg P&L: ~22% (exiting at 14:00)
- Potential avg P&L: ~46% (exiting at 12:00)
- Improvement: ~24 percentage points per trade
- At 5 trades/week × $2000/trade = $10,000 weekly exposure
- Value of improvement: $2,400/week potential (theoretical max)

**Realistic expectation:** Even capturing 25% of this improvement = $600/week = $2,400/month

**Break-even:** Less than 1 month of data subscription

**Risks:**
- Pattern may not hold in historical data
- Liquidity at 12:00 may be worse than 14:00
- Sample size (5 trades) is very small

### SQL Queries for Ongoing Monitoring

**Query 1: P&L Trajectory by Time Bucket**
```sql
-- Monitor if early exit pattern holds as we collect more data
WITH exit_day_snapshots AS (
    SELECT p.*
    FROM price_snapshots p
    JOIN trades t ON t.trade_id = p.trade_id
    WHERE t.status = 'exited'
      AND t.earnings_timing = 'BMO' 
      AND SUBSTR(p.ts, 1, 10) = t.earnings_date
)
SELECT 
    CAST(minutes_since_open / 30 AS INT) * 30 as mins_bucket,
    COUNT(*) as n_snapshots,
    COUNT(DISTINCT trade_id) as n_trades,
    ROUND(AVG(unrealized_pnl_pct) * 100, 1) as avg_pnl_pct,
    ROUND(MIN(unrealized_pnl_pct) * 100, 1) as min_pnl,
    ROUND(MAX(unrealized_pnl_pct) * 100, 1) as max_pnl
FROM exit_day_snapshots
WHERE unrealized_pnl_pct IS NOT NULL
GROUP BY mins_bucket
ORDER BY mins_bucket;
```

**Query 2: Per-Trade Peak Analysis**
```sql
-- Find when each trade peaked and how much we left on table
WITH exit_day_snapshots AS (
    SELECT p.*, t.ticker, t.earnings_timing, t.exit_pnl_pct, t.premium_paid
    FROM price_snapshots p
    JOIN trades t ON t.trade_id = p.trade_id
    WHERE t.status = 'exited'
      AND SUBSTR(p.ts, 1, 10) = t.earnings_date
),
trade_peaks AS (
    SELECT 
        trade_id, ticker, earnings_timing, premium_paid,
        MAX(unrealized_pnl_pct) as peak_pnl_pct,
        exit_pnl_pct as final_pnl_pct
    FROM exit_day_snapshots
    WHERE unrealized_pnl_pct IS NOT NULL
    GROUP BY trade_id
),
peak_times AS (
    SELECT e.trade_id, MIN(e.minutes_since_open) as peak_minutes
    FROM exit_day_snapshots e
    JOIN trade_peaks tp ON tp.trade_id = e.trade_id 
        AND e.unrealized_pnl_pct = tp.peak_pnl_pct
    GROUP BY e.trade_id
)
SELECT 
    tp.ticker,
    tp.earnings_timing,
    ROUND(tp.peak_pnl_pct * 100, 1) as peak_pnl_pct,
    pt.peak_minutes,
    TIME('09:30', '+' || pt.peak_minutes || ' minutes') as peak_time_et,
    ROUND(tp.final_pnl_pct * 100, 1) as final_pnl_pct,
    ROUND((tp.peak_pnl_pct - tp.final_pnl_pct) * 100, 1) as left_on_table_pct,
    ROUND((tp.peak_pnl_pct - tp.final_pnl_pct) * tp.premium_paid, 0) as left_on_table_usd
FROM trade_peaks tp
JOIN peak_times pt ON pt.trade_id = tp.trade_id
ORDER BY pt.peak_minutes;
```

**Query 3: Aggregate Summary**
```sql
-- Overall summary of exit timing optimization potential
WITH exit_day_snapshots AS (
    SELECT p.*, t.exit_pnl_pct, t.premium_paid
    FROM price_snapshots p
    JOIN trades t ON t.trade_id = p.trade_id
    WHERE t.status = 'exited'
      AND SUBSTR(p.ts, 1, 10) = t.earnings_date
),
trade_peaks AS (
    SELECT 
        trade_id, premium_paid,
        MAX(unrealized_pnl_pct) as peak_pnl_pct,
        exit_pnl_pct as final_pnl_pct
    FROM exit_day_snapshots
    WHERE unrealized_pnl_pct IS NOT NULL
    GROUP BY trade_id
),
peak_times AS (
    SELECT e.trade_id, MIN(e.minutes_since_open) as peak_minutes
    FROM exit_day_snapshots e
    JOIN trade_peaks tp ON tp.trade_id = e.trade_id 
        AND e.unrealized_pnl_pct = tp.peak_pnl_pct
    GROUP BY e.trade_id
)
SELECT 
    COUNT(*) as n_trades,
    ROUND(AVG(tp.peak_pnl_pct) * 100, 1) as avg_peak_pnl_pct,
    ROUND(AVG(tp.final_pnl_pct) * 100, 1) as avg_final_pnl_pct,
    ROUND(AVG(tp.peak_pnl_pct - tp.final_pnl_pct) * 100, 1) as avg_left_on_table_pct,
    ROUND(AVG(pt.peak_minutes), 0) as avg_peak_minutes,
    ROUND(AVG(pt.peak_minutes) / 60.0, 1) as avg_peak_hours,
    ROUND(SUM((tp.peak_pnl_pct - tp.final_pnl_pct) * tp.premium_paid), 0) as total_left_on_table_usd
FROM trade_peaks tp
JOIN peak_times pt ON pt.trade_id = tp.trade_id;
```

### Decision Criteria for ORATS Subscription

Subscribe to ORATS when ANY of:
1. ✅ 20+ trades confirm early exit pattern (avg peak < 180 min)
2. ✅ Total "left on table" exceeds $1,000 (ROI on $99 subscription)
3. ✅ Ready to retrain model with option P&L target

Do NOT subscribe if:
- Pattern doesn't hold (peak varies widely, no clear optimal time)
- Liquidity at earlier exit is significantly worse
- Sample shows early exit is only marginally better
