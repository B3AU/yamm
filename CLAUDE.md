# V1 PLAN — Earnings-Driven Options Volatility Strategy

## Executive Summary

Exploit volatility mispricing around earnings in semi-illiquid US equities. Use ML to predict the distribution of post-earnings moves, then trade defined-risk option structures when implied volatility misprices that distribution.

**Core edge:** Vol + tails, not direction.

**V1 in one sentence:** Earnings-only, T-1 close entry, T+1 close exit, strict option liquidity gates, straddles/strangles only, fixed position sizing, mechanical risk controls, execution-first validation.

---

## Phase 0: Execution Validation (FIRST PRIORITY)

**This phase comes before any ML work.**

### Objective
Validate that you can actually execute trades profitably before building models.

### Duration
2-4 weeks

### Activities
- Place small real orders (minimum viable size)
- Target 20-50 fills across different liquidity buckets
- No ML predictions - just test execution mechanics

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

## Event Definition

### Earnings Timing
For each stock:
- Earnings date (known in advance)
- Announcement timing: **Before Market Open (BMO)** or **After Market Close (AMC)**

Timing matters for exit:
- AMC earnings: Stock reacts at next open, you exit at next close
- BMO earnings: Stock reacts at open, you exit at same-day close

### Trade Window (Fixed for V1)
| Event | Timing |
|-------|--------|
| Entry | T-1 close (close before earnings day) |
| Exit | T+1 close (close after earnings reaction) |

**Do not vary this in V1.** Log counterfactual exits (next open, T+2) but don't trade them.

Rationale for T+1 close (not next open):
- Open spreads are pathological (widest of day)
- Post-earnings drift is real and often continues
- Full session to exit at reasonable price
- Simpler execution (no 9:30 AM urgency)

---

## Market-Implied Baseline

For each candidate, compute the market's expected move:

```
implied_move_pct = (ATM_call_mid + ATM_put_mid) / spot_price
```

This is approximately the expected |move| priced by the market.

Also compute:
- IV rank (current IV vs 52-week range)
- IV percentile (% of days IV was lower)
- ATM IV in absolute terms

These provide context but don't override the straddle-implied move.

---

## ML Model (V1)

### Objective
Predict the **distribution of post-earnings absolute returns**, focusing on tail probabilities.

### Target Variable
Primary: `|Close_T → Close_T+1|` (matches your exit timing)

Also log/model:
- `|Close_T → Open_T+1|` (gap move, cleaner earnings reaction)
- Keep both for future comparison

### Model Outputs
Quantiles of |return|:
- **q50** (median move)
- **q75** (upper quartile)
- **q90** (tail threshold)
- **q95** (extreme tail)

The key comparison: `predicted_q75` vs `implied_move`

### Feature Categories

**Event Context**
- Earnings timing (BMO / AMC) - one-hot
- Day of week
- Days until earnings (should be 1 for V1, but include for future)
- Earnings season density (how many other earnings this week)

**Historical Earnings Behavior**
- Historical |earnings moves| for this ticker (mean, std, max)
- Historical |earnings moves| vs implied (did it beat or miss implied?)
- Quarters since last "big" move (>2x implied)
- Earnings surprise history (beat/miss EPS frequency)

**Price / Volatility Regime**
- 5d, 10d, 20d realized volatility
- Realized vol vs implied vol (vol risk premium)
- Recent gap frequency (how "jumpy" is this stock?)
- Distance from 52-week high/low
- Recent trend (momentum features)
- Drawdown from recent peak

**Options Context (Light)**
- ATM IV
- IV rank / percentile
- Put-call skew proxy (25-delta put IV - ATM IV)
- Term structure slope (if available)

**Sector/Market Context**
- Sector (for sector-level effects)
- VIX level
- Recent sector earnings moves (if peers reported already)

### Modeling Notes
- **Walk-forward only:** No lookahead. Retrain quarterly or after N new samples.
- **Calibration is the key metric:** If predicted q90 = 12%, realized exceedance should be ~10%.
- **Start simple:** Gradient boosting (XGBoost/LightGBM) on tabular features before any deep learning.
- **Baseline:** Historical average |move| for this ticker. ML must beat this.

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

Suggested starting `cost_buffer`: 2-3% of spot (aggressive) to 5% (conservative).

**Do not trade if:**
- Edge < cost_buffer
- Liquidity gates not met
- Position limits would be exceeded
- Any kill switch is active

### No Continuous Sizing in V1
Do not scale position size by edge magnitude. Binary: trade or don't.

Rationale: Edge estimates are noisy early. Kelly-style sizing punishes miscalibration violently. Fixed sizing lets you learn without catastrophic errors.

---

## Trade Structures (V1)

### Primary: ATM Straddle
- Buy ATM call + ATM put
- Same strike, same expiration
- Delta-neutral at entry
- Profits from any large move

### Secondary: Strangle
- Buy OTM call + OTM put
- Cheaper than straddle
- Requires larger move to profit
- Use when straddle spread is too wide

### Structure Selection Logic
```
if ATM_straddle_spread <= 15% of mid:
    trade straddle
elif OTM_strangle_spread <= 12% of mid:
    trade strangle
else:
    no trade (liquidity insufficient)
```

### Not in V1: Directional Spreads
Debit call/put spreads require:
- Directional probability (not just |move| quantiles)
- Calibrated asymmetric tail forecasts

Add in V1.5 only after |move| model is calibrated and you develop directional signals.

---

## Entry Rules

### Timing
- Enter at T-1 close (last 30 minutes of session before earnings day)

### Order Type
- **Limit orders only**
- Never market orders

### Price Targeting
- Initial limit: `mid + 0.3 × spread` (slightly aggressive)
- If not filled in 5 minutes: `mid + 0.5 × spread`
- If still not filled: walk up to `mid + 0.7 × spread` max
- If not filled at max: **no trade** (log as "failed to fill")

### Fill Assumptions for Backtest
- Assume fill at `mid + α × spread` where α = 0.3-0.5
- Do NOT assume mid fills
- Reject any backtest that requires mid fills to be profitable

---

## Exit Rules

### Timing
- Exit at T+1 close (close of first full session after earnings)

### Order Type
- Limit orders, but be willing to hit bid if needed
- Target: `mid - 0.3 × spread`
- If approaching close and not filled: hit bid

### Early Exit (Optional)
- If position is up >100% intraday, consider taking profit
- If position is down >80% and earnings reaction is complete, consider cutting
- Log all early exits separately for analysis

### Forced Exit
- If liquidity disappears (no bid), mark to zero and log
- If earnings date changes and position no longer makes sense, exit immediately

---

## Position Sizing

### Fixed Risk Per Trade
- **Max loss per position: 0.25% of NAV**
- For a straddle, max loss = premium paid
- Position size = `0.0025 × NAV / straddle_premium`

### Sizing Progression
| Phase | Max Risk Per Trade |
|-------|-------------------|
| Phase 0 (validation) | Minimum contract size only |
| Phase 1 (early) | 0.25% NAV |
| Phase 2 (calibrated) | 0.50% NAV |
| Phase 3 (scaled) | 1.00% NAV |

Only progress when:
- Calibration is stable for 50+ trades
- Fill model matches reality
- No kill switches triggered recently

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

### Mechanical Kill Switches

These trigger automatically, not discretionally.

**1. Calibration Drift Monitor**
```
Rolling window: last 50 trades
Track: predicted q90 exceedance vs realized frequency
Expected: ~10% of trades exceed predicted q90

IF realized_exceedance > 2 × expected:
    → Cut position size by 50%
    → Flag for model review
    → Do not increase size until recalibrated

IF realized_exceedance > 3 × expected:
    → Halt new trades
    → Full model retrain required
```

**2. Drawdown Throttle**
```
Track: peak-to-trough drawdown (rolling)

IF drawdown > 3% NAV:
    → Freeze size increases
    → Continue trading at current size

IF drawdown > 5% NAV:
    → Reduce position size by 50%
    → Review for systematic issues

IF drawdown > 10% NAV:
    → Halt all new trades
    → Full system review required
```

**3. Execution Degradation**
```
Track: realized slippage vs modeled slippage

IF realized_slippage > modeled_slippage + 2% for 10 consecutive trades:
    → Tighten liquidity gates
    → Reduce trade count
    → Re-estimate fill model
```

---

## Logging Requirements

### For Every Trade
| Field | Description |
|-------|-------------|
| ticker | Symbol |
| earnings_date | Date of earnings |
| earnings_timing | BMO / AMC |
| entry_datetime | When order placed |
| entry_quoted_bid | Bid at decision |
| entry_quoted_ask | Ask at decision |
| entry_quoted_mid | Mid at decision |
| entry_limit_price | Your limit |
| entry_fill_price | Actual fill |
| entry_fill_time | Time to fill |
| structure | straddle / strangle |
| strikes | Strike(s) used |
| expiration | Option expiry |
| contracts | Number of contracts |
| premium_paid | Total premium |
| max_loss | = premium_paid |
| predicted_q50 | Model output |
| predicted_q75 | Model output |
| predicted_q90 | Model output |
| implied_move | Market's implied move |
| edge_q75 | predicted_q75 - implied_move |
| exit_datetime | When exited |
| exit_quoted_bid | Bid at exit |
| exit_quoted_ask | Ask at exit |
| exit_fill_price | Actual fill |
| exit_pnl | Realized P&L |
| realized_move | Actual |stock move| |
| spot_at_entry | Stock price at entry |
| spot_at_exit | Stock price at exit |

### For Every Non-Trade (CRITICAL)
Log candidates you passed on:
| Field | Description |
|-------|-------------|
| ticker | Symbol |
| earnings_date | Date |
| rejection_reason | Why not traded (spread, OI, edge, limits) |
| quoted_spread | What the spread was |
| quoted_oi | Open interest |
| predicted_edge | What edge would have been |
| counterfactual_pnl | What would have happened |

This prevents survivorship bias and tells you if gates are too tight/loose.

### Counterfactual Logging
For every trade, also log what would have happened with:
- Exit at next open (instead of next close)
- Entry at T-2 (instead of T-1)
- Different structure (strangle if you traded straddle)

---

## Backtesting Requirements

### Data Requirements
| Data | Source | Notes |
|------|--------|-------|
| Earnings dates + BMO/AMC | FMP | Have this |
| Historical prices | FMP | Have this |
| Historical option chains | ORATS / Polygon / CBOE / scrape | **Blocker** |
| Realized earnings moves | Derived from prices | Easy |
| Historical IV (ATM minimum) | Derived or purchased | Needed |

### Backtest Rules
- Use historical bid/ask, not just mid
- Fill assumption: `mid + 0.4 × spread`
- Walk-forward only (no future information)
- Minimum 2 years history (8 earnings cycles per stock)
- Include transaction costs explicitly
- Model non-fills (if spread was too wide, treat as no-trade)

### Backtest Credibility Checks
Reject backtest if:
- Profitable only at mid (fails at mid + 0.3×spread)
- Sharpe > 3 (probably overfit)
- Win rate > 70% (suspicious)
- Requires fills on <50 OI options

### If No Historical Options Data
Options data is expensive. Alternatives:
1. **Paper trade forward** with real-time chains + logging (fastest ground truth)
2. **Backtest move prediction only** - validate that predicted |move| > implied predicts profitable trades, without option-level P&L
3. **Synthetic backtest** - estimate historical straddle prices from realized vol and earnings history (less accurate)

Option 1 (paper trading forward) gives you real data in 1-2 months and is probably the best path.

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

---

## Timeline

| Phase | Duration | Focus |
|-------|----------|-------|
| Phase 0 | 2-4 weeks | Execution validation, fill model |
| Data & Features | 2-3 weeks | Historical data, feature engineering |
| Model V1 | 2-3 weeks | Train |move| quantile model, validate calibration |
| Paper Trading | 4-8 weeks | Full system, paper trades, logging |
| Phase 1 Live | 8-12 weeks | Small real money, validate everything |
| Review | 2 weeks | Analyze, fix issues, decide on Phase 2 |

**Total to V1 validation: 4-6 months**

---

## V2 Expansion Ideas (Out of Scope for V1)

Only consider after V1 is validated:

- **Directional spreads** - Add when you have calibrated P(up) model
- **Variable sizing** - Kelly-style when calibration is trusted
- **Short vol trades** - Sell straddles when predicted move < implied (requires iron condor for defined risk)
- **Earlier entry** - Test T-2, T-3 entry with higher theta cost
- **Earnings revision signals** - Trade on guidance changes, not just earnings
- **Cross-asset** - Apply to other event types (FDA, macro)
- **IV surface features** - Skew, term structure as model inputs
- **Portfolio optimization** - Greeks-based position limits

---

## Key Principles

1. **Execution dominates early.** Validate fills before anything else.

2. **Calibration over accuracy.** A well-calibrated model that says "10% chance of big move" is more valuable than an accurate point estimate.

3. **Fixed sizing until proven.** Don't scale with edge until you trust your edge estimates.

4. **Log everything, especially non-trades.** Survivorship bias is silent and deadly.

5. **Mechanical risk controls.** Kill switches trigger automatically, not when you "feel" something is wrong.

6. **Simple structures first.** Straddles before spreads. Complexity is V2.

7. **Assume fills are worse than quoted.** Any strategy that only works at mid is not a strategy.

8. **Small universe is fine.** 10 good trades beat 50 paper trades.

---

## Immediate Next Steps

1. **Validate options data access** - Can you get real-time chains? Historical? What's the cost?

2. **Build earnings calendar** - FMP earnings dates, verify BMO/AMC timing

3. **Screen for liquidity** - How many names actually pass your liquidity gates? Is universe viable?

4. **Phase 0 setup** - Broker connection, order routing, logging infrastructure

5. **First fill tests** - Place small orders, measure everything

Start here. Do not build ML models until Phase 0 is complete.
