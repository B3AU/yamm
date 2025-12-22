# News + Fundamentals Cross-Sectional Trading System

This document describes the **end-to-end design, assumptions, data alignment rules, model architecture,
and execution plan** for a daily equity trading system based on **fundamentals and news**, with strict
leakage control and cost awareness.

The goal is a **defensible, sellable, production-ready strategy**.

---

## 1. High-level objective

Build a system that:

- Trades **once per trading day**
- Selects a **basket of stocks** predicted to **outperform peers**
- Uses **slow, persistent signals** from fundamentals
- Uses **fast, transient signals** from news
- Incorporates **recent price context** for timing and risk
- Executes with **minimal transaction costs**
- Avoids **information leakage**
- Produces an **auditable live track record**

This is a **cross-sectional strategy**, not a time-series predictor.

---

## 2. Cross-sectional framing

At each trading day *t*, the model learns:

> Which stocks will outperform other stocks from **close(t) to close(t+1)**?

The model does **not** attempt to predict absolute market direction.

Benefits:
- Cancels market-wide noise
- More stable statistically
- Lower turnover
- Standard in professional equity ML

---

## 3. Trading cadence and execution (close→close)

### Decision frequency
- **Once per trading day**

### Execution mode (chosen)
**Market-On-Close (MOC)**

- Signals computed **before market close**
- Orders submitted as **MOC**
- Entry: **close(t)**
- Exit: **close(t+1)**

This aligns directly with the training target and avoids overnight execution uncertainty.

---

## 4. Time alignment and cutoff rules (critical)

### Timezone
All timestamps are converted to **US/Eastern**.

### Feature cutoff
- **15:30 ET**

This provides a safety buffer before the MOC cutoff (typically ~15:45 ET).

### News assignment rule

| News published | Assigned to trade date |
|---------------|------------------------|
| ≤ 15:30 ET on day *t* | Day *t* |
| > 15:30 ET on day *t* | Day *t+1* |

This rule is applied **consistently** in:
- Training
- Backtesting
- Live trading

No data after 15:30 ET is ever used for trades entered at close(t).

---

## 5. Universe definition

### Scope
- **US common stocks only**
  - NYSE
  - NASDAQ
  - NYSE American
- Explicitly exclude:
  - ETFs
  - Warrants
  - Preferred shares
  - Illiquid symbols

### Source of truth
- Official **exchange listings** (NASDAQ symbol files)
- Not derived from data vendors

### Liquidity filters
Applied before training and trading:
- Minimum price threshold (e.g. \$5)
- Minimum dollar volume (e.g. \$20M/day)

This ensures:
- Tight spreads
- Reliable MOC execution
- Fee efficiency on IBKR

---

## 6. Data sources

### Fundamentals (numeric)
- Quarterly financial statements
- Ratios:
  - Value (P/E, EV/EBITDA, FCF yield)
  - Quality (ROE, margins)
  - Growth (YoY revenue/earnings)
  - Leverage (debt ratios)
- Lagged to **availability date**

Fundamentals are used **directly as numeric features**, not embedded.

---

### News (text)
- Stock-level news
- Titles and bodies
- Embedded using a **local embedding model**
- Aggregated per `(symbol, trade_date)`

---

### Recent prices (numeric, lightweight)

#### Same-day features (using close(t) as proxy for 15:30 price)
The model runs at ~15:30 ET, so we have access to same-day price action.
We use close(t) as a proxy for the 15:30 price in training (highly correlated, ~30 min apart).

| Feature | Formula | Purpose |
|---------|---------|---------|
| `overnight_gap` | `open(t) / close(t-1) - 1` | Overnight reaction to news |
| `intraday_ret` | `close(t) / open(t) - 1` | Same-day continuation/reversal |

**Rationale**: If news causes a big gap up at open but price fades intraday, this signals possible overreaction.
The model needs to know if news alpha has already been priced in.

#### Historical features (up to close(t-1))
- Short-term returns (1–5 days)
- Short-term volatility
- Distance from recent highs/lows

Long-horizon technical indicators are explicitly avoided.

---

## 7. Leakage prevention

### Company anonymization
To prevent memorization of historical winners:

- Target company → `__TARGET__`
- Other companies → `__OTHER__`
- Mask:
  - Tickers
  - Company names
- Preserve:
  - Numbers
  - Percentages
  - Event language (beats, misses, guidance, lawsuits, etc.)

This forces the model to learn **patterns**, not identities.

---

## 8. Feature construction

### Cross-sectional normalization
All numeric features are normalized **per day**:
z[i,t] = (x[i,t] − mean_j x[j,t]) / std_j x[j,t]

This enforces cross-sectional learning.

---

### News aggregation
For each `(symbol, trade_date)`:
- Mean embedding
- Max embedding
- Optional scalars: news_count, recency-weighted count

---

### Final feature blocks
F[i,t] = numeric fundamentals
P[i,t] = recent price features
E[i,t] = aggregated news embedding

---

## 9. Model architecture (multi-branch, influence-controlled)

### Overview
A **multi-branch neural model** where each data modality is projected into a small latent space
before fusion.

This prevents the high-dimensional news embedding from dominating.

---

### Branch encoders

**Fundamentals encoder**
- MLP → latent `h_f` (default 32 dims)
- 1–2 layers max
- Dropout ~0.1

**Price encoder**
- MLP → latent `h_p` (default 16 dims)
- Dropout ~0.1

**News encoder**
- MLP → latent `h_n` (default 32 dims)
- Stronger dropout ~0.2–0.3
- Stronger regularization

---

### Influence control (mandatory)

Explicitly cap news impact:
h = concat([h_f, h_p, α * h_n])

- `α` default: **0.3**

Optional learned gating:
g = sigmoid(w · h_n + b)
h = concat([h_f, h_p, α * (g * h_n)])

Latent dimension size is **not** used to control influence.

---

### Output head
- Small MLP or linear layer
- Outputs scalar score `s[i,t]`

Scores are used **only for ranking** within day *t*.

---

## 10. Training objective

### Target (close→close)
For stock *i* on day *t*:

r[i,t+1] = log(close[i,t+1] / close[i,t])

---

### Cross-sectional loss (preferred)
**Pairwise ranking loss within each day**:

- Sample pairs `(i, j)` from same day *t*
- `y_ij = sign(r[i,t+1] − r[j,t+1])`
- Logistic pairwise loss

This directly matches basket trading.

---

### Alternative (baseline)
- Regression on demeaned returns:
- y[i,t] = r[i,t+1] − mean_j r[j,t+1]
---

## 11. Portfolio construction

- Rank stocks by `s[i,t]`
- Select **top-K** only (e.g. 10–30)
- Equal-weight or softmax weights
- Enforce **minimum order size** (e.g. \$2–3k)
- Compress or drop tiny positions to avoid fee drag

---

## 12. Execution and costs

### Broker
- Interactive Brokers (IBKR)

### Orders
- **Market-On-Close (MOC)** for entry
- Exit at next day close

### Cost control
- US equities only
- Few, large orders
- No intraday churn
- Tiered pricing

---

## 13. Evaluation

### Mandatory splits
- Time-based split
- Symbol holdout split

### Metrics
- Daily basket return
- Excess return vs benchmark
- Sharpe ratio
- Max drawdown
- Turnover-adjusted PnL
- Rank IC (Spearman per day)

---

## 14. Live proof and auditability

To make results credible and sellable:

- Broker execution logs
- Monthly broker statements
- Immutable signal snapshots (timestamped)
- Logged feature cutoff times
- Reproducible code + config hashes

Target: **12 months of clean live performance**.

---

## 15. Summary

This system is:

- Cross-sectional
- News-aware
- Fundamentally grounded
- Price-aware (lightly)
- Cost-efficient
- Leakage-safe
- Operationally simple
- Institutionally defensible

> **Fundamentals decide where to be invested.  
> News decides when to tilt.**

