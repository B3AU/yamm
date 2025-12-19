# News + Fundamentals Cross-Sectional Trading System

This document describes the **end-to-end design, assumptions, data alignment rules, and execution plan**
for a daily equity trading system based on **fundamentals and news**, with strict leakage control and
cost awareness.

The goal is a **defensible, sellable, production-ready strategy**.

---

## 1. High-level objective

Build a system that:

- Trades **once per day**
- Selects a **basket of stocks** predicted to **outperform peers**
- Uses **slow, persistent signals** from fundamentals
- Uses **fast, transient signals** from news
- Executes with **minimal transaction costs**
- Avoids **information leakage**
- Produces an **auditable live track record**

This is a **cross-sectional strategy**, not a time-series predictor.

---

## 2. Cross-sectional framing

At each trading day *t*, the model learns:

> Which stocks will outperform other stocks on day *t+1*?

Benefits:
- Cancels market-wide noise
- More stable statistically
- Lower turnover
- Standard in professional equity ML

---

## 3. Trading cadence

### Decision frequency
- Once per trading day

### Execution modes

#### Buy at open, sell at close (recommended)
- Compute signals after close of day *t-1*
- Buy at open of day *t*
- Sell at close of day *t*
- Clean alignment
- Lowest leakage risk

#### Market-On-Close (advanced)
- Compute signals before close of day *t*
- Submit MOC orders
- Buy at close of day *t*
- Sell at close of day *t+1*
- Requires strict cutoff discipline

---

## 4. Universe definition

### Scope
- US common stocks only (NYSE, NASDAQ, NYSE American)
- Exclude ETFs, warrants, preferred shares, illiquid symbols

### Source of truth
- Exchange listings (NASDAQ official symbol files)

### Liquidity filters
- Minimum price threshold
- Minimum dollar volume threshold

---

## 5. Data sources

### Fundamentals
- Quarterly financial statements
- Numeric ratios (value, quality, growth, leverage)
- Lagged to availability date

### News
- Stock-level news
- Titles and bodies
- Embedded using a local embedding model

---

## 6. Leakage prevention

### Company anonymization
- Target company replaced with `__TARGET__`
- Other companies replaced with `__OTHER__`
- Mask tickers and company names
- Keep numbers, percentages, event language

---

## 7. News-to-day alignment

All alignment is done in **US/Eastern time**.

### Feature cutoff
- **15:30 ET**

### Assignment rule
- News at or before 15:30 ET on day *t* applies to day *t*
- News after 15:30 ET applies to day *t+1*

---

## 8. Label definition

### Buy-at-open
- Features from day *t-1*
- Label: `open(t) → close(t)` return

### MOC
- Features up to 15:30 on day *t*
- Label: `close(t) → close(t+1)` return

Never mix schemes.

---

## 9. Feature construction

### Fundamentals
- Cross-sectionally normalized per day
- Winsorized

### News
- Aggregated per symbol per trade date
- Mean, max, count of embeddings

### Final feature vector
```
[numeric fundamentals + news embeddings + controls]
```

---

## 10. Model choice

### Recommended
- LightGBM or XGBoost
- Shallow trees
- Strong regularization

### Objective
- Ranking
- Or regression on demeaned returns

---

## 11. Portfolio construction

- Rank stocks daily
- Select top-K
- Equal-weight or softmax sizing
- Enforce minimum order size
- Compress small positions

---

## 12. Execution and costs

### Broker
- Interactive Brokers

### Cost minimization
- US equities only
- Few, large orders
- Avoid intraday churn

---

## 13. Evaluation

- Time-based split
- Symbol holdout split
- Metrics: excess return, Sharpe, drawdown, turnover-adjusted PnL

---

## 14. Live proof

- Broker statements
- Immutable trade logs
- Signal snapshots with timestamps
- Reproducible code and configs

---

## 15. Summary

The system is:
- Cross-sectional
- News-aware
- Fundamentally grounded
- Cost-efficient
- Leakage-safe
- Operationally simple

> **Fundamentals decide where to be invested.  
> News decides when to tilt.**
