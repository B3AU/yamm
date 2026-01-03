# YAMM Project Report: News-Based Cross-Sectional Equity Trading

**Project Duration:** 2021-2025 (data), Dec 2024 - Jan 2025 (development)
**Status:** Concluded - No viable trading strategy found
**Author:** Generated with Claude Code

---

## Executive Summary

This project attempted to build a daily cross-sectional equity trading system combining financial news embeddings with fundamental data. After extensive experimentation, **no profitable trading strategy was found that could be executed in practice**.

The core finding: **the model learns patterns that only exist in illiquid stocks where execution is impossible**. When constrained to tradeable (liquid) stocks, the signal either disappears or inverts.

### Key Numbers

| Metric | Illiquid ($1-8M vol) | Liquid ($50M+ vol) |
|--------|---------------------|-------------------|
| Short Sharpe | +1.46 | -0.72 |
| Annual Return (Short) | +231% gross | -73% |
| Tradeable? | No (spreads 3-10%) | Yes |

**Verdict:** The project produced valuable learnings about market microstructure and the limits of public information, but no deployable trading system.

---

## 1. Project Overview

### 1.1 Original Hypothesis

Financial news contains forward-looking information about stock performance. By:
1. Anonymizing news text (replacing company names with tokens like `__TARGET__`)
2. Embedding the anonymized text using a language model
3. Combining with fundamental data
4. Training a pairwise ranking model

We could predict which stocks would outperform their peers on a daily basis.

### 1.2 System Architecture

```
Raw News → Anonymization → Embeddings (768-dim)
                              ↓
Fundamentals (19 features) → Multi-Branch Neural Network → Stock Score
                              ↑
Price Features (9 features) ─┘
```

**Key Design Decisions:**
- Cross-sectional framing (relative performance, not absolute)
- Pairwise ranking loss with label smoothing
- 15:30 ET cutoff for same-day features (leakage prevention)
- News text anonymization to prevent ticker memorization

### 1.3 Data Sources

| Data Type | Source | Period | Size |
|-----------|--------|--------|------|
| News | Financial Modeling Prep | 2021-2025 | ~2M articles |
| Prices | FMP | 2021-2025 | 5,644 symbols |
| Fundamentals | FMP | 2021-2025 | Quarterly data |

**Final ML Dataset:** 2,092,929 rows (stock-day observations with news)

---

## 2. Model Development (Notebooks 2.x)

### 2.1 Architecture

The final model (`model_robust_optimized.pt`) used a multi-branch architecture:

```
Fundamental Branch: 19 → 32 → 32 (dropout 0.6)
Price Branch:       9 → 16 → 16 (dropout 0.3)
News Branch:      768 → 64 → 32 (dropout 0.1, alpha=0.8)
                         ↓
              Concatenate → 32 → 1 (ranking score)
```

### 2.2 Training Configuration

- **Optimizer:** AdamW (lr=2e-3, weight_decay=1e-3)
- **Loss:** Pairwise ranking with 5% label smoothing
- **Epochs:** 25 with cosine annealing
- **Batch Size:** 512
- **Selection Metric:** IC Sharpe on validation set

### 2.3 Validation Performance

On the validation set (2024-05-02 to 2024-10-21):
- IC Sharpe: 4.54
- Mean IC: 0.029

This looked promising - a strong ranking signal existed in the data.

---

## 3. Initial Strategy Testing (Notebooks 3.x - 4.x)

### 3.1 Basic Strategy Results

Testing on held-out data (2024-10-22 to 2025-12-18):

| Strategy | Annual Return | Sharpe |
|----------|--------------|--------|
| Long Top-5 | +1.7% | 0.07 |
| Short Bottom-5 | +231% | 1.46 |
| Long/Short | +116% | 1.17 |

**Observation:** The short side dominated returns. The model was much better at identifying losers than winners.

### 3.2 The Liquidity Problem (Notebook 4.5)

Stratifying by average daily dollar volume revealed:

| Volume Bucket | Short Return (Ann) | Sharpe |
|--------------|-------------------|--------|
| $1-2M | +485% | 1.89 |
| $2-4M | +329% | 1.67 |
| $4-6M | +279% | 1.54 |
| $6-8M | +177% | 1.21 |
| $8-10M | -616% | -0.83 |
| $10-20M | -112% | -0.62 |
| $50M+ | -73% | -0.72 |

**Critical Finding:** Alpha exists only in stocks with $1-8M daily volume. At $8M, the signal inverts.

### 3.3 Why This Matters

Stocks with $1-8M daily volume are practically untradeable:
- **Wide spreads:** 3-10% bid-ask spreads
- **Market impact:** Any trade moves the price
- **Borrow availability:** Hard-to-borrow or impossible to short
- **Position sizing:** Can't build meaningful positions

---

## 4. Tradability Analysis (Notebook 4.6)

### 4.1 Spread Estimation

Using OHLC data as a proxy for bid-ask spreads:
```
Estimated Spread ≈ (High - Low) / (2.5 × Close)
```

For $1-8M volume stocks:
- **Median spread:** 3.47%
- **Mean spread:** 4.89%

### 4.2 Borrow Cost Estimation

Without direct borrow data, estimated using stock characteristics:
- Penny stocks (<$5): +15% annual fee
- High volatility (>100% ann): +10% annual fee
- Erratic volume patterns: +5% annual fee

**Estimated annual borrow rate:** 35-55% for sweet spot stocks

### 4.3 Cost vs. Return Analysis

| Component | Daily Cost |
|-----------|-----------|
| Spread (entry + exit) | 6.94% |
| Market impact | 2.00% |
| Borrow cost | 0.14% |
| **Total** | **10.6%** |
| **Gross Return** | 4.4% |
| **Net Return** | **-6.2%** |

**Verdict:** Costs are 241% of gross returns. Strategy is NOT VIABLE.

---

## 5. Strategy Rethink (Notebook 5.0)

### 5.1 Alternatives Explored

| Approach | Result |
|----------|--------|
| Long-only (liquid) | -11.8% ann, Sharpe -0.50 |
| Short-only (liquid) | -168% ann, Sharpe -0.72 |
| Longer horizons (5d, 10d, 20d) | Degrades further |
| Combined L/S (liquid) | -90% ann, Sharpe -0.87 |

### 5.2 Signal Inversion

The model's predictions actively hurt performance in liquid stocks:
- Stocks the model ranks highly → underperform
- Stocks the model ranks poorly → outperform

This suggests the patterns learned apply specifically to illiquid market dynamics.

---

## 6. Liquid Universe Retraining (Notebook 5.1)

### 6.1 Hypothesis

Perhaps training specifically on liquid stocks would learn different (tradeable) patterns.

### 6.2 Methodology

- Filtered training data to $50M+ daily volume only
- Full hyperparameter optimization (81 configurations)
- Same architecture and training procedure

### 6.3 Results

| Metric | Original Model | Liquid-Trained |
|--------|---------------|----------------|
| IC Sharpe | 1.66 | 1.64 |
| Long Sharpe | -0.25 | **-1.37** |
| Short Sharpe | -0.72 | **-1.55** |
| Long Return | -5.8% | **-32.6%** |
| Short Return | -73.4% | **-147.2%** |

**Finding:** Training on liquid stocks made performance WORSE, not better. The model finds ranking correlation but systematically predicts the wrong direction.

---

## 7. Sector Aggregation (Notebook 5.2)

### 7.1 Hypothesis

Individual stock predictions are noisy. Aggregating to sector level might produce a cleaner signal tradeable via liquid sector ETFs.

### 7.2 Methodology

1. Score all stocks (including illiquid) with original model
2. Aggregate scores by sector each day
3. Trade sector ETFs (XLK, XLF, XLV, etc.) based on sector rankings

### 7.3 Results

- **Sector IC:** -0.0211
- **Sector IC Sharpe:** -0.89

The aggregation did not recover a tradeable signal. High-scoring sectors actually underperformed.

---

## 8. Key Findings

### 8.1 The Fundamental Problem

**Public information is already priced into liquid stocks.**

The news embeddings + fundamentals combination captures something real about stock performance. But:
- In illiquid stocks: Information is not efficiently priced → exploitable (but not executable)
- In liquid stocks: Information is rapidly incorporated → no edge

### 8.2 Market Microstructure

The project revealed significant insights about market structure:

1. **Liquidity is the gatekeeper:** Alpha opportunities exist precisely where execution is impossible
2. **Signal inversion:** Patterns in illiquid markets are negatively correlated with liquid market behavior
3. **Aggregation doesn't help:** Sector-level aggregation doesn't recover individual stock signal
4. **Costs dominate:** For any strategy involving illiquid stocks, costs swamp returns

### 8.3 Model Behavior

The neural network learned to identify stocks likely to underperform:
- Works well for losers in small/illiquid universe
- Actively counterproductive for liquid stocks
- Cannot be "fixed" by retraining on liquid data

---

## 9. Lessons Learned

### 9.1 Technical Lessons

1. **Test on tradeable universe first:** Should have stratified by liquidity from day one
2. **Cost modeling is essential:** Gross returns mean nothing without realistic cost estimates
3. **Validation IC ≠ Trading profits:** High IC doesn't guarantee real-world profitability
4. **Signal inversion is real:** A pattern that works in one regime can hurt in another

### 9.2 Strategic Lessons

1. **Public data has limits:** News and fundamentals are available to everyone
2. **Efficient markets are efficient:** For liquid stocks, information is priced quickly
3. **Edges are where competition isn't:** Illiquid stocks are inefficient because no one can trade them
4. **Small scale ≠ edge:** Retail can access same public data as institutions

### 9.3 Process Lessons

1. **Kill ideas faster:** Should have done liquidity analysis before full hyperparameter optimization
2. **Reality-check assumptions:** Assumed execution was possible; should have verified early
3. **Track experiments systematically:** The notebook structure helped but formal experiment tracking would be better

---

## 10. What Would Have Worked?

With hindsight, these approaches might have yielded better results:

### 10.1 Different Asset Classes

- **Options:** Illiquid options have wide spreads but can be traded (no borrow needed)
- **Crypto:** Inefficient markets with actual execution possibility
- **Private markets:** Information asymmetries exist but require different data

### 10.2 Different Time Horizons

- **Weekly/monthly rebalancing:** Lower turnover reduces cost impact
- **Event-driven:** Earnings, M&A announcements have documented drift effects

### 10.3 Different Data

- **Alternative data:** Satellite imagery, web traffic, job postings - truly differentiated
- **Private information:** Order flow, sentiment from non-public sources
- **Structural edges:** Index rebalancing, tax-loss harvesting windows

### 10.4 Different Framing

- **Risk management:** Use model to avoid disasters rather than pick winners
- **Factor timing:** Predict factor (momentum, value) performance, not individual stocks
- **Volatility prediction:** Predict vol for options pricing, not direction

---

## 11. Code & Data Artifacts

### 11.1 Key Files

| File | Purpose |
|------|---------|
| `anonymize_news.py` | News text anonymization |
| `trading/model.py` | Model inference wrapper |
| `data/model_robust_optimized.pt` | Trained model weights |
| `data/ml_dataset.pqt` | Full ML dataset |
| `data/symbol_sectors.pqt` | Symbol-to-sector mapping |

### 11.2 Notebooks

| Notebook | Purpose |
|----------|---------|
| 2.8 full opti.ipynb | Final hyperparameter optimization |
| 4.5 shortability analysis.ipynb | Liquidity stratification |
| 4.6 borrow cost analysis.ipynb | Cost estimation |
| 5.0 strategy rethink.ipynb | Alternative strategy exploration |
| 5.1 liquid universe training.ipynb | Liquid-only retraining |
| 5.2 sector aggregation test.ipynb | Sector ETF strategy |

### 11.3 Data Pipeline

```
FMP API → Raw Data → Feature Engineering → ML Dataset → Model Training → Backtesting
```

---

## 12. Conclusion

This project demonstrates a common failure mode in quantitative finance: **developing a model that works in backtests but cannot be implemented in practice**.

The news embedding approach successfully learned patterns predictive of stock returns. But these patterns exist only in the market segment where:
- Information flows slowly
- Few participants trade
- Execution is prohibitively expensive

For liquid, tradeable stocks, the public information in news and fundamentals is already reflected in prices. The efficient market hypothesis holds - at least for this data and approach.

### Final Verdict

**The project produced no viable trading strategy.** However, it provided:

1. **Validated the anonymization approach:** News embeddings do contain signal when properly constructed
2. **Mapped the liquidity-alpha relationship:** Clear understanding of where inefficiencies exist
3. **Realistic cost models:** Framework for evaluating strategy tradability
4. **Codebase for future work:** Clean data pipeline and model architecture

The most valuable outcome may be the lesson itself: **finding alpha is not enough; you must be able to execute it.**

---

## Appendix: Summary Statistics

### A.1 Dataset Statistics

- **Total rows:** 2,092,929
- **Unique symbols:** 3,506 (5,644 in price data)
- **Date range:** 2021-01-13 to 2025-12-18
- **Features:** 19 fundamental + 9 price + 768 embedding = 796 total

### A.2 Sector Distribution

| Sector | Rows | % |
|--------|------|---|
| Technology | 346,482 | 16.6% |
| Healthcare | 319,200 | 15.2% |
| Financial Services | 286,328 | 13.7% |
| Consumer Cyclical | 282,940 | 13.5% |
| Industrials | 272,950 | 13.0% |
| Energy | 116,109 | 5.5% |
| Real Estate | 115,784 | 5.5% |
| Basic Materials | 104,282 | 5.0% |
| Consumer Defensive | 99,342 | 4.7% |
| Communication Services | 87,520 | 4.2% |
| Utilities | 61,992 | 3.0% |

### A.3 Model Configuration (Final)

```python
ModelConfig(
    fund_hidden=32, price_hidden=16, news_hidden=64,
    fundamental_latent=32, price_latent=16, news_latent=32,
    fundamental_dropout=0.7, price_dropout=0.2, news_dropout=0.2,
    news_alpha=0.8,
    learning_rate=2e-3, weight_decay=1e-3, label_smoothing=0.05,
    n_epochs=25
)
```

---

*Report generated: January 2025*
