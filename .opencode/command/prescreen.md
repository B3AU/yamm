---
description: Screen today's tradeable earnings candidates
---

Run this command and show me the output:

```bash
python3 -m trading.earnings.test_screening --spread-threshold 50 --edge-threshold 0 $ARGUMENTS
```

Screens today's tradeable candidates (Today AMC + Tomorrow BMO) with relaxed thresholds (spread 50%, edge 0%). Connects to IBKR for live option quotes and runs ML predictions.

Common options:
- `--no-llm` - Skip LLM sanity check (faster)
- `--no-ibkr` - Skip IBKR, use mock data
- `-t AAPL MSFT` - Screen specific tickers only
