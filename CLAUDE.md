# CLAUDE.md

## Project Overview

Daily cross-sectional equity trading system combining fundamentals and news signals. Trades US stocks once per day, selecting a basket predicted to outperform peers.

## Key Design Principles

- **Leakage prevention is critical**: News text must be anonymized (`__TARGET__`, `__OTHER__`) to prevent the model from learning company-specific patterns
- **Cross-sectional framing**: Predicts relative performance between stocks, not absolute returns
- **Time alignment**: All timestamps in US/Eastern, 15:30 ET cutoff for same-day features

## Important Files

- `anonymize_news.py` - Text anonymization for news articles (masks company names, tickers, preserves semantics)
- `README.md` - Full system design document

## Code Conventions

- Python 3.10+ (uses `str | None` union syntax, lowercase `dict`/`list`/`tuple` generics)
- `from __future__ import annotations` for forward references
- Frozen dataclasses for configs
- Regex patterns defined as module-level constants with `re.VERBOSE` for complex patterns
- Factory functions to avoid closure-over-loop-variable issues
- Functional style preference

## Data Flow

1. Raw news → anonymized text (company names → tokens)
2. Anonymized text → embeddings (local embedding model)
3. Embeddings + fundamentals → daily feature vectors
4. Model ranks stocks → top-K portfolio

## Testing

Run smoke tests directly: `python3 <module>.py`
