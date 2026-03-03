# Sentinel

**Defense Stock Claim Analyzer** — classifies social media claims about defense stocks as exaggerated, accurate, or understated.

## What It Does

Sentinel scrapes tweets about defense stocks (LMT, RTX, NOC, etc.), fetches the actual price change and news catalysts, and labels each claim. It then trains ML classifiers and serves results through a Flask API for a SvelteKit frontend.

## Architecture

1. **Scrape**: Searches Twitter for defense ticker cashtags and company names via twscrape.
2. **Enrich**: Fetches price at tweet time + 24h later (yfinance) and news around the tweet (yfinance + DuckDuckGo).
3. **Label**: Rule-based labeling (exaggerated/accurate/understated) based on claimed vs actual direction + catalyst presence.
4. **Features**: Extracts text features (intensity, defense-specific, specificity) and news features (catalyst type, headline sentiment, embeddings).
5. **Models**: Naive baseline, classical (TF-IDF + LogReg/SVM), and neural (FinBERT + MiniLM fusion).
6. **API**: Flask serves `/api/feed`, `/api/analyze`, `/api/stats`, `/api/feed/stream` (SSE).

## Quick Start

```bash
uv sync
uv run setup
uv run collect --n 100
uv run serve
```

## CLI Commands

```
uv run setup       # Init DB, sanity check
uv run collect     # Scrape + label claims
uv run serve       # Start Flask API
uv run train       # Train ML models
uv run experiment  # Run news ablation
```

## Testing

```bash
uv run pytest tests/ -v
uv run pytest tests/ --cov=src --cov-report=html
uv run pytest tests/ -m "not slow"
```

Tests use mocks for external dependencies (Twitter, yfinance, LLM providers, DuckDuckGo) and run without API keys.

## Key Files

- `src/data/stocks.py` — Defense stock universe (ticker mapping)
- `src/data/models.py` — RawClaim + LabeledClaim dataclasses
- `src/data/labeler.py` — Claim labeling logic
- `src/data/db.py` — PostgreSQL persistence
- `src/scraper.py` — Twitter scraper (twscrape)
- `src/price_fetcher.py` — yfinance price lookups
- `src/news_fetcher.py` — News fetching + catalyst classification
- `src/features/` — Text and news feature extraction
- `src/models/` — Baseline, classical, neural classifiers
- `src/api/` — Flask API (app factory + routes)
- `src/cli.py` — Click CLI entry point

## Configuration

- **`config.yaml`**: App settings (LLM provider, labeling thresholds, model params)
- **`.env`**: Secrets (`OPENAI_API_KEY`, `GOOGLE_API_KEY`, `DATABASE_URL`)

## Package Manager

Uses **uv** exclusively. Never use `pip install` or `uv pip install`.
