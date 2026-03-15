# Sentinel

**Defense Stock Claim Analyzer** — builds a labeled dataset of social media claims about defense stocks, classifying each as exaggerated, accurate, or understated based on what actually happened to the stock price.

## What It Does

Sentinel scrapes tweets about defense stocks (LMT, RTX, NOC, etc.), fetches the actual 24h price change and news catalysts, then labels each claim using rule-based heuristics. The labeled data is stored in PostgreSQL and served through a Flask API.

This is a **retrospective data collection pipeline**, not a real-time prediction system. Labeling requires the 24h price window to have elapsed. Tweets less than 25 hours old are skipped during enrichment.

## Pipeline

1. **Scrape**: Searches Twitter for defense ticker cashtags (`$LMT`) and quoted company names (`"Lockheed Martin"`) via twscrape. Supports `--since`/`--until` for time-windowed historical scraping.
2. **Enrich**: Fetches price at tweet time + 24h later (yfinance) and news headlines around the tweet (yfinance + DuckDuckGo, ±48h window). Classifies news into catalyst types (contract > earnings > geopolitical > budget) by keyword matching.
3. **Label**: Rule-based labeling — compares claimed direction (keyword/emoji matching) against actual 24h price direction. No NLP models involved, just string matching and hardcoded thresholds.
4. **API**: Flask serves `/api/feed`, `/api/analyze`, `/api/stats`, `/api/feed/stream` (SSE).

## Quick Start

```bash
uv sync
uv run setup
uv run collect --days 7
uv run serve
```

## CLI Commands

```
uv run setup                   # Init DB, sanity check
uv run collect                 # Scrape yesterday's tweets, enrich + label
uv run collect --days 7        # Last 7 days, one day at a time
uv run collect -n 70 --days 7  # ~1000 tweets/day across all tickers
uv run collect --status        # Check background collection progress
uv run collect --stop          # Stop background collection
uv run enrich                  # Re-enrich existing claims
uv run enrich --days 7         # Re-enrich claims from last 7 days
uv run enrich --status         # Check background enrichment progress
uv run serve                   # Start Flask API
uv run train baseline          # Train naive baseline (majority class)
uv run train classical         # Train classical model (Optuna-tuned LR + XGBoost, 200 trials each)
uv run evaluate baseline       # Evaluate baseline on test set
uv run evaluate classical      # Evaluate classical model on test set
uv run predict classical "tweet text"  # Predict label for a tweet
```

## Testing

```bash
uv run pytest tests/ -v
uv run pytest tests/ --cov=src --cov-report=html
uv run pytest tests/ -m "not slow"
```

Tests use mocks for external dependencies (Twitter, yfinance, DuckDuckGo) and run without API keys.

## Key Files

- `src/cli.py` — Click CLI entry point
- `src/collector.py` — Background collection engine with status tracking
- `src/scraper.py` — Twitter scraper (twscrape, async)
- `src/price_fetcher.py` — yfinance price lookups with caching
- `src/news_fetcher.py` — News fetching + catalyst classification
- `src/data/labeler.py` — Rule-based claim labeling (keyword matching + thresholds)
- `src/data/models.py` — RawClaim + LabeledClaim dataclasses
- `src/data/stocks.py` — Defense stock universe (ticker mapping + resolution)
- `src/data/db.py` — PostgreSQL persistence
- `src/api/` — Flask API (app factory + routes)
- `src/config.py` — YAML + env configuration
- `src/models/` — ML models (BaseModel interface, feature extraction, training)
  - `baseline.py` — Naive majority class predictor
  - `classical.py` — Optuna-tuned LR + XGBoost ensemble
  - `features.py` — Tweet text → numerical feature extraction
  - `evaluate.py` — Metrics (accuracy, precision, recall, F1, confusion matrix)
  - `data.py` — Data loading and train/test splitting

## Documentation

- `docs/setup.md` — Full installation, PostgreSQL setup, Twitter accounts, deployment
- `docs/api.md` — API endpoint reference with examples
- `docs/decisions/` — Decision log documenting design choices and rationale
- `docs/project_framing.md` — High-level project framing for the course report

## Configuration

- **`config.yaml`**: App settings (labeling thresholds, scraping options)
- **`.env`**: Secrets (`DATABASE_URL`)

## Package Manager

Uses **uv** exclusively. Never use `pip install` or `uv pip install`.
