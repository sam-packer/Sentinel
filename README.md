# Sentinel

Defense Stock Claim Analyzer. Scrapes tweets about defense stocks, fetches the actual price movement and surrounding
news, then labels each claim as **exaggerated**, **accurate**, or **understated**.

Covers 15 public defense companies (LMT, RTX, NOC, GD, BA, LHX, HII, LDOS, SAIC, BAH, KTOS, PLTR, RKLB) and recognizes
two private ones (Anduril, Shield AI) in tweet text.

## Quick start

```bash
uv sync
cp config.example.yaml config.yaml
cp .env.example .env        # set DATABASE_URL
uv run setup
uv run collect --days 7
uv run serve
```

See [docs/setup.md](docs/setup.md) for full installation (PostgreSQL, Twitter accounts, deployment)
and [docs/api.md](docs/api.md) for API reference.

## Architecture

Sentinel is a four-stage retrospective pipeline. "Retrospective" because labeling requires the 24-hour price window to
have elapsed, so tweets less than 25 hours old are skipped during enrichment.

```
Twitter/X ──► Scrape ──► Enrich (price + news) ──► Label ──► PostgreSQL ──► Flask API
```

### Stage 1: Scrape

Searches Twitter/X for tweets mentioning defense stocks by cashtag (`$LMT`) and company name (`"Lockheed Martin"`)
using [twscrape](https://github.com/Telesphoreo/twscrape). All ticker searches run concurrently via `asyncio.gather`,
with twscrape's account pool handling rotation and rate limiting across the parallel tasks.

Each tweet is deduplicated by tweet ID, retweets are filtered out, and the ticker is resolved. If a tweet mentions
multiple defense stocks, the scraper picks the most specific match (longest company name found in text).

The `--days` flag scrapes each day independently rather than requesting a single wide time window. This prevents
Twitter from biasing results toward popular/recent tweets and gives more even temporal distribution.

### Stage 2: Enrich with price data

For each tweet, fetches two prices via yfinance:

- **Price at tweet time**, averaged over a 30-minute window centered on the tweet timestamp.
- **Price 24 hours later**, the first available trading candle at or after the 24h mark.

The 24h-later price deliberately uses the *next available* candle rather than interpolating. A Friday afternoon tweet
compares against Monday's open, not Friday's close again. This way the price reflects the actual information gap a
reader would have experienced.

Both prices try 5-minute candles first (most granular), fall back to hourly, then daily. The 30-minute averaging window
smooths out intraday noise.

### Stage 3: Enrich with news

Fetches news articles from yfinance ticker news and DuckDuckGo news search within a ±48-hour window around the tweet.
Articles are deduplicated by URL.

Headlines are scanned for keywords that indicate a catalyst, meaning a real news event that could explain a price move.
A tweet claiming "$LMT to the moon!" is more credible if there's a headline about a Pentagon contract than if the only
news is generic market commentary.

Four catalyst types are checked in priority order (first match wins):

| Catalyst     | Example keywords                                           |
|--------------|------------------------------------------------------------|
| Contract     | contract, award, pentagon, DoD, IDIQ, LRIP, billion-dollar |
| Earnings     | EPS, quarterly, beat/miss, guidance, revenue               |
| Geopolitical | war, conflict, missile, strike, Ukraine, Taiwan, NATO      |
| Budget       | NDAA, defense budget, appropriations, sequestration        |

The priority ordering matters: a headline mentioning both a "Pentagon contract" and "budget" is classified as
`contract`, because contract wins are more directly price-moving than general budget news.

### Stage 4: Label

Compares what the tweet *claimed* against what *actually happened*. This is entirely rule-based, no ML models, just
string matching and hardcoded thresholds.

The labeler first parses claimed direction from the tweet text. Bullish keywords (`moon`, `pump`, `surge`, `rally`,
`bullish`, `calls`, `long`) and emoji (🚀📈💎🔥💰🐂) count toward "up." Bearish keywords (`crash`, `dump`, `plunge`,
`short`, `puts`, `bearish`) and emoji (📉🔻💀🐻⚠️) count toward "down." If both signals are present, the result is
"neutral."

Then it determines actual direction from the 24h price change: above +0.5% is up, below -0.5% is down, otherwise
neutral.

It also scores tweet intensity from 0 to 1, based on exclamation marks, ALL CAPS words, emoji density, and superlatives
like "insane" or "massive". This feeds into the exaggeration score but is not stored separately.

The label is assigned by these rules:

| Condition                                          | Label       |
|----------------------------------------------------|-------------|
| Claimed direction opposite of actual               | exaggerated |
| Directional claim but price moved less than 2%     | exaggerated |
| Directional claim, no news catalyst, move under 4% | exaggerated |
| Direction matches, move at least 2%                | accurate    |
| Neutral tweet, price moved less than 2%            | accurate    |
| Neutral tweet, price moved 5% or more              | understated |
| Price moved 10%+ but tweet intensity below 0.3     | understated |

### Exaggeration score

Separately from the three-bucket label, each claim gets a continuous exaggeration score from 0.0 (spot-on) to 1.0
(completely wrong). The score is the sum of three independent components, each with a clear rationale:

**Direction mismatch: 0.0 or 0.5.** If the tweet claimed one direction and the stock went the other, that's half the
score by itself. This is the strongest signal because a wrong-direction call is fundamentally misleading regardless of
how big or small the move was. A calm wrong-direction tweet scores 0.5; a hyperbolic one scores higher because the other
components add on top.

**Magnitude gap: 0.0 to 0.3.** Measures how much the tweet's language intensity exceeds what the actual price move
justifies. A 5% daily move is large enough to justify even aggressive language for a single stock, so the gap is
`intensity * (1 - move/5%) * 0.3`. At 5%+, this component is zero. Below that, hype language on a small move gets
penalized proportionally. When directions mismatch, the full intensity counts because none of the move supports the
claim.

**Catalyst gap: 0.0 to 0.2.** Penalty for directional claims with no news catalyst. A tweet claiming "$LMT mooning"
with a Pentagon contract headline behind it is more credible than the same tweet with no relevant news. Calculated as
`intensity * 0.2`, but only when news was actually fetched (a failed news fetch doesn't count against the claim).

These three components sum to a maximum of 1.0. Because they're additive and independent, you can explain exactly why
any tweet scored what it did. For example, a score of 0.62 might break down as: 0.5 for wrong direction + 0.09 for
moderate language + 0.03 for weak catalyst backing.

## Claim data model

Each claim carries data from scraping, enrichment, and labeling. All fields are stored in PostgreSQL and returned by the
API.

Tweet fields:

| Field          | Description                                                                                                                 |
|----------------|-----------------------------------------------------------------------------------------------------------------------------|
| `tweet_id`     | Twitter's unique ID. Primary key, used for deduplication.                                                                   |
| `text`         | Full tweet text.                                                                                                            |
| `username`     | Twitter handle.                                                                                                             |
| `created_at`   | When the tweet was posted. Everything else is anchored to this timestamp.                                                   |
| `likes`        | Like count at scrape time.                                                                                                  |
| `retweets`     | Retweet count at scrape time.                                                                                               |
| `ticker`       | Resolved stock ticker (e.g. `LMT`). If a tweet mentions multiple defense stocks, the scraper picks the most specific match. |
| `company_name` | Full company name (e.g. `Lockheed Martin`), looked up from the ticker.                                                      |

Price enrichment fields:

| Field              | Description                                                                           |
|--------------------|---------------------------------------------------------------------------------------|
| `price_at_tweet`   | Stock price around tweet time, averaged over a 30-minute window.                      |
| `price_24h_later`  | Stock price ~24h later. Uses first available trading candle at or after the 24h mark. |
| `price_change_pct` | Percentage change: `((price_24h_later - price_at_tweet) / price_at_tweet) * 100`.     |

News enrichment fields:

| Field            | Description                                                                                               |
|------------------|-----------------------------------------------------------------------------------------------------------|
| `news_headlines` | Article titles found within ±48h of the tweet (yfinance + DuckDuckGo, deduplicated by URL). Can be empty. |
| `has_catalyst`   | Whether any headline matched a catalyst keyword category.                                                 |
| `catalyst_type`  | First matching category: `contract`, `earnings`, `geopolitical`, `budget`. Null if none matched.          |

Labeling fields:

| Field                | Description                                                                          |
|----------------------|--------------------------------------------------------------------------------------|
| `claimed_direction`  | Parsed from tweet text: `up`, `down`, or `neutral`.                                  |
| `actual_direction`   | From price change: above +0.5% is `up`, below -0.5% is `down`, otherwise `neutral`.  |
| `label`              | Final verdict: `exaggerated`, `accurate`, or `understated`.                          |
| `exaggeration_score` | 0.0 (accurate) to 1.0 (completely wrong). More granular than the three-bucket label. |
| `news_summary`       | First headline from `news_headlines`, used for display.                              |

## CLI

| Command                                   | Description                                     |
|-------------------------------------------|-------------------------------------------------|
| `uv run setup`                            | Init DB schema, sanity check labeler            |
| `uv run collect`                          | Scrape yesterday's tweets, enrich, label, store |
| `uv run collect --days 7`                 | Scrape last 7 days, one day at a time           |
| `uv run collect -n 100 --tickers LMT,RTX` | Specific tickers, 100 per ticker                |
| `uv run collect --background`             | Run in background                               |
| `uv run collect --status`                 | Check background progress                       |
| `uv run collect --stop`                   | Stop background collection                      |
| `uv run enrich`                           | Re-enrich all existing claims                   |
| `uv run enrich --days 7`                  | Re-enrich claims from the last 7 days           |
| `uv run enrich --unlabeled`               | Only enrich unlabeled claims                    |
| `uv run serve`                            | Start API (gunicorn, 4 workers)                 |
| `uv run serve --dev`                      | Flask dev server with hot reload                |

## Testing

```bash
uv run pytest tests/ -v
uv run pytest tests/ --cov=src --cov-report=html
```

All external dependencies are mocked. Tests run without API keys or a database.

## Project structure

```
src/
├── cli.py              # Click CLI entry points
├── collector.py         # Background collection engine with status tracking
├── scraper.py           # Twitter scraper (twscrape, async)
├── price_fetcher.py     # yfinance price lookups with caching
├── news_fetcher.py      # News fetching + catalyst classification
├── config.py            # YAML + env configuration
├── data/
│   ├── models.py        # RawClaim + LabeledClaim dataclasses
│   ├── labeler.py       # Rule-based claim labeling
│   ├── stocks.py        # Defense stock universe (ticker mapping)
│   └── db.py            # PostgreSQL persistence
└── api/
    ├── app.py           # Flask app factory
    └── routes.py        # API endpoints
```

## Configuration

`config.yaml` for app settings, `.env` for secrets. See [config.example.yaml](config.example.yaml) for all options.
