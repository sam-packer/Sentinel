# Sentinel

Defense Stock Claim Analyzer. Scrapes tweets about defense stocks, filters out bots, fetches the actual price movement
and surrounding news, then labels each claim as **exaggerated**, **accurate**, or **understated**. Tracks account-level
credibility to surface who's consistently right and who's full of it.

Covers 15 public defense companies (LMT, RTX, NOC, GD, BA, LHX, HII, LDOS, SAIC, BAH, KTOS, PLTR, RKLB) and recognizes
two private ones (Anduril, Shield AI) in tweet text.

## Quick start

```bash
uv sync
cp config.example.yaml config.yaml
cp .env.example .env        # set DATABASE_URL and ANTHROPIC_API_KEY
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
Twitter/X ──► Scrape ──► Classify accounts ──► Enrich (price + news) ──► Label ──► PostgreSQL ──► Flask API
```

### Stage 1: Scrape

Searches Twitter/X for tweets mentioning defense stocks by cashtag (`$LMT`) and company name (`"Lockheed Martin"`)
using [twscrape](https://github.com/Telesphoreo/twscrape). All ticker searches run concurrently via `asyncio.gather`,
with twscrape's account pool handling rotation and rate limiting across the parallel tasks.

Each tweet is deduplicated by tweet ID, retweets are filtered out, and the ticker is resolved. If a tweet mentions
multiple defense stocks, the scraper picks the most specific match (longest company name found in text).

The `--days` flag scrapes each day independently rather than requesting a single wide time window. This prevents
Twitter from biasing results toward popular/recent tweets and gives more even temporal distribution.

### Stage 2: Classify accounts

Before enriching tweets, Sentinel classifies each account as human or bot using LLM-as-judge (Claude). This runs
automatically during collection — each account is classified once and the result is cached in the `accounts` table.

The classifier receives the account's username and up to 5 sample tweets, then categorizes it as one of:

- **human** — real person sharing opinions or analysis
- **bot** — automated account reposting headlines, copying content, or auto-generating posts
- **garbage** — spam, scam, or irrelevant promotional content

All tweets are enriched regardless of account type. Human-only filtering happens downstream in the training data loader,
so bot and garbage tweets still get price/news data (useful for analysis) but are excluded from ML training and grifter
scoring.

Requires `ANTHROPIC_API_KEY` in `.env`. If not set or if bot detection is disabled in config, this step is skipped and
all accounts are treated as human.

### Stage 3: Enrich with price data

For each tweet, fetches two prices via yfinance:

- **Price at tweet time**, averaged over a 30-minute window centered on the tweet timestamp.
- **Price 24 hours later**, the first available trading candle at or after the 24h mark.

The 24h-later price deliberately uses the *next available* candle rather than interpolating. A Friday afternoon tweet
compares against Monday's open, not Friday's close again. This way the price reflects the actual information gap a
reader would have experienced.

Both prices try 5-minute candles first (most granular), fall back to hourly, then daily. The 30-minute averaging window
smooths out intraday noise.

### Stage 4: Enrich with news

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

### Stage 5: Label

Two independent labelers run on the same enriched tweets, writing to separate tables. Both are rule-based (no ML in the
labeling step). The enrichment loop fetches price and news once per tweet, then runs both labelers in a single pass.

**Naive labeler** (`src/data/labeler.py` → `naive_labeled_claims`): keyword/emoji matching for claimed direction, fixed
2% exaggeration threshold for all tickers. Simple and interpretable but can't handle negation, sarcasm, or non-claims.

**Improved labeler** (`src/data/improved_labeler.py` → `improved_labeled_claims`): adds negation detection (
proximity-based), sarcasm markers, non-claim filtering (job posts, questions, long-term theses, position disclosures,
past-tense recaps, informational content), and per-ticker volatility thresholds calibrated from historical data.

Both labelers parse claimed direction from keywords/emoji, compare against the actual 24h price change, and assign one
of three labels: **exaggerated**, **accurate**, or **understated**.

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

The above describes the naive labeler's scoring. The improved labeler has its own exaggeration score function
(`_compute_exaggeration_score_improved`) where the magnitude gap scales to 3x the ticker's median daily move instead of
a fixed 5%.

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
| `actual_direction`   | From price change: above +0.5% is `up`, below -0.5% is `down`, otherwise `neutral` (naive labeler). The improved labeler uses per-ticker thresholds for direction classification. |
| `label`              | Final verdict: `exaggerated`, `accurate`, or `understated`.                          |
| `exaggeration_score` | 0.0 (accurate) to 1.0 (completely wrong). More granular than the three-bucket label. |
| `news_summary`       | First headline from `news_headlines`, used for display.                              |

Account fields (stored in `accounts` table):

| Field                        | Description                                                                              |
|------------------------------|------------------------------------------------------------------------------------------|
| `username`                   | Twitter handle. Primary key.                                                             |
| `account_type`               | Account classification: `human`, `bot`, or `garbage` (classified by LLM-as-judge).       |
| `classification_reason`      | Why the account was classified this way. Includes type and explanation.                   |
| `naive_total_claims`         | Total labeled claims from this account (naive labeler).                                  |
| `naive_exaggerated_count`    | How many claims were labeled exaggerated (naive labeler).                                |
| `naive_grifter_score`        | Ratio of exaggerated to total claims (naive labeler). Null if fewer than 5 claims.       |
| `improved_total_claims`      | Total labeled claims from this account (improved labeler).                               |
| `improved_exaggerated_count` | How many claims were labeled exaggerated (improved labeler).                             |
| `improved_grifter_score`     | Ratio of exaggerated to total claims (improved labeler). Null if fewer than 5 claims.    |

## Commands

### setup

Initializes the database schema, creates directories, and runs a sanity check against hardcoded example tweets to
verify the labeler works.

```bash
uv run setup
```

### collect

Scrapes tweets about defense stocks, enriches them with price and news data, labels them, and stores results in
PostgreSQL. Each day is scraped independently for even temporal distribution. Only tweets older than 25 hours are
processed so the 24h price window has elapsed. Always runs both labelers (naive and improved) — there are no flags to
choose one.

```bash
uv run collect                              # yesterday's tweets, 50 per ticker
uv run collect --days 7                     # last 7 days, one day at a time
uv run collect -n 100 --days 30             # 100 per ticker per day, 30 days
uv run collect --tickers LMT,RTX --days 3   # specific tickers only
uv run collect --background                 # run in background
uv run collect --status                     # check background progress
uv run collect --stop                       # stop background collection
```

### enrich

Re-enriches existing raw claims with fresh price and news data. Runs both labelers by default. Use `--naive` or
`--improved` to run only one.

```bash
uv run enrich                      # enrich with both labelers
uv run enrich --naive              # enrich with naive labeler only → naive_labeled_claims
uv run enrich --improved           # enrich with improved labeler only → improved_labeled_claims
uv run enrich --days 7             # only claims from the last 7 days
uv run enrich --unlabeled          # only claims missing labels
uv run enrich --rejudge            # reclassify ALL accounts via LLM before enriching
uv run enrich --background         # run in background
uv run enrich --status             # check background progress
uv run enrich --stop               # stop background enrichment
```

### classify

Manually classify Twitter accounts as human or bot. This is a catch-up tool — accounts are automatically classified
during collection, but this command handles accounts that were scraped before bot detection was added.

```bash
uv run classify                        # classify unclassified accounts (up to 100)
uv run classify --limit 500            # classify more accounts
uv run classify --reclassify           # re-classify all accounts
```

### serve

Starts the Flask API server. Uses gunicorn with 4 workers by default.

```bash
uv run serve                       # gunicorn on 0.0.0.0:5000
uv run serve --port 8080           # custom port
uv run serve --dev                 # Flask dev server with hot reload
```

### train

Trains a model on labeled claims from the database. Loads data, splits into train/test sets with a fixed seed, trains,
saves the model to `models/<name>/`, and prints test set metrics.

Available models:

- `baseline` — Naive majority class predictor. Always predicts the most common label (~78% accuracy, 0% exaggeration
  recall). The floor any real model must beat.
- `classical` — Optuna-tuned TF-IDF + logistic regression. 200 Optuna trials with stratified 3-fold CV optimizing macro
  F1. Saves model weights, TF-IDF vectorizer, and top predictive words for interpretability.
- `neural` — Fine-tuned BERTweet (vinai/bertweet-base). 50 Optuna trials with stratified 3-fold CV optimizing macro F1.
  Tunes learning rate, weight decay, warmup, epochs, batch size, and dropout. Requires GPU.

Runs on both label sets by default. Use `--naive` or `--improved` to train on only one.

```bash
uv run train baseline                  # majority-class baseline on both label sets
uv run train classical --tune          # TF-IDF + LR on both label sets, run Optuna tuning
uv run train classical --naive --tune  # TF-IDF + LR on naive labels only, run Optuna tuning
uv run train classical --improved      # TF-IDF + LR on improved labels only (reuse saved params)
uv run train neural --naive --tune     # BERTweet on naive labels only (requires GPU)
uv run train neural --improved --tune  # BERTweet on improved labels only (requires GPU)
```

Model artifacts are saved to `models/<name>/<naive_labeler|improved_labeler>/`:

```
models/
├── baseline/
│   ├── naive_labeler/
│   │   ├── model.json          # majority class + class counts
│   │   ├── report.md           # full markdown evaluation report
│   │   ├── evaluation.json     # machine-readable metrics
│   │   └── mispredictions.json # every test set error with full context
│   └── improved_labeler/
│       └── ...
├── classical/
│   ├── naive_labeler/
│   │   ├── lr.pkl              # logistic regression weights
│   │   ├── tfidf.pkl           # TF-IDF vectorizer
│   │   ├── model.json          # hyperparams, top predictive words
│   │   ├── report.md
│   │   ├── evaluation.json
│   │   └── mispredictions.json
│   └── improved_labeler/
│       └── ...
└── neural/
    ├── naive_labeler/
    │   ├── model/              # BERTweet fine-tuned weights
    │   ├── tokenizer/          # BERTweet tokenizer files
    │   ├── model.json
    │   ├── report.md
    │   ├── evaluation.json
    │   └── mispredictions.json
    └── improved_labeler/
        └── ...
```

### evaluate

Evaluates a previously trained model on the test set. Uses the same seed and split as training so the test data
matches. Reports accuracy, per-class precision/recall/F1, and confusion matrix.

Runs on both label sets by default. Use `--naive` or `--improved` to evaluate on only one.

```bash
uv run evaluate baseline               # evaluate baseline on both label sets
uv run evaluate classical --naive      # evaluate classical on naive test set only
uv run evaluate classical --improved   # evaluate classical on improved test set only
uv run evaluate neural --naive         # evaluate neural on naive test set only
```

### predict

Runs inference on tweet text using a trained model. No price or news data needed - this is how you classify a tweet
before the 24h price window elapses.

```bash
uv run predict baseline '$LMT to the moon! 🚀🚀🚀'                  # uses naive labeler model by default
uv run predict classical --improved 'RTX awarded massive Pentagon contract'  # use improved labeler model
uv run predict neural --naive 'defense stocks looking weak here'             # explicitly use naive labeler model
```

Also available as an API endpoint at `POST /api/predict`. See [docs/api.md](docs/api.md).

## Testing

```bash
uv run pytest tests/ -v
uv run pytest tests/ --cov=src --cov-report=html
```

All external dependencies are mocked. Tests run without API keys or a database.

