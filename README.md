# Sentinel

Scrapes tweets about defense stocks, fetches the actual price movement and surrounding news, then labels each claim as
**exaggerated**, **accurate**, or **understated**. Stores everything in PostgreSQL and serves it through a Flask API.

Covers 15 public defense companies (LMT, RTX, NOC, GD, BA, LHX, HII, LDOS, SAIC, BAH, KTOS, PLTR, RKLB) and
recognizes two private ones (Anduril, Shield AI) in tweet text.

## How it works

The `collect` command runs a four-step pipeline:

**Scrape.** Searches Twitter/X for tweets mentioning defense stocks by cashtag (`$LMT`) and company name
(`"Lockheed Martin"`) using [twscrape](https://github.com/vladkens/twscrape). Filters out retweets, deduplicates by
tweet ID, and resolves which ticker the tweet is about.

**Enrich with price data.** For each tweet, fetches the stock price at tweet time and 24 hours later via yfinance.
Both prices are averaged over a 30-minute window for stability (tries 5-minute candles first, falls back to hourly,
then daily). The 24h-later price uses the first available trading candle at or after the target time, so a Friday
tweet compares against Monday's open instead of snapping back to Friday's close. Since this needs the full 24h window,
tweets less than 25 hours old are skipped.

**Enrich with news.** Fetches news articles from yfinance and DuckDuckGo within 48 hours before and after the tweet.
Deduplicates by URL and stores all matching headlines on the claim.

Headlines are then scanned for keywords that indicate a real news event ("catalyst") that could explain a price move.
A tweet claiming "$LMT to the moon!" is more credible if there's a headline about a Pentagon contract than if the
only news is generic market commentary. The labeler uses this: directional claims backed by a catalyst are judged
more leniently than claims with no news backing them.

Four catalyst types are checked in priority order (first match wins):

- Contract: contract, award, pentagon, DoD, IDIQ, LRIP, billion-dollar deal
- Earnings: EPS, quarterly, beat/miss, guidance, revenue
- Geopolitical: war, conflict, missile, strike, Ukraine, Taiwan, NATO, escalation
- Budget: NDAA, defense budget, appropriations, sequestration

**Label.** Compares what the tweet claimed against what actually happened:

1. Parses the tweet for directional language. Bullish keywords (moon, pump, surge, rally, bullish, calls, long) and
   emoji (rocket, chart-up, diamond, fire) count toward "up." Bearish keywords (crash, dump, plunge, short, puts,
   bearish) and emoji (chart-down, skull, bear) count toward "down." Mixed or absent signals resolve to "neutral."

2. Determines the actual direction from the 24h price change: above +0.5% is up, below -0.5% is down, otherwise
   neutral.

3. Scores the tweet's intensity from 0 to 1 based on exclamation marks, ALL CAPS words, emoji, and superlatives
   like "insane" or "massive."

4. Assigns a label:

| Condition                                            | Label       |
|------------------------------------------------------|-------------|
| Claimed direction opposite of actual                 | exaggerated |
| Directional claim but price moved less than 2%       | exaggerated |
| Directional claim, no news catalyst, move under 4%   | exaggerated |
| Direction matches, move at least 2%                  | accurate    |
| Neutral tweet, price moved less than 2%              | accurate    |
| Neutral tweet, price moved 5% or more               | understated |
| Price moved 10%+ but tweet intensity below 0.3       | understated |

5. Computes an exaggeration score from 0.0 (spot on) to 1.0 (completely wrong). Direction mismatches start at 0.7;
   loud language on a flat stock starts at 0.4; matching calls on real moves score near 0.0.

Labeled claims are stored in PostgreSQL (`raw_claims` + `labeled_claims` tables) with all price data, news headlines,
catalyst type, directions, and scores.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- PostgreSQL 17
- Twitter/X accounts with exported cookies (for scraping)

## Setup

### 1. Install PostgreSQL 17

```bash
sudo apt install -y postgresql-common
sudo /usr/share/postgresql-common/pgdg/apt.postgresql.org.sh
sudo apt install -y postgresql-17
```

### 2. Create the database

```bash
sudo -u postgres psql
```

```sql
CREATE USER sentinel WITH PASSWORD 'your-secure-password';
CREATE DATABASE sentinel_db OWNER sentinel;
\q
```

No extensions required.

### 3. Install dependencies

```bash
uv sync
```

### 4. Configure

```bash
cp config.example.yaml config.yaml
cp .env.example .env
```

Edit `.env`:

```
DATABASE_URL=postgresql://sentinel:your-secure-password@localhost:5432/sentinel_db
OPENAI_API_KEY=sk-...       # if using OpenAI provider
GOOGLE_API_KEY=...           # if using Google provider
```

### 5. Add Twitter accounts

Sentinel uses [twscrape](https://github.com/vladkens/twscrape) for scraping. You need at least one Twitter/X account
with exported cookies.

1. Log into Twitter/X in your browser
2. Export cookies as JSON using a browser extension (e.g., "Cookie-Editor")
3. Add the account:

```bash
uv run python add_account.py <twitter_username> <cookies.json>
```

Multiple accounts improve rate limit handling. Proxies can be configured in `config.yaml`:

```yaml
twitter:
  proxies:
    - socks5://user:pass@host1:port
    - socks5://user:pass@host2:port
```

### 6. Initialize and run

```bash
uv run setup
uv run collect -n 100
uv run serve
```

## CLI

| Command                             | Description                             |
|-------------------------------------|-----------------------------------------|
| `setup`                             | Init DB schema, sanity check labeler    |
| `collect`                           | Scrape, enrich, label, store            |
| `collect -n 100 --tickers LMT,RTX`  | Specific tickers                        |
| `collect --background`              | Run in background                       |
| `collect --since 2d`                | Only tweets from the last 2 days        |
| `collect --market-open yesterday`   | Tweets around yesterday's open (9:00-10:30 AM ET) |
| `status`                            | Check background collection progress    |
| `stop`                              | Stop background collection              |
| `serve`                             | Start API (gunicorn, 4 workers)         |
| `serve --dev`                       | Flask dev server with hot reload        |

All commands use `uv run`.

### Time filtering

`--since`, `--until`, and `--market-open` control which tweets are scraped. Absolute times are interpreted as US
Eastern (ET). Relative durations (`1h`, `30m`, `2d`, `1w`) are subtracted from now. `--market-open` accepts `today`,
`yesterday`, or `YYYY-MM-DD` and sets a 9:00-10:30 AM ET window.

Since labeling needs the 24h price outcome, tweets under 25 hours old are skipped during enrichment.

```bash
uv run collect -n 50 --since 2026-03-02T09:00 --until 2026-03-02T11:00
uv run collect -n 100 --market-open yesterday
uv run collect -n 20 --since 30m --tickers LMT
```

## API

Served via gunicorn at `/api`.

| Method | Endpoint           | Description                              |
|--------|--------------------|------------------------------------------|
| `POST` | `/api/analyze`     | Label a tweet using current price data   |
| `GET`  | `/api/feed`        | Paginated labeled claims, newest first   |
| `GET`  | `/api/feed/stream` | SSE stream of new claims                 |
| `GET`  | `/api/stats`       | Label distribution, top tickers          |
| `GET`  | `/api/stocks`      | Defense stock universe                   |
| `GET`  | `/api/health`      | DB connectivity check                    |

`/api/feed` takes `limit` (default 50, max 200), `offset`, and optional `label` filter.

`/api/analyze` takes `{"text": "...", "ticker": "LMT"}`. Ticker is optional and will be resolved from the text.
When `live_news_fetch` is enabled, it fetches current price and news before labeling.

```bash
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "$LMT to the moon!", "ticker": "LMT"}'

curl http://localhost:5000/api/feed?limit=20
curl http://localhost:5000/api/stats
```

## Testing

```bash
uv run pytest tests/ -v
uv run pytest tests/ --cov=src --cov-report=html
```

All external dependencies are mocked. Tests run without API keys or a database.

## Configuration

`config.yaml` for settings, `.env` for secrets. Environment variables override YAML.

```yaml
app:
  port: 5000
  live_news_fetch: true

twitter:
  db_path: accounts.db
  proxies: []

llm:
  provider: google
  openai_model: gpt-4o
  google_model: gemini-2.0-flash

scraping:
  limit_per_ticker: 50
  search_timeout: 300

labeling:
  exaggeration_threshold: 0.02   # 2% minimum significant move
  news_window_hours: 48

logging:
  level: INFO
```

## Claim fields

Each claim carries data from scraping, enrichment, and labeling. All of these are stored in PostgreSQL and
returned by the API.

**From the tweet:**

| Field | Description |
|-------|-------------|
| `tweet_id` | Twitter's unique ID. Primary key, used for deduplication. |
| `text` | Full tweet text. |
| `username` | Twitter handle that posted it. |
| `created_at` | When the tweet was posted. Everything else is anchored to this timestamp. |
| `likes` | Like count at scrape time. Stored but not used in labeling. |
| `retweets` | Retweet count at scrape time. Stored but not used in labeling. |
| `ticker` | Resolved stock ticker (e.g. "LMT"). If a tweet mentions multiple defense stocks, the scraper picks the most specific match. |
| `company_name` | Full company name (e.g. "Lockheed Martin"), looked up from the ticker. |

**From price enrichment:**

| Field | Description |
|-------|-------------|
| `price_at_tweet` | Stock price around when the tweet was posted, averaged over a 30-minute window. |
| `price_24h_later` | Stock price ~24 hours later. Uses the first available trading candle at or after the 24h mark, so a Friday tweet gets Monday's price instead of Friday's close again. |
| `price_change_pct` | Percentage change between those two prices: `((price_24h_later - price_at_tweet) / price_at_tweet) * 100`. |

**From news enrichment:**

| Field | Description |
|-------|-------------|
| `news_headlines` | JSON array of article titles found within 48 hours before or after the tweet (from yfinance and DuckDuckGo, deduplicated by URL). Can be empty. |
| `has_catalyst` | True if any headline matched a catalyst keyword category. |
| `catalyst_type` | Which category matched first in priority order: `contract`, `earnings`, `geopolitical`, `budget`. Null if nothing matched or no news was found. |

**From labeling:**

| Field | Description |
|-------|-------------|
| `claimed_direction` | What the tweet predicted, from keyword/emoji matching. "up" (bullish language like "moon", "surge", rocket emoji), "down" (bearish like "crash", "dump"), or "neutral" (no signal or conflicting signals). |
| `actual_direction` | What the stock actually did. Above +0.5% = "up", below -0.5% = "down", otherwise "neutral". |
| `label` | Final verdict: "exaggerated" (wrong or overblown), "accurate" (matched reality), or "understated" (underplayed a real move). |
| `exaggeration_score` | 0.0 (nailed it) to 1.0 (completely wrong). Direction mismatches start at 0.7, hype on a flat stock starts at 0.4, correct calls on real moves score near 0.0. More granular than the three-bucket label. |
| `news_summary` | First headline from `news_headlines`, used for display. Empty if no news found. |

The intensity score (0-1, based on exclamation marks, caps, emoji, superlatives like "insane" or "massive")
feeds into the exaggeration score but is not stored separately.

## Deployment

### Dedicated user setup

```bash
sudo useradd -m -s /bin/bash sentinel
sudo mkdir -p /opt/sentinel
sudo chown sentinel:sentinel /opt/sentinel
sudo -iu sentinel

curl -LsSf https://astral.sh/uv/install.sh | sh
git clone <repo-url> /opt/sentinel/app
cd /opt/sentinel/app
cp config.example.yaml config.yaml
cp .env.example .env
editor .env
chmod 600 .env
uv sync
uv run setup
uv run python add_account.py <username> <cookies.json>
exit
```

### systemd

```bash
sudo cp /opt/sentinel/app/systemd/sentinel.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now sentinel.service
```

### Environment variables

| Variable          | Required        | Description                        |
|-------------------|-----------------|------------------------------------|
| `DATABASE_URL`    | Yes             | PostgreSQL connection string       |
| `OPENAI_API_KEY`  | If using OpenAI | For analysis explanations          |
| `GOOGLE_API_KEY`  | If using Google | For analysis explanations          |
| `TWITTER_PROXIES` | No              | Comma-separated proxy list         |
