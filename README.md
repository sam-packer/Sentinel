# Sentinel

Defense Stock Claim Analyzer — scrapes social media claims about defense stocks, fetches actual price data and news
catalysts, and classifies each claim as **exaggerated**, **accurate**, or **understated**.

Tracks 15 defense companies (LMT, RTX, NOC, GD, BA, LHX, HII, LDOS, BAH, KTOS, PLTR, and more) across Twitter/X.

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
CREATE
USER sentinel WITH PASSWORD 'your-secure-password';
CREATE
DATABASE sentinel_db OWNER sentinel;
\q
```

No extensions are required — Sentinel uses plain PostgreSQL without pgvector or any other extras.

### 3. Install dependencies

```bash
uv sync
```

### 4. Configure

```bash
cp config.example.yaml config.yaml
cp .env.example .env
```

Edit `.env` with your secrets:

```
DATABASE_URL=postgresql://sentinel:your-secure-password@localhost:5432/sentinel_db
OPENAI_API_KEY=sk-...       # if using OpenAI provider
GOOGLE_API_KEY=...           # if using Google provider
```

See `config.example.yaml` for all application settings (model params, labeling thresholds, scraping options).

### 5. Add Twitter accounts

Sentinel uses [twscrape](https://github.com/vladkens/twscrape) to scrape tweets. You need at least one Twitter/X account
with exported cookies.

1. Log into Twitter/X in your browser
2. Export cookies as JSON using a browser extension (e.g., "Cookie-Editor" or "Get cookies.txt")
3. Add the account:

```bash
uv run python add_account.py <twitter_username> <cookies.json>
```

You can add multiple accounts for better rate limit handling. If you use proxies, configure them in `config.yaml`:

```yaml
twitter:
  proxies:
    - socks5://user:pass@host1:port
    - socks5://user:pass@host2:port
```

Proxies are assigned to accounts in round-robin order.

### 6. Initialize and run

```bash
# Init DB schema and sanity check the labeler
uv run setup

# Scrape and label claims
uv run collect --n 100

# Start the API server (gunicorn, 4 workers)
uv run serve
```

The API will be available at `http://localhost:5000`.

## CLI Reference

| Command                             | Description                                                 |
|-------------------------------------|-------------------------------------------------------------|
| `setup`                             | Create directories, init DB schema, sanity check labeler    |
| `collect`                           | Scrape tweets, fetch prices/news, label claims, store in DB |
| `collect --n 100 --tickers LMT,RTX` | Collect for specific tickers                                |
| `collect --background`              | Run collection in the background                            |
| `status`                            | Check background collection progress                        |
| `stop`                              | Stop a background collection                                |
| `serve`                             | Start API server via gunicorn                               |
| `serve --dev`                       | Start Flask dev server (hot reload)                         |
| `serve --port 8080 --workers 8`     | Custom port and worker count                                |
| `train`                             | Train ML models on collected data                           |
| `train --model classical`           | Train only the classical model                              |
| `experiment`                        | Run news feature ablation experiment                        |

All commands are run with `uv run`, e.g. `uv run setup`, `uv run collect --n 100`.

## API

| Method | Endpoint                      | Description                                               |
|--------|-------------------------------|-----------------------------------------------------------|
| `POST` | `/api/analyze`                | Analyze a tweet, returns label + confidence + explanation |
| `GET`  | `/api/feed`                   | Paginated labeled claims, newest first                    |
| `GET`  | `/api/feed?label=exaggerated` | Filter by label                                           |
| `GET`  | `/api/feed/stream`            | SSE stream of new claims                                  |
| `GET`  | `/api/stats`                  | Label distribution, top tickers, most exaggerated users   |
| `GET`  | `/api/stocks`                 | Defense stock universe                                    |
| `GET`  | `/api/health`                 | DB ping + model status                                    |

### Examples

```bash
# Analyze a claim
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "$LMT to the moon!", "ticker": "LMT"}'

# Get the feed
curl http://localhost:5000/api/feed?limit=20&offset=0

# Get stats
curl http://localhost:5000/api/stats
```

## ML Models

Sentinel trains three classifiers on labeled claims:

1. **Baseline** — Majority class classifier (always predicts the most common label)
2. **Classical** — TF-IDF + handcrafted features fed into Logistic Regression / SVM with grid search over regularization
3. **Neural** — FinBERT (tweet text) + MiniLM (news headlines) + scalar features through a fusion head

The neural model uses differential learning rates (2e-5 for FinBERT, 1e-3 for the classification head) with early
stopping on validation macro F1.

## Testing

```bash
uv run pytest tests/ -v                          # all tests
uv run pytest tests/ --cov=src --cov-report=html  # with coverage
uv run pytest tests/ -m "not slow"                # skip slow tests
```

Tests mock all external dependencies (Twitter, yfinance, LLM providers, DuckDuckGo) and run without API keys or a
database.

## Deployment

### Create a dedicated user

```bash
# Create user with home directory
sudo useradd -m -s /bin/bash sentinel
sudo mkdir -p /opt/sentinel
sudo chown sentinel:sentinel /opt/sentinel

# Switch to sentinel user
sudo -iu sentinel

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone or copy the repo
git clone <repo-url> /opt/sentinel/app

# Set up config and secrets
cd /opt/sentinel/app
cp config.example.yaml config.yaml
cp .env.example .env
editor .env
chmod 600 .env

# Install dependencies and initialize
uv sync
uv run setup

# Add Twitter accounts
uv run python add_account.py <username> <cookies.json>

# Exit back to root
exit
```

### Install systemd service

```bash
sudo cp /opt/sentinel/app/systemd/sentinel.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now sentinel.service

# Verify
sudo systemctl status sentinel
sudo journalctl -u sentinel -f
```

### Environment variables

| Variable          | Required        | Description                                             |
|-------------------|-----------------|---------------------------------------------------------|
| `DATABASE_URL`    | Yes             | PostgreSQL connection string                            |
| `OPENAI_API_KEY`  | If using OpenAI | For LLM-based analysis explanations                     |
| `GOOGLE_API_KEY`  | If using Google | For Gemini-based analysis explanations                  |
| `TWITTER_PROXIES` | No              | Comma-separated proxy list (alternative to config.yaml) |
