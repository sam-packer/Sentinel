# API reference

Sentinel runs a Flask API via gunicorn at the `/api` prefix.

```bash
uv run serve                            # gunicorn, 4 workers
uv run serve --dev                      # Flask dev server with hot reload
uv run serve --host 0.0.0.0 --port 8080 # custom bind
uv run serve --workers 8                # custom worker count
```

## Endpoints

| Method | Endpoint           | Description                            |
|--------|--------------------|----------------------------------------|
| `GET`  | `/api/health`      | DB connectivity check                  |
| `GET`  | `/api/stocks`      | Defense stock universe                 |
| `GET`  | `/api/feed`        | Paginated labeled claims, newest first |
| `GET`  | `/api/feed/stream` | SSE stream of new claims               |
| `GET`  | `/api/stats`       | Label distribution, top tickers        |
| `POST` | `/api/predict`     | ML model inference on tweet text       |
| `GET`  | `/api/accounts`              | Account credibility listing            |
| `GET`  | `/api/accounts/:username`    | Account detail with claim history      |
| `GET`  | `/api/stocks/:ticker/feed`   | Per-stock tweet feed, bot-filtered     |
| `GET`  | `/api/stocks/:ticker/stats`  | Stock-level aggregates                 |
| `GET`  | `/api/leaderboard`           | Top grifters and high-signal accounts  |

## GET /api/feed

Paginated list of labeled claims, newest first.

Query parameters:

| Param    | Default | Description                                      |
|----------|---------|--------------------------------------------------|
| `limit`  | 50      | Max results (capped at 200)                      |
| `offset` | 0       | Pagination offset                                |
| `label`  | none    | Filter: `exaggerated`, `accurate`, `understated` |

```bash
curl http://localhost:5000/api/feed?limit=20
curl http://localhost:5000/api/feed?label=exaggerated&limit=10
```

## GET /api/feed/stream

Server-Sent Events endpoint for live claim updates. Emits new `LabeledClaim` objects as JSON whenever the pipeline
inserts a new row. Sends a heartbeat comment every 15 seconds.

```bash
curl -N http://localhost:5000/api/feed/stream
```

## GET /api/stats

Aggregate statistics: label counts, catalyst type distribution, top tickers.

```bash
curl http://localhost:5000/api/stats
```

## GET /api/stocks

Returns the defense stock universe, all tracked tickers and company names.

## POST /api/predict

Classify tweet text using a trained ML model. Pure text-based inference, no price or news data needed. This is how you
predict exaggerated vs accurate before the 24h price window elapses.

Request body:

```json
{
  "text": "$LMT to the moon! 🚀🚀🚀",
  "model": "classical",
  "username": "hypetrader"
}
```

`model` is optional. If omitted, uses the first available trained model. Trained models are loaded into memory at
server startup.

`username` is optional. If provided and the account exists in the database, the response includes their credibility info.

Response:

```json
{
  "label": "exaggerated",
  "confidence": 0.8734,
  "model": "classical",
  "available_models": ["baseline", "classical"],
  "account": {
    "username": "hypetrader",
    "grifter_score": 0.75,
    "grifter_category": "mostly_wrong",
    "total_claims": 12
  }
}
```

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "$LMT to the moon!", "model": "classical"}'
```

## GET /api/accounts

List accounts with credibility scores, sortable and filterable.

Query parameters:

| Param       | Default          | Description                                          |
|-------------|------------------|------------------------------------------------------|
| `sort_by`   | `grifter_score`  | Sort field: `grifter_score`, `total_claims`, `username`, `last_seen` |
| `order`     | `desc`           | Sort order: `asc` or `desc`                          |
| `min_claims`| 0                | Minimum total claims to include                      |
| `is_bot`    | none             | Filter by bot status: `true` or `false`              |
| `limit`     | 50               | Max results (capped at 200)                          |
| `offset`    | 0                | Pagination offset                                    |

```bash
curl http://localhost:5000/api/accounts?min_claims=5&sort_by=grifter_score&order=desc
curl http://localhost:5000/api/accounts?is_bot=true
```

## GET /api/accounts/:username

Account detail with credibility metrics and claim history.

```bash
curl http://localhost:5000/api/accounts/hypetrader
```

Returns 404 if the account is not in the database.

## GET /api/stocks/:ticker/feed

Per-stock tweet feed with predictions and credibility info. Bot accounts are excluded by default.

Query parameters:

| Param          | Default | Description                                 |
|----------------|---------|---------------------------------------------|
| `limit`        | 50      | Max results (capped at 200)                 |
| `offset`       | 0       | Pagination offset                           |
| `exclude_bots` | `true`  | Whether to exclude tweets from bot accounts |

```bash
curl http://localhost:5000/api/stocks/LMT/feed
curl http://localhost:5000/api/stocks/RTX/feed?exclude_bots=false
```

Returns 404 if the ticker is not in the defense stock universe.

## GET /api/stocks/:ticker/stats

Stock-level aggregates: total claims, label distribution, exaggeration rate, top catalysts, average price change.

```bash
curl http://localhost:5000/api/stocks/LMT/stats
```

Returns 404 if the ticker is not in the defense stock universe.

## GET /api/leaderboard

Top accounts by credibility. Returns accounts with 5+ claims, excluding bots.

Query parameters:

| Param      | Default    | Description                                   |
|------------|------------|-----------------------------------------------|
| `category` | `grifters` | `grifters` (highest grifter score) or `signal` (lowest) |
| `limit`    | 20         | Max results (capped at 100)                   |

```bash
curl http://localhost:5000/api/leaderboard
curl http://localhost:5000/api/leaderboard?category=signal&limit=10
```

## GET /api/health

Returns `200` with `{"status": "healthy"}` if the database is reachable, `503` with `{"status": "degraded"}` otherwise.
