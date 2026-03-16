# API reference

Sentinel runs a Flask API via gunicorn at the `/api` prefix.

```bash
uv run serve                            # gunicorn, 4 workers
uv run serve --dev                      # Flask dev server with hot reload
uv run serve --host 0.0.0.0 --port 8080 # custom bind
uv run serve --workers 8                # custom worker count
```

## Endpoints

| Method | Endpoint                    | Description                            |
|--------|-----------------------------|----------------------------------------|
| `GET`  | `/api/health`               | DB connectivity check                  |
| `GET`  | `/api/stocks`               | Defense stock universe                 |
| `GET`  | `/api/feed`                 | Paginated labeled claims, newest first |
| `GET`  | `/api/feed/stream`          | SSE stream of new claims               |
| `GET`  | `/api/stats`                | Label distribution, top tickers        |
| `POST` | `/api/predict`              | ML model inference on a tweet URL      |
| `GET`  | `/api/accounts`             | Account credibility listing            |
| `GET`  | `/api/accounts/:username`   | Account detail with claim history      |
| `GET`  | `/api/stocks/:ticker/feed`  | Per-stock tweet feed, bot-filtered     |
| `GET`  | `/api/stocks/:ticker/stats` | Stock-level aggregates                 |
| `GET`  | `/api/leaderboard`          | Top grifters and high-signal accounts  |

## Common parameters

### `labels` query parameter

Available on all endpoints except `/api/health`, `/api/stocks`, and `/api/predict`.

| Value      | Table queried             |
|------------|---------------------------|
| `naive`    | `naive_labeled_claims`    |
| `improved` | `improved_labeled_claims` |

Default: `naive`.

Controls which labeled claims table is queried. Every response from an endpoint that accepts this parameter includes a
`"labels"` field confirming which labeler was used.

```bash
curl http://localhost:5000/api/feed?labels=improved
```

## GET /api/feed

Paginated list of labeled claims, newest first.

Query parameters:

| Param    | Default | Description                                      |
|----------|---------|--------------------------------------------------|
| `limit`  | 50      | Max results (capped at 200)                      |
| `offset` | 0       | Pagination offset                                |
| `label`  | none    | Filter: `exaggerated`, `accurate`, `understated` |
| `labels` | `naive` | Labeler: `naive` or `improved`                   |

```bash
curl http://localhost:5000/api/feed?limit=20
curl http://localhost:5000/api/feed?label=exaggerated&labels=improved&limit=10
```

## GET /api/feed/stream

Server-Sent Events endpoint for live claim updates. Emits new `LabeledClaim` objects as JSON whenever the pipeline
inserts a new row. Sends a heartbeat comment every 15 seconds.

Accepts the `labels` query parameter to select which labeler's claims are streamed.

```bash
curl -N http://localhost:5000/api/feed/stream
curl -N http://localhost:5000/api/feed/stream?labels=improved
```

## GET /api/stats

Aggregate statistics: label counts, catalyst type distribution, top tickers. The response includes a `"labels"` field
confirming which labeler was used.

Query parameters:

| Param    | Default | Description                    |
|----------|---------|--------------------------------|
| `labels` | `naive` | Labeler: `naive` or `improved` |

```bash
curl http://localhost:5000/api/stats
curl http://localhost:5000/api/stats?labels=improved
```

## GET /api/stocks

Returns the defense stock universe, all tracked tickers and company names.

## POST /api/predict

Classify a tweet using a trained ML model. Pure text-based inference, no price or news data needed. This is how you
predict exaggerated vs accurate before the 24h price window elapses.

Request body:

```json
{
  "url": "https://x.com/hypetrader/status/123456",
  "model": "classical",
  "labels": "naive"
}
```

| Field    | Required | Default         | Description                                                                                      |
|----------|----------|-----------------|--------------------------------------------------------------------------------------------------|
| `url`    | yes      | --              | Tweet URL. Example: `https://x.com/user/status/123456`                                           |
| `model`  | no       | first available | Model key like `classical/naive_labeler` or bare name like `classical`                           |
| `labels` | no       | `naive`         | `naive` or `improved`. Resolves bare model names (e.g. `classical` -> `classical/naive_labeler`) |

When `model` is a bare name (e.g. `classical`), the `labels` field determines which variant is used:
`classical/naive_labeler` or `classical/improved_labeler`.

Response:

```json
{
  "label": "exaggerated",
  "confidence": 0.8734,
  "model": "classical/naive_labeler",
  "available_models": [
    "baseline/naive_labeler",
    "baseline/improved_labeler",
    "classical/naive_labeler",
    "classical/improved_labeler",
    "neural/naive_labeler",
    "neural/improved_labeler"
  ],
  "account": {
    "username": "hypetrader",
    "naive": {
      "grifter_score": 0.75,
      "grifter_category": "mostly_wrong",
      "total_claims": 12
    },
    "improved": {
      "grifter_score": 0.62,
      "grifter_category": "mixed",
      "total_claims": 12
    }
  }
}
```

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"url": "https://x.com/hypetrader/status/123456", "model": "classical"}'

curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"url": "https://x.com/hypetrader/status/123456", "model": "classical", "labels": "improved"}'
```

## GET /api/accounts

List accounts with credibility scores, sortable and filterable. The response always includes both naive and improved
grifter scores per account.

Query parameters:

| Param          | Default             | Description                                                                                                                                         |
|----------------|---------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| `sort_by`      | depends on `labels` | `naive_grifter_score` (default for naive) or `improved_grifter_score` (default for improved). Also accepts `total_claims`, `username`, `last_seen`. |
| `order`        | `desc`              | Sort order: `asc` or `desc`                                                                                                                         |
| `min_claims`   | 5                   | Minimum total claims to include                                                                                                                     |
| `account_type` | none                | Filter by account type: `human`, `bot`, `garbage`                                                                                                   |
| `limit`        | 50                  | Max results (capped at 200)                                                                                                                         |
| `offset`       | 0                   | Pagination offset                                                                                                                                   |
| `labels`       | `naive`             | Labeler: `naive` or `improved`                                                                                                                      |

```bash
curl http://localhost:5000/api/accounts?min_claims=5&sort_by=naive_grifter_score&order=desc
curl http://localhost:5000/api/accounts?account_type=bot
curl http://localhost:5000/api/accounts?labels=improved&sort_by=improved_grifter_score
```

## GET /api/accounts/:username

Account detail with credibility metrics and claim history. The `labels` parameter controls which labeled claims are
returned.

Query parameters:

| Param    | Default | Description                    |
|----------|---------|--------------------------------|
| `labels` | `naive` | Labeler: `naive` or `improved` |

```bash
curl http://localhost:5000/api/accounts/hypetrader
curl http://localhost:5000/api/accounts/hypetrader?labels=improved
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
| `labels`       | `naive` | Labeler: `naive` or `improved`              |

```bash
curl http://localhost:5000/api/stocks/LMT/feed
curl http://localhost:5000/api/stocks/RTX/feed?exclude_bots=false&labels=improved
```

Returns 404 if the ticker is not in the defense stock universe.

## GET /api/stocks/:ticker/stats

Stock-level aggregates: total claims, label distribution, exaggeration rate, top catalysts, average price change.

Query parameters:

| Param    | Default | Description                    |
|----------|---------|--------------------------------|
| `labels` | `naive` | Labeler: `naive` or `improved` |

```bash
curl http://localhost:5000/api/stocks/LMT/stats
curl http://localhost:5000/api/stocks/LMT/stats?labels=improved
```

Returns 404 if the ticker is not in the defense stock universe.

## GET /api/leaderboard

Top accounts by credibility. Returns accounts with 5+ claims, excluding bots.

Query parameters:

| Param      | Default    | Description                                             |
|------------|------------|---------------------------------------------------------|
| `category` | `grifters` | `grifters` (highest grifter score) or `signal` (lowest) |
| `limit`    | 20         | Max results (capped at 100)                             |
| `labels`   | `naive`    | Labeler: `naive` or `improved`                          |

```bash
curl http://localhost:5000/api/leaderboard
curl http://localhost:5000/api/leaderboard?category=signal&limit=10&labels=improved
```

## GET /api/health

Returns `200` with `{"status": "healthy"}` if the database is reachable, `503` with `{"status": "degraded"}` otherwise.
