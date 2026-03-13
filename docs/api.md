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
| `POST` | `/api/analyze`     | Label a tweet using current price data |

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

## POST /api/analyze

Analyze a single tweet text against current market data.

Request body:

```json
{
  "tweet_text": "$LMT to the moon!",
  "ticker": "LMT"
}
```

`ticker` is optional and will be resolved from the text if not provided.

When `live_news_fetch` is enabled in config, the endpoint fetches current price and news before labeling.

Response:

```json
{
  "label": "exaggerated",
  "confidence": 0.4,
  "ticker": "LMT",
  "company_name": "Lockheed Martin",
  "claimed_direction": "up",
  "actual_direction": "neutral",
  "has_catalyst": false,
  "catalyst_type": null,
  "news_headlines": [],
  "price_change_24h": 0.44,
  "explanation": "Claim direction: up, actual: neutral. 24h price move: +0.44%. No news catalyst found. The claim overstates the move or lacks supporting evidence."
}
```

```bash
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"tweet_text": "$LMT to the moon!", "ticker": "LMT"}'
```

## GET /api/health

Returns `200` with `{"status": "healthy"}` if the database is reachable, `503` with `{"status": "degraded"}` otherwise.
