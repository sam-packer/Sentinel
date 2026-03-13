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
  "model": "baseline"
}
```

`model` is optional. If omitted, uses the first available trained model. Trained models are loaded into memory at
server startup.

Response:

```json
{
  "label": "accurate",
  "model": "baseline",
  "available_models": [
    "baseline"
  ]
}
```

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "$LMT to the moon!", "model": "baseline"}'
```

## GET /api/health

Returns `200` with `{"status": "healthy"}` if the database is reachable, `503` with `{"status": "degraded"}` otherwise.
