"""
API route handlers for Sentinel.

Endpoints:
  POST /api/predict     — ML model inference on tweet text
  GET  /api/feed        — Paginated labeled claims (newest first)
  GET  /api/feed/stream — SSE stream of new claims
  GET  /api/stats       — Aggregate statistics
  GET  /api/stocks      — Defense stock universe
  GET  /api/health      — Health check
"""

import json
import logging
import time
from datetime import datetime

from flask import Blueprint, Response, current_app, jsonify, request, stream_with_context

from ..data.db import SentinelDB
from ..data.stocks import DEFENSE_STOCKS, TICKER_NAMES

logger = logging.getLogger("sentinel.api.routes")

api_bp = Blueprint("api", __name__)


def _get_db() -> SentinelDB:
    """Get database instance, creating if needed."""
    if not hasattr(current_app, "_sentinel_db"):
        db = SentinelDB(current_app.config["DATABASE_URL"])
        db.connect()
        current_app._sentinel_db = db
    return current_app._sentinel_db


@api_bp.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    try:
        db = _get_db()
        db_ok = db.ping()
    except Exception:
        db_ok = False

    status = "healthy" if db_ok else "degraded"
    code = 200 if db_ok else 503

    return jsonify({
        "status": status,
        "database": "connected" if db_ok else "disconnected",
        "timestamp": datetime.utcnow().isoformat(),
    }), code


@api_bp.route("/stocks", methods=["GET"])
def stocks():
    """Return the defense stock universe."""
    return jsonify({
        "stocks": DEFENSE_STOCKS,
        "ticker_names": TICKER_NAMES,
    })


@api_bp.route("/feed", methods=["GET"])
def feed():
    """Paginated list of labeled claims, newest first.

    Query params:
      limit  — max results (default 50)
      offset — pagination offset (default 0)
      label  — optional filter: exaggerated, accurate, understated
    """
    try:
        limit = min(int(request.args.get("limit", 50)), 200)
        offset = int(request.args.get("offset", 0))
        label_filter = request.args.get("label")
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid query parameters"}), 400

    if label_filter and label_filter not in ("exaggerated", "accurate", "understated"):
        return jsonify({"error": "Invalid label filter"}), 400

    try:
        db = _get_db()
        claims = db.get_feed(limit=limit, offset=offset, label=label_filter)
        return jsonify({"claims": claims, "count": len(claims)})
    except Exception as e:
        logger.error(f"Feed query failed: {e}")
        return jsonify({"error": "Database error"}), 500


@api_bp.route("/feed/stream", methods=["GET"])
def feed_stream():
    """Server-Sent Events endpoint for real-time claim updates.

    The SvelteKit frontend subscribes to this for live updates.
    Emits new LabeledClaim as JSON whenever the scraper pipeline
    inserts a new row.
    """
    def generate():
        db = _get_db()
        last_id = db.get_latest_claim_id()

        while True:
            try:
                new_claims = db.get_claims_since(last_id) if last_id else []

                for claim in new_claims:
                    yield f"data: {json.dumps(claim)}\n\n"
                    last_id = claim.get("tweet_id", last_id)

                # Heartbeat every 15 seconds
                yield f": heartbeat {int(time.time())}\n\n"
                time.sleep(5)

            except GeneratorExit:
                break
            except Exception as e:
                logger.error(f"SSE stream error: {e}")
                yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
                time.sleep(10)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@api_bp.route("/stats", methods=["GET"])
def stats():
    """Aggregate statistics about labeled claims."""
    try:
        db = _get_db()
        return jsonify(db.get_stats())
    except Exception as e:
        logger.error(f"Stats query failed: {e}")
        return jsonify({"error": "Database error"}), 500


@api_bp.route("/predict", methods=["POST"])
def predict():
    """Predict a label for tweet text using a trained ML model.

    Body: { "text": str, "model": str? }
    Returns: { "label": str, "model": str, "available_models": list }

    Unlike /analyze, this runs pure text-based inference with no price
    or news data. The whole point is predicting before the 24h window.
    """
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "text is required"}), 400

    models = current_app.config.get("MODELS", {})
    if not models:
        return jsonify({"error": "No trained models available. Run 'uv run train <model>' first."}), 503

    model_name = data.get("model")
    available = list(models.keys())

    if model_name:
        if model_name not in models:
            return jsonify({
                "error": f"Model '{model_name}' not available",
                "available_models": available,
            }), 404
        model = models[model_name]
    else:
        # Default to first available model
        model_name = available[0]
        model = models[model_name]

    label = model.predict(data["text"])

    return jsonify({
        "label": label,
        "model": model_name,
        "available_models": available,
    })
