"""
API route handlers for Sentinel.

Endpoints:
  POST /api/analyze     — Analyze a tweet, return label + confidence
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
from ..data.stocks import DEFENSE_STOCKS, TICKER_NAMES, resolve_ticker
from ..data.labeler import label_claim, parse_direction
from ..data.models import RawClaim

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


@api_bp.route("/analyze", methods=["POST"])
def analyze():
    """Analyze a single tweet text.

    Body: { "tweet_text": str, "ticker": str? }
    Returns: label, confidence, probabilities, ticker, etc.
    """
    data = request.get_json()
    if not data or "tweet_text" not in data:
        return jsonify({"error": "tweet_text is required"}), 400

    tweet_text = data["tweet_text"]
    ticker = data.get("ticker")

    # Resolve ticker from text if not provided
    if not ticker:
        ticker = resolve_ticker(tweet_text)

    if not ticker:
        return jsonify({"error": "No defense stock ticker found in text"}), 400

    company = TICKER_NAMES.get(ticker, ticker)
    claimed = parse_direction(tweet_text)

    # Build a minimal RawClaim for labeling
    raw = RawClaim(
        tweet_id=0,
        text=tweet_text,
        username="api_user",
        created_at=datetime.utcnow(),
        likes=0,
        retweets=0,
        ticker=ticker,
        company_name=company,
    )

    # Fetch live price and news if enabled
    try:
        from ..config import config
        if config.app.live_news_fetch:
            from ..price_fetcher import PriceFetcher
            from ..news_fetcher import fetch_news_for_claim, classify_catalyst
            import asyncio

            pf = PriceFetcher()
            move = pf.get_price_change(ticker, raw.created_at, hours=24)
            raw.price_at_tweet = move.price_at
            raw.price_24h_later = move.price_after
            raw.price_change_pct = move.change_pct

            # Fetch news synchronously
            loop = asyncio.new_event_loop()
            try:
                articles = loop.run_until_complete(
                    fetch_news_for_claim(ticker, company, raw.created_at)
                )
                raw.news_headlines = [a["title"] for a in articles if a.get("title")]
                raw.has_catalyst, raw.catalyst_type = classify_catalyst(raw.news_headlines)
            finally:
                loop.close()
    except Exception as e:
        logger.warning(f"Live data fetch failed: {e}")

    labeled = label_claim(raw)

    return jsonify({
        "label": labeled.label,
        "confidence": round(1.0 - labeled.exaggeration_score, 3),
        "probabilities": {
            "exaggerated": round(labeled.exaggeration_score, 3),
            "accurate": round(1.0 - labeled.exaggeration_score, 3) if labeled.label == "accurate" else 0.0,
            "understated": 0.0,
        },
        "ticker": labeled.ticker,
        "company_name": labeled.company_name,
        "claimed_direction": labeled.claimed_direction,
        "actual_direction": labeled.actual_direction,
        "has_catalyst": labeled.has_catalyst,
        "catalyst_type": labeled.catalyst_type,
        "news_headlines": labeled.news_headlines,
        "price_change_24h": labeled.price_change_pct,
        "explanation": _build_explanation(labeled),
    })


def _build_explanation(labeled) -> str:
    """Build a human-readable explanation of the label."""
    parts = []

    parts.append(
        f"Claim direction: {labeled.claimed_direction}, "
        f"actual: {labeled.actual_direction}."
    )

    if labeled.price_change_pct is not None:
        parts.append(f"24h price move: {labeled.price_change_pct:+.2f}%.")

    if labeled.has_catalyst:
        parts.append(f"News catalyst: {labeled.catalyst_type}.")
    else:
        parts.append("No news catalyst found.")

    if labeled.label == "exaggerated":
        parts.append("The claim overstates the move or lacks supporting evidence.")
    elif labeled.label == "understated":
        parts.append("The actual move was larger than the claim suggests.")
    else:
        parts.append("The claim aligns with the actual price movement.")

    return " ".join(parts)
