"""
API route handlers for Sentinel.

Endpoints:
  POST /api/predict              — ML model inference on tweet text
  GET  /api/feed                 — Paginated labeled claims (newest first)
  GET  /api/feed/stream          — SSE stream of new claims
  GET  /api/stats                — Aggregate statistics
  GET  /api/stocks               — Defense stock universe
  GET  /api/stocks/<ticker>/feed — Per-stock claim feed
  GET  /api/stocks/<ticker>/stats — Per-stock aggregates
  GET  /api/accounts             — Account listing with credibility scores
  GET  /api/accounts/<username>  — Account detail + claim history
  GET  /api/leaderboard          — Top grifters / best signal accounts
  GET  /api/health               — Health check
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


def _serialize_account(account) -> dict:
    """Convert an Account dataclass to a JSON-safe dict."""
    return {
        "username": account.username,
        "account_type": account.account_type,
        "classification_reason": account.classification_reason,
        "total_claims": account.total_claims,
        "exaggerated_count": account.exaggerated_count,
        "accurate_count": account.accurate_count,
        "understated_count": account.understated_count,
        "grifter_score": account.grifter_score,
        "grifter_category": _grifter_category(account.grifter_score),
        "first_seen": account.first_seen.isoformat() if account.first_seen else None,
        "last_seen": account.last_seen.isoformat() if account.last_seen else None,
        "classified_at": account.classified_at.isoformat() if account.classified_at else None,
    }


def _grifter_category(score) -> str:
    """Map a grifter score to a human-readable category."""
    if score is None:
        return "unrated"
    if score < 0.2:
        return "high_signal"
    if score < 0.5:
        return "normal"
    if score < 0.8:
        return "mostly_wrong"
    return "grifter"


def _get_db() -> SentinelDB:
    """Get database instance, creating if needed."""
    if "sentinel_db" not in current_app.extensions:
        db = SentinelDB(current_app.config["DATABASE_URL"])
        db.connect()
        current_app.extensions["sentinel_db"] = db
    return current_app.extensions["sentinel_db"]


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


@api_bp.route("/accounts", methods=["GET"])
def accounts():
    """List accounts with credibility scores.

    Query params:
      sort_by    — field to sort by (default "grifter_score")
      order      — "asc" or "desc" (default "desc")
      min_claims — minimum claims to include (default 5)
      account_type — optional filter: human, bot, garbage
      limit      — max results (default 50)
      offset     — pagination offset (default 0)
    """
    try:
        sort_by = request.args.get("sort_by", "grifter_score")
        order = request.args.get("order", "desc")
        min_claims = int(request.args.get("min_claims", 5))
        limit = min(int(request.args.get("limit", 50)), 200)
        offset = int(request.args.get("offset", 0))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid query parameters"}), 400

    if order not in ("asc", "desc"):
        return jsonify({"error": "Invalid order, must be 'asc' or 'desc'"}), 400

    account_type = request.args.get("account_type")
    if account_type and account_type not in ("human", "bot", "garbage"):
        return jsonify({"error": "Invalid account_type filter. Must be: human, bot, garbage"}), 400

    try:
        db = _get_db()
        account_list = db.get_accounts(
            sort_by=sort_by,
            order=order,
            min_claims=min_claims,
            account_type=account_type,
            limit=limit,
            offset=offset,
        )
        return jsonify({
            "accounts": [_serialize_account(a) for a in account_list],
            "count": len(account_list),
        })
    except Exception as e:
        logger.error(f"Accounts query failed: {e}")
        return jsonify({"error": "Database error"}), 500


@api_bp.route("/accounts/<username>", methods=["GET"])
def account_detail(username):
    """Account detail with claim history.

    Returns account info plus their labeled claims.
    """
    try:
        db = _get_db()
        account = db.get_account(username)
    except Exception as e:
        logger.error(f"Account lookup failed: {e}")
        return jsonify({"error": "Database error"}), 500

    if account is None:
        return jsonify({"error": f"Account '{username}' not found"}), 404

    try:
        claims = db.get_account_claims(username, limit=200, offset=0)
        return jsonify({
            "account": _serialize_account(account),
            "claims": claims,
            "claim_count": len(claims),
        })
    except Exception as e:
        logger.error(f"Account claims query failed: {e}")
        return jsonify({"error": "Database error"}), 500


@api_bp.route("/stocks/<ticker>/feed", methods=["GET"])
def stock_feed(ticker):
    """Per-stock claim feed.

    Query params:
      limit        — max results (default 50)
      offset       — pagination offset (default 0)
      exclude_bots — exclude bot accounts (default true)
    """
    ticker = ticker.upper()
    if ticker not in TICKER_NAMES:
        return jsonify({"error": f"Unknown ticker '{ticker}'"}), 404

    try:
        limit = min(int(request.args.get("limit", 50)), 200)
        offset = int(request.args.get("offset", 0))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid query parameters"}), 400

    exclude_bots = request.args.get("exclude_bots", "true").lower() in ("true", "1", "yes")

    try:
        db = _get_db()
        claims = db.get_stock_feed(
            ticker=ticker,
            limit=limit,
            offset=offset,
            exclude_bots=exclude_bots,
        )
        return jsonify({
            "claims": claims,
            "count": len(claims),
            "ticker": ticker,
        })
    except Exception as e:
        logger.error(f"Stock feed query failed: {e}")
        return jsonify({"error": "Database error"}), 500


@api_bp.route("/stocks/<ticker>/stats", methods=["GET"])
def stock_stats(ticker):
    """Per-stock aggregate statistics."""
    ticker = ticker.upper()
    if ticker not in TICKER_NAMES:
        return jsonify({"error": f"Unknown ticker '{ticker}'"}), 404

    try:
        db = _get_db()
        stock_data = db.get_stock_stats(ticker)
        return jsonify({
            "ticker": ticker,
            "company_name": TICKER_NAMES[ticker],
            **stock_data,
        })
    except Exception as e:
        logger.error(f"Stock stats query failed: {e}")
        return jsonify({"error": "Database error"}), 500


@api_bp.route("/leaderboard", methods=["GET"])
def leaderboard():
    """Account leaderboard by category.

    Query params:
      category — "grifters" or "signal" (default "grifters")
      limit    — max results (default 20)
    """
    category = request.args.get("category", "grifters")
    if category not in ("grifters", "signal"):
        return jsonify({"error": "Invalid category, must be 'grifters' or 'signal'"}), 400

    try:
        limit = min(int(request.args.get("limit", 20)), 100)
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid query parameters"}), 400

    try:
        db = _get_db()
        account_list = db.get_leaderboard(category=category, limit=limit)
        return jsonify({
            "category": category,
            "accounts": account_list,
        })
    except Exception as e:
        logger.error(f"Leaderboard query failed: {e}")
        return jsonify({"error": "Database error"}), 500


@api_bp.route("/predict", methods=["POST"])
def predict():
    """Predict a label for tweet text using a trained ML model.

    Body: { "text": str, "model": str?, "username": str? }
    Returns: { "label": str, "confidence": float, "model": str,
               "available_models": list, "account": dict | null }

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

    result = model.predict_proba(data["text"])

    # Look up account credibility if username provided
    account_info = None
    username = data.get("username")
    if username:
        try:
            db = _get_db()
            account = db.get_account(username)
            if account:
                account_info = {
                    "username": account.username,
                    "grifter_score": account.grifter_score,
                    "grifter_category": _grifter_category(account.grifter_score),
                }
        except Exception as e:
            logger.error(f"Account lookup in predict failed: {e}")

    return jsonify({
        "label": result["label"],
        "confidence": result["confidence"],
        "model": model_name,
        "available_models": available,
        "account": account_info,
    })
