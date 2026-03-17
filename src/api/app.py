# This file was developed with the assistance of Claude Code and Opus 4.6.

"""
Flask application factory for Sentinel API.

Creates and configures the Flask app with CORS, database connection,
and API route registration.
"""

import logging

from flask import Flask
from flask_cors import CORS

from .limiter import limiter

logger = logging.getLogger("sentinel.api")


def create_app(database_url: str | None = None) -> Flask:
    """Create and configure the Flask application.

    Args:
        database_url: PostgreSQL connection string. Falls back to config/env if None.

    Returns:
        Configured Flask app.
    """
    app = Flask(__name__)

    # Enable CORS for all origins (SvelteKit dev server on different port)
    CORS(app)

    # Initialize rate limiter with the app (in-memory storage)
    limiter.init_app(app)

    # Store database URL in app config
    if database_url is None:
        from ..config import config
        database_url = config.database.url

    app.config["DATABASE_URL"] = database_url

    # Load trained models into memory
    from ..models import MODEL_REGISTRY, load_model

    loaded_models = {}
    for name in MODEL_REGISTRY:
        for labels in ("naive", "improved"):
            key = f"{name}/{labels}_labeler"
            try:
                loaded_models[key] = load_model(name, labels=labels)
                logger.info(f"Loaded model: {key}")
            except FileNotFoundError:
                logger.info(f"No trained model for '{key}', skipping")
            except ImportError as e:
                logger.warning(f"Cannot load '{key}': missing dependency ({e})")

        # Fallback: try loading from flat directory (backward compat)
        if not any(k.startswith(f"{name}/") for k in loaded_models):
            try:
                loaded_models[name] = load_model(name)
                logger.info(f"Loaded model: {name} (legacy flat path)")
            except (FileNotFoundError, ImportError):
                pass
    app.config["MODELS"] = loaded_models

    # Register API routes
    from .routes import api_bp
    app.register_blueprint(api_bp, url_prefix="/api")

    logger.info("Sentinel Flask app created")
    return app
