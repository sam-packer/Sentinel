"""
Flask application factory for Sentinel API.

Creates and configures the Flask app with CORS, database connection,
and API route registration.
"""

import logging

from flask import Flask
from flask_cors import CORS

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

    # Store database URL in app config
    if database_url is None:
        from ..config import config
        database_url = config.database.url

    app.config["DATABASE_URL"] = database_url

    # Register API routes
    from .routes import api_bp
    app.register_blueprint(api_bp, url_prefix="/api")

    logger.info("Sentinel Flask app created")
    return app
