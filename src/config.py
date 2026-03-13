"""
Configuration module for Sentinel.

Loads application settings from config.yaml and secrets from environment variables.
"""

import logging
import os
import contextvars
from dataclasses import dataclass, field
from pathlib import Path
import yaml
from dotenv import load_dotenv

load_dotenv()

# Context variable for worker ID logging
worker_context = contextvars.ContextVar("worker_id", default=None)


class WorkerLogFilter(logging.Filter):
    """Filter to inject worker ID into log records."""
    def filter(self, record):
        worker_id = worker_context.get()
        if worker_id is not None:
            record.worker_info = f" [Worker {worker_id}]"
        else:
            record.worker_info = ""
        return True


CONFIG_FILE = Path(__file__).parent.parent / "config.yaml"


def _load_yaml_config() -> dict:
    """Load configuration from YAML file."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return yaml.safe_load(f) or {}
    return {}


_yaml_config = _load_yaml_config()


def _get_yaml(section: str, key: str, default=None):
    """Get a value from the YAML config."""
    return _yaml_config.get(section, {}).get(key, default)


def _get_yaml_section(section: str, default=None):
    """Get an entire section from the YAML config."""
    return _yaml_config.get(section, default or {})


def _get_proxies() -> list[str]:
    """Get proxies from YAML or .env."""
    env_proxies = os.getenv("TWITTER_PROXIES", "")
    if env_proxies:
        return [p.strip() for p in env_proxies.split(",") if p.strip()]
    return _get_yaml("twitter", "proxies", []) or []


@dataclass
class TwitterConfig:
    """Twitter/X credentials and settings."""
    db_path: str = field(default_factory=lambda: _get_yaml("twitter", "db_path", "accounts.db"))
    proxies: list[str] = field(default_factory=_get_proxies)


@dataclass
class DatabaseConfig:
    """PostgreSQL database configuration."""
    url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", ""))


@dataclass
class LabelingConfig:
    """Claim labeling thresholds."""
    exaggeration_threshold: float = field(
        default_factory=lambda: _get_yaml("labeling", "exaggeration_threshold", 0.02)
    )
    news_window_hours: int = field(
        default_factory=lambda: _get_yaml("labeling", "news_window_hours", 48)
    )


@dataclass
class AppConfig:
    """Application-level settings."""
    log_level: str = field(
        default_factory=lambda: _get_yaml("logging", "level", "INFO")
    )
    scrape_limit_per_ticker: int = field(
        default_factory=lambda: _get_yaml("scraping", "limit_per_ticker", 50)
    )
    search_timeout: int = field(
        default_factory=lambda: _get_yaml("scraping", "search_timeout", 300)
    )
    port: int = field(
        default_factory=lambda: _get_yaml("app", "port", 5000)
    )
    live_news_fetch: bool = field(
        default_factory=lambda: _get_yaml("app", "live_news_fetch", True)
    )


@dataclass
class Config:
    """Main configuration container for Sentinel."""
    twitter: TwitterConfig = field(default_factory=TwitterConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)
    app: AppConfig = field(default_factory=AppConfig)

    def setup_logging(self) -> logging.Logger:
        """Configure and return the application logger."""
        root = logging.getLogger()
        if root.handlers:
            for handler in root.handlers:
                root.removeHandler(handler)

        logging.basicConfig(
            level=getattr(logging, self.app.log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s%(worker_info)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        for handler in logging.getLogger().handlers:
            handler.addFilter(WorkerLogFilter())

        return logging.getLogger("sentinel")

    def validate(self) -> list[str]:
        """Validate configuration, return list of error messages."""
        errors = []

        if not CONFIG_FILE.exists():
            errors.append(f"Config file not found: {CONFIG_FILE}")

        if not self.database.url:
            errors.append("DATABASE_URL is required in .env")

        return errors


# Global configuration instance
config = Config()
