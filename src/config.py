"""
Configuration module for Sentinel.

Loads application settings from config.yaml and secrets from environment variables.
"""

import logging
import os
import contextvars
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

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


@dataclass
class TwitterConfig:
    """Twitter/X credentials and settings."""
    db_path: str = field(default_factory=lambda: _get_yaml("twitter", "db_path", "accounts.db"))
    proxies: list[str] = field(default_factory=lambda: _get_proxies())


def _get_proxies() -> list[str]:
    """Get proxies from YAML or .env."""
    env_proxies = os.getenv("TWITTER_PROXIES", "")
    if env_proxies:
        return [p.strip() for p in env_proxies.split(",") if p.strip()]
    return _get_yaml("twitter", "proxies", []) or []


@dataclass
class OpenAIConfig:
    """OpenAI API configuration."""
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model: str = field(default_factory=lambda: _get_yaml("llm", "openai_model", "gpt-4o"))


@dataclass
class GoogleConfig:
    """Google Generative AI configuration."""
    api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    model: str = field(default_factory=lambda: _get_yaml("llm", "google_model", "gemini-2.0-flash"))


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
class NeuralConfig:
    """Neural model training configuration."""
    base_model: str = field(
        default_factory=lambda: _get_yaml("neural", "base_model", "ProsusAI/finbert")
    )
    fallback_model: str = field(
        default_factory=lambda: _get_yaml("neural", "fallback_model", "distilbert-base-uncased")
    )
    news_encoder: str = field(
        default_factory=lambda: _get_yaml("neural", "news_encoder", "sentence-transformers/all-MiniLM-L6-v2")
    )
    max_length: int = field(
        default_factory=lambda: _get_yaml("neural", "max_length", 128)
    )
    batch_size: int = field(
        default_factory=lambda: _get_yaml("neural", "batch_size", 16)
    )
    learning_rate_bert: float = field(
        default_factory=lambda: _get_yaml("neural", "learning_rate_bert", 2.0e-5)
    )
    learning_rate_head: float = field(
        default_factory=lambda: _get_yaml("neural", "learning_rate_head", 1.0e-3)
    )
    max_epochs: int = field(
        default_factory=lambda: _get_yaml("neural", "max_epochs", 10)
    )
    patience: int = field(
        default_factory=lambda: _get_yaml("neural", "patience", 3)
    )


@dataclass
class AppConfig:
    """Application-level settings."""
    llm_provider: Literal["openai", "google"] = field(
        default_factory=lambda: _get_yaml("llm", "provider", "openai")
    )
    log_level: str = field(
        default_factory=lambda: _get_yaml("logging", "level", "INFO")
    )
    spacy_model: str = field(
        default_factory=lambda: _get_yaml("app", "spacy_model", "en_core_web_lg")
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
    default_model: str = field(
        default_factory=lambda: _get_yaml("models", "default", "neural")
    )


@dataclass
class Config:
    """Main configuration container for Sentinel."""
    twitter: TwitterConfig = field(default_factory=TwitterConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    google: GoogleConfig = field(default_factory=GoogleConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)
    neural: NeuralConfig = field(default_factory=NeuralConfig)
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

        if self.app.llm_provider == "openai" and not self.openai.api_key:
            errors.append("OPENAI_API_KEY is required when using OpenAI provider")
        elif self.app.llm_provider == "google" and not self.google.api_key:
            errors.append("GOOGLE_API_KEY is required when using Google provider")

        if not self.database.url:
            errors.append("DATABASE_URL is required in .env")

        return errors


# Global configuration instance
config = Config()
