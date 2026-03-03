"""Tests for Sentinel configuration."""

import logging
from unittest.mock import patch

from src.config import (
    Config, LabelingConfig, NeuralConfig, AppConfig,
    WorkerLogFilter, worker_context, _get_yaml, _get_yaml_section,
)


class TestYamlConfigLoading:
    def test_get_yaml_returns_default_for_missing_key(self):
        result = _get_yaml("nonexistent", "key", "default_value")
        assert result == "default_value"

    def test_get_yaml_section_returns_empty_dict_for_missing(self):
        result = _get_yaml_section("nonexistent")
        assert result == {}


class TestTwitterConfig:
    def test_twitter_config_defaults(self):
        from src.config import TwitterConfig
        config = TwitterConfig()
        assert config.db_path == "accounts.db"

    def test_twitter_config_proxies_from_env(self, monkeypatch):
        monkeypatch.setenv("TWITTER_PROXIES", "http://proxy1:8080,http://proxy2:8080")
        from src.config import _get_proxies
        proxies = _get_proxies()
        assert len(proxies) == 2


class TestOpenAIConfig:
    def test_openai_config_has_model(self):
        from src.config import OpenAIConfig
        config = OpenAIConfig()
        assert config.model is not None
        assert len(config.model) > 0

    def test_openai_config_uses_env_api_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
        from src.config import OpenAIConfig
        config = OpenAIConfig()
        assert config.api_key == "test-key-123"


class TestGoogleConfig:
    def test_google_config_has_model(self):
        from src.config import GoogleConfig
        config = GoogleConfig()
        assert config.model is not None
        assert "gemini" in config.model.lower()


class TestSentinelConfig:
    def test_default_config(self):
        cfg = Config()
        assert cfg.app.port == 5000
        assert cfg.labeling.exaggeration_threshold == 0.02
        assert cfg.labeling.news_window_hours == 48

    def test_validation_no_db(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        cfg = Config()
        cfg.database.url = ""
        errors = cfg.validate()
        assert any("DATABASE_URL" in e for e in errors)

    def test_labeling_config_defaults(self):
        lc = LabelingConfig()
        assert lc.exaggeration_threshold == 0.02
        assert lc.news_window_hours == 48

    def test_neural_config_defaults(self):
        nc = NeuralConfig()
        assert nc.base_model == "ProsusAI/finbert"
        assert nc.batch_size == 16
        assert nc.patience == 3

    def test_app_config_defaults(self):
        ac = AppConfig()
        assert ac.default_model == "neural"
        assert ac.live_news_fetch is True


class TestConfigValidation:
    def test_validate_missing_openai_key(self):
        cfg = Config()
        cfg.app.llm_provider = "openai"
        cfg.openai.api_key = ""
        cfg.database.url = "postgresql://test"
        errors = cfg.validate()
        assert any("OPENAI_API_KEY" in e for e in errors)

    def test_validate_missing_google_key(self):
        cfg = Config()
        cfg.app.llm_provider = "google"
        cfg.google.api_key = ""
        cfg.database.url = "postgresql://test"
        errors = cfg.validate()
        assert any("GOOGLE_API_KEY" in e for e in errors)


class TestWorkerLogFilter:
    def test_filter_adds_worker_info(self):
        filter_instance = WorkerLogFilter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="test message", args=(), exc_info=None,
        )

        filter_instance.filter(record)
        assert record.worker_info == ""

        worker_context.set(5)
        filter_instance.filter(record)
        assert record.worker_info == " [Worker 5]"
        worker_context.set(None)
