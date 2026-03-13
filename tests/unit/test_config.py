"""Tests for Sentinel configuration."""

import logging

from src.config import (
    Config, LabelingConfig, AppConfig,
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


class TestSentinelConfig:
    def test_default_config(self):
        cfg = Config()
        assert cfg.app.port == 5000
        assert cfg.labeling.exaggeration_threshold == 0.02
        assert cfg.labeling.news_window_hours == 48

    def test_validation_no_db(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "")
        cfg = Config()
        cfg.database.url = ""
        errors = cfg.validate()
        assert any("DATABASE_URL" in e for e in errors)

    def test_labeling_config_defaults(self):
        lc = LabelingConfig()
        assert lc.exaggeration_threshold == 0.02
        assert lc.news_window_hours == 48

    def test_app_config_defaults(self):
        ac = AppConfig()
        assert ac.live_news_fetch is True


class TestConfigValidation:
    def test_validate_passes_with_db_url(self):
        cfg = Config()
        cfg.database.url = "postgresql://test"
        errors = cfg.validate()
        db_errors = [e for e in errors if "DATABASE_URL" in e]
        assert len(db_errors) == 0


class TestWorkerLogFilter:
    def test_filter_adds_worker_info(self):
        filter_instance = WorkerLogFilter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="test message", args=(), exc_info=None,
        )

        filter_instance.filter(record)
        assert record.worker_info == ""  # pylint: disable=no-member

        worker_context.set(5)
        filter_instance.filter(record)
        assert record.worker_info == " [Worker 5]"  # pylint: disable=no-member
        worker_context.set(None)
