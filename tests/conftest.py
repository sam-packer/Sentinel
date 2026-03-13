"""
Shared pytest fixtures for Sentinel tests.

Provides sample data, mock external services, and configuration fixtures.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.fixtures import make_raw_claim, make_labeled_claim, make_defense_claims


# =============================================================================
# Core Fixtures
# =============================================================================

@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory."""
    return tmp_path


# =============================================================================
# Sample Data Fixtures
# =============================================================================

@pytest.fixture
def sample_raw_claim() -> "RawClaim":
    """Single sample RawClaim."""
    return make_raw_claim()


@pytest.fixture
def sample_labeled_claim() -> "LabeledClaim":
    """Single sample LabeledClaim."""
    return make_labeled_claim()


@pytest.fixture
def defense_claims() -> list:
    """Batch of varied defense stock claims."""
    return make_defense_claims(count=5)


@pytest.fixture
def sample_config_yaml(tmp_path):
    """Create a sample config.yaml file."""
    config_path = tmp_path / "config.yaml"
    config_content = """
scraping:
  limit_per_ticker: 50
  search_timeout: 300

labeling:
  exaggeration_threshold: 0.02
  news_window_hours: 48

app:
  port: 5000
  live_news_fetch: true

logging:
  level: DEBUG
"""
    config_path.write_text(config_content)
    return config_path


# =============================================================================
# Mock External Services
# =============================================================================

@pytest.fixture
def mock_yfinance():
    """Mock yfinance.Ticker for price fetcher tests."""
    with patch("yfinance.Ticker") as mock_ticker_class:
        mock_ticker = MagicMock()

        mock_history = MagicMock()
        mock_history.empty = False

        mock_row = MagicMock()
        mock_row.__getitem__ = lambda self, key: {
            "Close": 450.0,
            "Volume": 1000000,
        }.get(key, 0)

        mock_history.iloc.__getitem__ = lambda idx: mock_row
        mock_history.__len__ = lambda self: 10
        mock_history.index = MagicMock()
        mock_history.index.__sub__ = lambda self, other: MagicMock()

        mock_ticker.history.return_value = mock_history
        mock_ticker.news = [
            {"title": "LMT wins Pentagon contract", "link": "https://example.com/1", "publisher": "Reuters"},
        ]

        mock_ticker_class.return_value = mock_ticker
        yield mock_ticker_class


@pytest.fixture
def mock_twscrape_api():
    """Mock twscrape.API for scraper tests."""
    with patch("twscrape.API") as mock_api_class:
        mock_api = MagicMock()
        mock_api.pool.stats = AsyncMock(
            return_value={"active": 3, "total": 5, "locked": 2}
        )
        mock_api.pool.reset_locks = AsyncMock()

        async def mock_search(*args, **kwargs):
            for i in range(3):
                mock_tweet = MagicMock()
                mock_tweet.id = 1234567890 + i
                mock_tweet.rawContent = f"$LMT looking bullish! Defense stocks rally #{i}"
                mock_tweet.user = MagicMock(username=f"user{i}", displayname=f"User {i}")
                mock_tweet.date = datetime.now()
                mock_tweet.likeCount = 100
                mock_tweet.retweetCount = 50
                mock_tweet.replyCount = 10
                mock_tweet.viewCount = 1000
                mock_tweet.lang = "en"
                mock_tweet.hashtags = ["defense"]
                yield mock_tweet

        mock_api.search = mock_search
        mock_api_class.return_value = mock_api
        yield mock_api_class


@pytest.fixture
def mock_ddgs():
    """Mock DuckDuckGo search."""
    with patch("ddgs.DDGS") as mock_ddgs_class:
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.news.return_value = [
            {
                "title": "Lockheed Martin wins $2B radar contract",
                "body": "Pentagon awards major radar system deal.",
                "url": "https://example.com/lmt-contract",
                "source": "Reuters",
                "date": "2024-06-15",
            },
        ]
        mock_ddgs_class.return_value = mock_ddgs_instance
        yield mock_ddgs_class


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set mock environment variables for testing."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost:5432/sentinel_test")
