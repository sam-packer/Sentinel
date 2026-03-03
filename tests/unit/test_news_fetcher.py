"""Tests for defense news fetcher."""

import pytest
from datetime import datetime, timezone

from src.news_fetcher import classify_catalyst, _parse_article_age_hours, _filter_by_window


class TestClassifyCatalyst:
    def test_contract_catalyst(self):
        headlines = ["Pentagon awards $5B contract to Lockheed Martin"]
        has, ctype = classify_catalyst(headlines)
        assert has is True
        assert ctype == "contract"

    def test_earnings_catalyst(self):
        headlines = ["RTX beats Q4 earnings estimates, guidance raised"]
        has, ctype = classify_catalyst(headlines)
        assert has is True
        assert ctype == "earnings"

    def test_geopolitical_catalyst(self):
        headlines = ["NATO deploys troops near Ukraine border amid conflict"]
        has, ctype = classify_catalyst(headlines)
        assert has is True
        assert ctype == "geopolitical"

    def test_budget_catalyst(self):
        headlines = ["NDAA defense budget passes Senate with $886B allocation"]
        has, ctype = classify_catalyst(headlines)
        assert has is True
        assert ctype == "budget"

    def test_no_catalyst(self):
        headlines = ["Stock market closes higher on broad gains"]
        has, ctype = classify_catalyst(headlines)
        assert has is False
        assert ctype is None

    def test_empty_headlines(self):
        has, ctype = classify_catalyst([])
        assert has is False
        assert ctype is None

    def test_priority_contract_over_geopolitical(self):
        headlines = ["Pentagon contract awarded amid military conflict escalation"]
        has, ctype = classify_catalyst(headlines)
        assert has is True
        assert ctype == "contract"  # contract > geopolitical


class TestParseArticleAgeHours:
    def test_article_before_reference(self):
        ref = datetime(2024, 6, 15, 14, 0, tzinfo=timezone.utc)
        age = _parse_article_age_hours("2024-06-15T08:00:00Z", reference=ref)
        assert age == pytest.approx(6.0)

    def test_article_after_reference(self):
        """Article published AFTER the tweet should return positive hours, not 0."""
        ref = datetime(2024, 6, 15, 14, 0, tzinfo=timezone.utc)
        age = _parse_article_age_hours("2024-06-15T17:00:00Z", reference=ref)
        assert age == pytest.approx(3.0)

    def test_exact_match(self):
        ref = datetime(2024, 6, 15, 14, 0, tzinfo=timezone.utc)
        age = _parse_article_age_hours("2024-06-15T14:00:00Z", reference=ref)
        assert age == pytest.approx(0.0)

    def test_invalid_date(self):
        assert _parse_article_age_hours("not a date") is None

    def test_empty_string(self):
        assert _parse_article_age_hours("") is None


class TestFilterByWindow:
    def test_symmetric_window(self):
        """Articles both before and after the tweet within the window should pass."""
        tweet_time = datetime(2024, 6, 15, 14, 0, tzinfo=timezone.utc)
        articles = [
            {"title": "Before", "date": "2024-06-15T10:00:00Z"},  # 4h before
            {"title": "After", "date": "2024-06-15T18:00:00Z"},   # 4h after
            {"title": "Too old", "date": "2024-06-12T14:00:00Z"}, # 72h before
        ]
        result = _filter_by_window(articles, tweet_time, window_hours=6)
        assert len(result) == 2
        titles = [a["title"] for a in result]
        assert "Before" in titles
        assert "After" in titles
        assert "Too old" not in titles

    def test_article_after_tweet_outside_window_excluded(self):
        """Article published well after the tweet should be excluded."""
        tweet_time = datetime(2024, 6, 15, 14, 0, tzinfo=timezone.utc)
        articles = [
            {"title": "Way after", "date": "2024-06-18T14:00:00Z"},  # 72h after
        ]
        result = _filter_by_window(articles, tweet_time, window_hours=48)
        assert len(result) == 0

    def test_empty_articles(self):
        tweet_time = datetime(2024, 6, 15, 14, 0, tzinfo=timezone.utc)
        result = _filter_by_window([], tweet_time, window_hours=48)
        assert result == []
