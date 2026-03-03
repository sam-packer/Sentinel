"""Tests for defense news fetcher."""

from src.news_fetcher import classify_catalyst


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
