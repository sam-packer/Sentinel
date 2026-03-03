"""Tests for data models."""

from datetime import datetime

from src.data.models import RawClaim, LabeledClaim
from tests.fixtures import make_raw_claim, make_labeled_claim


class TestRawClaim:
    def test_creation(self):
        claim = make_raw_claim()
        assert claim.tweet_id == 1234567890
        assert claim.ticker == "LMT"
        assert claim.company_name == "Lockheed Martin"

    def test_defaults(self):
        claim = RawClaim(
            tweet_id=1, text="test", username="u",
            created_at=datetime.now(), likes=0, retweets=0,
            ticker="BA", company_name="Boeing",
        )
        assert claim.price_at_tweet is None
        assert claim.news_headlines == []
        assert claim.has_catalyst is False
        assert claim.catalyst_type is None


class TestLabeledClaim:
    def test_from_raw(self):
        raw = make_raw_claim()
        labeled = LabeledClaim.from_raw(
            raw,
            label="exaggerated",
            claimed_direction="up",
            actual_direction="neutral",
            exaggeration_score=0.7,
            news_summary="No catalyst found",
        )
        assert labeled.label == "exaggerated"
        assert labeled.ticker == raw.ticker
        assert labeled.tweet_id == raw.tweet_id

    def test_make_labeled_claim(self):
        claim = make_labeled_claim(label="understated")
        assert claim.label == "understated"
        assert isinstance(claim.exaggeration_score, float)
