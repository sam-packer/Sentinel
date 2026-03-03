"""Tests for news feature extraction."""

from src.features.news_features import extract_news_features, NEWS_FEATURE_NAMES
from tests.fixtures import make_labeled_claim


class TestNewsFeatures:
    def test_basic_extraction(self):
        claim = make_labeled_claim(
            has_catalyst=True, catalyst_type="contract",
            news_headlines=["LMT wins $5B radar contract"],
        )
        # Need to set these on the raw claim before building labeled
        features = extract_news_features(claim)
        assert isinstance(features, dict)

    def test_all_feature_names_present(self):
        claim = make_labeled_claim()
        features = extract_news_features(claim)
        for name in NEWS_FEATURE_NAMES:
            assert name in features, f"Missing feature: {name}"

    def test_catalyst_one_hot(self):
        claim = make_labeled_claim(has_catalyst=True, catalyst_type="earnings")
        features = extract_news_features(claim)
        assert features["has_catalyst"] == 1
        assert features["catalyst_earnings"] == 1
        assert features["catalyst_contract"] == 0

    def test_no_catalyst(self):
        claim = make_labeled_claim(has_catalyst=False, catalyst_type=None)
        features = extract_news_features(claim)
        assert features["has_catalyst"] == 0
        assert all(
            features[f"catalyst_{t}"] == 0
            for t in ["contract", "earnings", "geopolitical", "budget"]
        )

    def test_headline_count(self):
        claim = make_labeled_claim(
            news_headlines=["Headline 1", "Headline 2", "Headline 3"],
        )
        features = extract_news_features(claim)
        assert features["headline_count"] == 3

    def test_empty_headlines(self):
        claim = make_labeled_claim(news_headlines=[])
        features = extract_news_features(claim)
        assert features["headline_count"] == 0
        assert features["avg_headline_sentiment"] == 0.0
