"""Tests for text feature extraction."""

from src.features.text_features import extract_text_features, FEATURE_NAMES
from tests.fixtures import make_raw_claim


class TestTextFeatures:
    def test_basic_extraction(self):
        claim = make_raw_claim(text="$LMT surging! Buy now! 🚀")
        features = extract_text_features(claim)
        assert isinstance(features, dict)
        assert features["exclamation_count"] == 2
        assert features["emoji_bullish_count"] >= 1

    def test_all_feature_names_present(self):
        claim = make_raw_claim()
        features = extract_text_features(claim)
        for name in FEATURE_NAMES:
            assert name in features, f"Missing feature: {name}"

    def test_defense_specific_features(self):
        claim = make_raw_claim(
            text="Pentagon awards $5B contract for F-35 program to Lockheed Martin"
        )
        features = extract_text_features(claim)
        assert features["mentions_contract"] == 1
        assert features["has_specific_program"] == 1
        assert features["dollar_amount_mentioned"] == 1

    def test_geopolitical_features(self):
        claim = make_raw_claim(text="NATO troops deploy near Ukraine border, defense stocks rally")
        features = extract_text_features(claim)
        assert features["mentions_geopolitical"] == 1

    def test_caps_ratio(self):
        claim = make_raw_claim(text="LMT IS GOING TO THE MOON!!!")
        features = extract_text_features(claim)
        assert features["caps_ratio"] > 0.3
        assert features["all_caps_word_count"] >= 2

    def test_specificity_features(self):
        claim = make_raw_claim(text="$LMT hitting $500 by EOW, up 5% today #defense @MilTwit")
        features = extract_text_features(claim)
        assert features["has_percentage"] == 1
        assert features["has_price_target"] == 1
        assert features["has_timeframe"] == 1
        assert features["hashtag_count"] >= 1
        assert features["mention_count"] >= 1

    def test_engagement_features(self):
        claim = make_raw_claim(likes=1000, retweets=500)
        features = extract_text_features(claim)
        assert features["log_likes"] > 0
        assert features["log_retweets"] > 0

    def test_with_spacy(self, spacy_nlp):
        claim = make_raw_claim(text="Lockheed Martin signed a $10B deal with NATO in Germany")
        features = extract_text_features(claim, nlp=spacy_nlp)
        assert features["entity_count"] > 0
