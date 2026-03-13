"""Tests for claim labeling logic."""

from src.data.labeler import parse_direction, label_claim, compute_exaggeration_score
from tests.fixtures import make_raw_claim


class TestParseDirection:
    def test_bullish_keywords(self):
        assert parse_direction("$LMT to the moon! 🚀") == "up"
        assert parse_direction("Bullish on RTX, loading up") == "up"
        assert parse_direction("NOC is going to surge") == "up"

    def test_bearish_keywords(self):
        assert parse_direction("$BA is crashing, sell now! 📉") == "down"
        assert parse_direction("Bearish on LMT, buying puts") == "down"
        assert parse_direction("RTX about to dump hard") == "down"

    def test_neutral(self):
        assert parse_direction("Watching $LMT closely today") == "neutral"
        assert parse_direction("NOC reports earnings tomorrow") == "neutral"

    def test_conflicting_signals(self):
        assert parse_direction("$LMT could moon or crash from here 🚀📉") == "neutral"


class TestLabelClaim:
    def test_accurate_bullish_with_move(self):
        raw = make_raw_claim(
            text="$LMT is surging! 🚀",
            price_change_pct=3.5,
            has_catalyst=True, catalyst_type="contract",
        )
        labeled = label_claim(raw)
        assert labeled.label == "accurate"
        assert labeled.claimed_direction == "up"

    def test_exaggerated_bullish_tiny_move(self):
        raw = make_raw_claim(
            text="$LMT TO THE MOON!!! 🚀🚀🚀 INSANE!!!",
            price_change_pct=0.1,
            has_catalyst=False,
        )
        labeled = label_claim(raw)
        assert labeled.label == "exaggerated"

    def test_exaggerated_wrong_direction(self):
        raw = make_raw_claim(
            text="$BA is crashing! Get out! 📉📉",
            price_change_pct=2.5,
            has_catalyst=True, catalyst_type="earnings",
        )
        labeled = label_claim(raw)
        assert labeled.label == "exaggerated"
        assert labeled.claimed_direction == "down"
        assert labeled.actual_direction == "up"

    def test_understated_neutral_big_move(self):
        raw = make_raw_claim(
            text="NOC looks interesting here. Watching.",
            price_change_pct=8.0,
            has_catalyst=True, catalyst_type="geopolitical",
        )
        labeled = label_claim(raw)
        assert labeled.label == "understated"

    def test_accurate_neutral_small_move(self):
        raw = make_raw_claim(
            text="$GD moving sideways. Nothing special.",
            price_change_pct=0.3,
        )
        labeled = label_claim(raw)
        assert labeled.label == "accurate"
        assert labeled.claimed_direction == "neutral"

    def test_no_price_data(self):
        raw = make_raw_claim(
            text="$LMT mooning!!!",
            price_at_tweet=None,
            price_24h_later=None,
            price_change_pct=None,
        )
        labeled = label_claim(raw)
        assert labeled.label == "exaggerated"

    def test_news_summary_from_headlines(self):
        raw = make_raw_claim(
            text="$RTX surging!",
            price_change_pct=4.0,
            news_headlines=["RTX wins $5B missile contract", "Defense stocks rally"],
            has_catalyst=True, catalyst_type="contract",
        )
        labeled = label_claim(raw)
        assert labeled.news_summary == "RTX wins $5B missile contract"


class TestNoNewsPenalty:
    """No-catalyst penalty should only apply when news was actually found."""

    def test_no_news_directional_claim_not_penalized(self):
        """Empty news_headlines means fetch failed — should NOT label as exaggerated."""
        raw = make_raw_claim(
            text="$LMT is surging! 🚀",
            price_change_pct=3.0,
            price_at_tweet=450.0,
            price_24h_later=463.5,
            news_headlines=[],
            has_catalyst=False,
        )
        labeled = label_claim(raw)
        assert labeled.label == "accurate"

    def test_news_found_no_catalyst_still_penalized(self):
        """Non-empty headlines but no catalyst keywords should still be penalized."""
        raw = make_raw_claim(
            text="$LMT is surging! 🚀",
            price_change_pct=3.0,
            price_at_tweet=450.0,
            price_24h_later=463.5,
            news_headlines=["Stock market closes higher on broad gains"],
            has_catalyst=False,
        )
        labeled = label_claim(raw)
        assert labeled.label == "exaggerated"

    def test_exaggeration_score_respects_news_available(self):
        """compute_exaggeration_score should not penalize when news_available=False."""
        score_with_news = compute_exaggeration_score(
            "up", "up", 1.5, False, 0.3, news_available=True,
        )
        score_no_news = compute_exaggeration_score(
            "up", "up", 1.5, False, 0.3, news_available=False,
        )
        assert score_no_news <= score_with_news


class TestExaggerationScore:
    def test_direction_mismatch_high_score(self):
        # Direction mismatch = 0.5, magnitude = 0.5*0.3 = 0.15, catalyst = 0
        score = compute_exaggeration_score("up", "down", -5.0, True, 0.5)
        assert score >= 0.5

    def test_direction_mismatch_intense_language(self):
        # Direction mismatch = 0.5, magnitude = 1.0*0.3 = 0.3, no catalyst = 1.0*0.2
        score = compute_exaggeration_score("up", "down", -5.0, False, 1.0)
        assert score == 1.0

    def test_accurate_low_score(self):
        # Direction match, 5% move justifies language, has catalyst
        # All three components are 0
        score = compute_exaggeration_score("up", "up", 5.0, True, 0.3)
        assert score == 0.0

    def test_no_price_uncertain(self):
        score = compute_exaggeration_score("up", "neutral", None, False, 0.5)
        assert score == 0.5

    def test_components_are_additive(self):
        # No mismatch, small move (1%), no catalyst, moderate intensity
        # direction = 0, magnitude = 0.5 * (1 - 1/5) * 0.3 = 0.12, catalyst = 0.5 * 0.2 = 0.1
        score = compute_exaggeration_score("up", "up", 1.0, False, 0.5)
        assert abs(score - 0.22) < 0.01

    def test_neutral_claim_scores_zero(self):
        # Neutral claims have no magnitude or catalyst gap
        score = compute_exaggeration_score("neutral", "neutral", 0.5, False, 0.0)
        assert score == 0.0
