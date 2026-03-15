"""Tests for predict_proba model enhancement."""

import pytest

from src.models.classical import ClassicalModel


# Use minimal trials in tests
TEST_TRIALS = 5


class TestClassicalPredictProba:
    """Tests for ClassicalModel.predict_proba."""

    @pytest.fixture
    def trained_model(self):
        """Create a small trained classical model."""
        model = ClassicalModel()
        texts = [
            "$LMT to the moon! 🚀🚀🚀",
            "MASSIVE rally incoming for $RTX!!!",
            "$NOC pump incoming, load up now!",
            "RTX awarded $2B Pentagon contract",
            "Lockheed reports steady Q4 earnings",
            "General Dynamics maintains guidance",
        ] * 5  # repeat for minimum viable training
        labels = ["exaggerated", "exaggerated", "exaggerated", "accurate", "accurate", "accurate"] * 5
        model.train(texts, labels, n_trials=TEST_TRIALS)
        return model

    def test_returns_dict_with_label_and_confidence(self, trained_model):
        result = trained_model.predict_proba("$LMT to the moon! 🚀")
        assert "label" in result
        assert "confidence" in result
        assert result["label"] in ("exaggerated", "accurate")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_confidence_is_float(self, trained_model):
        result = trained_model.predict_proba("RTX contract win")
        assert isinstance(result["confidence"], float)

    def test_high_confidence_on_obvious_text(self, trained_model):
        result = trained_model.predict_proba("$LMT MASSIVE PUMP 🚀🚀🚀 TO THE MOON!!!")
        assert result["confidence"] > 0.5

    def test_raises_when_not_trained(self):
        model = ClassicalModel()
        with pytest.raises(RuntimeError, match="Model has not been trained"):
            model.predict_proba("test")
