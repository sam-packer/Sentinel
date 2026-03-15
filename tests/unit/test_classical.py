"""Tests for the classical ML model (TF-IDF + Logistic Regression)."""

import pytest

from src.models.classical import ClassicalModel

# Use minimal trials in tests — just enough to verify the pipeline works
TEST_TRIALS = 5


class TestClassicalModel:
    def _make_training_data(self):
        """Create synthetic training data with clear signal."""
        texts = [
            # Exaggerated-style tweets
            "$LMT to the moon! 🚀🚀🚀 MASSIVE!!!",
            "$RTX CRASHING HARD dump now 📉📉📉",
            "$NOC 🚀🚀🚀 INSANE rally incoming!!!",
            "$LMT MOONING!!! Huge breakout 🔥🔥🔥",
            "$GD crash dump everything 💀💀💀!!!",
            "$RTX 🚀 MASSIVE surge!!!! Crazy!!!",
            "$NOC is CRASHING plummeting hard 📉📉",
            "$LMT going to MOON 🚀 insane pump!!!",
            "$BA about to EXPLODE higher 🚀🚀",
            "$HII dump this garbage stock now!!!",
            # Accurate-style tweets
            "Lockheed Martin reported Q3 earnings today",
            "$RTX up 1.2% on contract news",
            "Northrop awarded $2B contract per DoD release",
            "$LMT holding steady after Pentagon announcement",
            "General Dynamics sees modest gains this quarter",
            "RTX earnings beat estimates by 3 cents",
            "$NOC flat after budget discussions",
            "Defense sector mixed on new appropriations",
            "$GD trades sideways on light volume",
            "$LMT up slightly on broad market rally",
            "RTX reports steady backlog growth",
            "Northrop guidance in line with expectations",
            "Boeing defense revenue in line with street",
            "SAIC wins routine IT services contract",
            "Leidos quarterly results meet expectations",
        ]
        labels = ["exaggerated"] * 10 + ["accurate"] * 15
        return texts, labels

    def test_train_returns_metadata(self):
        model = ClassicalModel()
        texts, labels = self._make_training_data()
        meta = model.train(texts, labels, n_trials=TEST_TRIALS)

        assert "tuning" in meta
        assert meta["tuning"]["n_trials"] == TEST_TRIALS
        assert meta["tuning"]["objective"] == "f1_macro"
        assert "best_params" in meta["tuning"]
        assert "tfidf" in meta
        assert meta["tfidf"]["vocab_size"] > 0
        assert "interpretability" in meta
        assert "lr_top_exaggerated_words" in meta["interpretability"]
        assert "lr_top_accurate_words" in meta["interpretability"]
        assert meta["train_size"] == 25

    def test_predict_returns_valid_label(self):
        model = ClassicalModel()
        texts, labels = self._make_training_data()
        model.train(texts, labels, n_trials=TEST_TRIALS)
        prediction = model.predict("$LMT 🚀🚀🚀 MOONING!!!")
        assert prediction in ("exaggerated", "accurate")

    def test_predict_batch(self):
        model = ClassicalModel()
        texts, labels = self._make_training_data()
        model.train(texts, labels, n_trials=TEST_TRIALS)
        predictions = model.predict_batch(["test one", "test two"])
        assert len(predictions) == 2
        assert all(p in ("exaggerated", "accurate") for p in predictions)

    def test_predict_raises_without_training(self):
        model = ClassicalModel()
        with pytest.raises(RuntimeError):
            model.predict("test")

    def test_save_and_load(self, tmp_path):
        model = ClassicalModel()
        texts, labels = self._make_training_data()
        model.train(texts, labels, n_trials=TEST_TRIALS)

        save_dir = tmp_path / "classical"
        model.save(save_dir)

        assert (save_dir / "lr.pkl").exists()
        assert (save_dir / "tfidf.pkl").exists()
        assert (save_dir / "model.json").exists()

        loaded = ClassicalModel()
        loaded.load(save_dir)

        test_text = "$LMT 🚀🚀🚀 MOONING!!!"
        assert model.predict(test_text) == loaded.predict(test_text)

    def test_hype_vs_calm_separation(self):
        model = ClassicalModel()
        texts, labels = self._make_training_data()
        model.train(texts, labels, n_trials=TEST_TRIALS)

        hype_preds = model.predict_batch([
            "$LMT MOONING 🚀🚀🚀!!! INSANE!!!",
            "$RTX CRASHING HARD 📉📉 dump!!!",
        ])
        calm_preds = model.predict_batch([
            "Lockheed Martin up 0.5% today",
            "RTX reports quarterly earnings",
        ])

        assert "exaggerated" in hype_preds
        assert "accurate" in calm_preds
