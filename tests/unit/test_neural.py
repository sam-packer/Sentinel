"""Tests for the neural (BERTweet) model.

These tests use minimal trials and a tiny dataset to verify the
pipeline works end-to-end without requiring a GPU or long runtime.
The real training happens on GPU with 50 Optuna trials.
"""

import pytest

from src.models.neural import NeuralModel, LABEL2ID, ID2LABEL

# Minimal trials — just verify the pipeline connects
TEST_TRIALS = 2


class TestLabelMapping:
    def test_label2id_consistent(self):
        assert LABEL2ID["accurate"] == 0
        assert LABEL2ID["exaggerated"] == 1

    def test_id2label_inverse(self):
        for label, idx in LABEL2ID.items():
            assert ID2LABEL[idx] == label


class TestNeuralModel:
    def _make_training_data(self):
        """Tiny dataset for pipeline verification."""
        texts = [
            "$LMT to the moon! 🚀🚀🚀 MASSIVE!!!",
            "$RTX CRASHING HARD dump now 📉📉📉",
            "$NOC 🚀🚀🚀 INSANE rally incoming!!!",
            "$LMT MOONING!!! Huge breakout 🔥🔥🔥",
            "$GD crash dump everything 💀💀💀!!!",
            "$RTX 🚀 MASSIVE surge!!!! Crazy!!!",
            "Lockheed Martin reported Q3 earnings today",
            "$RTX up 1.2% on contract news",
            "Northrop awarded $2B contract per DoD release",
            "$LMT holding steady after Pentagon announcement",
            "General Dynamics sees modest gains this quarter",
            "RTX earnings beat estimates by 3 cents",
            "$NOC flat after budget discussions",
            "Defense sector mixed on new appropriations",
            "$GD trades sideways on light volume",
        ]
        labels = ["exaggerated"] * 6 + ["accurate"] * 9
        return texts, labels

    @pytest.mark.slow
    def test_train_returns_metadata(self):
        model = NeuralModel()
        texts, labels = self._make_training_data()
        meta = model.train(texts, labels, n_trials=TEST_TRIALS)

        assert "tuning" in meta
        assert meta["tuning"]["n_trials"] == TEST_TRIALS
        assert meta["tuning"]["objective"] == "f1_macro"
        assert "best_params" in meta["tuning"]
        assert meta["model_name"] == "vinai/bertweet-base"
        assert meta["train_size"] == 15

    @pytest.mark.slow
    def test_predict_returns_valid_label(self):
        model = NeuralModel()
        texts, labels = self._make_training_data()
        model.train(texts, labels, n_trials=TEST_TRIALS)
        prediction = model.predict("$LMT 🚀🚀🚀 MOONING!!!")
        assert prediction in ("exaggerated", "accurate")

    @pytest.mark.slow
    def test_predict_batch(self):
        model = NeuralModel()
        texts, labels = self._make_training_data()
        model.train(texts, labels, n_trials=TEST_TRIALS)
        predictions = model.predict_batch(["test one", "test two"])
        assert len(predictions) == 2
        assert all(p in ("exaggerated", "accurate") for p in predictions)

    def test_predict_raises_without_training(self):
        model = NeuralModel()
        with pytest.raises(RuntimeError):
            model.predict("test")

    @pytest.mark.slow
    def test_save_and_load(self, tmp_path):
        model = NeuralModel()
        texts, labels = self._make_training_data()
        model.train(texts, labels, n_trials=TEST_TRIALS)

        save_dir = tmp_path / "neural"
        model.save(save_dir)

        assert (save_dir / "model").exists()
        assert (save_dir / "tokenizer").exists()
        assert (save_dir / "model.json").exists()

        loaded = NeuralModel()
        loaded.load(save_dir)

        test_text = "$LMT 🚀🚀🚀 MOONING!!!"
        assert model.predict(test_text) == loaded.predict(test_text)
