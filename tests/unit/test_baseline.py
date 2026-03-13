"""Tests for the naive baseline model."""

from src.models.baseline import MajorityClassModel
from src.models.evaluate import compute_metrics, format_metrics
from src.models.data import prepare_split


class TestMajorityClassModel:
    def test_train_picks_majority(self):
        model = MajorityClassModel()
        texts = ["a", "b", "c", "d", "e"]
        labels = ["accurate", "accurate", "accurate", "exaggerated", "exaggerated"]
        metadata = model.train(texts, labels)
        assert metadata["majority_class"] == "accurate"
        assert metadata["total_samples"] == 5

    def test_predict_returns_majority(self):
        model = MajorityClassModel()
        model.train(["a", "b", "c"], ["exaggerated", "exaggerated", "accurate"])
        assert model.predict("anything") == "exaggerated"

    def test_predict_batch(self):
        model = MajorityClassModel()
        model.train(["a", "b", "c"], ["accurate", "accurate", "exaggerated"])
        predictions = model.predict_batch(["x", "y", "z"])
        assert predictions == ["accurate", "accurate", "accurate"]

    def test_predict_raises_without_training(self):
        model = MajorityClassModel()
        try:
            model.predict("test")
            assert False, "Should have raised RuntimeError"
        except RuntimeError:
            pass

    def test_save_and_load(self, tmp_path):
        model = MajorityClassModel()
        model.train(["a", "b", "c"], ["accurate", "accurate", "exaggerated"])

        model.save(tmp_path / "baseline")

        loaded = MajorityClassModel()
        loaded.load(tmp_path / "baseline")
        assert loaded.predict("test") == "accurate"
        assert loaded._class_counts == {"accurate": 2, "exaggerated": 1}  # pylint: disable=protected-access


class TestComputeMetrics:
    def test_perfect_predictions(self):
        predictions = ["accurate", "exaggerated", "accurate"]
        labels = ["accurate", "exaggerated", "accurate"]
        metrics = compute_metrics(predictions, labels)
        assert metrics["accuracy"] == 1.0
        assert metrics["per_class"]["accurate"]["f1"] == 1.0

    def test_all_wrong(self):
        predictions = ["exaggerated", "exaggerated"]
        labels = ["accurate", "accurate"]
        metrics = compute_metrics(predictions, labels)
        assert metrics["accuracy"] == 0.0

    def test_majority_class_baseline(self):
        # Simulates what the baseline model does
        labels = ["accurate"] * 78 + ["exaggerated"] * 22
        predictions = ["accurate"] * 100
        metrics = compute_metrics(predictions, labels)
        assert metrics["accuracy"] == 0.78
        assert metrics["per_class"]["exaggerated"]["recall"] == 0.0
        assert metrics["per_class"]["accurate"]["recall"] == 1.0

    def test_confusion_matrix_structure(self):
        predictions = ["accurate", "exaggerated", "accurate", "accurate"]
        labels = ["accurate", "accurate", "exaggerated", "accurate"]
        metrics = compute_metrics(predictions, labels)
        cm = metrics["confusion_matrix"]
        assert cm["accurate"]["accurate"] == 2
        assert cm["accurate"]["exaggerated"] == 1
        assert cm["exaggerated"]["accurate"] == 1
        assert cm["exaggerated"]["exaggerated"] == 0


class TestFormatMetrics:
    def test_format_returns_string(self):
        metrics = compute_metrics(
            ["accurate", "accurate", "exaggerated"],
            ["accurate", "exaggerated", "exaggerated"],
        )
        output = format_metrics(metrics)
        assert "Accuracy:" in output
        assert "Confusion matrix" in output


class TestPrepareSplit:
    def test_filters_understated(self):
        claims = [
            {"text": "a", "label": "accurate"},
            {"text": "b", "label": "exaggerated"},
            {"text": "c", "label": "understated"},
            {"text": "d", "label": "accurate"},
            {"text": "e", "label": "accurate"},
        ]
        split = prepare_split(claims, test_size=0.4, seed=42)
        all_labels = split.train_labels + split.test_labels
        assert "understated" not in all_labels
        assert len(all_labels) == 4

    def test_deterministic_split(self):
        claims = [{"text": f"t{i}", "label": "accurate"} for i in range(100)]
        split1 = prepare_split(claims, seed=42)
        split2 = prepare_split(claims, seed=42)
        assert split1.train_texts == split2.train_texts
        assert split1.test_texts == split2.test_texts

    def test_different_seed_different_split(self):
        claims = [{"text": f"t{i}", "label": "accurate"} for i in range(100)]
        split1 = prepare_split(claims, seed=42)
        split2 = prepare_split(claims, seed=99)
        assert split1.train_texts != split2.train_texts
