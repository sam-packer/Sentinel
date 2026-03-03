"""Tests for majority class baseline model."""

import numpy as np

from src.models.baseline import MajorityClassifier


class TestMajorityClassifier:
    def test_fit_and_predict(self):
        clf = MajorityClassifier()
        labels = ["exaggerated"] * 10 + ["accurate"] * 5 + ["understated"] * 2
        clf.fit(None, labels)
        assert clf.majority_class == "exaggerated"

        preds = clf.predict(np.zeros((3, 1)))
        assert list(preds) == ["exaggerated"] * 3

    def test_predict_proba(self):
        clf = MajorityClassifier()
        labels = ["accurate"] * 6 + ["exaggerated"] * 3 + ["understated"] * 1
        clf.fit(None, labels)

        probs = clf.predict_proba(np.zeros((2, 1)))
        assert probs.shape == (2, 3)
        assert np.allclose(probs[0].sum(), 1.0)

    def test_score(self):
        clf = MajorityClassifier()
        labels = ["accurate"] * 8 + ["exaggerated"] * 2
        clf.fit(None, labels)

        score = clf.score(np.zeros((10, 1)), labels)
        assert score == 0.8  # 8 out of 10 are "accurate"
