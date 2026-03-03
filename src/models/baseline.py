"""
Naive baseline classifier for Sentinel.

Always predicts the majority class. Used as a floor for comparison.
Implements the sklearn interface (fit, predict, predict_proba).
"""

import numpy as np
from collections import Counter


class MajorityClassifier:
    """Predicts the most common label in the training set."""

    def __init__(self):
        self.majority_class: str | None = None
        self.class_distribution: dict[str, float] = {}
        self.classes_: list[str] = []

    def fit(self, X, y: list[str]) -> "MajorityClassifier":
        """Fit by finding the majority class.

        Args:
            X: Ignored (present for sklearn interface compatibility).
            y: List of label strings.

        Returns:
            self
        """
        counts = Counter(y)
        self.majority_class = counts.most_common(1)[0][0]
        total = sum(counts.values())
        self.classes_ = sorted(counts.keys())
        self.class_distribution = {
            cls: counts[cls] / total for cls in self.classes_
        }
        return self

    def predict(self, X) -> np.ndarray:
        """Predict majority class for all inputs.

        Args:
            X: Input features (ignored).

        Returns:
            Array of majority class predictions.
        """
        n = len(X) if hasattr(X, "__len__") else X.shape[0]
        return np.array([self.majority_class] * n)

    def predict_proba(self, X) -> np.ndarray:
        """Return class probabilities (based on training distribution).

        Args:
            X: Input features (ignored).

        Returns:
            (n_samples, n_classes) array of probabilities.
        """
        n = len(X) if hasattr(X, "__len__") else X.shape[0]
        probs = [self.class_distribution.get(cls, 0.0) for cls in self.classes_]
        return np.tile(probs, (n, 1))

    def score(self, X, y: list[str]) -> float:
        """Return accuracy."""
        preds = self.predict(X)
        return float(np.mean(preds == np.array(y)))
