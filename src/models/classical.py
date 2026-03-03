"""
Classical ML classifier for Sentinel.

TF-IDF (ngram 1-2, 10k features) on tweet text + StandardScaled linguistic/news
scalar features via ColumnTransformer -> LogisticRegression or LinearSVC.
GridSearchCV over C values, picking best by macro F1.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

logger = logging.getLogger("sentinel.models.classical")


class ClassicalClassifier:
    """TF-IDF + scalar features -> LogisticRegression/SVM classifier."""

    def __init__(self, model_type: str = "logreg"):
        """Initialize classifier.

        Args:
            model_type: "logreg" for LogisticRegression, "svm" for LinearSVC.
        """
        self.model_type = model_type
        self.pipeline: Pipeline | None = None
        self.feature_names_: list[str] = []
        self._scalar_feature_names: list[str] = []

    def fit(
        self,
        texts: list[str],
        scalar_features: np.ndarray,
        labels: list[str],
        scalar_feature_names: list[str] | None = None,
    ) -> "ClassicalClassifier":
        """Train the classifier with grid search over C values.

        Args:
            texts: List of tweet text strings.
            scalar_features: (n_samples, n_features) array of engineered features.
            labels: List of label strings.
            scalar_feature_names: Names of scalar features (for feature importance).

        Returns:
            self
        """
        self._scalar_feature_names = scalar_feature_names or []

        # Build transformer: TF-IDF on text + StandardScaler on scalars
        tfidf = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("tfidf", tfidf, 0),  # column 0 = text
                ("scalar", StandardScaler(), list(range(1, scalar_features.shape[1] + 1))),
            ],
            remainder="drop",
        )

        # Combine text and scalar features into single array
        # We'll pass text as column 0 and scalar features as remaining columns
        # Build X as mixed array
        X = self._combine_features(texts, scalar_features)

        if self.model_type == "svm":
            clf = LinearSVC(class_weight="balanced", max_iter=5000, dual="auto")
            param_grid = {"classifier__C": [0.1, 1.0, 10.0]}
        else:
            clf = LogisticRegression(
                class_weight="balanced", max_iter=1000, solver="lbfgs",
            )
            param_grid = {"classifier__C": [0.1, 1.0, 10.0]}

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ])

        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=min(5, len(labels)),
            scoring="f1_macro",
            n_jobs=-1,
            refit=True,
        )
        grid.fit(X, labels)

        self.pipeline = grid.best_estimator_
        logger.info(
            f"Best {self.model_type} C={grid.best_params_['classifier__C']}, "
            f"macro F1={grid.best_score_:.4f}"
        )
        return self

    def predict(self, texts: list[str], scalar_features: np.ndarray) -> np.ndarray:
        """Predict labels for new data.

        Args:
            texts: Tweet text strings.
            scalar_features: Engineered features array.

        Returns:
            Array of predicted label strings.
        """
        if self.pipeline is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        X = self._combine_features(texts, scalar_features)
        return self.pipeline.predict(X)

    def predict_proba(self, texts: list[str], scalar_features: np.ndarray) -> np.ndarray | None:
        """Predict class probabilities (LogisticRegression only).

        Returns None for SVM since it doesn't natively support probabilities.
        """
        if self.pipeline is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        if self.model_type == "svm":
            return None
        X = self._combine_features(texts, scalar_features)
        return self.pipeline.predict_proba(X)

    def get_top_features(self, n: int = 20) -> list[tuple[str, float]]:
        """Get top N most important features by absolute coefficient.

        Returns:
            List of (feature_name, coefficient) tuples, sorted by importance.
        """
        if self.pipeline is None:
            return []

        clf = self.pipeline.named_steps["classifier"]
        preprocessor = self.pipeline.named_steps["preprocessor"]

        try:
            if hasattr(clf, "coef_"):
                coefs = np.abs(clf.coef_).mean(axis=0)  # average over classes
            else:
                return []

            # Get feature names from preprocessor
            try:
                feature_names = preprocessor.get_feature_names_out()
            except Exception:
                feature_names = [f"feature_{i}" for i in range(len(coefs))]

            ranked = sorted(
                zip(feature_names, coefs),
                key=lambda x: x[1],
                reverse=True,
            )
            return [(str(name), float(coef)) for name, coef in ranked[:n]]
        except Exception as e:
            logger.warning(f"Failed to extract top features: {e}")
            return []

    def save(self, path: str) -> None:
        """Save trained model to disk."""
        if self.pipeline is None:
            raise RuntimeError("No model to save.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"pipeline": self.pipeline, "model_type": self.model_type},
            path,
        )
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> "ClassicalClassifier":
        """Load a trained model from disk."""
        data = joblib.load(path)
        self.pipeline = data["pipeline"]
        self.model_type = data["model_type"]
        logger.info(f"Model loaded from {path}")
        return self

    @staticmethod
    def _combine_features(texts: list[str], scalar_features: np.ndarray) -> np.ndarray:
        """Combine text and scalar features into a single structured array.

        The ColumnTransformer expects a 2D array-like where column 0 is text
        and remaining columns are scalar features.
        """
        n = len(texts)
        # Create object array to hold mixed types
        X = np.empty((n, 1 + scalar_features.shape[1]), dtype=object)
        for i, text in enumerate(texts):
            X[i, 0] = text
        X[:, 1:] = scalar_features
        return X
