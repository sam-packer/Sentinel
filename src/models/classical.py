# This file was developed with the assistance of Claude Code and Opus 4.6.

"""
Classical ML model: Optuna-tuned logistic regression with TF-IDF.

Uses TF-IDF on raw tweet text as features — no hand-crafted features
that could leak the labeling rules. Hyperparameters are tuned with
Optuna using stratified 3-fold CV and macro F1 as the objective.

We evaluated XGBoost as well but it scored poorly on high-dimensional
sparse TF-IDF features (0.58 macro F1 vs LR's 0.76). Tree-based
models are a poor fit for bag-of-words representations — they must
split on individual word dimensions, which is inefficient when the
vocabulary is 5,000 terms wide and most entries are zero. LR handles
this natively via a dot product over word weights.
"""

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import optuna
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from . import BaseModel

logger = logging.getLogger("sentinel.models.classical")

# Keep Optuna from flooding the logs — we log our own summaries
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Cross-validation config
N_SPLITS = 3
CV_SEED = 42


def _make_cv() -> StratifiedKFold:
    """Create the cross-validation splitter."""
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_SEED)


def _tune_lr(X, y: np.ndarray, n_trials: int) -> dict:
    """Tune logistic regression hyperparameters with Optuna.

    X can be sparse (from TF-IDF). LR handles sparse matrices natively.

    Search space:
        C: Inverse regularization strength (1e-4 to 1e2, log scale).
        l1_ratio: Mix of L1 vs L2 regularization (0.0–1.0).
    """
    def objective(trial: optuna.Trial) -> float:
        C = trial.suggest_float("C", 1e-4, 1e2, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)

        model = LogisticRegression(
            C=C,
            l1_ratio=l1_ratio,
            solver="saga",
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        )

        scores = cross_val_score(
            model, X, y, cv=_make_cv(), scoring="f1_macro", n_jobs=-1
        )
        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
        study_name="lr_tuning",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    logger.info(
        f"LR tuning complete: best macro F1 = {best.value:.4f} "
        f"(trial {best.number}/{n_trials})"
    )
    logger.info(f"LR best params: {best.params}")

    return {
        "best_params": best.params,
        "best_score": round(best.value, 4),
        "n_trials": n_trials,
        "best_trial_number": best.number,
    }


class ClassicalModel(BaseModel):
    """Optuna-tuned logistic regression with TF-IDF features.

    Uses TF-IDF on raw tweet text as the feature representation.
    No hand-crafted features — the model learns which words and
    bigrams are predictive of exaggeration from the data.
    """

    def __init__(self):
        self._lr: LogisticRegression | None = None
        self._tfidf: TfidfVectorizer | None = None
        self._training_meta: dict = {}

    @property
    def name(self) -> str:
        return "classical"

    def train(
        self,
        texts: list[str],
        labels: list[str],
        *,
        n_trials: int = 200,
        saved_params: dict | None = None,
    ) -> dict:
        # TF-IDF vectorization — learns vocabulary from training data
        self._tfidf = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            stop_words="english",
        )
        X = self._tfidf.fit_transform(texts)
        y = np.array(labels)

        logger.info(f"TF-IDF vocabulary size: {len(self._tfidf.vocabulary_)}")

        # Tune or reuse hyperparameters
        if saved_params is not None:
            lr_tuning = saved_params
            logger.info(f"Using saved params: {lr_tuning['best_params']}")
        else:
            logger.info(f"Tuning LR ({n_trials} trials, {N_SPLITS}-fold CV, macro F1)...")
            lr_tuning = _tune_lr(X, y, n_trials)

        # Train final model on full training set with best params
        self._lr = LogisticRegression(
            **lr_tuning["best_params"],
            solver="saga",
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        )
        self._lr.fit(X, y)

        # Extract interpretability data — top words pushing toward
        # each class, ready for a paper table
        feature_names = self._tfidf.get_feature_names_out()
        coefs = self._lr.coef_[0]

        top_k = 20
        top_exaggerated_idx = coefs.argsort()[-top_k:][::-1]
        top_accurate_idx = coefs.argsort()[:top_k]

        top_exaggerated_words = {
            feature_names[i]: round(float(coefs[i]), 4)
            for i in top_exaggerated_idx
        }
        top_accurate_words = {
            feature_names[i]: round(float(coefs[i]), 4)
            for i in top_accurate_idx
        }

        train_acc = self._lr.score(X, y)
        logger.info(f"Train accuracy: {train_acc:.3f}")
        logger.info(f"Top words → exaggerated: {list(top_exaggerated_words.items())[:5]}")
        logger.info(f"Top words → accurate: {list(top_accurate_words.items())[:5]}")

        self._training_meta = {
            "train_size": len(texts),
            "tfidf": {
                "max_features": 5000,
                "ngram_range": [1, 2],
                "vocab_size": len(self._tfidf.vocabulary_),
            },
            "tuning": {
                "n_trials": n_trials,
                "cv_folds": N_SPLITS,
                "objective": "f1_macro",
                **lr_tuning,
            },
            "interpretability": {
                "lr_top_exaggerated_words": top_exaggerated_words,
                "lr_top_accurate_words": top_accurate_words,
                "lr_intercept": round(float(self._lr.intercept_[0]), 4),
            },
            "train_accuracy": round(train_acc, 4),
        }
        return self._training_meta

    def predict(self, text: str) -> str:
        if self._lr is None:
            raise RuntimeError(
                "Model has not been trained. Run 'uv run train classical' first."
            )
        X = self._tfidf.transform([text])
        return str(self._lr.predict(X)[0])

    def predict_proba(self, text: str) -> dict:
        """Predict label with confidence from logistic regression probabilities."""
        if self._lr is None:
            raise RuntimeError(
                "Model has not been trained. Run 'uv run train classical' first."
            )
        X = self._tfidf.transform([text])
        probas = self._lr.predict_proba(X)[0]
        label_idx = probas.argmax()
        return {
            "label": str(self._lr.classes_[label_idx]),
            "confidence": round(float(probas[label_idx]), 4),
        }

    def predict_batch(self, texts: list[str]) -> list[str]:
        if self._lr is None:
            raise RuntimeError(
                "Model has not been trained. Run 'uv run train classical' first."
            )
        X = self._tfidf.transform(texts)
        return self._lr.predict(X).tolist()

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)

        with open(directory / "lr.pkl", "wb") as f:
            pickle.dump(self._lr, f)
        with open(directory / "tfidf.pkl", "wb") as f:
            pickle.dump(self._tfidf, f)

        # Save best hyperparameters separately for --skip-tuning
        if "tuning" in self._training_meta:
            with open(directory / "best_params.json", "w") as f:
                json.dump(self._training_meta["tuning"], f, indent=2)

        meta = {
            "classes": self._lr.classes_.tolist(),
            "training_meta": self._training_meta,
        }
        with open(directory / "model.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Saved to {directory}")

    def load(self, directory: Path) -> None:
        with open(directory / "lr.pkl", "rb") as f:
            self._lr = pickle.load(f)
        with open(directory / "tfidf.pkl", "rb") as f:
            self._tfidf = pickle.load(f)

        with open(directory / "model.json") as f:
            meta = json.load(f)
        self._training_meta = meta.get("training_meta", {})

        logger.info(f"Loaded from {directory}")
