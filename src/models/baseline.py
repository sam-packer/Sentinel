# This file was developed with the assistance of Claude Code and Opus 4.6.

"""
Naive baseline model: always predicts the majority class.

This is the floor that any real model needs to beat. With the current
data (~78% accurate, ~22% exaggerated), this model gets 78% accuracy
by always predicting 'accurate'. It has zero recall on the exaggerated
class, which is the one we actually care about detecting.
"""

import json
import logging
from collections import Counter
from pathlib import Path

from . import BaseModel

logger = logging.getLogger("sentinel.models.baseline")


class MajorityClassModel(BaseModel):
    """Always predicts the most common label from the training set."""

    def __init__(self):
        self._majority_class: str | None = None
        self._class_counts: dict[str, int] = {}

    @property
    def name(self) -> str:
        return "baseline"

    def train(self, texts: list[str], labels: list[str]) -> dict:
        counts = Counter(labels)
        self._majority_class = counts.most_common(1)[0][0]
        self._class_counts = dict(counts)

        total = len(labels)
        logger.info(
            f"Majority class: '{self._majority_class}' "
            f"({counts[self._majority_class]}/{total}, "
            f"{counts[self._majority_class] / total:.1%})"
        )

        return {
            "majority_class": self._majority_class,
            "class_counts": self._class_counts,
            "total_samples": total,
        }

    def predict(self, text: str) -> str:
        if self._majority_class is None:
            raise RuntimeError("Model has not been trained. Run 'uv run train baseline' first.")
        return self._majority_class

    def predict_batch(self, texts: list[str]) -> list[str]:
        if self._majority_class is None:
            raise RuntimeError("Model has not been trained. Run 'uv run train baseline' first.")
        return [self._majority_class] * len(texts)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "model.json"
        with open(path, "w") as f:
            json.dump({
                "majority_class": self._majority_class,
                "class_counts": self._class_counts,
            }, f, indent=2)
        logger.info(f"Saved to {path}")

    def load(self, directory: Path) -> None:
        path = directory / "model.json"
        with open(path) as f:
            data = json.load(f)
        self._majority_class = data["majority_class"]
        self._class_counts = data["class_counts"]
        logger.info(f"Loaded from {path} (majority: {self._majority_class})")
