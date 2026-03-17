# This file was developed with the assistance of Claude Code and Opus 4.6.

"""
ML models for Sentinel claim classification.

All models implement the same BaseModel interface so they can be
trained, evaluated, and served interchangeably.
"""

import importlib
from abc import ABC, abstractmethod
from pathlib import Path

MODEL_REGISTRY = {
    "baseline": "src.models.baseline:MajorityClassModel",
    "classical": "src.models.classical:ClassicalModel",
    "neural": "src.models.neural:NeuralModel",
}

MODEL_DIR = Path("models")


def load_model(name: str, labels: str = "naive") -> "BaseModel":
    """Import, instantiate, and load a trained model by name.

    Args:
        name: Model name from MODEL_REGISTRY.
        labels: Label set subdirectory ("naive" or "improved").

    Raises FileNotFoundError if no saved model exists.
    Raises KeyError if the model name is not in the registry.
    """
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unknown model '{name}'. Available: {available}")

    module_path, class_name = MODEL_REGISTRY[name].rsplit(":", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    instance = cls()

    model_dir = MODEL_DIR / instance.name / f"{labels}_labeler"
    if not model_dir.exists():
        # Fallback: check old flat path for backward compatibility
        flat_dir = MODEL_DIR / instance.name
        if flat_dir.exists():
            model_dir = flat_dir
        else:
            raise FileNotFoundError(
                f"No trained model at {model_dir}/ or {flat_dir}/. "
                f"Run 'uv run train {name}' first."
            )

    instance.load(model_dir)
    return instance


class BaseModel(ABC):
    """Interface that all Sentinel models must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short name for this model (e.g. 'baseline', 'classical', 'neural')."""

    @abstractmethod
    def train(self, texts: list[str], labels: list[str]) -> dict:
        """Train the model on labeled tweet texts.

        Args:
            texts: Tweet text strings.
            labels: Corresponding labels ('exaggerated' or 'accurate').

        Returns:
            Dict of training metadata (e.g. class distribution, majority class).
        """

    @abstractmethod
    def predict(self, text: str) -> str:
        """Predict label for a single tweet.

        Returns:
            'exaggerated' or 'accurate'.
        """

    def predict_proba(self, text: str) -> dict:
        """Predict label with confidence score.

        Returns:
            Dict with 'label' (str) and 'confidence' (float 0.0-1.0).
        """
        return {"label": self.predict(text), "confidence": 1.0}

    def predict_batch(self, texts: list[str]) -> list[str]:
        """Predict labels for multiple tweets. Override for efficiency."""
        return [self.predict(text) for text in texts]

    @abstractmethod
    def save(self, directory: Path) -> None:
        """Save model to a directory."""

    @abstractmethod
    def load(self, directory: Path) -> None:
        """Load model from a directory."""
