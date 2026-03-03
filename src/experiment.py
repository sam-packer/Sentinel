"""
News feature ablation experiment for Sentinel.

Trains the classical model under 4 conditions across 5 random seeds:
  1. Text only (TF-IDF)
  2. Text + linguistic features
  3. Text + linguistic + scalar news features
  4. Text + linguistic + full news features (including headline embeddings)

Records macro F1 and per-class F1 per condition per seed.
Plots bar chart with error bars.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from .models.classical import ClassicalClassifier

logger = logging.getLogger("sentinel.experiment")


def run_ablation(
    texts: list[str],
    text_features: np.ndarray,
    news_scalar_features: np.ndarray,
    headline_embeddings: np.ndarray,
    labels: list[str],
    output_dir: str = "data/outputs",
    n_seeds: int = 5,
) -> dict:
    """Run the news feature ablation experiment.

    Args:
        texts: Tweet text strings.
        text_features: (n, text_feat_dim) linguistic features.
        news_scalar_features: (n, news_feat_dim) scalar news features.
        headline_embeddings: (n, 384) headline embeddings.
        labels: Label strings.
        output_dir: Where to save the plot.
        n_seeds: Number of random seeds.

    Returns:
        Dict with results per condition and seed.
    """
    conditions = {
        "Text only": lambda: np.zeros((len(texts), 0)),
        "Text + linguistic": lambda: text_features,
        "Text + ling. + news scalar": lambda: np.hstack([text_features, news_scalar_features]),
        "Text + ling. + full news": lambda: np.hstack([
            text_features, news_scalar_features, headline_embeddings,
        ]),
    }

    results: dict[str, dict] = {}

    for condition_name, feature_fn in conditions.items():
        logger.info(f"Running condition: {condition_name}")
        condition_results = {"macro_f1": [], "per_class_f1": []}

        for seed in range(n_seeds):
            scalar_feats = feature_fn()

            # If no scalar features, use a dummy column
            if scalar_feats.shape[1] == 0:
                scalar_feats = np.zeros((len(texts), 1))

            X_train_texts, X_test_texts, \
                X_train_scalar, X_test_scalar, \
                y_train, y_test = train_test_split(
                    texts, scalar_feats, labels,
                    test_size=0.2, random_state=seed, stratify=labels,
                )

            clf = ClassicalClassifier(model_type="logreg")
            clf.fit(
                list(X_train_texts), X_train_scalar, list(y_train),
            )

            preds = clf.predict(list(X_test_texts), X_test_scalar)

            macro_f1 = f1_score(y_test, preds, average="macro")
            per_class = f1_score(
                y_test, preds, average=None,
                labels=["exaggerated", "accurate", "understated"],
            )

            condition_results["macro_f1"].append(macro_f1)
            condition_results["per_class_f1"].append(per_class)

            logger.info(f"  seed={seed}, macro F1={macro_f1:.4f}")

        results[condition_name] = condition_results

    # Plot
    _plot_ablation(results, output_dir)
    return results


def _plot_ablation(results: dict, output_dir: str) -> None:
    """Create bar chart with error bars for ablation results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    conditions = list(results.keys())
    means = [np.mean(results[c]["macro_f1"]) for c in conditions]
    stds = [np.std(results[c]["macro_f1"]) for c in conditions]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(conditions))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color="#2563eb", alpha=0.8)

    ax.set_xlabel("Feature Condition")
    ax.set_ylabel("Macro F1 Score")
    ax.set_title("News Feature Ablation — Classical Model (5 seeds)")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=15, ha="right")
    ax.set_ylim(0, 1.0)

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.02,
            f"{mean:.3f}",
            ha="center", va="bottom", fontsize=10,
        )

    plt.tight_layout()
    save_path = output_path / "news_ablation.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Ablation plot saved to {save_path}")
