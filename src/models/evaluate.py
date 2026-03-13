"""
Model evaluation metrics.

Computes accuracy, precision, recall, F1, and confusion matrix.
Same evaluation code runs for all three model approaches so
comparisons are apples-to-apples.
"""

from collections import Counter


def compute_metrics(predictions: list[str], labels: list[str]) -> dict:
    """Compute classification metrics for binary exaggerated/accurate.

    Returns a dict with overall accuracy, per-class precision/recall/f1,
    and a confusion matrix.
    """
    if len(predictions) != len(labels):
        raise ValueError(
            f"predictions ({len(predictions)}) and labels ({len(labels)}) must have the same length"
        )

    classes = sorted(set(labels) | set(predictions))
    total = len(labels)

    # Overall accuracy
    correct = sum(p == l for p, l in zip(predictions, labels))
    accuracy = correct / total if total > 0 else 0.0

    # Per-class metrics
    per_class = {}
    for cls in classes:
        tp = sum(p == cls and l == cls for p, l in zip(predictions, labels))
        fp = sum(p == cls and l != cls for p, l in zip(predictions, labels))
        fn = sum(p != cls and l == cls for p, l in zip(predictions, labels))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class[cls] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(1 for l in labels if l == cls),
        }

    # Confusion matrix as nested dict: confusion[actual][predicted] = count
    confusion = {actual: {pred: 0 for pred in classes} for actual in classes}
    for pred, label in zip(predictions, labels):
        confusion[label][pred] += 1

    return {
        "accuracy": round(accuracy, 4),
        "total": total,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def format_metrics(metrics: dict) -> str:
    """Format metrics dict into a readable string for CLI output."""
    lines = []
    lines.append(f"Accuracy: {metrics['accuracy']:.1%} ({metrics['total']} samples)")
    lines.append("")

    # Per-class table
    lines.append(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    lines.append("-" * 57)
    for cls, m in metrics["per_class"].items():
        lines.append(
            f"{cls:<15} {m['precision']:>10.4f} {m['recall']:>10.4f} "
            f"{m['f1']:>10.4f} {m['support']:>10}"
        )

    # Confusion matrix
    classes = sorted(metrics["confusion_matrix"].keys())
    lines.append("")
    lines.append("Confusion matrix (rows = actual, columns = predicted):")
    header = f"{'':>15}" + "".join(f"{c:>15}" for c in classes)
    lines.append(header)
    for actual in classes:
        row = f"{actual:>15}" + "".join(
            f"{metrics['confusion_matrix'][actual][pred]:>15}" for pred in classes
        )
        lines.append(row)

    return "\n".join(lines)
