"""
Model evaluation metrics.

Computes accuracy, precision, recall, F1, and confusion matrix.
Same evaluation code runs for all three model approaches so
comparisons are apples-to-apples.
"""


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


def format_report(
    model_name: str,
    labels: str,
    train_size: int,
    test_size: int,
    seed: int,
    metrics: dict,
    training_meta: dict | None = None,
    mispredictions: list[dict] | None = None,
) -> str:
    """Generate a markdown report for a training run.

    Captures everything useful while training state is still in memory:
    metrics, hyperparameters, top features, class distribution, error
    breakdown, and tuning history.
    """
    import json
    from datetime import datetime

    lines = []
    lines.append(f"# {model_name} — {labels} labels")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    # Summary table
    f1_values = [m["f1"] for m in metrics["per_class"].values()]
    macro_f1 = sum(f1_values) / len(f1_values) if f1_values else 0

    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Label source | `{labels}_labeled_claims` |")
    lines.append(f"| Train size | {train_size} |")
    lines.append(f"| Test size | {test_size} |")
    lines.append(f"| Seed | {seed} |")
    lines.append(f"| **Accuracy** | **{metrics['accuracy']:.1%}** |")
    lines.append(f"| **Macro F1** | **{macro_f1:.4f}** |")

    # Add train accuracy if available
    if training_meta and "train_accuracy" in training_meta:
        lines.append(f"| Train accuracy | {training_meta['train_accuracy']:.1%} |")

    lines.append("")

    # Class distribution in test set
    lines.append("## Test set class distribution")
    lines.append("")
    total = metrics["total"]
    for cls, m in metrics["per_class"].items():
        pct = m["support"] / total * 100 if total > 0 else 0
        lines.append(f"- **{cls}**: {m['support']} ({pct:.1f}%)")
    lines.append("")

    # Per-class metrics
    lines.append("## Per-class metrics")
    lines.append("")
    lines.append("| Class | Precision | Recall | F1 | Support |")
    lines.append("|-------|-----------|--------|-----|---------|")
    for cls, m in metrics["per_class"].items():
        lines.append(
            f"| {cls} | {m['precision']:.4f} | {m['recall']:.4f} | "
            f"{m['f1']:.4f} | {m['support']} |"
        )
    lines.append("")

    # Confusion matrix
    classes = sorted(metrics["confusion_matrix"].keys())
    lines.append("## Confusion matrix")
    lines.append("")
    lines.append("Rows = actual, columns = predicted.")
    lines.append("")
    header = "| | " + " | ".join(classes) + " |"
    separator = "|---|" + "|".join("---:" for _ in classes) + "|"
    lines.append(header)
    lines.append(separator)
    for actual in classes:
        row_vals = " | ".join(
            str(metrics["confusion_matrix"][actual][pred]) for pred in classes
        )
        lines.append(f"| **{actual}** | {row_vals} |")
    lines.append("")

    # Error breakdown
    total_errors = total - sum(
        metrics["confusion_matrix"][c][c] for c in classes
    )
    if total_errors > 0:
        lines.append("## Error breakdown")
        lines.append("")
        lines.append(f"Total mispredictions: {total_errors}/{total} ({100*total_errors/total:.1f}%)")
        lines.append("")
        for actual in classes:
            for pred in classes:
                if actual != pred:
                    count = metrics["confusion_matrix"][actual][pred]
                    if count > 0:
                        lines.append(
                            f"- **{actual} predicted as {pred}**: {count} "
                            f"({100*count/total_errors:.1f}% of errors)"
                        )
        lines.append("")

    # Training metadata
    if training_meta:
        # Hyperparameters (tuning results)
        tuning = training_meta.get("tuning", {})
        if tuning:
            lines.append("## Hyperparameter tuning")
            lines.append("")
            if "best_cv_score" in tuning:
                lines.append(f"- Best CV macro F1: **{tuning['best_cv_score']:.4f}**")
            if "best_trial" in tuning:
                n = tuning.get("n_trials", "?")
                lines.append(f"- Best trial: {tuning['best_trial']}/{n}")
            lines.append(f"- CV folds: {tuning.get('cv_folds', N_SPLITS)}")
            lines.append("")

            best_params = tuning.get("best_params", {})
            if best_params:
                lines.append("### Best parameters")
                lines.append("")
                lines.append("| Parameter | Value |")
                lines.append("|-----------|-------|")
                for k, v in best_params.items():
                    if isinstance(v, float):
                        lines.append(f"| {k} | {v:.6g} |")
                    else:
                        lines.append(f"| {k} | {v} |")
                lines.append("")

        # Top predictive words (classical model)
        interp = training_meta.get("interpretability", {})
        if interp:
            lines.append("## Top predictive features")
            lines.append("")

            top_exag = interp.get("lr_top_exaggerated_words", {})
            if top_exag:
                lines.append("### Words pushing toward exaggerated")
                lines.append("")
                lines.append("| Word | Coefficient |")
                lines.append("|------|-------------|")
                for word, coef in list(top_exag.items())[:10]:
                    lines.append(f"| {word} | {coef:+.4f} |")
                lines.append("")

            top_acc = interp.get("lr_top_accurate_words", {})
            if top_acc:
                lines.append("### Words pushing toward accurate")
                lines.append("")
                lines.append("| Word | Coefficient |")
                lines.append("|------|-------------|")
                for word, coef in list(top_acc.items())[:10]:
                    lines.append(f"| {word} | {coef:+.4f} |")
                lines.append("")

            if "lr_intercept" in interp:
                lines.append(f"LR intercept: {interp['lr_intercept']:.4f}")
                lines.append("")

        # TF-IDF info
        tfidf = training_meta.get("tfidf", {})
        if tfidf:
            lines.append("## TF-IDF vectorizer")
            lines.append("")
            lines.append(f"- Max features: {tfidf.get('max_features')}")
            lines.append(f"- N-gram range: {tfidf.get('ngram_range')}")
            lines.append(f"- Vocabulary size: {tfidf.get('vocab_size')}")
            lines.append("")

        # Model info (neural)
        if "model_name" in training_meta:
            lines.append("## Model architecture")
            lines.append("")
            lines.append(f"- Base model: `{training_meta['model_name']}`")
            if "max_length" in training_meta:
                lines.append(f"- Max sequence length: {training_meta['max_length']}")
            lines.append("")

    # Misprediction examples
    if mispredictions:
        lines.append("## Mispredictions")
        lines.append("")
        lines.append(f"Total: {len(mispredictions)} errors")
        lines.append("")

        # Group by error type
        false_exag = [m for m in mispredictions if m["predicted"] == "exaggerated" and m["actual"] == "accurate"]
        false_acc = [m for m in mispredictions if m["predicted"] == "accurate" and m["actual"] == "exaggerated"]

        if false_exag:
            lines.append(f"### False exaggerated ({len(false_exag)} — predicted exaggerated, was accurate)")
            lines.append("")
            # Sort by absolute price change to show the most interesting ones
            sorted_fe = sorted(
                false_exag,
                key=lambda m: abs(m.get("price_change_pct") or 0),
                reverse=True,
            )
            for m in sorted_fe[:10]:
                pct = f"{m['price_change_pct']:+.2f}%" if m.get("price_change_pct") is not None else "N/A"
                catalyst = m.get("catalyst_type") or "none"
                lines.append(f"**@{m.get('username', '?')}** | {m.get('ticker', '?')} | price: {pct} | catalyst: {catalyst}")
                lines.append(f"> {m['text'][:200]}")
                lines.append("")

        if false_acc:
            lines.append(f"### False accurate ({len(false_acc)} — predicted accurate, was exaggerated)")
            lines.append("")
            sorted_fa = sorted(
                false_acc,
                key=lambda m: m.get("exaggeration_score") or 0,
                reverse=True,
            )
            for m in sorted_fa[:10]:
                pct = f"{m['price_change_pct']:+.2f}%" if m.get("price_change_pct") is not None else "N/A"
                score = m.get("exaggeration_score", 0)
                lines.append(f"**@{m.get('username', '?')}** | {m.get('ticker', '?')} | price: {pct} | exag score: {score:.3f}")
                lines.append(f"> {m['text'][:200]}")
                lines.append("")

    if training_meta:
        # Full metadata dump for reference
        lines.append("## Raw training metadata")
        lines.append("")
        lines.append("<details>")
        lines.append("<summary>Click to expand</summary>")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(training_meta, indent=2, default=str))
        lines.append("```")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    return "\n".join(lines)


# Used by format_report when tuning info doesn't include cv_folds
N_SPLITS = 3
