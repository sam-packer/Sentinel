"""
Categorize labeling errors by failure mode.

Pulls all labeled claims, runs the classical model predictions,
and categorizes each misprediction into failure mode buckets.
This tells us which improvements to the labeler would have the most impact.

Failure modes checked:
1. Long-term thesis: bullish keywords + "long", "longterm", "hold", "invest" — not a 24h prediction
2. Past tense recap: reporting what already happened, not predicting ("ripped", "surged today")
3. Informational/analytical: educational content, watchlists, analysis without prediction
4. Position disclosure: "added", "bought", "holding" — stating a position, not predicting direction
5. Question not assertion: "will $LMT moon?" is not a claim
6. Volatile ticker + tight threshold: move < 2% on a volatile ticker (KTOS, RKLB, PLTR)
7. Negation missed: "not", "n't", "unlikely" near bullish/bearish keywords
8. Sarcasm: eye-roll emoji, "/s", "sure", "totally" near hype words
9. Non-claim: job posting, corporate PR, news link without commentary
10. Unclassifiable: doesn't fit any category — the label or prediction is just wrong
"""

import re
import sys
import os
from pathlib import Path
from collections import Counter

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.data.db import SentinelDB
from src.models.data import load_labeled_claims, prepare_split, EXCLUDE_LABELS
from src.models.classical import ClassicalModel
import random

# Volatile tickers — typically move 2%+ on a normal day
VOLATILE_TICKERS = {"KTOS", "RKLB", "PLTR", "RKLB", "HII"}

LONGTERM_KEYWORDS = {
    "long term", "longterm", "long-term", "hold", "holding", "hodl",
    "invest", "investing", "investment", "portfolio", "position",
    "years", "decade", "believer", "conviction",
}

PAST_TENSE_KEYWORDS = {
    "ripped", "surged", "mooned", "crashed", "dumped", "dropped",
    "rallied", "pumped", "plunged", "tanked", "soared",
    "today's", "today", "yesterday", "last week", "this week",
    "winners today", "losers today", "recap",
}

POSITION_KEYWORDS = {
    "added", "bought", "picked up", "loading", "loaded",
    "own", "i own", "my position", "my shares",
    "trimmed", "sold", "selling",
}

QUESTION_PATTERN = re.compile(r"\?\s*$|\bwill\b.*\?|\bshould\b.*\?|\bwould\b.*\?", re.IGNORECASE)

INFORMATIONAL_KEYWORDS = {
    "watch", "watching", "to watch", "keep an eye",
    "analysis", "overview", "breakdown", "thread",
    "key companies", "companies to watch", "names to watch",
    "sector", "industry",
}

NEGATION_PATTERN = re.compile(
    r"\b(not|n't|no|never|unlikely|doubt|don't|doesn't|won't|wouldn't|isn't|aren't)\b"
    r".{0,20}"
    r"\b(moon|surge|rally|rip|breakout|bullish|crash|dump|plunge|bearish)\b",
    re.IGNORECASE,
)

SARCASM_MARKERS = {"🙄", "/s", "sure,", "yeah sure", "totally", "definitely", "right,"}

NON_CLAIM_KEYWORDS = {
    "hiring", "job", "position open", "apply", "career",
    "press release", "media advisory",
    "#hiring", "#job", "#nowhiring",
}


def categorize(text, ticker, price_change_pct, claimed_direction, label, predicted):
    """Return list of failure mode categories for a misprediction."""
    lowered = text.lower()
    categories = []

    # Only categorize actual mispredictions
    if predicted == label:
        return []

    # 1. Long-term thesis
    if any(kw in lowered for kw in LONGTERM_KEYWORDS):
        categories.append("long_term_thesis")

    # 2. Past tense recap
    if any(kw in lowered for kw in PAST_TENSE_KEYWORDS):
        categories.append("past_tense_recap")

    # 3. Informational/analytical
    if any(kw in lowered for kw in INFORMATIONAL_KEYWORDS):
        categories.append("informational")

    # 4. Position disclosure
    if any(kw in lowered for kw in POSITION_KEYWORDS):
        categories.append("position_disclosure")

    # 5. Question
    if QUESTION_PATTERN.search(text):
        categories.append("question_not_claim")

    # 6. Volatile ticker + tight threshold
    if (ticker in VOLATILE_TICKERS
            and price_change_pct is not None
            and abs(price_change_pct) < 3.0
            and claimed_direction != "neutral"):
        categories.append("volatile_ticker_tight_threshold")

    # 7. Negation
    if NEGATION_PATTERN.search(text):
        categories.append("negation_missed")

    # 8. Sarcasm
    if any(marker in lowered or marker in text for marker in SARCASM_MARKERS):
        categories.append("sarcasm")

    # 9. Non-claim (job posting, PR, etc.)
    if any(kw in lowered for kw in NON_CLAIM_KEYWORDS):
        categories.append("non_claim")

    # 10. If nothing matched, it's unclassifiable
    if not categories:
        categories.append("unclassifiable")

    return categories


def main():
    config.setup_logging()
    db = SentinelDB(config.database.url)
    db.connect()

    claims = load_labeled_claims(db)
    split = prepare_split(claims, test_size=0.2, seed=42)

    # Recreate full claim dicts for test set
    filtered = [c for c in claims if c["label"] not in EXCLUDE_LABELS]
    rng = random.Random(42)
    rng.shuffle(filtered)
    split_idx = int(len(filtered) * 0.8)
    test_claims = filtered[split_idx:]

    # Load model and predict
    model = ClassicalModel()
    model_dir = Path(__file__).parent.parent / "models" / "classical"
    model.load(model_dir)

    predictions = []
    for text in split.test_texts:
        predictions.append(model.predict(text))

    # Categorize mispredictions
    category_counts = Counter()
    category_examples = {}
    total_errors = 0
    false_exaggerated = 0
    false_accurate = 0

    for i, (pred, actual) in enumerate(zip(predictions, split.test_labels)):
        if pred == actual:
            continue

        total_errors += 1
        claim = test_claims[i]

        if pred == "exaggerated" and actual == "accurate":
            false_exaggerated += 1
        else:
            false_accurate += 1

        cats = categorize(
            text=claim["text"],
            ticker=claim["ticker"],
            price_change_pct=claim.get("price_change_pct"),
            claimed_direction=claim.get("claimed_direction"),
            label=actual,
            predicted=pred,
        )

        for cat in cats:
            category_counts[cat] += 1
            if cat not in category_examples:
                category_examples[cat] = []
            if len(category_examples[cat]) < 3:
                category_examples[cat].append({
                    "text": claim["text"][:200],
                    "ticker": claim["ticker"],
                    "predicted": pred,
                    "actual": actual,
                    "price_change": claim.get("price_change_pct"),
                    "claimed_dir": claim.get("claimed_direction"),
                    "username": claim.get("username"),
                })

    db.close()

    # Report
    print(f"Test set: {split.test_size} claims")
    print(f"Mispredictions: {total_errors} ({100*total_errors/split.test_size:.1f}%)")
    print(f"  False exaggerated (predicted exag, was accurate): {false_exaggerated}")
    print(f"  False accurate (predicted accurate, was exag):    {false_accurate}")

    print(f"\n{'=' * 80}")
    print("FAILURE MODE DISTRIBUTION")
    print(f"{'=' * 80}")
    print(f"\nNote: one misprediction can have multiple categories.")
    print(f"{'unclassifiable' !r} means no pattern matched — needs manual review.\n")

    for cat, count in category_counts.most_common():
        pct = 100 * count / total_errors
        bar_len = int(pct / 2)
        bar = "\u2588" * bar_len
        print(f"  {cat:<35} {count:>4} ({pct:>5.1f}%) {bar}")

    for cat, count in category_counts.most_common():
        examples = category_examples[cat]
        print(f"\n{'─' * 80}")
        print(f"{cat.upper()} — {count} errors ({100*count/total_errors:.1f}%)")
        print(f"{'─' * 80}")

        for ex in examples:
            pct = f"{ex['price_change']:+.2f}%" if ex['price_change'] is not None else "N/A"
            print(f"\n  @{ex['username']} | {ex['ticker']} | price: {pct}")
            print(f"  predicted: {ex['predicted']}, actual: {ex['actual']}, claimed: {ex['claimed_dir']}")
            print(f"  {ex['text']}")


if __name__ == "__main__":
    main()
