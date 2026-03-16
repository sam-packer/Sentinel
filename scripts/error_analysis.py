"""
Error analysis script for the paper.

Loads the trained classical model, runs predictions on the test set,
and finds the most interesting mispredictions with full context
(tweet text, ticker, price change, catalyst, etc.).
"""

import sys
import os
from pathlib import Path

# Fix Windows console encoding for emoji
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.data.db import SentinelDB
from src.models.data import load_labeled_claims, prepare_split
from src.models.classical import ClassicalModel


def main():
    config.setup_logging()
    db = SentinelDB(config.database.url)
    db.connect()

    # Load data with the same split used for training
    # load_labeled_claims already filters out bot/garbage accounts
    # (WHERE a.account_type IS NULL OR a.account_type = 'human')
    claims = load_labeled_claims(db)
    split = prepare_split(claims, test_size=0.2, seed=42)

    print(f"Total claims loaded (humans only): {len(claims)}")
    print(f"Train: {split.train_size}, Test: {split.test_size}")

    # We need the full claim dicts for the test set to get context
    # Recreate the filtering and splitting to get the full dicts
    from src.models.data import EXCLUDE_LABELS
    import random

    filtered = [c for c in claims if c["label"] not in EXCLUDE_LABELS]
    rng = random.Random(42)
    rng.shuffle(filtered)
    split_idx = int(len(filtered) * 0.8)
    test_claims = filtered[split_idx:]

    # Verify alignment
    assert len(test_claims) == split.test_size
    assert [c["text"] for c in test_claims] == split.test_texts

    # Load trained model and predict
    model = ClassicalModel()
    model_dir = Path(__file__).parent.parent / "models" / "classical"
    model.load(model_dir)
    predictions = []
    confidences = []
    for text in split.test_texts:
        result = model.predict_proba(text)
        predictions.append(result["label"])
        confidences.append(result["confidence"])

    # Find mispredictions
    mispredictions = []
    for i, (pred, actual) in enumerate(zip(predictions, split.test_labels)):
        if pred != actual:
            claim = test_claims[i]
            mispredictions.append({
                "text": claim["text"],
                "ticker": claim["ticker"],
                "company_name": claim["company_name"],
                "username": claim["username"],
                "predicted": pred,
                "actual": actual,
                "confidence": confidences[i],
                "price_change_pct": claim.get("price_change_pct"),
                "has_catalyst": claim.get("has_catalyst"),
                "catalyst_type": claim.get("catalyst_type"),
                "claimed_direction": claim.get("claimed_direction"),
                "actual_direction": claim.get("actual_direction"),
                "exaggeration_score": claim.get("exaggeration_score"),
                "created_at": str(claim.get("created_at", "")),
            })

    print(f"\nTotal mispredictions: {len(mispredictions)} / {split.test_size} "
          f"({100*len(mispredictions)/split.test_size:.1f}%)")

    # Categorize mispredictions
    false_exaggerated = [m for m in mispredictions if m["predicted"] == "exaggerated" and m["actual"] == "accurate"]
    false_accurate = [m for m in mispredictions if m["predicted"] == "accurate" and m["actual"] == "exaggerated"]

    print(f"\nFalse exaggerated (predicted exaggerated, was accurate): {len(false_exaggerated)}")
    print(f"False accurate (predicted accurate, was exaggerated): {len(false_accurate)}")

    # Show the most interesting mispredictions
    print("\n" + "=" * 80)
    print("FALSE EXAGGERATED — Model thought these were exaggerated, but they were accurate")
    print("(Hype language that happened to be right)")
    print("=" * 80)

    # Sort by price change to find the ones with biggest actual moves
    fe_with_price = [m for m in false_exaggerated if m["price_change_pct"] is not None]
    fe_with_price.sort(key=lambda x: abs(x["price_change_pct"]), reverse=True)

    for m in fe_with_price[:10]:
        print(f"\n--- @{m['username']} | {m['ticker']} | {m['created_at'][:10]} ---")
        print(f"Text: {m['text'][:200]}")
        print(f"Price change: {m['price_change_pct']:+.2f}%")
        print(f"Claimed: {m['claimed_direction']} | Actual: {m['actual_direction']}")
        print(f"Catalyst: {m['has_catalyst']} ({m['catalyst_type']})")
        print(f"Exaggeration score: {m['exaggeration_score']:.3f}")
        print(f"Model confidence: {m['confidence']:.3f}")

    print("\n" + "=" * 80)
    print("FALSE ACCURATE — Model thought these were accurate, but they were exaggerated")
    print("(Subtle exaggeration the model missed)")
    print("=" * 80)

    # Sort by exaggeration score to find the most clearly exaggerated ones the model missed
    false_accurate.sort(key=lambda x: x.get("exaggeration_score", 0), reverse=True)

    for m in false_accurate[:10]:
        print(f"\n--- @{m['username']} | {m['ticker']} | {m['created_at'][:10]} ---")
        print(f"Text: {m['text'][:200]}")
        print(f"Price change: {m['price_change_pct']:+.2f}%" if m['price_change_pct'] else "Price change: N/A")
        print(f"Claimed: {m['claimed_direction']} | Actual: {m['actual_direction']}")
        print(f"Catalyst: {m['has_catalyst']} ({m['catalyst_type']})")
        print(f"Exaggeration score: {m['exaggeration_score']:.3f}")

    # Also show some with catalysts for the "catalyst-backed hype" story
    print("\n" + "=" * 80)
    print("FALSE EXAGGERATED WITH CATALYST — Model missed that a real catalyst backed the claim")
    print("=" * 80)

    fe_catalyst = [m for m in false_exaggerated if m["has_catalyst"]]
    for m in fe_catalyst[:5]:
        print(f"\n--- @{m['username']} | {m['ticker']} | {m['created_at'][:10]} ---")
        print(f"Text: {m['text'][:200]}")
        print(f"Price change: {m['price_change_pct']:+.2f}%" if m['price_change_pct'] else "Price change: N/A")
        print(f"Catalyst type: {m['catalyst_type']}")
        print(f"Exaggeration score: {m['exaggeration_score']:.3f}")

    db.close()


if __name__ == "__main__":
    main()
