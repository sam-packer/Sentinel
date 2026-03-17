# This file was developed with the assistance of Claude Code and Opus 4.6.

"""
Data loading and splitting for model training.

Pulls labeled claims from PostgreSQL, filters out understated (too few
samples to train on), and splits into train/test sets with a deterministic
seed for reproducibility across model comparisons.
"""

import logging
import random
from dataclasses import dataclass

from ..data.db import SentinelDB

logger = logging.getLogger("sentinel.models.data")

EXCLUDE_LABELS = {"understated"}


@dataclass
class TrainTestSplit:
    """Train/test data ready for model consumption."""
    train_texts: list[str]
    train_labels: list[str]
    test_texts: list[str]
    test_labels: list[str]

    @property
    def train_size(self) -> int:
        return len(self.train_texts)

    @property
    def test_size(self) -> int:
        return len(self.test_texts)


VALID_LABEL_TABLES = ("naive_labeled_claims", "improved_labeled_claims")


def load_labeled_claims(
    db: SentinelDB,
    label_table: str = "naive_labeled_claims",
) -> list[dict]:
    """Load all labeled claims from the database.

    Args:
        db: Database connection.
        label_table: Which label table to read from
            ("naive_labeled_claims" or "improved_labeled_claims").

    Returns a list of dicts with 'text' and 'label' keys (plus all
    other fields from the joined raw_claims table).
    """
    if label_table not in VALID_LABEL_TABLES:
        raise ValueError(f"Invalid label_table: {label_table}")

    query = f"""
        SELECT r.tweet_id, r.text, r.username, r.created_at,
               r.ticker, r.company_name, r.price_change_pct,
               r.has_catalyst, r.catalyst_type,
               l.label, l.claimed_direction, l.actual_direction,
               l.exaggeration_score
        FROM {label_table} l
        JOIN raw_claims r ON r.tweet_id = l.tweet_id
        LEFT JOIN accounts a ON r.username = a.username
        WHERE (a.account_type IS NULL OR a.account_type = 'human')
          AND r.created_at >= r.scraped_at - INTERVAL '90 days'
          AND r.price_change_pct IS NOT NULL
        ORDER BY r.created_at
    """
    columns, rows = db.execute_query(query)

    logger.info(f"Loaded {len(rows)} claims from {label_table} (bot accounts excluded)")

    return [dict(zip(columns, row)) for row in rows]


def prepare_split(
    claims: list[dict],
    test_size: float = 0.2,
    seed: int = 42,
) -> TrainTestSplit:
    """Filter and split claims into train/test sets.

    Filters out understated claims (too few to train on), then does a
    stratified-ish shuffle split. Uses a fixed seed so all three model
    approaches train and test on the exact same data.
    """
    # Filter out excluded labels
    filtered = [c for c in claims if c["label"] not in EXCLUDE_LABELS]
    logger.info(
        f"Filtered {len(claims)} -> {len(filtered)} claims "
        f"(dropped {len(claims) - len(filtered)} {EXCLUDE_LABELS})"
    )

    # Shuffle deterministically
    rng = random.Random(seed)
    rng.shuffle(filtered)

    # Split
    split_idx = int(len(filtered) * (1 - test_size))
    train = filtered[:split_idx]
    test = filtered[split_idx:]

    return TrainTestSplit(
        train_texts=[c["text"] for c in train],
        train_labels=[c["label"] for c in train],
        test_texts=[c["text"] for c in test],
        test_labels=[c["label"] for c in test],
    )
