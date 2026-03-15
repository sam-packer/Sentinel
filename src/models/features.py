"""
Feature extraction for classical ML models.

Converts raw tweet text into numerical features that capture linguistic
intensity, claimed direction, and stylistic signals — all derivable
from the text alone, no price or news data needed.
"""

import re

from ..data.labeler import _UP_SIGNALS, _UP_EMOJI, _DOWN_SIGNALS, _DOWN_EMOJI


def extract_features(text: str) -> dict[str, float]:
    """Extract numerical features from a tweet's text.

    Returns a dict of feature_name -> float value. Every tweet
    produces the same set of keys, so the output can be fed
    directly into a scikit-learn DictVectorizer.
    """
    lowered = text.lower()
    words = text.split()

    # Direction signals
    up_signals = sum(1 for kw in _UP_SIGNALS if kw in lowered)
    down_signals = sum(1 for kw in _DOWN_SIGNALS if kw in lowered)
    up_emojis = sum(1 for e in _UP_EMOJI if e in text)
    down_emojis = sum(1 for e in _DOWN_EMOJI if e in text)

    # Claimed direction as numeric: +1 up, -1 down, 0 neutral
    total_up = up_signals + up_emojis
    total_down = down_signals + down_emojis
    if total_up > 0 and total_down > 0:
        direction = 0.0
    elif total_up > total_down:
        direction = 1.0
    elif total_down > total_up:
        direction = -1.0
    else:
        direction = 0.0

    # Intensity features
    exclamation_count = text.count("!")
    caps_words = sum(1 for w in words if w.isupper() and len(w) > 2)
    caps_ratio = caps_words / len(words) if words else 0.0
    emoji_count = up_emojis + down_emojis

    superlatives = ["insane", "massive", "huge", "crazy", "unbelievable", "incredible"]
    superlative_count = sum(1 for s in superlatives if s in lowered)

    # Structural features
    word_count = len(words)
    has_url = 1.0 if re.search(r"https?://", text) else 0.0
    has_question_mark = 1.0 if "?" in text else 0.0

    return {
        "direction": direction,
        "up_signal_count": float(up_signals),
        "down_signal_count": float(down_signals),
        "emoji_count": float(emoji_count),
        "exclamation_count": float(exclamation_count),
        "caps_ratio": caps_ratio,
        "superlative_count": float(superlative_count),
        "word_count": float(word_count),
        "has_url": has_url,
        "has_question_mark": has_question_mark,
    }
