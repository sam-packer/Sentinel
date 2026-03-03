"""
Claim labeling engine for Sentinel.

Parses directional sentiment from tweet text, compares against actual price
moves, and classifies claims as exaggerated, accurate, or understated.
"""

import logging
import re
from typing import Literal

from .models import RawClaim, LabeledClaim

logger = logging.getLogger("sentinel.labeler")

# Direction signal keywords
_UP_SIGNALS = {
    "moon", "mooning", "pump", "surge", "rally", "run", "rip",
    "skyrocket", "breakout", "bullish", "ath", "load up", "loading up",
    "calls", "long", "ripping", "soaring",
}
_UP_EMOJI = {"🚀", "📈", "💎", "🔥", "💰", "🐂"}

_DOWN_SIGNALS = {
    "crash", "crashing", "dump", "collapse", "drop", "plunge", "bleed",
    "short", "puts", "bearish", "get out", "getting out",
    "tanking", "plummeting", "cratering",
}
_DOWN_EMOJI = {"📉", "🔻", "💀", "🐻", "⚠️"}


def parse_direction(text: str) -> Literal["up", "down", "neutral"]:
    """Parse claimed direction from tweet text.

    Uses keyword and emoji matching. If both bullish and bearish
    signals are present, returns "neutral".
    """
    lowered = text.lower()

    up_score = sum(1 for kw in _UP_SIGNALS if kw in lowered)
    up_score += sum(1 for e in _UP_EMOJI if e in text)

    down_score = sum(1 for kw in _DOWN_SIGNALS if kw in lowered)
    down_score += sum(1 for e in _DOWN_EMOJI if e in text)

    if up_score > 0 and down_score > 0:
        return "neutral"
    if up_score > down_score:
        return "up"
    if down_score > up_score:
        return "down"
    return "neutral"


def _actual_direction(price_change_pct: float | None) -> Literal["up", "down", "neutral"]:
    """Determine actual direction from price change."""
    if price_change_pct is None:
        return "neutral"
    if price_change_pct > 0.5:
        return "up"
    if price_change_pct < -0.5:
        return "down"
    return "neutral"


def _intensity_score(text: str) -> float:
    """Rough measure of how intense/hyperbolic the tweet language is (0-1)."""
    lowered = text.lower()
    score = 0.0

    # Exclamation marks
    score += min(text.count("!") * 0.1, 0.3)

    # ALL CAPS words
    words = text.split()
    caps_words = sum(1 for w in words if w.isupper() and len(w) > 2)
    score += min(caps_words * 0.05, 0.2)

    # Rocket/fire emoji
    score += min(sum(1 for e in _UP_EMOJI | _DOWN_EMOJI if e in text) * 0.1, 0.3)

    # Superlatives
    superlatives = ["insane", "massive", "huge", "crazy", "unbelievable", "incredible"]
    score += min(sum(0.1 for s in superlatives if s in lowered), 0.2)

    return min(score, 1.0)


def compute_exaggeration_score(
    claimed: Literal["up", "down", "neutral"],
    actual: Literal["up", "down", "neutral"],
    price_change_pct: float | None,
    has_catalyst: bool,
    intensity: float,
    *,
    news_available: bool = True,
) -> float:
    """Compute exaggeration score from 0.0 (accurate) to 1.0 (wildly off).

    Takes into account direction match, magnitude, catalyst presence,
    and language intensity.
    """
    if price_change_pct is None:
        return 0.5  # uncertain

    abs_move = abs(price_change_pct)

    # Direction mismatch is the strongest signal
    if claimed != "neutral" and actual != "neutral" and claimed != actual:
        return min(0.7 + intensity * 0.3, 1.0)

    # Strong language but tiny move
    if claimed != "neutral" and abs_move < 1.0:
        return min(0.4 + intensity * 0.4, 0.9)

    # No catalyst but claiming big move (only when we actually found news)
    if not has_catalyst and news_available and claimed != "neutral" and abs_move < 2.0:
        return min(0.3 + intensity * 0.3, 0.8)

    # Direction matches and move is meaningful
    if claimed != "neutral" and claimed == actual and abs_move >= 2.0:
        return max(0.0, 0.2 - abs_move * 0.02)

    # Neutral claim, small move — accurate
    if claimed == "neutral" and abs_move < 2.0:
        return 0.1

    return 0.3  # default moderate


def label_claim(
    raw: RawClaim,
    exaggeration_threshold: float = 0.02,
) -> LabeledClaim:
    """Label a RawClaim as exaggerated, accurate, or understated.

    Args:
        raw: A RawClaim with price and news data populated.
        exaggeration_threshold: Minimum price move (as fraction) to count
            as significant. Default 0.02 = 2%.

    Returns:
        LabeledClaim with label, directions, and exaggeration score.
    """
    claimed = parse_direction(raw.text)
    actual = _actual_direction(raw.price_change_pct)
    intensity = _intensity_score(raw.text)

    threshold_pct = exaggeration_threshold * 100  # convert to percentage
    abs_move = abs(raw.price_change_pct) if raw.price_change_pct is not None else 0.0

    # Default label
    label: Literal["exaggerated", "accurate", "understated"] = "accurate"

    if raw.price_change_pct is None:
        # No price data — can't label definitively
        label = "accurate" if claimed == "neutral" else "exaggerated"

    elif claimed == "neutral" and abs_move < threshold_pct:
        # Neutral tweet, small move → accurate
        label = "accurate"

    elif claimed == "neutral" and abs_move >= 5.0:
        # Neutral tweet but large move → understated
        label = "understated"

    elif claimed != "neutral" and actual != "neutral" and claimed != actual:
        # Wrong direction → exaggerated
        label = "exaggerated"

    elif claimed != "neutral" and abs_move < threshold_pct:
        # Strong language but tiny move → exaggerated
        label = "exaggerated"

    elif (
        claimed != "neutral"
        and not raw.has_catalyst
        and raw.news_headlines  # only penalize when news was actually found
        and abs_move < threshold_pct * 2
    ):
        # News found but no catalyst backing a directional claim → exaggerated
        label = "exaggerated"

    elif claimed != "neutral" and claimed == actual and abs_move >= threshold_pct:
        # Direction matches and move is significant → accurate
        label = "accurate"

    elif abs_move >= 10.0 and intensity < 0.3:
        # Huge move but calm tweet → understated
        label = "understated"

    exaggeration_score = compute_exaggeration_score(
        claimed, actual, raw.price_change_pct, raw.has_catalyst, intensity,
        news_available=bool(raw.news_headlines),
    )

    news_summary = raw.news_headlines[0] if raw.news_headlines else ""

    return LabeledClaim.from_raw(
        raw,
        label=label,
        claimed_direction=claimed,
        actual_direction=actual,
        exaggeration_score=round(exaggeration_score, 3),
        news_summary=news_summary,
    )
