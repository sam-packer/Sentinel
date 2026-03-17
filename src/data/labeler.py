# This file was developed with the assistance of Claude Code and Opus 4.6.

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

# Regex to strip punctuation from word boundaries for single-word matching
_WORD_SPLIT_RE = re.compile(r"[a-z0-9']+")


def _count_keyword_matches(text: str, keywords: set[str]) -> int:
    """Count keyword matches using word-boundary-aware matching.

    Single-word keywords are matched against individual words (with
    punctuation stripped) to avoid false positives like "run" matching
    "runtime". Multi-word keywords use substring matching, which is
    correct for phrases like "load up" or "getting out".
    """
    lowered = text.lower()
    words = set(_WORD_SPLIT_RE.findall(lowered))

    single_word = {kw for kw in keywords if " " not in kw}
    multi_word = keywords - single_word

    count = len(words & single_word)
    count += sum(1 for kw in multi_word if kw in lowered)
    return count


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
    up_score = _count_keyword_matches(text, _UP_SIGNALS)
    up_score += sum(1 for e in _UP_EMOJI if e in text)

    down_score = _count_keyword_matches(text, _DOWN_SIGNALS)
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
    superlatives = {"insane", "massive", "huge", "crazy", "unbelievable", "incredible"}
    score += min(_count_keyword_matches(text, superlatives) * 0.1, 0.2)

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

    The score is the sum of three independent components:

    Direction mismatch (0.0 or 0.5): whether the claimed direction is
    opposite the actual price movement. This is the single strongest
    signal because getting the direction wrong means the claim is
    fundamentally misleading regardless of magnitude. Worth half the
    total score on its own.

    Magnitude gap (0.0 to 0.3): how much the tweet's language intensity
    exceeds what the actual price move justifies. A 5% daily move is
    large enough to justify even aggressive language for a single stock,
    so the gap is calculated as intensity * (1 - move/5%) * 0.3. Below
    5%, intense language is increasingly unsupported. When directions
    mismatch, the full intensity counts because none of the move supports
    the claim.

    Catalyst gap (0.0 to 0.2): penalty for directional claims with no
    news catalyst. A tweet claiming "$LMT mooning" without any contract,
    earnings, or geopolitical news is more likely hype than informed
    analysis. Only applied when news was actually fetched (empty results
    from a failed fetch don't count against the claim).

    These sum to a maximum of 1.0. Each component is independently
    interpretable, so you can explain why any given tweet scored what
    it did.
    """
    if price_change_pct is None:
        return 0.5  # no price data, can't assess

    abs_move = abs(price_change_pct)

    # Component 1: Direction mismatch (0.0 or 0.5)
    # Claimed one direction, stock went the other.
    direction_score = 0.0
    if claimed != "neutral" and actual != "neutral" and claimed != actual:
        direction_score = 0.5

    # Component 2: Magnitude gap (0.0 to 0.3)
    # How much the language oversells the actual move.
    # At 5%+ move, even intense language is justified (gap = 0).
    # Below 5%, the gap scales with intensity.
    # When directions mismatch, none of the move supports the claim,
    # so the full intensity applies.
    magnitude_score = 0.0
    if claimed != "neutral":
        if direction_score > 0:
            magnitude_score = intensity * 0.3
        else:
            move_ratio = min(abs_move / 5.0, 1.0)
            magnitude_score = intensity * (1.0 - move_ratio) * 0.3

    # Component 3: Catalyst gap (0.0 to 0.2)
    # Directional claim with no news backing it up.
    catalyst_score = 0.0
    if claimed != "neutral" and not has_catalyst and news_available:
        catalyst_score = intensity * 0.2

    return min(direction_score + magnitude_score + catalyst_score, 1.0)


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
