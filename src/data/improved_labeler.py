"""
Improved claim labeling engine for Sentinel.

Addresses the top failure modes found in error analysis of the naive labeler
(137 mispredictions):

- Long-term thesis (31%): "I'm a believer! $RKLB longterm" no longer flagged
  as exaggerated — detected as non-predictive and neutralized.
- Position disclosure (22%): "added some $LDOS" no longer treated as a
  directional prediction.
- Informational/analytical (15%): watchlists, sector analysis neutralized.
- Volatile ticker threshold (14%): per-ticker thresholds replace the fixed 2%
  cutoff (e.g., KTOS uses 2.15% to match its median daily move).
- Past tense recap (11%): "ripped today" detected as backward-looking, not
  a forward prediction.
- Sarcasm (4%): eye-roll emoji and sarcastic phrasing neutralize direction.
- Questions (3%): "will $LMT moon?" treated as neutral, not bullish.
- Negation (1%): "NOT going to moon" correctly flips direction.

Reuses keyword lists, emoji sets, _intensity_score, _actual_direction, and
compute_exaggeration_score from the naive labeler. Only parse_direction and
label_claim are replaced.
"""

import logging
import re
from typing import Literal

from .labeler import (
    _UP_SIGNALS,
    _UP_EMOJI,
    _DOWN_SIGNALS,
    _DOWN_EMOJI,
    _actual_direction,
    _intensity_score,
    compute_exaggeration_score,
    parse_direction,
)
from .models import RawClaim, LabeledClaim

logger = logging.getLogger("sentinel.improved_labeler")

# ---------------------------------------------------------------------------
# Per-ticker volatility thresholds (median absolute daily move)
# ---------------------------------------------------------------------------
TICKER_THRESHOLDS: dict[str, float] = {
    "KTOS": 0.0215,
    "BA": 0.0216,
    "PLTR": 0.0176,
    "BAH": 0.0137,
    "SAIC": 0.0136,
    "HII": 0.0170,
    "RKLB": 0.0132,
    "LDOS": 0.0066,
    "LMT": 0.0123,
    "NOC": 0.0077,
    "RTX": 0.0060,
    "LHX": 0.0068,
    "GD": 0.0085,
}
DEFAULT_THRESHOLD: float = 0.02

# ---------------------------------------------------------------------------
# Pre-classification pattern sets
# ---------------------------------------------------------------------------
_LONG_TERM_KEYWORDS = {
    "long term", "longterm", "long-term", "hold", "holding", "hodl",
    "invest", "investing", "investment", "portfolio", "years", "decade",
    "believer", "conviction",
}

_POSITION_PATTERNS = re.compile(
    r"\b(added|bought|picked up|i own|my position|my shares|trimmed|sold some)\b",
    re.IGNORECASE,
)

_INFORMATIONAL_KEYWORDS = {
    "to watch", "watching", "keep an eye", "analysis", "overview",
    "breakdown", "thread", "companies to watch", "names to watch",
    "sector update",
}

_PAST_TENSE_DIRECTIONAL = {
    "ripped", "surged", "mooned", "crashed", "dumped", "rallied",
    "pumped", "tanked", "soared",
}

_PAST_TENSE_CONTEXT = {"today", "yesterday", "this week", "recap"}

_QUESTION_PATTERNS = re.compile(
    r"(^|\s)(will|should|would)\s.+\?", re.IGNORECASE,
)

_NON_CLAIM_KEYWORDS = {
    "hiring", "job", "position open", "apply", "career",
    "#hiring", "#nowhiring", "press release", "media advisory",
}

_SARCASM_EMOJI = {"🙄"}

_SARCASM_PATTERNS = re.compile(
    r"(/s\b|^sure,?\s|yeah\s+sure|totally\s+(moon|crash|dump|rip|surg|bull|bear)"
    r"|definitely\s+(moon|crash|dump|rip|surg|bull|bear|insane|massive|huge))",
    re.IGNORECASE,
)

_NEGATION_WORDS = {
    "not", "n't", "no", "never", "unlikely", "doubt",
    "don't", "doesn't", "won't", "wouldn't", "isn't", "aren't",
}


# ---------------------------------------------------------------------------
# Pre-classification checks
# ---------------------------------------------------------------------------

def _is_long_term_thesis(lowered: str) -> bool:
    """Detect long-term position statements that aren't 24h predictions."""
    has_long_term = any(kw in lowered for kw in _LONG_TERM_KEYWORDS)
    has_bullish = any(kw in lowered for kw in _UP_SIGNALS)
    return has_long_term and has_bullish


def _is_position_disclosure(text: str) -> bool:
    """Detect tweets disclosing a buy/sell, not predicting direction."""
    return bool(_POSITION_PATTERNS.search(text))


def _is_informational(lowered: str) -> bool:
    """Detect watchlists, sector analysis, and other non-predictive tweets."""
    return any(kw in lowered for kw in _INFORMATIONAL_KEYWORDS)


def _is_past_tense_recap(lowered: str) -> bool:
    """Detect backward-looking recaps ('ripped today')."""
    has_past = any(kw in lowered for kw in _PAST_TENSE_DIRECTIONAL)
    has_context = any(kw in lowered for kw in _PAST_TENSE_CONTEXT)
    return has_past and has_context


def _is_question(text: str) -> bool:
    """Detect questions that aren't assertions."""
    if text.rstrip().endswith("?"):
        return True
    return bool(_QUESTION_PATTERNS.search(text))


def _is_non_claim(lowered: str) -> bool:
    """Detect job postings, press releases, and other non-claims."""
    return any(kw in lowered for kw in _NON_CLAIM_KEYWORDS)


def _is_sarcastic(text: str) -> bool:
    """Detect sarcasm markers that invert or neutralize sentiment."""
    if any(e in text for e in _SARCASM_EMOJI):
        return True
    return bool(_SARCASM_PATTERNS.search(text))


def _classify_non_predictive(text: str) -> str | None:
    """Return a reason string if the tweet is non-predictive, else None."""
    lowered = text.lower()

    checks = [
        (_is_non_claim(lowered), "non-claim (job/press)"),
        (_is_question(text), "question"),
        (_is_sarcastic(text), "sarcasm"),
        (_is_long_term_thesis(lowered), "long-term thesis"),
        (_is_position_disclosure(text), "position disclosure"),
        (_is_informational(lowered), "informational/analytical"),
        (_is_past_tense_recap(lowered), "past tense recap"),
    ]
    for matched, reason in checks:
        if matched:
            return reason
    return None


# ---------------------------------------------------------------------------
# Negation-aware direction parsing
# ---------------------------------------------------------------------------

def _find_keyword_positions(lowered: str, words: list[str], keywords: set[str]) -> list[int]:
    """Find word indices where a keyword starts.

    Handles multi-word keywords by checking substrings starting at each
    word position.
    """
    positions = []
    for i, _ in enumerate(words):
        for kw in keywords:
            kw_words = kw.split()
            if i + len(kw_words) <= len(words):
                candidate = " ".join(words[i:i + len(kw_words)])
                if candidate == kw:
                    positions.append(i)
    return positions


def _has_negation_nearby(words: list[str], keyword_idx: int, window: int = 5) -> bool:
    """Check if a negation word appears within `window` words before keyword_idx."""
    start = max(0, keyword_idx - window)
    for i in range(start, keyword_idx):
        word = words[i].strip(".,!?;:'\"")
        if word in _NEGATION_WORDS:
            return True
        # Check for contractions like "isn't", "won't" etc.
        if "'t" in words[i] or "\u2019t" in words[i]:
            return True
    return False


def _score_with_negation(
    lowered: str,
    text: str,
    keywords: set[str],
    emoji_set: set[str],
) -> tuple[int, int]:
    """Score keyword matches, returning (positive_hits, negated_hits).

    positive_hits: keywords matched without nearby negation.
    negated_hits: keywords matched WITH nearby negation (should count for
    the opposite direction).
    """
    words = lowered.split()
    positions = _find_keyword_positions(lowered, words, keywords)

    positive = 0
    negated = 0
    for pos in positions:
        if _has_negation_nearby(words, pos):
            negated += 1
        else:
            positive += 1

    # Emoji are never negated
    positive += sum(1 for e in emoji_set if e in text)

    return positive, negated


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_direction_improved(text: str) -> tuple[Literal["up", "down", "neutral"], str]:
    """Parse claimed direction with NLP improvements.

    Returns (direction, reason) where reason explains why the direction
    was set (for debugging/auditing).
    """
    # Step 1: Check if tweet is non-predictive
    non_predictive_reason = _classify_non_predictive(text)
    if non_predictive_reason is not None:
        return "neutral", non_predictive_reason

    # Step 2: Negation-aware keyword scoring
    lowered = text.lower()

    up_pos, up_neg = _score_with_negation(lowered, text, _UP_SIGNALS, _UP_EMOJI)
    down_pos, down_neg = _score_with_negation(lowered, text, _DOWN_SIGNALS, _DOWN_EMOJI)

    # Negated bullish keywords count as bearish and vice versa
    effective_up = up_pos + down_neg
    effective_down = down_pos + up_neg

    if effective_up > 0 and effective_down > 0:
        return "neutral", "mixed signals"
    if effective_up > effective_down:
        reason = "bullish keywords"
        if down_neg > 0:
            reason += " (negated bearish)"
        return "up", reason
    if effective_down > effective_up:
        reason = "bearish keywords"
        if up_neg > 0:
            reason += " (negated bullish)"
        return "down", reason

    return "neutral", "no directional signal"


def label_claim_improved(raw: RawClaim) -> LabeledClaim:
    """Label a RawClaim using improved rules.

    Same return type as the naive labeler for compatibility.
    Writes to the improved_labeled_claims table.
    """
    claimed, reason = parse_direction_improved(raw.text)
    actual = _actual_direction(raw.price_change_pct)
    intensity = _intensity_score(raw.text)

    # Per-ticker threshold instead of fixed 2%
    threshold = TICKER_THRESHOLDS.get(raw.ticker, DEFAULT_THRESHOLD)
    threshold_pct = threshold * 100

    abs_move = abs(raw.price_change_pct) if raw.price_change_pct is not None else 0.0

    # Compare with naive labeler for logging
    naive_direction = parse_direction(raw.text)
    if naive_direction != claimed:
        logger.info(
            "Improved labeler disagrees on direction for tweet %s: "
            "naive=%s, improved=%s (reason: %s)",
            raw.tweet_id, naive_direction, claimed, reason,
        )

    # Default label
    label: Literal["exaggerated", "accurate", "understated"] = "accurate"

    if raw.price_change_pct is None:
        label = "accurate" if claimed == "neutral" else "exaggerated"

    elif claimed == "neutral" and abs_move < threshold_pct:
        label = "accurate"

    elif claimed == "neutral" and abs_move >= 5.0:
        label = "understated"

    elif claimed != "neutral" and actual != "neutral" and claimed != actual:
        label = "exaggerated"

    elif claimed != "neutral" and abs_move < threshold_pct:
        label = "exaggerated"

    elif (
        claimed != "neutral"
        and not raw.has_catalyst
        and raw.news_headlines
        and abs_move < threshold_pct * 2
    ):
        label = "exaggerated"

    elif claimed != "neutral" and claimed == actual and abs_move >= threshold_pct:
        label = "accurate"

    elif abs_move >= 10.0 and intensity < 0.3:
        label = "understated"

    # Log when improved label differs from what naive would produce
    # (direction differences already logged above; this catches threshold effects)
    naive_label = _naive_label_for_comparison(raw, naive_direction, actual, intensity)
    if naive_label != label:
        logger.info(
            "Improved labeler disagrees on label for tweet %s: "
            "naive=%s, improved=%s (ticker=%s, threshold=%.4f, reason=%s)",
            raw.tweet_id, naive_label, label, raw.ticker, threshold, reason,
        )

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


def _naive_label_for_comparison(
    raw: RawClaim,
    claimed: Literal["up", "down", "neutral"],
    actual: Literal["up", "down", "neutral"],
    intensity: float,
) -> Literal["exaggerated", "accurate", "understated"]:
    """Reproduce the naive labeler's logic for comparison logging.

    Uses the fixed 2% threshold to match the naive labeler's behavior.
    """
    threshold_pct = 2.0
    abs_move = abs(raw.price_change_pct) if raw.price_change_pct is not None else 0.0

    if raw.price_change_pct is None:
        return "accurate" if claimed == "neutral" else "exaggerated"
    if claimed == "neutral" and abs_move < threshold_pct:
        return "accurate"
    if claimed == "neutral" and abs_move >= 5.0:
        return "understated"
    if claimed != "neutral" and actual != "neutral" and claimed != actual:
        return "exaggerated"
    if claimed != "neutral" and abs_move < threshold_pct:
        return "exaggerated"
    if (
        claimed != "neutral"
        and not raw.has_catalyst
        and raw.news_headlines
        and abs_move < threshold_pct * 2
    ):
        return "exaggerated"
    if claimed != "neutral" and claimed == actual and abs_move >= threshold_pct:
        return "accurate"
    if abs_move >= 10.0 and intensity < 0.3:
        return "understated"
    return "accurate"
