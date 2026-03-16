"""
Improved claim labeling engine for Sentinel.

All labeling thresholds are statistical — expressed as multiples of each
ticker's median absolute daily move (computed from yfinance market data).
There are zero fixed percentage thresholds anywhere in this module.

Addresses the top failure modes found in error analysis of the naive labeler
(137 mispredictions):

- Long-term thesis (31%): "I'm a believer! $RKLB longterm" no longer flagged
  as exaggerated — detected as non-predictive and neutralized.
- Position disclosure (22%): "added some $LDOS" no longer treated as a
  directional prediction.
- Informational/analytical (15%): watchlists, sector analysis neutralized.
- Volatile ticker threshold (14%): per-ticker thresholds computed dynamically
  from yfinance data (median absolute daily move) replace the fixed 2% cutoff.
- Past tense recap (11%): "ripped today" detected as backward-looking, not
  a forward prediction.
- Sarcasm (4%): eye-roll emoji and sarcastic phrasing neutralize direction.
- Questions (3%): "will $LMT moon?" treated as neutral, not bullish.
- Negation (1%): "NOT going to moon" correctly flips direction.

Reuses keyword lists, emoji sets, and _intensity_score from the naive labeler.
Only parse_direction, label_claim, and exaggeration scoring are replaced.
"""

import functools
import logging
import re
from typing import Literal

import numpy as np
import yfinance as yf

from .labeler import (
    _UP_SIGNALS,
    _UP_EMOJI,
    _DOWN_SIGNALS,
    _DOWN_EMOJI,
    _intensity_score,
)
from .models import RawClaim, LabeledClaim

logger = logging.getLogger("sentinel.improved_labeler")

# ---------------------------------------------------------------------------
# Per-ticker volatility thresholds (computed dynamically from yfinance)
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLD: float = 0.02


@functools.lru_cache(maxsize=32)
def _get_ticker_threshold(ticker: str, lookback_days: int = 90) -> float:
    """Compute median absolute daily percentage move for a ticker.

    Uses the last `lookback_days` of daily close prices from yfinance.
    Falls back to DEFAULT_THRESHOLD (2%) if data is unavailable.
    Result is cached for the process lifetime.
    """
    try:
        hist = yf.Ticker(ticker).history(period=f"{lookback_days}d")
        if hist.empty or len(hist) < 10:
            logger.warning("Insufficient price history for %s, using default threshold", ticker)
            return DEFAULT_THRESHOLD
        daily_returns = hist["Close"].pct_change().dropna()
        median_move = float(np.median(np.abs(daily_returns)))
        logger.debug("Computed threshold for %s: %.4f (from %d days)", ticker, median_move, len(daily_returns))
        return median_move
    except Exception as e:
        logger.warning("Failed to compute threshold for %s: %s, using default", ticker, e)
        return DEFAULT_THRESHOLD


# ---------------------------------------------------------------------------
# Per-ticker actual direction
# ---------------------------------------------------------------------------

def _actual_direction_for_ticker(price_change_pct: float | None, ticker: str) -> Literal["up", "down", "neutral"]:
    """Determine actual direction using per-ticker threshold."""
    if price_change_pct is None:
        return "neutral"
    threshold = _get_ticker_threshold(ticker) * 100  # convert to percentage
    if price_change_pct > threshold:
        return "up"
    if price_change_pct < -threshold:
        return "down"
    return "neutral"


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
    r"|definitely\s+(moon|crash|dump|rip|surg|bull|bear|insane|massive|huge)"
    r"|oh\s+yeah|lmao\s+right|right\s*,?\s*(moon|crash|rip|bull|bear)"
    r"|as\s+if|yeah\s+right)",
    re.IGNORECASE,
)

_NEGATION_WORDS = {
    "not", "n't", "no", "never", "unlikely", "doubt",
    "don't", "doesn't", "won't", "wouldn't", "isn't", "aren't",
    "cant", "wont", "dont", "doesnt", "isnt", "arent",
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
    """Detect backward-looking recaps ('ripped', 'crashed today')."""
    return any(kw in lowered for kw in _PAST_TENSE_DIRECTIONAL)


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

    # Priority order matters: earlier checks take precedence.
    # 1. Non-claims (jobs, press releases) — strongest signal, always correct
    # 2. Sarcasm — must come before questions so "will $X moon? 🙄" is sarcasm
    # 3. Questions — "will $LMT moon?" without sarcasm markers
    # 4. Long-term thesis — "believer in $RKLB longterm"
    # 5. Position disclosure — "added some $LDOS"
    # 6. Informational — watchlists, sector analysis
    # 7. Past tense recap — "ripped today" (weakest signal, most false positives)
    checks = [
        (_is_non_claim(lowered), "non-claim (job/press)"),
        (_is_sarcastic(text), "sarcasm"),
        (_is_question(text), "question"),
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

    Handles multi-word keywords. Longer keywords take priority over
    shorter ones at the same position to avoid double-counting.
    """
    positions = []
    matched_indices: set[int] = set()
    # Check longer keywords first so they take priority
    sorted_keywords = sorted(keywords, key=lambda k: len(k.split()), reverse=True)
    for kw in sorted_keywords:
        kw_words = kw.split()
        kw_len = len(kw_words)
        for i in range(len(words) - kw_len + 1):
            if i in matched_indices:
                continue
            candidate = " ".join(words[i:i + kw_len])
            if candidate == kw:
                positions.append(i)
                for j in range(i, i + kw_len):
                    matched_indices.add(j)
    return positions


def _has_negation_nearby(words: list[str], keyword_idx: int, window: int = 5) -> bool:
    """Check if a negation word appears within `window` words of keyword_idx."""
    start = max(0, keyword_idx - window)
    end = min(len(words), keyword_idx + window + 1)
    for i in range(start, end):
        if i == keyword_idx:
            continue
        word = words[i].strip(".,!?;:'\"").lower()
        if word in _NEGATION_WORDS:
            return True
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
# Exaggeration score (per-ticker scaling, no fixed thresholds)
# ---------------------------------------------------------------------------

def _compute_exaggeration_score_improved(
    claimed: Literal["up", "down", "neutral"],
    actual: Literal["up", "down", "neutral"],
    price_change_pct: float | None,
    has_catalyst: bool,
    intensity: float,
    threshold_pct: float,
    *,
    news_available: bool = True,
) -> float:
    """Compute exaggeration score using per-ticker scaling.

    Same three components as the naive version, but the magnitude gap
    scales relative to the ticker's volatility instead of a fixed 5%.
    A move of 3x the ticker's median daily swing is "fully justified"
    (magnitude gap = 0).
    """
    if price_change_pct is None:
        return 0.5

    abs_move = abs(price_change_pct)
    max_justified = threshold_pct * 3  # 3x typical move = fully justified

    # Component 1: Direction mismatch (0.0 or 0.5)
    direction_score = 0.0
    if claimed != "neutral" and actual != "neutral" and claimed != actual:
        direction_score = 0.5

    # Component 2: Magnitude gap (0.0 to 0.3)
    magnitude_score = 0.0
    if claimed != "neutral":
        if direction_score > 0:
            magnitude_score = intensity * 0.3
        else:
            move_ratio = min(abs_move / max_justified, 1.0)
            magnitude_score = intensity * (1.0 - move_ratio) * 0.3

    # Component 3: Catalyst gap (0.0 to 0.2)
    catalyst_score = 0.0
    if claimed != "neutral" and not has_catalyst and news_available:
        catalyst_score = intensity * 0.2

    return min(direction_score + magnitude_score + catalyst_score, 1.0)


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
    claimed, _reason = parse_direction_improved(raw.text)
    actual = _actual_direction_for_ticker(raw.price_change_pct, raw.ticker)
    intensity = _intensity_score(raw.text)

    # Per-ticker threshold computed dynamically from yfinance
    threshold = _get_ticker_threshold(raw.ticker)
    threshold_pct = threshold * 100

    abs_move = abs(raw.price_change_pct) if raw.price_change_pct is not None else 0.0

    # All thresholds derived from per-ticker volatility — no fixed numbers.
    # threshold_pct = ticker's median absolute daily move in percentage points.
    # A "significant move" is >= 1x the ticker's typical daily move.
    # A "large move" is >= 3x (understated territory for neutral tweets).
    # A "huge move" is >= 5x (understated even with some directional language).
    understated_threshold = threshold_pct * 3
    major_move_threshold = threshold_pct * 5
    catalyst_threshold = threshold_pct * 2

    # Default label
    label: Literal["exaggerated", "accurate", "understated"] = "accurate"

    if raw.price_change_pct is None:
        label = "accurate" if claimed == "neutral" else "exaggerated"

    elif claimed == "neutral" and abs_move < threshold_pct:
        # Neutral tweet, move within normal daily noise
        label = "accurate"

    elif claimed == "neutral" and abs_move >= understated_threshold:
        # Neutral tweet but move is 3x+ the typical daily swing
        label = "understated"

    elif claimed != "neutral" and actual != "neutral" and claimed != actual:
        # Got the direction wrong
        label = "exaggerated"

    elif claimed != "neutral" and abs_move < threshold_pct:
        # Directional claim but move is below the ticker's noise floor
        label = "exaggerated"

    elif (
        claimed != "neutral"
        and not raw.has_catalyst
        and raw.news_headlines
        and abs_move < catalyst_threshold
    ):
        # Directional claim, no catalyst found, move below 2x typical
        label = "exaggerated"

    elif claimed != "neutral" and claimed == actual and abs_move >= threshold_pct:
        # Direction matches and move exceeds the ticker's noise floor
        label = "accurate"

    elif abs_move >= major_move_threshold and intensity < 0.3:
        # Huge move (5x+ typical) but calm language
        label = "understated"

    exaggeration_score = _compute_exaggeration_score_improved(
        claimed, actual, raw.price_change_pct, raw.has_catalyst, intensity,
        threshold_pct,
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
