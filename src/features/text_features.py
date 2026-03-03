"""
Text and linguistic feature extraction for Sentinel.

Extracts intensity, defense-specific, and specificity features from
claim text for use in classical and neural models.
"""

import math
import re
from typing import Any

from ..data.models import RawClaim

# Defense program names for has_specific_program feature
_DEFENSE_PROGRAMS = [
    "f-35", "f-22", "f/a-18", "b-21", "b-2", "kc-46",
    "himars", "patriot", "thaad", "javelin", "stinger",
    "abrams", "bradley", "aegis", "virginia class", "columbia class",
    "sentinel", "minuteman", "trident", "harpoon", "tomahawk",
    "mq-25", "mq-9", "rq-180", "ngad", "cca",
    "ch-53k", "v-22", "osprey", "blackhawk", "apache",
]

# Competitor mentions (defense companies mentioning each other)
_COMPETITOR_NAMES = [
    "lockheed", "raytheon", "northrop", "boeing", "general dynamics",
    "l3harris", "huntington", "leidos", "saic", "booz allen",
    "kratos", "palantir", "anduril", "shield ai", "rocket lab",
]

# Emoji sets
_BULLISH_EMOJI = {"🚀", "📈", "💎", "🔥", "💰", "🐂", "💪", "🎯"}
_BEARISH_EMOJI = {"📉", "🔻", "💀", "🐻", "⚠️", "🩸", "💩"}

# Superlatives
_SUPERLATIVES = [
    "insane", "massive", "huge", "crazy", "unbelievable", "incredible",
    "unprecedented", "enormous", "tremendous", "explosive", "parabolic",
    "absurd", "ridiculous", "unstoppable", "historic",
]

# Catalyst keyword groups (reused from labeler for consistency)
_CONTRACT_KEYWORDS = [
    "contract", "award", "awarded", "pentagon", "dod",
    "idiq", "lrip", "billion", "deal",
]
_GEOPOLITICAL_KEYWORDS = [
    "war", "conflict", "missile", "strike", "invasion", "ukraine",
    "taiwan", "iran", "nato", "troops", "military", "attack",
]
_EARNINGS_KEYWORDS = [
    "earnings", "eps", "quarterly", "beat", "miss", "guidance",
    "revenue", "profit",
]
_BUDGET_KEYWORDS = [
    "ndaa", "defense budget", "appropriations", "defense spending",
]

# Regex patterns
_DOLLAR_AMOUNT_RE = re.compile(r"\$[\d,.]+\s*[BMKbmk]?\b")
_PERCENTAGE_RE = re.compile(r"\d+(\.\d+)?%")
_PRICE_TARGET_RE = re.compile(r"\$\d+(\.\d+)?(?!\w)")  # $450, $12.50 but not $LMT
_TIMEFRAME_RE = re.compile(
    r"\b(by|before|within|eod|eow|end of|next week|next month|tomorrow|today|this week)\b",
    re.IGNORECASE,
)
_HASHTAG_RE = re.compile(r"#\w+")
_MENTION_RE = re.compile(r"@\w+")
_CASHTAG_RE = re.compile(r"\$[A-Z]{1,5}\b")


def extract_text_features(claim: RawClaim, nlp=None) -> dict[str, Any]:
    """Extract linguistic features from claim text.

    Args:
        claim: RawClaim with text and metadata.
        nlp: Optional spaCy Language model for NER features.
             If None, NER features are set to 0.

    Returns:
        Dict of feature name -> value.
    """
    text = claim.text
    lowered = text.lower()
    words = text.split()

    features: dict[str, Any] = {}

    # === Intensity features ===
    features["exclamation_count"] = text.count("!")
    features["caps_ratio"] = (
        sum(1 for c in text if c.isupper()) / max(len(text), 1)
    )
    features["emoji_bullish_count"] = sum(1 for e in _BULLISH_EMOJI if e in text)
    features["emoji_bearish_count"] = sum(1 for e in _BEARISH_EMOJI if e in text)
    features["superlative_count"] = sum(1 for s in _SUPERLATIVES if s in lowered)
    features["all_caps_word_count"] = sum(
        1 for w in words if w.isupper() and len(w) > 2 and not _CASHTAG_RE.match(w)
    )

    # === Defense-specific features ===
    features["mentions_contract"] = int(
        any(kw in lowered for kw in _CONTRACT_KEYWORDS)
    )
    features["mentions_geopolitical"] = int(
        any(kw in lowered for kw in _GEOPOLITICAL_KEYWORDS)
    )
    features["mentions_earnings"] = int(
        any(kw in lowered for kw in _EARNINGS_KEYWORDS)
    )
    features["mentions_budget"] = int(
        any(kw in lowered for kw in _BUDGET_KEYWORDS)
    )
    features["dollar_amount_mentioned"] = int(bool(_DOLLAR_AMOUNT_RE.search(text)))
    features["has_specific_program"] = int(
        any(prog in lowered for prog in _DEFENSE_PROGRAMS)
    )
    features["mentions_competitor"] = int(
        sum(1 for comp in _COMPETITOR_NAMES if comp in lowered) > 1
    )

    # === Specificity features ===
    features["has_percentage"] = int(bool(_PERCENTAGE_RE.search(text)))
    features["has_price_target"] = int(bool(_PRICE_TARGET_RE.search(text)))
    features["has_timeframe"] = int(bool(_TIMEFRAME_RE.search(text)))
    features["hashtag_count"] = len(_HASHTAG_RE.findall(text))
    features["mention_count"] = len(_MENTION_RE.findall(text))

    # === Engagement features ===
    features["log_likes"] = math.log1p(claim.likes)
    features["log_retweets"] = math.log1p(claim.retweets)

    # === NER features (optional) ===
    if nlp is not None:
        doc = nlp(text)
        features["entity_count"] = len(doc.ents)
        features["has_org_entity"] = int(any(e.label_ == "ORG" for e in doc.ents))
        features["has_money_entity"] = int(any(e.label_ == "MONEY" for e in doc.ents))
        features["has_gpe_entity"] = int(any(e.label_ == "GPE" for e in doc.ents))
    else:
        features["entity_count"] = 0
        features["has_org_entity"] = 0
        features["has_money_entity"] = 0
        features["has_gpe_entity"] = 0

    return features


# Ordered list of feature names for consistent vectorization
FEATURE_NAMES = [
    "exclamation_count", "caps_ratio", "emoji_bullish_count", "emoji_bearish_count",
    "superlative_count", "all_caps_word_count",
    "mentions_contract", "mentions_geopolitical", "mentions_earnings", "mentions_budget",
    "dollar_amount_mentioned", "has_specific_program", "mentions_competitor",
    "has_percentage", "has_price_target", "has_timeframe", "hashtag_count", "mention_count",
    "log_likes", "log_retweets",
    "entity_count", "has_org_entity", "has_money_entity", "has_gpe_entity",
]
