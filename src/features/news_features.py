"""
News-based feature extraction for Sentinel.

Extracts scalar features from news context (catalyst presence, headline
sentiment, counts) and optional 384-dim headline embeddings for the
neural model.
"""

import logging
import re
from typing import Any

from ..data.models import LabeledClaim

logger = logging.getLogger("sentinel.features.news")

# Catalyst type one-hot keys
_CATALYST_TYPES = ["contract", "earnings", "geopolitical", "budget"]

_DOLLAR_AMOUNT_RE = re.compile(r"\$[\d,.]+\s*[BMKbmk]?\b")


def _vader_sentiment(text: str) -> float:
    """Get VADER compound sentiment score for text. Returns 0 on failure."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        return analyzer.polarity_scores(text)["compound"]
    except Exception:
        return 0.0


def extract_news_features(claim: LabeledClaim) -> dict[str, Any]:
    """Extract scalar news features from a labeled claim.

    Args:
        claim: LabeledClaim with news_headlines and catalyst fields populated.

    Returns:
        Dict of feature name -> value.
    """
    features: dict[str, Any] = {}

    features["has_catalyst"] = int(claim.has_catalyst)

    # One-hot catalyst type
    for ct in _CATALYST_TYPES:
        features[f"catalyst_{ct}"] = int(claim.catalyst_type == ct)

    features["headline_count"] = len(claim.news_headlines)

    # Average headline sentiment (VADER)
    if claim.news_headlines:
        sentiments = [_vader_sentiment(h) for h in claim.news_headlines]
        features["avg_headline_sentiment"] = sum(sentiments) / len(sentiments)
    else:
        features["avg_headline_sentiment"] = 0.0

    # Dollar amounts in headlines
    features["headline_mentions_dollar_amount"] = int(
        any(_DOLLAR_AMOUNT_RE.search(h) for h in claim.news_headlines)
    ) if claim.news_headlines else 0

    return features


def encode_headlines(
    headlines: list[str],
    model=None,
) -> list[float]:
    """Encode concatenated headlines into a 384-dim embedding.

    Uses sentence-transformers/all-MiniLM-L6-v2 (frozen).
    Used by the neural model only.

    Args:
        headlines: List of headline strings.
        model: Pre-loaded SentenceTransformer model. If None, loads lazily.

    Returns:
        384-dimensional embedding as list of floats.
        Returns zero vector if no headlines or on error.
    """
    dim = 384

    if not headlines:
        return [0.0] * dim

    combined = " ".join(headlines)

    try:
        if model is None:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")

        embedding = model.encode(combined, convert_to_numpy=True)
        return embedding.tolist()

    except Exception as e:
        logger.warning(f"Headline encoding failed: {e}")
        return [0.0] * dim


# Ordered list of scalar news feature names
NEWS_FEATURE_NAMES = [
    "has_catalyst",
    "catalyst_contract", "catalyst_earnings", "catalyst_geopolitical", "catalyst_budget",
    "headline_count",
    "avg_headline_sentiment",
    "headline_mentions_dollar_amount",
]
