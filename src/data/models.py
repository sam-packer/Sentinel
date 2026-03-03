"""
Core data models for Sentinel.

RawClaim: a tweet mentioning a defense stock, enriched with price and news data.
LabeledClaim: a RawClaim that has been classified as exaggerated/accurate/understated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


@dataclass
class RawClaim:
    """A tweet about a defense stock, enriched with price and news context."""
    tweet_id: int
    text: str
    username: str
    created_at: datetime
    likes: int
    retweets: int
    ticker: str
    company_name: str
    price_at_tweet: float | None = None
    price_24h_later: float | None = None
    price_change_pct: float | None = None
    news_headlines: list[str] = field(default_factory=list)
    has_catalyst: bool = False
    catalyst_type: str | None = None  # "contract" | "geopolitical" | "earnings" | "budget"


@dataclass
class LabeledClaim:
    """A RawClaim with a classification label attached."""
    tweet_id: int
    text: str
    username: str
    created_at: datetime
    likes: int
    retweets: int
    ticker: str
    company_name: str
    price_at_tweet: float | None
    price_24h_later: float | None
    price_change_pct: float | None
    news_headlines: list[str]
    has_catalyst: bool
    catalyst_type: str | None
    # Label fields
    label: Literal["exaggerated", "accurate", "understated"]
    claimed_direction: Literal["up", "down", "neutral"]
    actual_direction: Literal["up", "down", "neutral"]
    exaggeration_score: float  # 0.0 (perfectly accurate) to 1.0 (wildly off)
    news_summary: str  # one-line top headline for display

    @classmethod
    def from_raw(
        cls,
        raw: RawClaim,
        *,
        label: Literal["exaggerated", "accurate", "understated"],
        claimed_direction: Literal["up", "down", "neutral"],
        actual_direction: Literal["up", "down", "neutral"],
        exaggeration_score: float,
        news_summary: str = "",
    ) -> LabeledClaim:
        """Create a LabeledClaim from a RawClaim plus label fields."""
        return cls(
            tweet_id=raw.tweet_id,
            text=raw.text,
            username=raw.username,
            created_at=raw.created_at,
            likes=raw.likes,
            retweets=raw.retweets,
            ticker=raw.ticker,
            company_name=raw.company_name,
            price_at_tweet=raw.price_at_tweet,
            price_24h_later=raw.price_24h_later,
            price_change_pct=raw.price_change_pct,
            news_headlines=raw.news_headlines,
            has_catalyst=raw.has_catalyst,
            catalyst_type=raw.catalyst_type,
            label=label,
            claimed_direction=claimed_direction,
            actual_direction=actual_direction,
            exaggeration_score=exaggeration_score,
            news_summary=news_summary,
        )
