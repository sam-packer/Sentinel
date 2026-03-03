"""Test fixtures and factories for Sentinel tests."""

from datetime import datetime

from src.data.models import RawClaim, LabeledClaim


def make_raw_claim(
    tweet_id: int = 1234567890,
    text: str = "$LMT looking bullish! 🚀 Lockheed just got a huge contract",
    username: str = "defensetrader",
    created_at: datetime | None = None,
    likes: int = 100,
    retweets: int = 50,
    ticker: str = "LMT",
    company_name: str = "Lockheed Martin",
    price_at_tweet: float | None = 450.0,
    price_24h_later: float | None = 455.0,
    price_change_pct: float | None = 1.11,
    news_headlines: list[str] | None = None,
    has_catalyst: bool = False,
    catalyst_type: str | None = None,
) -> RawClaim:
    """Create a sample RawClaim for testing."""
    return RawClaim(
        tweet_id=tweet_id,
        text=text,
        username=username,
        created_at=created_at or datetime(2024, 6, 15, 14, 30, 0),
        likes=likes,
        retweets=retweets,
        ticker=ticker,
        company_name=company_name,
        price_at_tweet=price_at_tweet,
        price_24h_later=price_24h_later,
        price_change_pct=price_change_pct,
        news_headlines=news_headlines or [],
        has_catalyst=has_catalyst,
        catalyst_type=catalyst_type,
    )


def make_labeled_claim(
    label: str = "accurate",
    claimed_direction: str = "up",
    actual_direction: str = "up",
    exaggeration_score: float = 0.1,
    news_summary: str = "",
    **raw_kwargs,
) -> LabeledClaim:
    """Create a sample LabeledClaim for testing."""
    raw = make_raw_claim(**raw_kwargs)
    return LabeledClaim.from_raw(
        raw,
        label=label,
        claimed_direction=claimed_direction,
        actual_direction=actual_direction,
        exaggeration_score=exaggeration_score,
        news_summary=news_summary,
    )


def make_defense_claims(count: int = 5) -> list[RawClaim]:
    """Create a batch of varied defense stock claims."""
    templates = [
        {
            "text": "$LMT to the moon! 🚀🚀🚀 Lockheed about to explode!",
            "ticker": "LMT", "company_name": "Lockheed Martin",
            "price_change_pct": 0.44, "has_catalyst": False,
        },
        {
            "text": "RTX awarded $2B Pentagon contract for next-gen radar",
            "ticker": "RTX", "company_name": "RTX Corporation",
            "price_change_pct": 5.0, "has_catalyst": True, "catalyst_type": "contract",
            "news_headlines": ["RTX wins $2B Pentagon radar contract"],
        },
        {
            "text": "NOC looks interesting here. Watching.",
            "ticker": "NOC", "company_name": "Northrop Grumman",
            "price_change_pct": 8.33, "has_catalyst": True, "catalyst_type": "geopolitical",
        },
        {
            "text": "$BA is crashing! Get out now! 📉📉📉",
            "ticker": "BA", "company_name": "Boeing",
            "price_change_pct": -0.3, "has_catalyst": False,
        },
        {
            "text": "PLTR earnings beat expectations. Q4 revenue up 20%.",
            "ticker": "PLTR", "company_name": "Palantir Technologies",
            "price_change_pct": 12.5, "has_catalyst": True, "catalyst_type": "earnings",
            "news_headlines": ["Palantir beats Q4 earnings estimates"],
        },
    ]

    claims = []
    for i, tmpl in enumerate(templates[:count]):
        claims.append(make_raw_claim(
            tweet_id=9000000 + i,
            price_at_tweet=100.0,
            price_24h_later=100.0 * (1 + tmpl["price_change_pct"] / 100),
            **tmpl,
        ))
    return claims
