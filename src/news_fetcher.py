"""
Defense news fetcher for Sentinel.

Fetches news articles around a claim's timestamp using yfinance ticker news
and DuckDuckGo news search. Classifies news catalysts by type.
"""

import asyncio
import logging
from datetime import datetime, timezone

from dateutil import parser as dateutil_parser

from .data.stocks import TICKER_NAMES

logger = logging.getLogger("sentinel.news_fetcher")

# Catalyst keyword sets, ordered by priority.
_CATALYST_KEYWORDS: dict[str, list[str]] = {
    "contract": [
        "contract", "award", "awarded", "pentagon", "dod", "department of defense",
        "idiq", "lrip", "billion", "million dollar", "deal",
    ],
    "earnings": [
        "earnings", "eps", "quarterly", "beat", "miss", "guidance",
        "q1", "q2", "q3", "q4", "revenue", "profit",
    ],
    "geopolitical": [
        "war", "conflict", "missile", "strike", "invasion", "ukraine",
        "taiwan", "iran", "nato", "troops", "military", "attack",
        "ceasefire", "escalation", "sanctions",
    ],
    "budget": [
        "ndaa", "defense budget", "appropriations", "continuing resolution",
        "defense spending", "sequestration",
    ],
}

# Priority order: contract > earnings > geopolitical > budget
_CATALYST_PRIORITY = ["contract", "earnings", "geopolitical", "budget"]


def classify_catalyst(headlines: list[str]) -> tuple[bool, str | None]:
    """Classify catalyst type from headline keywords.

    Args:
        headlines: List of news headline strings.

    Returns:
        (has_catalyst, catalyst_type) where catalyst_type is one of
        "contract", "earnings", "geopolitical", "budget", or None.
    """
    if not headlines:
        return False, None

    combined = " ".join(headlines).lower()

    for catalyst_type in _CATALYST_PRIORITY:
        keywords = _CATALYST_KEYWORDS[catalyst_type]
        if any(kw in combined for kw in keywords):
            return True, catalyst_type

    return False, None


def _parse_article_age_hours(date_str: str, reference: datetime | None = None) -> float | None:
    """Parse a date string and return absolute time distance in hours from reference time.

    Returns the unsigned distance so articles both before and after the
    reference are handled symmetrically.
    """
    if not date_str:
        return None
    try:
        parsed = dateutil_parser.parse(date_str, fuzzy=True)
        ref = reference or datetime.now(timezone.utc)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=timezone.utc)
        delta = ref - parsed
        return abs(delta.total_seconds() / 3600)
    except (ValueError, OverflowError):
        return None


def _filter_by_window(
    articles: list[dict],
    tweet_time: datetime,
    window_hours: int,
    date_key: str = "date",
) -> list[dict]:
    """Filter articles to those within ±window_hours of tweet_time."""
    filtered = []
    for article in articles:
        age = _parse_article_age_hours(
            str(article.get(date_key, "")),
            reference=tweet_time,
        )
        if age is not None and age <= window_hours:
            filtered.append(article)
    return filtered


async def fetch_news_for_claim(
    ticker: str,
    company_name: str | None = None,
    tweet_time: datetime | None = None,
    window_hours: int = 48,
) -> list[dict]:
    """Fetch news articles relevant to a defense stock claim.

    Two sources (both free, no API keys):
    1. yfinance Ticker.news — filtered to ±window around tweet time
    2. DuckDuckGo news search — defense-specific query

    Args:
        ticker: Defense stock ticker.
        company_name: Company name for search queries.
        tweet_time: When the tweet was posted (for time windowing).
        window_hours: Hours ± tweet_time to include articles.

    Returns:
        Deduplicated list of article dicts with keys: title, url, source, date.
    """
    if company_name is None:
        company_name = TICKER_NAMES.get(ticker, ticker)

    articles: list[dict] = []
    seen_urls: set[str] = set()

    # Source 1: yfinance ticker news
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        yf_news = t.news or []
        for item in yf_news:
            url = item.get("link", item.get("url", ""))
            if url in seen_urls:
                continue
            seen_urls.add(url)
            articles.append({
                "title": item.get("title", ""),
                "url": url,
                "source": item.get("publisher", item.get("source", "Unknown")),
                "date": item.get("providerPublishTime", item.get("publishedDate", "")),
            })
    except Exception as e:
        logger.warning(f"yfinance news fetch failed for {ticker}: {e}")

    # Source 2: DuckDuckGo news
    try:
        from ddgs import DDGS

        loop = asyncio.get_running_loop()
        query = f"{company_name} {ticker} defense"

        def run_ddg():
            ddgs = DDGS()
            return ddgs.news(query, max_results=5)

        results = await loop.run_in_executor(None, run_ddg)

        for r in results or []:
            url = r.get("url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)
            articles.append({
                "title": r.get("title", ""),
                "url": url,
                "source": r.get("source", "Unknown"),
                "date": r.get("date", ""),
            })
    except Exception as e:
        logger.warning(f"DuckDuckGo news fetch failed for {ticker}: {e}")

    # Filter by time window if tweet_time provided
    if tweet_time is not None:
        articles = _filter_by_window(articles, tweet_time, window_hours)

    logger.info(f"Fetched {len(articles)} news articles for {ticker}")
    return articles
