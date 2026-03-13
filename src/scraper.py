"""
Defense Stock Tweet Scraper for Sentinel.

Searches Twitter/X for tweets mentioning defense company names and tickers,
converts them to RawClaim objects with resolved tickers.

Uses twscrape for async scraping with automatic account rotation and rate limiting.
"""

import asyncio
import logging
from contextlib import aclosing
from dataclasses import dataclass, field
from datetime import datetime

from twscrape import API
from twscrape.models import Tweet

from .data.stocks import TICKER_NAMES, resolve_ticker, get_public_tickers
from .data.models import RawClaim

logger = logging.getLogger("sentinel.scraper")


@dataclass
class ScrapedTweet:
    """Normalized tweet data structure (internal to scraper)."""
    id: int
    text: str
    username: str
    display_name: str
    created_at: datetime
    likes: int
    retweets: int
    replies: int
    views: int | None
    language: str | None
    is_retweet: bool
    hashtags: list[str] = field(default_factory=list)

    @classmethod
    def from_twscrape(cls, tweet: Tweet) -> "ScrapedTweet":
        """Create ScrapedTweet from twscrape Tweet object."""
        hashtags = list(tweet.hashtags) if tweet.hashtags else []
        return cls(
            id=tweet.id,
            text=tweet.rawContent,
            username=tweet.user.username if tweet.user else "unknown",
            display_name=tweet.user.displayname if tweet.user else "Unknown",
            created_at=tweet.date,
            likes=tweet.likeCount or 0,
            retweets=tweet.retweetCount or 0,
            replies=tweet.replyCount or 0,
            views=tweet.viewCount,
            language=tweet.lang,
            is_retweet=tweet.rawContent.startswith("RT @") if tweet.rawContent else False,
            hashtags=hashtags,
        )

    def to_raw_claim(self, ticker: str, company_name: str) -> RawClaim:
        """Convert to a RawClaim with resolved ticker."""
        return RawClaim(
            tweet_id=self.id,
            text=self.text,
            username=self.username,
            created_at=self.created_at,
            likes=self.likes,
            retweets=self.retweets,
            ticker=ticker,
            company_name=company_name,
        )


def _build_search_queries(ticker: str) -> list[str]:
    """Build search queries for a defense ticker.

    For each ticker, we search the cashtag ($LMT) and the full company name.
    """
    queries = [f"${ticker}"]

    name = TICKER_NAMES.get(ticker)
    if name:
        queries.append(f'"{name}"')

    return queries


class DefenseStockScraper:
    """Scrapes tweets about defense stocks using twscrape."""

    def __init__(self, db_path: str = "accounts.db"):
        self.db_path = db_path
        self._api: API | None = None
        logger.info(f"DefenseStockScraper initialized with database: {db_path}")

    async def _get_api(self) -> API:
        """Get or create the twscrape API instance."""
        if self._api is None:
            self._api = API(self.db_path)
        return self._api

    async def get_account_stats(self) -> dict:
        """Get statistics about the account pool."""
        api = await self._get_api()
        stats = await api.pool.stats()
        logger.debug(f"Account pool stats: {stats}")
        return stats

    async def fix_locks(self) -> None:
        """Reset account locks after interrupted runs."""
        try:
            api = await self._get_api()
            await api.pool.reset_locks()
            logger.info("Account locks reset")
        except Exception as e:
            logger.error(f"Failed to reset locks: {e}")

    async def search_tweets(
        self,
        query: str,
        limit: int = 50,
        lang: str = "en",
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[ScrapedTweet]:
        """Search for tweets matching a query.

        twscrape handles rate limiting, account rotation, and request pacing
        internally — no manual delays or batching needed.

        Args:
            query: Search query (cashtag, company name, etc.).
            limit: Maximum number of tweets to retrieve.
            lang: Language filter.
            since: Only return tweets after this time (inclusive).
            until: Only return tweets before this time (exclusive).

        Returns:
            List of ScrapedTweet objects.
        """
        api = await self._get_api()
        tweets: list[ScrapedTweet] = []
        search_query = f"{query} lang:{lang}"

        if since is not None:
            search_query += f" since_time:{int(since.timestamp())}"
        if until is not None:
            search_query += f" until_time:{int(until.timestamp())}"

        logger.info(f"Searching: '{search_query}' (limit: {limit})")

        try:
            async with aclosing(api.search(search_query, limit=limit)) as gen:
                async for tweet in gen:
                    try:
                        tweets.append(ScrapedTweet.from_twscrape(tweet))
                    except Exception as e:
                        logger.warning(f"Failed to parse tweet {tweet.id}: {e}")

            logger.info(f"Retrieved {len(tweets)} tweets for: {query}")
            return tweets

        except Exception as e:
            logger.error(f"Error searching '{query}': {e}")
            return tweets

    async def scrape_defense_claims(
        self,
        tickers: list[str] | None = None,
        limit_per_ticker: int = 50,
        on_ticker_done: callable = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[RawClaim]:
        """Scrape tweets about defense stocks and convert to RawClaims.

        Launches all ticker searches concurrently via asyncio.gather — twscrape's
        account pool handles rotation and rate limiting across the parallel tasks.

        Args:
            tickers: Tickers to scrape. Defaults to all public defense tickers.
            limit_per_ticker: Max tweets per ticker.
            on_ticker_done: Optional callback(ticker, claims_found, tickers_done, tickers_total)
                called after each ticker completes.
            since: Only return tweets after this time.
            until: Only return tweets before this time.

        Returns:
            Deduplicated list of RawClaims.
        """
        if tickers is None:
            tickers = get_public_tickers()

        time_desc = ""
        if since or until:
            parts = []
            if since:
                parts.append(f"since {since.isoformat()}")
            if until:
                parts.append(f"until {until.isoformat()}")
            time_desc = f" ({', '.join(parts)})"
        logger.info(f"Scraping {len(tickers)} defense tickers, {limit_per_ticker} tweets each{time_desc}")

        seen_tweet_ids: set[int] = set()
        all_claims: list[RawClaim] = []

        async def process_ticker(ticker: str) -> list[RawClaim]:
            queries = _build_search_queries(ticker)
            company = TICKER_NAMES.get(ticker, ticker)
            ticker_tweets: list[ScrapedTweet] = []

            for query in queries:
                results = await self.search_tweets(
                    query, limit=limit_per_ticker // len(queries),
                    since=since, until=until,
                )
                ticker_tweets.extend(results)

            claims = []
            for tweet in ticker_tweets:
                if tweet.id in seen_tweet_ids:
                    continue
                if tweet.is_retweet:
                    continue
                seen_tweet_ids.add(tweet.id)

                resolved = resolve_ticker(tweet.text)
                if resolved is None:
                    resolved = ticker
                resolved_name = TICKER_NAMES.get(resolved, company)
                claims.append(tweet.to_raw_claim(resolved, resolved_name))

            logger.info(f"${ticker}: {len(claims)} claims from {len(ticker_tweets)} tweets")
            return claims

        tasks = [process_ticker(ticker) for ticker in tickers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, (ticker, result) in enumerate(zip(tickers, results)):
            if isinstance(result, Exception):
                logger.error(f"Failed to scrape ${ticker}: {result}")
                continue
            all_claims.extend(result)

            if on_ticker_done:
                on_ticker_done(ticker, len(all_claims), i + 1, len(tickers))

        logger.info(f"Scraping complete: {len(all_claims)} total claims from {len(tickers)} tickers")
        return all_claims

    async def close(self) -> None:
        """Clean up resources."""
        logger.debug("Scraper resources cleaned up")
