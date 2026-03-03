"""
Defense Stock Tweet Scraper for Sentinel.

Searches Twitter/X for tweets mentioning defense company names and tickers,
converts them to RawClaim objects with resolved tickers.

Uses twscrape for async scraping with account pool management and rate limiting.
"""

import asyncio
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime

from twscrape import API
from twscrape.models import Tweet

from .config import worker_context
from .data.stocks import DEFENSE_STOCKS, TICKER_NAMES, resolve_ticker, get_public_tickers
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
        timeout: int = 300,
    ) -> list[ScrapedTweet]:
        """Search for tweets matching a query.

        Args:
            query: Search query (cashtag, company name, etc.).
            limit: Maximum number of tweets to retrieve.
            lang: Language filter.
            timeout: Maximum time in seconds.

        Returns:
            List of ScrapedTweet objects.
        """
        api = await self._get_api()
        tweets: list[ScrapedTweet] = []
        search_query = f"{query} lang:{lang}"

        # Check account availability
        try:
            stats = await api.pool.stats()
            active = stats.get("active", 0)
            total = stats.get("total", 0)
            if total > 0 and active == 0:
                logger.warning(
                    f"All {total} accounts are rate-limited. "
                    "Increasing timeout to allow waiting..."
                )
                timeout = max(timeout, 960)
        except Exception as e:
            logger.debug(f"Could not check account availability: {e}")

        if limit > 100:
            logger.warning(
                f"High tweet limit ({limit}) may trigger rate limits. "
                "Consider reducing limit_per_ticker."
            )

        logger.info(f"Searching: '{search_query}' (limit: {limit}, timeout: {timeout}s)")
        safety_timeout = 1200

        try:
            raw_tweets = []
            jitter = random.uniform(1, 5)
            await asyncio.sleep(jitter)

            async def fetch_with_delays():
                count = 0
                async for tweet in api.search(search_query, limit=limit):
                    raw_tweets.append(tweet)
                    count += 1
                    if count % 15 == 0:
                        delay = random.uniform(10, 15)
                        logger.debug(f"Search '{query}': {count} tweets, pacing {delay:.1f}s")
                        await asyncio.sleep(delay)
                return raw_tweets

            await asyncio.wait_for(fetch_with_delays(), timeout=safety_timeout)

            for tweet in raw_tweets:
                try:
                    scraped = ScrapedTweet.from_twscrape(tweet)
                    tweets.append(scraped)
                except Exception as e:
                    logger.warning(f"Failed to parse tweet {tweet.id}: {e}")

            logger.info(f"Retrieved {len(tweets)} tweets for: {query}")
            return tweets

        except asyncio.TimeoutError:
            logger.error(f"Timeout for '{query}' after {safety_timeout}s")
            return tweets
        except Exception as e:
            logger.error(f"Error searching '{query}': {e}")
            return []

    async def scrape_defense_claims(
        self,
        tickers: list[str] | None = None,
        limit_per_ticker: int = 50,
        on_ticker_done: callable = None,
    ) -> list[RawClaim]:
        """Scrape tweets about defense stocks and convert to RawClaims.

        Args:
            tickers: Tickers to scrape. Defaults to all public defense tickers.
            limit_per_ticker: Max tweets per ticker.
            on_ticker_done: Optional callback(ticker, claims_found, tickers_done, tickers_total)
                called after each ticker batch completes.

        Returns:
            Deduplicated list of RawClaims.
        """
        if tickers is None:
            tickers = get_public_tickers()

        logger.info(f"Scraping {len(tickers)} defense tickers, {limit_per_ticker} tweets each")
        all_claims: list[RawClaim] = []
        seen_tweet_ids: set[int] = set()

        stats = await self.get_account_stats()
        active_count = max(stats.get("active", 1), 1)
        workers_per_batch = min(active_count, 2)

        # Process tickers in batches to manage rate limits
        ticker_index = 0
        batch_number = 0

        while ticker_index < len(tickers):
            batch_tickers = tickers[ticker_index:ticker_index + workers_per_batch]

            async def process_ticker(wid: int, ticker: str):
                worker_context.set(wid)
                queries = _build_search_queries(ticker)
                company = TICKER_NAMES.get(ticker, ticker)
                ticker_tweets: list[ScrapedTweet] = []

                for query in queries:
                    results = await self.search_tweets(
                        query, limit=limit_per_ticker // len(queries)
                    )
                    ticker_tweets.extend(results)

                claims = []
                for tweet in ticker_tweets:
                    if tweet.id in seen_tweet_ids:
                        continue
                    if tweet.is_retweet:
                        continue
                    seen_tweet_ids.add(tweet.id)

                    # Resolve ticker from text (may match a different defense stock)
                    resolved = resolve_ticker(tweet.text)
                    if resolved is None:
                        resolved = ticker
                    resolved_name = TICKER_NAMES.get(resolved, company)

                    claims.append(tweet.to_raw_claim(resolved, resolved_name))

                logger.info(f"${ticker}: {len(claims)} claims from {len(ticker_tweets)} tweets")
                return claims

            base_slot = (batch_number * workers_per_batch) % active_count
            worker_slots = [(base_slot + i) % active_count for i in range(len(batch_tickers))]

            tasks = [
                process_ticker(worker_slots[i], ticker)
                for i, ticker in enumerate(batch_tickers)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for ticker, result in zip(batch_tickers, results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to scrape ${ticker}: {result}")
                    continue
                all_claims.extend(result)

            ticker_index += len(batch_tickers)
            batch_number += 1

            if on_ticker_done:
                last_ticker = batch_tickers[-1]
                on_ticker_done(last_ticker, len(all_claims), ticker_index, len(tickers))

            if ticker_index < len(tickers):
                await asyncio.sleep(5)

        logger.info(f"Scraping complete: {len(all_claims)} total claims from {len(tickers)} tickers")
        return all_claims

    async def close(self) -> None:
        """Clean up resources."""
        logger.debug("Scraper resources cleaned up")
