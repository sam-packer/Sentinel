"""
Defense stock price fetcher for Sentinel.

Uses yfinance to fetch historical prices at specific timestamps and compute
price moves over 24h windows. Handles market-closed hours by using nearest
available close.
"""

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import NamedTuple

import yfinance as yf

from .data.stocks import get_public_tickers

logger = logging.getLogger("sentinel.price_fetcher")


class PriceMove(NamedTuple):
    """Result of a price move calculation."""
    price_at: float | None
    price_after: float | None
    change_pct: float | None


class PriceFetcher:
    """Fetches historical prices for defense stocks via yfinance."""

    def __init__(self, cache_ttl_seconds: int = 300):
        self._cache: dict[str, tuple[float, any]] = {}  # key -> (expire_time, data)
        self._ttl = cache_ttl_seconds

    def _cache_get(self, key: str):
        """Get value from cache if not expired."""
        if key in self._cache:
            expire, data = self._cache[key]
            if time.time() < expire:
                return data
            del self._cache[key]
        return None

    def _cache_set(self, key: str, data):
        """Store value in cache."""
        self._cache[key] = (time.time() + self._ttl, data)

    def _validate_ticker(self, ticker: str) -> bool:
        """Check that ticker is in our defense stock universe."""
        return ticker in get_public_tickers()

    def get_price_at_time(self, ticker: str, dt: datetime) -> float | None:
        """Get the closing price nearest to the given datetime.

        For market hours: returns the nearest hourly close.
        For after-hours: returns the last available close.

        Args:
            ticker: Defense stock ticker (e.g. "LMT").
            dt: Target datetime.

        Returns:
            Price as float, or None if unavailable.
        """
        if not self._validate_ticker(ticker):
            logger.warning(f"Ticker {ticker} not in defense stock universe")
            return None

        cache_key = f"price:{ticker}:{dt.strftime('%Y%m%d%H')}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            # Fetch 5 days of hourly data around the target time
            start = dt - timedelta(days=3)
            end = dt + timedelta(days=1)

            t = yf.Ticker(ticker)
            hist = t.history(start=start, end=end, interval="1h")

            if hist.empty:
                # Fallback to daily data
                hist = t.history(start=start, end=end + timedelta(days=1), interval="1d")
                if hist.empty:
                    logger.warning(f"No price data for {ticker} around {dt}")
                    return None

            # Find nearest row to target datetime
            target_ts = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
            # Convert index to UTC for comparison
            idx = hist.index
            if hasattr(idx, 'tz_localize'):
                try:
                    idx = idx.tz_localize('UTC')
                except TypeError:
                    idx = idx.tz_convert('UTC')

            # Find closest timestamp
            deltas = abs(idx - target_ts)
            closest_idx = deltas.argmin()
            price = float(hist.iloc[closest_idx]["Close"])

            self._cache_set(cache_key, price)
            return price

        except Exception as e:
            logger.error(f"Failed to fetch price for {ticker} at {dt}: {e}")
            return None

    def get_price_change(
        self, ticker: str, dt: datetime, hours: int = 24
    ) -> PriceMove:
        """Get price at tweet time and N hours later, compute percent change.

        Args:
            ticker: Defense stock ticker.
            dt: Tweet timestamp.
            hours: Window to measure (default 24h).

        Returns:
            PriceMove(price_at, price_after, change_pct).
        """
        price_at = self.get_price_at_time(ticker, dt)
        if price_at is None:
            return PriceMove(None, None, None)

        price_after = self.get_price_at_time(ticker, dt + timedelta(hours=hours))
        if price_after is None:
            return PriceMove(price_at, None, None)

        change_pct = ((price_after - price_at) / price_at) * 100
        return PriceMove(price_at, price_after, round(change_pct, 4))

    def get_current_price(self, ticker: str) -> float | None:
        """Get the latest available price for a ticker."""
        if not self._validate_ticker(ticker):
            return None

        cache_key = f"current:{ticker}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="1d", interval="1m")
            if hist.empty:
                hist = t.history(period="5d", interval="1d")
            if hist.empty:
                return None

            price = float(hist.iloc[-1]["Close"])
            self._cache_set(cache_key, price)
            return price

        except Exception as e:
            logger.error(f"Failed to fetch current price for {ticker}: {e}")
            return None
