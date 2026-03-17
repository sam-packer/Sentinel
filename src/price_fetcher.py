# This file was developed with the assistance of Claude Code and Opus 4.6.

"""
Defense stock price fetcher for Sentinel.

Uses yfinance to fetch historical prices at specific timestamps and compute
price moves over 24h windows. Averages prices within a configurable window
for stability, and handles weekends/holidays by finding the next available
trading session.
"""

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import NamedTuple

import pandas as pd
import yfinance as yf

from .data.stocks import get_public_tickers

logger = logging.getLogger("sentinel.price_fetcher")


class PriceMove(NamedTuple):
    """Result of a price move calculation."""
    price_at: float | None
    price_after: float | None
    change_pct: float | None
    volume_at_tweet: float | None = None


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

    @staticmethod
    def _ensure_utc_index(hist: pd.DataFrame) -> pd.DatetimeIndex:
        """Convert a yfinance DataFrame index to UTC for comparison."""
        idx = hist.index
        if hasattr(idx, 'tz_localize'):
            try:
                idx = idx.tz_localize('UTC')
            except TypeError:
                idx = idx.tz_convert('UTC')
        return idx

    def _fetch_candles(
        self, ticker: str, start: datetime, end: datetime,
    ) -> tuple[pd.DataFrame, str]:
        """Fetch candle data, trying 5m -> 1h -> 1d intervals.

        Uses 5-minute candles when the target is within yfinance's 60-day
        limit for intraday data, giving more data points for window averaging.
        Falls back to hourly, then daily.
        """
        t = yf.Ticker(ticker)

        # Try 5-minute candles if within yfinance's ~60-day limit
        now = datetime.now(tz=timezone.utc)
        start_utc = start.replace(tzinfo=timezone.utc) if start.tzinfo is None else start
        days_ago = (now - start_utc).days
        if days_ago <= 58:
            hist = t.history(start=start, end=end, interval="5m")
            if not hist.empty:
                return hist, "5m"

        # Hourly
        hist = t.history(start=start, end=end, interval="1h")
        if not hist.empty:
            return hist, "1h"

        # Daily fallback
        hist = t.history(start=start, end=end + timedelta(days=1), interval="1d")
        return hist, "1d"

    def _average_in_window(
        self, hist: pd.DataFrame, target_ts: datetime, window_minutes: int = 30,
    ) -> float | None:
        """Average Close prices within +/- window_minutes/2 of target_ts.

        Falls back to single nearest candle if no candles fall within the window.
        """
        target = target_ts.replace(tzinfo=timezone.utc) if target_ts.tzinfo is None else target_ts
        idx = self._ensure_utc_index(hist)

        half_window = timedelta(minutes=window_minutes / 2)
        mask = (idx >= target - half_window) & (idx <= target + half_window)
        subset = hist.loc[mask]

        if len(subset) > 0:
            return float(subset["Close"].mean())

        # Fallback: nearest single candle
        deltas = abs(idx - target)
        closest_idx = deltas.argmin()
        return float(hist.iloc[closest_idx]["Close"])

    def _volume_at_time(
        self, hist: pd.DataFrame, target_ts: datetime, window_minutes: int = 30,
    ) -> float | None:
        """Get total volume within +/- window_minutes/2 of target_ts."""
        target = target_ts.replace(tzinfo=timezone.utc) if target_ts.tzinfo is None else target_ts
        idx = self._ensure_utc_index(hist)

        half_window = timedelta(minutes=window_minutes / 2)
        mask = (idx >= target - half_window) & (idx <= target + half_window)
        subset = hist.loc[mask]

        if len(subset) > 0 and "Volume" in subset.columns:
            return float(subset["Volume"].sum())
        return None

    def get_price_at_time(
        self, ticker: str, dt: datetime, window_minutes: int = 30,
    ) -> float | None:
        """Get the average closing price within +/-window_minutes/2 of dt.

        Tries 5m candles (if recent enough), then 1h, then 1d.
        Falls back to nearest single candle if no candles fall in the window.

        Args:
            ticker: Defense stock ticker (e.g. "LMT").
            dt: Target datetime.
            window_minutes: Width of averaging window in minutes.

        Returns:
            Price as float, or None if unavailable.
        """
        if not self._validate_ticker(ticker):
            logger.warning(f"Ticker {ticker} not in defense stock universe")
            return None

        cache_key = f"price:{ticker}:{dt.strftime('%Y%m%d%H%M')}:w{window_minutes}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            start = dt - timedelta(days=3)
            end = dt + timedelta(days=1)

            hist, _interval = self._fetch_candles(ticker, start, end)
            if hist.empty:
                logger.warning(f"No price data for {ticker} around {dt}")
                return None

            price = self._average_in_window(hist, dt, window_minutes)
            if price is not None:
                self._cache_set(cache_key, price)
            return price

        except Exception as e:
            logger.error(f"Failed to fetch price for {ticker} at {dt}: {e}")
            return None

    def get_price_after_time(
        self, ticker: str, dt: datetime, window_minutes: int = 30,
    ) -> float | None:
        """Get the average price from the first available candle(s) at or after dt.

        Handles weekends and holidays: if dt falls on a non-trading day, finds
        the next trading session. Fetches up to dt+5 days to cover long weekends.

        Args:
            ticker: Defense stock ticker.
            dt: Target datetime (find first candle at or after this).
            window_minutes: Width of forward averaging window in minutes.

        Returns:
            Price as float, or None if unavailable.
        """
        if not self._validate_ticker(ticker):
            return None

        cache_key = f"price_after:{ticker}:{dt.strftime('%Y%m%d%H%M')}:w{window_minutes}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            start = dt - timedelta(hours=1)  # small lookback for near-exact matches
            end = dt + timedelta(days=5)     # cover long weekends / holidays

            hist, _interval = self._fetch_candles(ticker, start, end)
            if hist.empty:
                logger.warning(f"No price data for {ticker} after {dt}")
                return None

            target = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
            idx = self._ensure_utc_index(hist)

            # Filter to candles at or after target
            after_mask = idx >= target
            if after_mask.any():
                first_after = idx[after_mask][0]
                # Average candles within a forward window from the first available candle
                half_window = timedelta(minutes=window_minutes / 2)
                window_mask = (idx >= first_after) & (idx <= first_after + half_window)
                subset = hist.loc[window_mask]
                if len(subset) > 0:
                    price = float(subset["Close"].mean())
                else:
                    price = float(hist.loc[after_mask].iloc[0]["Close"])
            else:
                # All candles are before target (unlikely with 5-day range)
                logger.warning(f"No candles found at or after {dt} for {ticker}")
                price = float(hist.iloc[-1]["Close"])

            self._cache_set(cache_key, price)
            return price

        except Exception as e:
            logger.error(f"Failed to fetch price after {dt} for {ticker}: {e}")
            return None

    def get_price_change(
        self, ticker: str, dt: datetime, hours: int = 24,
        window_minutes: int = 30,
    ) -> PriceMove:
        """Get price at tweet time and N hours later, compute percent change.

        Uses windowed averaging for both prices. The "after" price uses
        get_price_after_time so weekends/holidays resolve to the next
        available trading session instead of snapping back to the previous close.

        Args:
            ticker: Defense stock ticker.
            dt: Tweet timestamp.
            hours: Window to measure (default 24h).
            window_minutes: Width of averaging window in minutes.

        Returns:
            PriceMove(price_at, price_after, change_pct).
        """
        price_at = self.get_price_at_time(ticker, dt, window_minutes=window_minutes)
        if price_at is None:
            return PriceMove(None, None, None)

        price_after = self.get_price_after_time(
            ticker, dt + timedelta(hours=hours), window_minutes=window_minutes,
        )
        if price_after is None:
            return PriceMove(price_at, None, None)

        change_pct = ((price_after - price_at) / price_at) * 100

        # Capture volume at tweet time
        volume = None
        try:
            start = dt - timedelta(days=3)
            end = dt + timedelta(days=1)
            hist, _interval = self._fetch_candles(ticker, start, end)
            if not hist.empty:
                volume = self._volume_at_time(hist, dt, window_minutes)
        except Exception:
            pass  # volume is nice-to-have, don't fail the whole enrichment

        return PriceMove(price_at, price_after, round(change_pct, 4), volume)
