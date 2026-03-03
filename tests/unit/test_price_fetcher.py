"""Tests for defense stock price fetcher."""

import pandas as pd
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from src.price_fetcher import PriceFetcher, PriceMove


def _make_candle_df(timestamps, prices):
    """Build a DataFrame mimicking yfinance output."""
    idx = pd.DatetimeIndex(timestamps, tz="UTC")
    return pd.DataFrame({"Close": prices}, index=idx)


def _make_empty_df():
    return pd.DataFrame(columns=["Close"])


# ---------------------------------------------------------------------------
# _fetch_candles interval fallback
# ---------------------------------------------------------------------------

class TestFetchCandles:
    @patch("yfinance.Ticker")
    @patch("src.price_fetcher.get_public_tickers", return_value=["LMT"])
    def test_tries_5m_when_recent(self, _mock_tickers, mock_yf):
        """Within 58 days of now, should try 5m interval first."""
        pf = PriceFetcher()
        dt = datetime.now(tz=timezone.utc) - timedelta(days=10)
        candles = _make_candle_df([dt], [450.0])

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = candles
        mock_yf.return_value = mock_ticker

        hist, interval = pf._fetch_candles("LMT", dt - timedelta(days=1), dt + timedelta(days=1))
        assert interval == "5m"
        # First call should be with interval="5m"
        mock_ticker.history.assert_called_once()
        call_kwargs = mock_ticker.history.call_args
        assert call_kwargs[1].get("interval") == "5m" or call_kwargs.kwargs.get("interval") == "5m"

    @patch("yfinance.Ticker")
    @patch("src.price_fetcher.get_public_tickers", return_value=["LMT"])
    def test_skips_5m_when_old(self, _mock_tickers, mock_yf):
        """Older than 58 days, should skip 5m and try 1h."""
        pf = PriceFetcher()
        dt = datetime.now(tz=timezone.utc) - timedelta(days=90)
        candles = _make_candle_df([dt], [450.0])

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = candles
        mock_yf.return_value = mock_ticker

        hist, interval = pf._fetch_candles("LMT", dt - timedelta(days=1), dt + timedelta(days=1))
        assert interval == "1h"

    @patch("yfinance.Ticker")
    @patch("src.price_fetcher.get_public_tickers", return_value=["LMT"])
    def test_falls_back_to_1h(self, _mock_tickers, mock_yf):
        """If 5m returns empty, should fall back to 1h."""
        pf = PriceFetcher()
        dt = datetime.now(tz=timezone.utc) - timedelta(days=5)
        candles_1h = _make_candle_df([dt], [450.0])

        mock_ticker = MagicMock()
        mock_ticker.history.side_effect = [_make_empty_df(), candles_1h]
        mock_yf.return_value = mock_ticker

        hist, interval = pf._fetch_candles("LMT", dt - timedelta(days=1), dt + timedelta(days=1))
        assert interval == "1h"
        assert not hist.empty

    @patch("yfinance.Ticker")
    @patch("src.price_fetcher.get_public_tickers", return_value=["LMT"])
    def test_falls_back_to_1d(self, _mock_tickers, mock_yf):
        """If 5m and 1h are empty, should fall back to 1d."""
        pf = PriceFetcher()
        dt = datetime.now(tz=timezone.utc) - timedelta(days=5)
        candles_1d = _make_candle_df([dt], [450.0])

        mock_ticker = MagicMock()
        mock_ticker.history.side_effect = [_make_empty_df(), _make_empty_df(), candles_1d]
        mock_yf.return_value = mock_ticker

        hist, interval = pf._fetch_candles("LMT", dt - timedelta(days=1), dt + timedelta(days=1))
        assert interval == "1d"
        assert not hist.empty


# ---------------------------------------------------------------------------
# _average_in_window
# ---------------------------------------------------------------------------

class TestAverageInWindow:
    def test_averages_multiple_candles(self):
        """Multiple candles within window should be averaged."""
        pf = PriceFetcher()
        base = datetime(2024, 6, 14, 14, 0, tzinfo=timezone.utc)
        timestamps = [base + timedelta(minutes=i * 5) for i in range(-2, 4)]
        prices = [449.0, 450.0, 451.0, 452.0, 450.5, 451.5]
        hist = _make_candle_df(timestamps, prices)

        result = pf._average_in_window(hist, base, window_minutes=30)
        expected = sum(prices) / len(prices)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_single_candle_in_window(self):
        """Only one candle within window returns that price."""
        pf = PriceFetcher()
        base = datetime(2024, 6, 14, 14, 0, tzinfo=timezone.utc)
        hist = _make_candle_df([base], [450.0])

        result = pf._average_in_window(hist, base, window_minutes=30)
        assert result == 450.0

    def test_fallback_to_nearest_when_window_empty(self):
        """No candles in window should fall back to nearest candle."""
        pf = PriceFetcher()
        target = datetime(2024, 6, 14, 14, 0, tzinfo=timezone.utc)
        # Candle 2 hours away (well outside 30-min window)
        far_candle = target - timedelta(hours=2)
        hist = _make_candle_df([far_candle], [445.0])

        result = pf._average_in_window(hist, target, window_minutes=30)
        assert result == 445.0


# ---------------------------------------------------------------------------
# get_price_after_time (weekend/holiday handling)
# ---------------------------------------------------------------------------

class TestGetPriceAfterTime:
    @patch("src.price_fetcher.get_public_tickers", return_value=["LMT"])
    def test_friday_tweet_returns_monday_price(self, _mock_tickers):
        """Saturday target (Friday tweet + 24h) should return Monday's price."""
        pf = PriceFetcher()
        friday_3pm = datetime(2024, 6, 14, 15, 0, tzinfo=timezone.utc)
        monday_930 = datetime(2024, 6, 17, 9, 30, tzinfo=timezone.utc)
        monday_935 = datetime(2024, 6, 17, 9, 35, tzinfo=timezone.utc)
        monday_940 = datetime(2024, 6, 17, 9, 40, tzinfo=timezone.utc)

        candles = _make_candle_df(
            [friday_3pm, monday_930, monday_935, monday_940],
            [450.0, 453.0, 454.0, 452.0],
        )

        with patch.object(pf, "_fetch_candles", return_value=(candles, "5m")):
            saturday_3pm = datetime(2024, 6, 15, 15, 0, tzinfo=timezone.utc)
            price = pf.get_price_after_time("LMT", saturday_3pm, window_minutes=30)

        # Should average Monday's candles (453 + 454 + 452) / 3, NOT Friday's 450
        assert price is not None
        assert price == pytest.approx((453.0 + 454.0 + 452.0) / 3, rel=1e-4)

    @patch("src.price_fetcher.get_public_tickers", return_value=["LMT"])
    def test_exact_match(self, _mock_tickers):
        """Target time with an exact candle should use it."""
        pf = PriceFetcher()
        target = datetime(2024, 6, 14, 14, 0, tzinfo=timezone.utc)
        candles = _make_candle_df([target], [450.0])

        with patch.object(pf, "_fetch_candles", return_value=(candles, "1h")):
            price = pf.get_price_after_time("LMT", target)

        assert price == 450.0

    @patch("src.price_fetcher.get_public_tickers", return_value=["LMT"])
    def test_returns_none_for_empty_data(self, _mock_tickers):
        """No data at all should return None."""
        pf = PriceFetcher()
        target = datetime(2024, 6, 14, 14, 0, tzinfo=timezone.utc)

        with patch.object(pf, "_fetch_candles", return_value=(_make_empty_df(), "1d")):
            price = pf.get_price_after_time("LMT", target)

        assert price is None

    @patch("src.price_fetcher.get_public_tickers", return_value=["LMT"])
    def test_invalid_ticker_returns_none(self, _mock_tickers):
        """Ticker not in universe should return None."""
        _mock_tickers.return_value = ["RTX"]  # LMT not in universe
        pf = PriceFetcher()
        target = datetime(2024, 6, 14, 14, 0, tzinfo=timezone.utc)
        assert pf.get_price_after_time("LMT", target) is None


# ---------------------------------------------------------------------------
# get_price_change end-to-end
# ---------------------------------------------------------------------------

class TestGetPriceChange:
    @patch("src.price_fetcher.get_public_tickers", return_value=["LMT"])
    def test_friday_tweet_not_zero_change(self, _mock_tickers):
        """A Friday tweet should NOT produce 0% change from weekend snap-back."""
        pf = PriceFetcher()
        friday_2pm = datetime(2024, 6, 14, 14, 0, tzinfo=timezone.utc)
        monday_2pm = datetime(2024, 6, 17, 14, 0, tzinfo=timezone.utc)

        friday_candles = _make_candle_df([friday_2pm], [450.0])
        monday_candles = _make_candle_df([monday_2pm], [460.0])

        def mock_fetch(ticker, start, end):
            # Return different candles based on the date range
            all_candles = _make_candle_df(
                [friday_2pm, monday_2pm], [450.0, 460.0],
            )
            return all_candles, "1h"

        with patch.object(pf, "_fetch_candles", side_effect=mock_fetch):
            move = pf.get_price_change("LMT", friday_2pm, hours=24)

        assert move.price_at == pytest.approx(450.0)
        assert move.price_after == pytest.approx(460.0)
        assert move.change_pct == pytest.approx(2.2222, rel=1e-3)

    @patch("src.price_fetcher.get_public_tickers", return_value=["LMT"])
    def test_normal_weekday(self, _mock_tickers):
        """Normal weekday should work as before."""
        pf = PriceFetcher()
        tuesday_2pm = datetime(2024, 6, 11, 14, 0, tzinfo=timezone.utc)
        wednesday_2pm = datetime(2024, 6, 12, 14, 0, tzinfo=timezone.utc)

        def mock_fetch(ticker, start, end):
            all_candles = _make_candle_df(
                [tuesday_2pm, wednesday_2pm], [450.0, 455.0],
            )
            return all_candles, "1h"

        with patch.object(pf, "_fetch_candles", side_effect=mock_fetch):
            move = pf.get_price_change("LMT", tuesday_2pm, hours=24)

        assert move.price_at == pytest.approx(450.0)
        assert move.price_after == pytest.approx(455.0)
        expected_pct = ((455.0 - 450.0) / 450.0) * 100
        assert move.change_pct == pytest.approx(expected_pct, rel=1e-3)

    @patch("src.price_fetcher.get_public_tickers", return_value=["LMT"])
    def test_none_when_no_data(self, _mock_tickers):
        """No price data should return all None."""
        pf = PriceFetcher()
        dt = datetime(2024, 6, 14, 14, 0, tzinfo=timezone.utc)

        with patch.object(pf, "_fetch_candles", return_value=(_make_empty_df(), "1d")):
            move = pf.get_price_change("LMT", dt)

        assert move == PriceMove(None, None, None)
