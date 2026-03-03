"""Tests for defense stock registry."""

from src.data.stocks import (
    DEFENSE_STOCKS,
    TICKER_NAMES,
    resolve_ticker,
    get_public_tickers,
    is_private,
    company_name,
)


class TestDefenseStocks:
    def test_all_public_tickers_have_names(self):
        tickers = get_public_tickers()
        for t in tickers:
            assert t in TICKER_NAMES, f"Ticker {t} missing from TICKER_NAMES"

    def test_known_tickers(self):
        assert DEFENSE_STOCKS["lmt"] == "LMT"
        assert DEFENSE_STOCKS["raytheon"] == "RTX"
        assert DEFENSE_STOCKS["northrop grumman"] == "NOC"

    def test_private_companies(self):
        assert DEFENSE_STOCKS["anduril"] is None
        assert DEFENSE_STOCKS["shield ai"] is None
        assert is_private("anduril")
        assert not is_private("lmt")

    def test_get_public_tickers(self):
        tickers = get_public_tickers()
        assert "LMT" in tickers
        assert "RTX" in tickers
        assert None not in tickers

    def test_resolve_ticker_cashtag(self):
        assert resolve_ticker("$LMT is looking good today") == "LMT"
        assert resolve_ticker("Buying $RTX calls") == "RTX"

    def test_resolve_ticker_name(self):
        assert resolve_ticker("Lockheed Martin wins big contract") == "LMT"
        assert resolve_ticker("northrop grumman earnings beat") == "NOC"

    def test_resolve_ticker_none(self):
        assert resolve_ticker("Just a random tweet about cats") is None
        assert resolve_ticker("AAPL to the moon") is None

    def test_resolve_ticker_longest_match(self):
        # "lockheed martin" should match before "lockheed"
        assert resolve_ticker("lockheed martin stock") == "LMT"

    def test_company_name(self):
        assert company_name("LMT") == "Lockheed Martin"
        assert company_name("UNKNOWN") == "UNKNOWN"
