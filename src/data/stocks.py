"""
Defense stock universe for Sentinel.

Maps company names, aliases, and tickers to canonical ticker symbols.
Private companies (Anduril, Shield AI) are tracked but have no price data.
"""

import re

# Alias -> canonical ticker mapping.
# None means the company is private (detect but skip price fetch).
DEFENSE_STOCKS: dict[str, str | None] = {
    # Big 5 Primes
    "lockheed": "LMT", "lockheed martin": "LMT", "lmt": "LMT",
    "raytheon": "RTX", "rtx": "RTX", "rtx corporation": "RTX",
    "northrop": "NOC", "northrop grumman": "NOC", "noc": "NOC",
    "general dynamics": "GD", "gd": "GD",
    "boeing": "BA", "ba": "BA",
    # Mid-Tier
    "huntington ingalls": "HII", "hii": "HII",
    "l3harris": "LHX", "l3 harris": "LHX", "lhx": "LHX",
    "leidos": "LDOS", "ldos": "LDOS",
    "saic": "SAIC",
    "booz allen": "BAH", "booz allen hamilton": "BAH", "bah": "BAH",
    # Emerging / Tech-Adjacent
    "kratos": "KTOS", "ktos": "KTOS",
    "palantir": "PLTR", "pltr": "PLTR",
    "anduril": None,
    "shield ai": None,
    "rocket lab": "RKLB", "rklb": "RKLB",
}

# Canonical ticker -> company name (for display).
TICKER_NAMES: dict[str, str] = {
    "LMT": "Lockheed Martin",
    "RTX": "RTX Corporation",
    "NOC": "Northrop Grumman",
    "GD": "General Dynamics",
    "BA": "Boeing",
    "HII": "Huntington Ingalls",
    "LHX": "L3Harris Technologies",
    "LDOS": "Leidos",
    "SAIC": "SAIC",
    "BAH": "Booz Allen Hamilton",
    "KTOS": "Kratos Defense",
    "PLTR": "Palantir Technologies",
    "RKLB": "Rocket Lab",
}

# Precompute for fast matching: sorted longest-first so "lockheed martin"
# matches before "lockheed".
_SORTED_ALIASES = sorted(DEFENSE_STOCKS.keys(), key=len, reverse=True)

# Cashtag pattern: $LMT, $RTX, etc.
_CASHTAG_RE = re.compile(r"\$([A-Z]{2,5})\b")


def resolve_ticker(text: str) -> str | None:
    """Scan text for any defense stock alias or cashtag, return canonical ticker.

    Returns the first match found (longest alias wins). Returns None if no
    defense stock is mentioned.
    """
    lowered = text.lower()

    # Check cashtags first (most explicit signal)
    for match in _CASHTAG_RE.finditer(text):
        symbol = match.group(1).lower()
        if symbol in DEFENSE_STOCKS:
            return DEFENSE_STOCKS[symbol]

    # Check aliases (longest first)
    for alias in _SORTED_ALIASES:
        if alias in lowered:
            return DEFENSE_STOCKS[alias]

    return None


def get_public_tickers() -> list[str]:
    """Return list of public defense stock tickers (those with yfinance symbols)."""
    return sorted({v for v in DEFENSE_STOCKS.values() if v is not None})


def is_private(ticker_or_alias: str) -> bool:
    """Check if a company is private (no public ticker)."""
    lowered = ticker_or_alias.lower()
    if lowered in DEFENSE_STOCKS:
        return DEFENSE_STOCKS[lowered] is None
    return False


def company_name(ticker: str) -> str:
    """Return display name for a ticker, or the ticker itself if unknown."""
    return TICKER_NAMES.get(ticker, ticker)
