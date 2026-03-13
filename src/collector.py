"""
Background-capable collection engine for Sentinel.

Handles scraping, enriching, labeling, and storing defense stock claims.
Writes progress to a JSON status file so other processes can monitor it.
"""

import json
import logging
import os
import signal
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import config

logger = logging.getLogger("sentinel.collector")


# ---------------------------------------------------------------------------
# File paths — parameterized by command name ("collect" or "enrich")
# ---------------------------------------------------------------------------

def _status_file(name: str) -> Path:
    return Path(f"data/{name}_status.json")


def _pid_file(name: str) -> Path:
    return Path(f"data/{name}.pid")


def _log_file(name: str) -> Path:
    return Path(f"data/{name}.log")


# ---------------------------------------------------------------------------
# Status tracking
# ---------------------------------------------------------------------------

@dataclass
class CollectionStatus:
    pid: int = 0
    state: str = "pending"  # pending, scraping, enriching, completed, failed, stopped
    started_at: str | None = None
    tickers: list[str] = field(default_factory=list)
    n_per_ticker: int = 0
    phase: str = ""
    # Scraping progress
    tickers_scraped: int = 0
    tickers_total: int = 0
    scrape_tweets_found: int = 0
    current_ticker: str | None = None
    # Enrichment progress
    scraped: int = 0
    enriched: int = 0
    labeled: int = 0
    failed: int = 0
    last_update: str | None = None
    error: str | None = None
    finished_at: str | None = None


def _update_status(status: CollectionStatus, name: str) -> None:
    """Write status to JSON file atomically."""
    sf = _status_file(name)
    status.last_update = datetime.now().isoformat()
    sf.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp = tempfile.mkstemp(dir=sf.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(asdict(status), f, indent=2)
        Path(tmp).replace(sf)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_status(name: str) -> CollectionStatus | None:
    """Read the current status, or None if no status file exists."""
    sf = _status_file(name)
    if not sf.exists():
        return None
    try:
        data = json.loads(sf.read_text())
        return CollectionStatus(**data)
    except (json.JSONDecodeError, TypeError):
        return None


def is_running(name: str) -> bool:
    """Check if a background process is currently running."""
    pf = _pid_file(name)
    if not pf.exists():
        return False
    try:
        pid = int(pf.read_text().strip())
        if sys.platform == "win32":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x1000, False, pid)
            if handle == 0:
                raise ProcessLookupError()
            kernel32.CloseHandle(handle)
        else:
            os.kill(pid, 0)
        return True
    except (ValueError, ProcessLookupError, PermissionError, OSError):
        pf.unlink(missing_ok=True)
        return False


def _setup_logging(name: str) -> None:
    """Add a file handler so --status can show recent output."""
    lf = _log_file(name)
    lf.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(lf, mode="w")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    )
    logging.getLogger().addHandler(file_handler)


def _setup_signals(status: CollectionStatus, name: str):
    """Install signal handlers for clean shutdown. Returns a shutdown flag checker."""
    shutdown_requested = False

    def _handle_signal(signum, frame):
        nonlocal shutdown_requested
        shutdown_requested = True
        status.state = "stopped"
        status.finished_at = datetime.now().isoformat()
        _update_status(status, name)
        _pid_file(name).unlink(missing_ok=True)
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _handle_signal)
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, _handle_signal)

    def is_shutdown():
        return shutdown_requested

    return is_shutdown


def _filter_mature_claims(claims, logger_instance=logger):
    """Remove tweets less than 25h old (24h price window hasn't elapsed)."""
    now = datetime.now(tz=timezone.utc)
    min_age = timedelta(hours=25)
    mature = []
    skipped = 0
    for claim in claims:
        tweet_time = claim.created_at
        if tweet_time.tzinfo is None:
            tweet_time = tweet_time.replace(tzinfo=timezone.utc)
        if now - tweet_time < min_age:
            skipped += 1
        else:
            mature.append(claim)

    if skipped:
        logger_instance.warning(
            f"Skipped {skipped} tweets less than 25h old — "
            f"the 24h price window hasn't elapsed yet. "
            f"Proceeding with {len(mature)} mature tweets."
        )
    return mature


async def _enrich_and_label(claims, db, status, name, is_shutdown):
    """Shared enrichment loop used by both collect and enrich pipelines."""
    from .data.labeler import label_claim
    from .news_fetcher import classify_catalyst, fetch_news_for_claim
    from .price_fetcher import PriceFetcher

    pf = PriceFetcher()

    for i, claim in enumerate(claims):
        if is_shutdown():
            break

        try:
            move = pf.get_price_change(claim.ticker, claim.created_at, hours=24)
            claim.price_at_tweet = move.price_at
            claim.price_24h_later = move.price_after
            claim.price_change_pct = move.change_pct

            articles = await fetch_news_for_claim(
                claim.ticker, claim.company_name, claim.created_at,
            )
            claim.news_headlines = [a["title"] for a in articles if a.get("title")]
            claim.has_catalyst, claim.catalyst_type = classify_catalyst(claim.news_headlines)

            status.enriched = i + 1
            status.current_ticker = claim.ticker

            labeled = label_claim(claim)
            status.labeled += 1

            if db:
                db.insert_labeled_claim(labeled)

            _update_status(status, name)

        except Exception as e:
            status.failed += 1
            _update_status(status, name)
            logger.warning(f"Failed to process claim {claim.tweet_id}: {e}")


# ---------------------------------------------------------------------------
# Collection pipeline (scrape + enrich)
# ---------------------------------------------------------------------------

async def run_collection(
    n_per_ticker: int,
    ticker_list: list[str],
    database_url: str | None,
    since: datetime | None = None,
    until: datetime | None = None,
    daily: bool = False,
) -> None:
    """Run the full collection pipeline with status tracking."""
    from .data.db import SentinelDB
    from .scraper import DefenseStockScraper

    name = "collect"
    _setup_logging(name)

    status = CollectionStatus(
        pid=os.getpid(),
        state="scraping",
        started_at=datetime.now().isoformat(),
        tickers=ticker_list,
        n_per_ticker=n_per_ticker,
        phase="scraping",
    )

    pf = _pid_file(name)
    pf.parent.mkdir(parents=True, exist_ok=True)
    pf.write_text(str(os.getpid()))

    is_shutdown = _setup_signals(status, name)

    try:
        status.tickers_total = len(ticker_list)
        _update_status(status, name)

        # 1. Scrape
        scraper = DefenseStockScraper(config.twitter.db_path)

        def _on_ticker_done(ticker, total_claims, tickers_done, tickers_total):
            status.current_ticker = ticker
            status.tickers_scraped = tickers_done
            status.tickers_total = tickers_total
            status.scrape_tweets_found = total_claims
            _update_status(status, name)

        if daily and since and until:
            # Split into per-day windows for even temporal distribution
            claims = []
            seen_ids: set[int] = set()
            day_start = since
            day_count = 0
            while day_start < until:
                day_end = min(day_start + timedelta(days=1), until)
                day_count += 1
                logger.info(
                    f"Daily scrape: day {day_count}, "
                    f"{day_start.strftime('%Y-%m-%d')} to {day_end.strftime('%Y-%m-%d')}"
                )
                status.phase = f"scraping day {day_count} ({day_start.strftime('%Y-%m-%d')})"
                _update_status(status, name)

                day_claims = await scraper.scrape_defense_claims(
                    tickers=ticker_list,
                    limit_per_ticker=n_per_ticker,
                    on_ticker_done=_on_ticker_done,
                    since=day_start,
                    until=day_end,
                )
                # Deduplicate across days
                for c in day_claims:
                    if c.tweet_id not in seen_ids:
                        seen_ids.add(c.tweet_id)
                        claims.append(c)

                logger.info(
                    f"Day {day_count} ({day_start.strftime('%Y-%m-%d')}): "
                    f"{len(day_claims)} claims"
                )
                day_start = day_end
        else:
            claims = await scraper.scrape_defense_claims(
                tickers=ticker_list,
                limit_per_ticker=n_per_ticker,
                on_ticker_done=_on_ticker_done,
                since=since,
                until=until,
            )

        status.scraped = len(claims)
        status.scrape_tweets_found = len(claims)
        status.phase = "enriching"
        status.state = "enriching"
        status.current_ticker = None
        _update_status(status, name)
        logger.info(f"Scraped {len(claims)} raw claims")

        # 2. Filter immature tweets
        claims = _filter_mature_claims(claims)

        # 3. Connect DB and skip already-stored tweets
        db = None
        if database_url:
            db = SentinelDB(database_url)
            db.connect()
            db.init_schema()

        if db:
            all_ids = [c.tweet_id for c in claims]
            existing = db.get_existing_tweet_ids(all_ids)
            if existing:
                claims = [c for c in claims if c.tweet_id not in existing]
                logger.info(
                    f"Skipped {len(existing)} already-scraped tweets, "
                    f"{len(claims)} new tweets to enrich"
                )

        status.scraped = len(claims)

        # 4. Enrich + label
        await _enrich_and_label(claims, db, status, name, is_shutdown)

        if db:
            db.close()

        status.state = "completed"
        status.phase = "done"
        status.finished_at = datetime.now().isoformat()
        status.current_ticker = None
        _update_status(status, name)
        _pid_file(name).unlink(missing_ok=True)
        logger.info(f"Collection complete: {status.labeled} labeled, {status.failed} failed")

    except Exception as e:
        status.state = "failed"
        status.error = str(e)
        status.finished_at = datetime.now().isoformat()
        _update_status(status, name)
        _pid_file(name).unlink(missing_ok=True)
        logger.error(f"Collection failed: {e}")
        raise


# ---------------------------------------------------------------------------
# Enrichment pipeline (DB claims only, no scraping)
# ---------------------------------------------------------------------------

async def run_enrichment(
    database_url: str,
    tickers: list[str] | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    unlabeled_only: bool = False,
) -> None:
    """Re-enrich existing raw claims without scraping.

    Reads claims from the database, fetches fresh price/news data,
    re-labels, and updates the database.
    """
    from .data.db import SentinelDB

    name = "enrich"
    _setup_logging(name)

    status = CollectionStatus(
        pid=os.getpid(),
        state="enriching",
        started_at=datetime.now().isoformat(),
        phase="enriching",
    )

    pf = _pid_file(name)
    pf.parent.mkdir(parents=True, exist_ok=True)
    pf.write_text(str(os.getpid()))

    is_shutdown = _setup_signals(status, name)

    try:
        db = SentinelDB(database_url)
        db.connect()
        db.init_schema()

        claims = db.get_raw_claims(
            tickers=tickers, since=since, until=until,
            unlabeled_only=unlabeled_only,
        )

        if not claims:
            logger.info("No claims to enrich")
            db.close()
            status.state = "completed"
            status.phase = "done"
            status.finished_at = datetime.now().isoformat()
            _update_status(status, name)
            _pid_file(name).unlink(missing_ok=True)
            return

        claims = _filter_mature_claims(claims)
        status.scraped = len(claims)
        _update_status(status, name)
        logger.info(f"Enriching {len(claims)} claims")

        await _enrich_and_label(claims, db, status, name, is_shutdown)

        db.close()

        status.state = "completed"
        status.phase = "done"
        status.finished_at = datetime.now().isoformat()
        status.current_ticker = None
        _update_status(status, name)
        _pid_file(name).unlink(missing_ok=True)
        logger.info(f"Enrichment complete: {status.labeled} labeled, {status.failed} failed")

    except Exception as e:
        status.state = "failed"
        status.error = str(e)
        status.finished_at = datetime.now().isoformat()
        _update_status(status, name)
        _pid_file(name).unlink(missing_ok=True)
        logger.error(f"Enrichment failed: {e}")
        raise
