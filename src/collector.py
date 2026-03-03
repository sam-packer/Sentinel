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

STATUS_FILE = Path("data/collect_status.json")
PID_FILE = Path("data/collect.pid")
LOG_FILE = Path("data/collect.log")


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


def _update_status(status: CollectionStatus) -> None:
    """Write status to JSON file atomically."""
    status.last_update = datetime.now().isoformat()
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file then rename for atomicity
    fd, tmp = tempfile.mkstemp(dir=STATUS_FILE.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(asdict(status), f, indent=2)
        Path(tmp).replace(STATUS_FILE)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_status() -> CollectionStatus | None:
    """Read the current collection status, or None if no status file exists."""
    if not STATUS_FILE.exists():
        return None
    try:
        data = json.loads(STATUS_FILE.read_text())
        return CollectionStatus(**data)
    except (json.JSONDecodeError, TypeError):
        return None


def is_running() -> bool:
    """Check if a collection process is currently running."""
    if not PID_FILE.exists():
        return False
    try:
        pid = int(PID_FILE.read_text().strip())
        if sys.platform == "win32":
            # On Windows, os.kill(pid, 0) doesn't work reliably
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if handle == 0:
                raise ProcessLookupError()
            kernel32.CloseHandle(handle)
        else:
            os.kill(pid, 0)
        return True
    except (ValueError, ProcessLookupError, PermissionError, OSError):
        PID_FILE.unlink(missing_ok=True)
        return False


async def run_collection(
    n_per_ticker: int,
    ticker_list: list[str],
    database_url: str | None,
    since: datetime | None = None,
    until: datetime | None = None,
) -> None:
    """Run the full collection pipeline with status tracking."""
    from .data.db import SentinelDB
    from .data.labeler import label_claim
    from .news_fetcher import classify_catalyst, fetch_news_for_claim
    from .price_fetcher import PriceFetcher
    from .scraper import DefenseStockScraper

    # Set up file logging so `status` can show recent output
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(LOG_FILE, mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(file_handler)

    status = CollectionStatus(
        pid=os.getpid(),
        state="scraping",
        started_at=datetime.now().isoformat(),
        tickers=ticker_list,
        n_per_ticker=n_per_ticker,
        phase="scraping",
    )

    # Write PID file
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))

    # Signal handler for clean shutdown
    shutdown_requested = False

    def _handle_signal(signum, frame):
        nonlocal shutdown_requested
        shutdown_requested = True
        status.state = "stopped"
        status.finished_at = datetime.now().isoformat()
        _update_status(status)
        PID_FILE.unlink(missing_ok=True)
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _handle_signal)
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, _handle_signal)

    try:
        status.tickers_total = len(ticker_list)
        _update_status(status)

        # 1. Scrape
        scraper = DefenseStockScraper(config.twitter.db_path)

        def _on_ticker_done(ticker, total_claims, tickers_done, tickers_total):
            status.current_ticker = ticker
            status.tickers_scraped = tickers_done
            status.tickers_total = tickers_total
            status.scrape_tweets_found = total_claims
            _update_status(status)

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
        _update_status(status)
        logger.info(f"Scraped {len(claims)} raw claims")

        # 2. Enrich + label
        # Filter out tweets that are too recent for the 24h price window.
        # Without the full window, the labeler can't compute the actual price
        # move and labels will be unreliable.
        now = datetime.now(tz=timezone.utc)
        min_age = timedelta(hours=25)  # 24h + 1h buffer for market close
        mature_claims = []
        skipped_recent = 0
        for claim in claims:
            tweet_time = claim.created_at
            if tweet_time.tzinfo is None:
                tweet_time = tweet_time.replace(tzinfo=timezone.utc)
            if now - tweet_time < min_age:
                skipped_recent += 1
            else:
                mature_claims.append(claim)

        if skipped_recent > 0:
            logger.warning(
                f"Skipped {skipped_recent} tweets less than 25h old — "
                f"the 24h price window hasn't elapsed yet so labels would be unreliable. "
                f"Proceeding with {len(mature_claims)} mature tweets."
            )
        claims = mature_claims

        pf = PriceFetcher()

        db = None
        if database_url:
            db = SentinelDB(database_url)
            db.connect()
            db.init_schema()

        # Skip tweets already in the database
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

        for i, claim in enumerate(claims):
            if shutdown_requested:
                break

            try:
                # Price
                move = pf.get_price_change(claim.ticker, claim.created_at, hours=24)
                claim.price_at_tweet = move.price_at
                claim.price_24h_later = move.price_after
                claim.price_change_pct = move.change_pct

                # News
                articles = await fetch_news_for_claim(
                    claim.ticker, claim.company_name, claim.created_at,
                )
                claim.news_headlines = [a["title"] for a in articles if a.get("title")]
                claim.has_catalyst, claim.catalyst_type = classify_catalyst(claim.news_headlines)

                status.enriched = i + 1
                status.current_ticker = claim.ticker

                # Label
                labeled = label_claim(claim)
                status.labeled += 1

                if db:
                    db.insert_labeled_claim(labeled)

                _update_status(status)

            except Exception as e:
                status.failed += 1
                _update_status(status)
                logger.warning(f"Failed to process claim {claim.tweet_id}: {e}")

        if db:
            db.close()

        status.state = "completed"
        status.phase = "done"
        status.finished_at = datetime.now().isoformat()
        status.current_ticker = None
        _update_status(status)
        PID_FILE.unlink(missing_ok=True)
        logger.info(f"Collection complete: {status.labeled} labeled, {status.failed} failed")

    except SystemExit:
        raise
    except Exception as e:
        status.state = "failed"
        status.error = str(e)
        status.finished_at = datetime.now().isoformat()
        _update_status(status)
        PID_FILE.unlink(missing_ok=True)
        logger.error(f"Collection failed: {e}")
        raise


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
    from .data.labeler import label_claim
    from .news_fetcher import classify_catalyst, fetch_news_for_claim
    from .price_fetcher import PriceFetcher

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
        return

    # Filter out tweets too recent for 24h price window
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
        logger.warning(
            f"Skipped {skipped} tweets less than 25h old, "
            f"{len(mature)} to enrich"
        )

    logger.info(f"Enriching {len(mature)} claims")
    pf = PriceFetcher()
    labeled_count = 0
    failed_count = 0

    for i, claim in enumerate(mature):
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

            labeled = label_claim(claim)
            db.insert_labeled_claim(labeled)
            labeled_count += 1

            if (i + 1) % 25 == 0:
                logger.info(f"Enriched {i + 1}/{len(mature)}")

        except Exception as e:
            failed_count += 1
            logger.warning(f"Failed to enrich claim {claim.tweet_id}: {e}")

    db.close()
    logger.info(f"Enrichment complete: {labeled_count} labeled, {failed_count} failed")
