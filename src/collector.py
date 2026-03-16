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
    # Classification progress
    accounts_classified: int = 0
    accounts_total: int = 0
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


def _is_market_hours(dt: datetime) -> bool:
    """Check if a datetime falls within US stock market hours (9:30 AM - 4:00 PM ET)."""
    try:
        from zoneinfo import ZoneInfo
        et = ZoneInfo("America/New_York")
    except ImportError:
        et = timezone(timedelta(hours=-5))

    et_dt = dt.astimezone(et)
    if et_dt.weekday() >= 5:  # Saturday or Sunday
        return False
    market_open = et_dt.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = et_dt.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= et_dt <= market_close


MAX_TWEET_AGE = timedelta(days=90)


def _filter_mature_claims(claims, logger_instance=logger):
    """Remove tweets less than 25h old or stale outliers (created >90 days before scrape)."""
    now = datetime.now(tz=timezone.utc)
    min_age = timedelta(hours=25)
    mature = []
    too_new = 0
    too_old = 0
    for claim in claims:
        tweet_time = claim.created_at
        if tweet_time.tzinfo is None:
            tweet_time = tweet_time.replace(tzinfo=timezone.utc)
        if now - tweet_time < min_age:
            too_new += 1
        elif now - tweet_time > MAX_TWEET_AGE:
            too_old += 1
        else:
            mature.append(claim)

    if too_new:
        logger_instance.warning(
            f"Skipped {too_new} tweets less than 25h old — "
            f"the 24h price window hasn't elapsed yet."
        )
    if too_old:
        logger_instance.warning(
            f"Skipped {too_old} stale tweets older than {MAX_TWEET_AGE.days} days."
        )
    if too_new or too_old:
        logger_instance.info(f"Proceeding with {len(mature)} tweets.")
    return mature


async def _enrich_and_label(claims, db, status, name, is_shutdown, labeler="naive"):
    """Shared enrichment loop used by both collect and enrich pipelines.

    Args:
        labeler: "naive" or "improved" — which labeling system to use.
    """
    if labeler == "improved":
        from .data.improved_labeler import label_claim_improved as do_label
        label_table = "improved_labeled_claims"
    else:
        from .data.labeler import label_claim as do_label
        label_table = "naive_labeled_claims"

    from .news_fetcher import classify_catalyst, fetch_news_for_claim
    from .price_fetcher import PriceFetcher

    pf = PriceFetcher()
    logger.info(f"Using {labeler} labeler → {label_table}")

    for i, claim in enumerate(claims):
        if is_shutdown():
            break

        try:
            move = pf.get_price_change(claim.ticker, claim.created_at, hours=24)
            claim.price_at_tweet = move.price_at
            claim.price_24h_later = move.price_after
            claim.price_change_pct = move.change_pct
            claim.volume_at_tweet = move.volume_at_tweet
            claim.posted_during_market_hours = _is_market_hours(claim.created_at)

            articles = await fetch_news_for_claim(
                claim.ticker, claim.company_name, claim.created_at,
            )
            claim.news_headlines = [a["title"] for a in articles if a.get("title")]
            claim.has_catalyst, claim.catalyst_type = classify_catalyst(claim.news_headlines)

            status.enriched = i + 1
            status.current_ticker = claim.ticker

            labeled = do_label(claim)
            status.labeled += 1

            if db:
                db.insert_labeled_claim(labeled, label_table=label_table)

            _update_status(status, name)

        except Exception as e:
            status.failed += 1
            _update_status(status, name)
            logger.warning(f"Failed to process claim {claim.tweet_id}: {e}")


def _classify_new_accounts(claims, db, status, name, rejudge=False):
    """Classify any unclassified accounts from the current batch.

    Checks which usernames in the batch don't have a bot classification yet,
    classifies them via LLM-as-judge, and saves the results. Returns the set
    of bot usernames so the caller can filter them out.

    If rejudge=True, reclassifies ALL accounts regardless of prior classification.
    """
    from .bot_detector import classify_accounts_batch
    from .config import config as app_config
    from .data.models import Account

    if not app_config.bot_detection.enabled:
        logger.info("Bot detection disabled, skipping account classification")
        return set()

    if rejudge:
        logger.warning(
            "Reclassifying all accounts (--rejudge). "
            "This will call the LLM for every account."
        )

    # Find unique usernames that haven't been classified
    usernames = {c.username for c in claims}
    unclassified = []
    bot_usernames = set()

    for username in usernames:
        account = db.get_account(username)
        if not rejudge and account and account.classified_at is not None:
            if account.account_type != "human":
                bot_usernames.add(username)
            continue
        # Gather sample tweets for this account from the current batch
        sample_tweets = [c.text for c in claims if c.username == username][:5]
        unclassified.append({"username": username, "sample_tweets": sample_tweets})

    if not unclassified:
        logger.info(f"All {len(usernames)} accounts already classified, {len(bot_usernames)} known bots")
        return bot_usernames

    logger.info(f"Classifying {len(unclassified)} new accounts...")
    status.phase = f"classifying {len(unclassified)} accounts"
    status.accounts_total = len(unclassified)
    status.accounts_classified = 0
    _update_status(status, name)

    newly_classified = 0

    def _on_classified(username, classification, idx, total):
        nonlocal newly_classified
        # Save immediately so classifications survive crashes
        account = db.get_account(username) or Account(username=username)
        account.account_type = classification.account_type
        account.classification_reason = (
            f"{classification.account_type}: {classification.reason}"
        )
        account.classified_at = datetime.now(timezone.utc)
        db.upsert_account(account)

        if classification.is_filtered:
            bot_usernames.add(username)
        newly_classified += 1

        status.accounts_classified = idx + 1
        _update_status(status, name)
        logger.info(
            f"  @{username}: {classification.account_type.upper()} "
            f"({classification.confidence:.0%}) — {classification.reason}"
        )

    classify_accounts_batch(
        unclassified,
        model=app_config.bot_detection.model,
        on_classified=_on_classified,
    )

    logger.info(
        f"Account classification done: {newly_classified} classified, "
        f"{len(bot_usernames)} total bots"
    )
    return bot_usernames


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

        # 1. Connect DB early so scraped tweets are persisted incrementally
        db = None
        if database_url:
            db = SentinelDB(database_url)
            db.connect()
            db.init_schema()

        # 2. Scrape and persist each batch immediately
        scraper = DefenseStockScraper(config.twitter.db_path)

        def _on_ticker_done(ticker, total_claims, tickers_done, tickers_total):
            status.current_ticker = ticker
            status.tickers_scraped = tickers_done
            status.tickers_total = tickers_total
            status.scrape_tweets_found = total_claims
            _update_status(status, name)

        def _persist_claims(batch):
            """Insert a batch of claims to the DB so they survive crashes."""
            if not db:
                return
            persisted = 0
            for claim in batch:
                db.insert_raw_claim(claim)
                persisted += 1
            if persisted:
                logger.info(f"Persisted {persisted} raw claims to DB")

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
                new_day_claims = []
                for c in day_claims:
                    if c.tweet_id not in seen_ids:
                        seen_ids.add(c.tweet_id)
                        claims.append(c)
                        new_day_claims.append(c)

                _persist_claims(new_day_claims)

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
            _persist_claims(claims)

        status.scraped = len(claims)
        status.scrape_tweets_found = len(claims)
        status.phase = "enriching"
        status.state = "enriching"
        status.current_ticker = None
        _update_status(status, name)
        logger.info(f"Scraped {len(claims)} raw claims")

        # 3. Filter immature tweets
        claims = _filter_mature_claims(claims)

        # 4. Skip already-enriched tweets
        if db:
            all_ids = [c.tweet_id for c in claims]
            existing = db.get_existing_tweet_ids(all_ids)
            if existing:
                claims = [c for c in claims if c.tweet_id not in existing]
                logger.info(
                    f"Skipped {len(existing)} already-enriched tweets, "
                    f"{len(claims)} new tweets to enrich"
                )

        status.scraped = len(claims)

        # 5. Classify accounts and filter bots
        if db:
            bot_usernames = _classify_new_accounts(claims, db, status, name)
            if bot_usernames:
                pre_filter = len(claims)
                claims = [c for c in claims if c.username not in bot_usernames]
                logger.info(
                    f"Filtered {pre_filter - len(claims)} bot tweets, "
                    f"{len(claims)} human tweets to enrich"
                )
                status.scraped = len(claims)

        # 6. Enrich + label
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
    rejudge: bool = False,
    labeler: str = "naive",
) -> None:
    """Re-enrich existing raw claims without scraping.

    Reads claims from the database, fetches fresh price/news data,
    re-labels, and updates the database.

    Args:
        labeler: "naive" or "improved" — which labeling system to use.
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

        # Classify accounts and filter bots
        bot_usernames = _classify_new_accounts(claims, db, status, name, rejudge=rejudge)
        if bot_usernames:
            pre_filter = len(claims)
            claims = [c for c in claims if c.username not in bot_usernames]
            logger.info(
                f"Filtered {pre_filter - len(claims)} bot tweets, "
                f"{len(claims)} human tweets to enrich"
            )

        status.scraped = len(claims)
        status.phase = "enriching"
        _update_status(status, name)
        logger.info(f"Enriching {len(claims)} claims with {labeler} labeler")

        await _enrich_and_label(claims, db, status, name, is_shutdown, labeler=labeler)

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
