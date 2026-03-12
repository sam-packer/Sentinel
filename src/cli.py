"""
CLI entry points for Sentinel.

Each function is a standalone click command exposed via pyproject.toml [project.scripts].

Usage:
  uv run setup                        — Initialize DB, sanity check
  uv run collect --n 100              — Scrape and label claims
  uv run collect --n 100 --background — Run collection in background
  uv run collect --status             — Check collection progress
  uv run collect --stop               — Stop background collection
  uv run enrich                       — Re-enrich existing claims
  uv run enrich --background          — Run enrichment in background
  uv run enrich --status              — Check enrichment progress
  uv run enrich --stop                — Stop background enrichment
  uv run serve                        — Start API server
"""

import asyncio
import logging
import os
import signal
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import click

from .config import config

# US Eastern timezone (ET) for market hours
try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
except ImportError:
    from datetime import tzinfo
    # Fallback: EST = UTC-5 (ignores DST)
    ET = timezone(timedelta(hours=-5))

logger = logging.getLogger("sentinel.cli")


def _init():
    """Common initialization for all commands."""
    config.setup_logging()


# ---------------------------------------------------------------------------
# Shared helpers for --status and --stop
# ---------------------------------------------------------------------------

def _show_status(name: str) -> None:
    """Display status for a background task (collect or enrich)."""
    from .collector import _log_file, is_running, read_status

    st = read_status(name)
    if st is None:
        click.echo(f"No {name} has been run yet.")
        return

    running = is_running(name)

    if running and st.state in ("scraping", "enriching"):
        started = datetime.fromisoformat(st.started_at)
        elapsed = datetime.now() - started
        minutes = int(elapsed.total_seconds() // 60)
        seconds = int(elapsed.total_seconds() % 60)

        click.echo(f"Sentinel {name.title()} Status")
        click.echo(f"  State:    {st.state}")
        click.echo(f"  Started:  {minutes}m {seconds}s ago")
        click.echo(f"  Phase:    {st.phase}")

        if st.state == "scraping":
            pct = (st.tickers_scraped / st.tickers_total * 100) if st.tickers_total > 0 else 0
            click.echo(f"  Tickers:  {st.tickers_scraped}/{st.tickers_total} ({pct:.0f}%)")
            click.echo(f"  Tweets:   {st.scrape_tweets_found} found so far")
            if st.current_ticker:
                click.echo(f"  Current:  {st.current_ticker}")
        else:
            total = st.scraped
            done = st.enriched
            pct = (done / total * 100) if total > 0 else 0
            click.echo(f"  Progress: {done}/{total} claims enriched ({pct:.0f}%)")
            if st.current_ticker:
                click.echo(f"  Current:  {st.current_ticker}")
            click.echo(f"  Labeled:  {st.labeled}")
            click.echo(f"  Failed:   {st.failed}")

        click.echo(f"  PID:      {st.pid}")

        lf = _log_file(name)
        if lf.exists():
            click.echo("")
            click.echo("Recent log output:")
            try:
                lines = lf.read_text().splitlines()
                for line in lines[-10:]:
                    click.echo(f"  {line}")
            except Exception:
                pass
    else:
        click.echo(f"No {name} running.")
        state_label = st.state
        if not running and st.state in ("scraping", "enriching"):
            state_label = "interrupted"

        finished = ""
        if st.finished_at:
            finished = f" ({st.finished_at[:16].replace('T', ' ')})"
        elif st.started_at:
            finished = f" ({st.started_at[:16].replace('T', ' ')})"

        click.echo(
            f"Last run: {state_label} — "
            f"{st.scraped} scraped, {st.labeled} labeled, {st.failed} failed"
            f"{finished}"
        )

        if st.error:
            click.echo(f"Error: {st.error}")


def _stop_background(name: str) -> None:
    """Stop a background task (collect or enrich)."""
    from .collector import _pid_file, _update_status, is_running, read_status

    if not is_running(name):
        click.echo(f"No {name} is currently running.")
        return

    pf = _pid_file(name)
    try:
        pid = int(pf.read_text().strip())
    except (ValueError, FileNotFoundError):
        click.echo("Could not read PID file.")
        return

    click.echo(f"Stopping {name} (PID {pid})...")

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        click.echo("Process already exited.")
        pf.unlink(missing_ok=True)
        return

    st = read_status(name)
    if st and st.state not in ("completed", "failed", "stopped"):
        st.state = "stopped"
        st.finished_at = datetime.now().isoformat()
        _update_status(st, name)

    pf.unlink(missing_ok=True)
    click.echo(f"{name.title()} stopped.")


def _format_time(dt: datetime) -> str:
    """Format a datetime for display, showing the timezone abbreviation."""
    et_dt = dt.astimezone(ET)
    return et_dt.strftime("%Y-%m-%d %H:%M %Z")


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------

@click.command()
def setup():
    """Initialize database, download models, and run sanity checks."""
    _init()

    click.echo("=== Sentinel Setup ===")

    # 1. Create directories
    for d in ["data/outputs", "models/classical", "models/neural"]:
        Path(d).mkdir(parents=True, exist_ok=True)
    click.echo("[1/5] Directories created")

    # 2. Initialize database
    if config.database.url:
        try:
            from .data.db import SentinelDB
            db = SentinelDB(config.database.url)
            db.connect()
            db.init_schema()
            db.close()
            click.echo("[2/5] Database schema initialized")
        except Exception as e:
            click.echo(f"[2/5] Database setup failed: {e}", err=True)
            click.echo("       Set DATABASE_URL in .env and ensure PostgreSQL is running.")
    else:
        click.echo("[2/5] Skipped (DATABASE_URL not set)")

    # 3. Check spaCy model
    try:
        import spacy
        spacy.load(config.app.spacy_model)
        click.echo(f"[3/5] spaCy model '{config.app.spacy_model}' loaded")
    except OSError:
        click.echo(f"[3/5] spaCy model not found. Run: python -m spacy download {config.app.spacy_model}")

    # 4. Sanity check with hardcoded examples
    from .data.labeler import label_claim
    from .data.models import RawClaim

    examples = [
        RawClaim(
            tweet_id=1, text="$LMT to the moon! 🚀🚀🚀 Lockheed about to explode!",
            username="hypetrader", created_at=datetime.now(),
            likes=100, retweets=50, ticker="LMT", company_name="Lockheed Martin",
            price_at_tweet=450.0, price_24h_later=452.0, price_change_pct=0.44,
            has_catalyst=False,
        ),
        RawClaim(
            tweet_id=2, text="RTX awarded $2B Pentagon contract for next-gen radar systems",
            username="defensenews", created_at=datetime.now(),
            likes=200, retweets=100, ticker="RTX", company_name="RTX Corporation",
            price_at_tweet=100.0, price_24h_later=105.0, price_change_pct=5.0,
            news_headlines=["RTX wins $2B Pentagon radar contract"],
            has_catalyst=True, catalyst_type="contract",
        ),
        RawClaim(
            tweet_id=3, text="NOC looks interesting here. Watching.",
            username="calmtrader", created_at=datetime.now(),
            likes=10, retweets=2, ticker="NOC", company_name="Northrop Grumman",
            price_at_tweet=480.0, price_24h_later=520.0, price_change_pct=8.33,
            has_catalyst=True, catalyst_type="geopolitical",
        ),
    ]

    click.echo("[4/5] Sanity check:")
    for raw in examples:
        labeled = label_claim(raw)
        click.echo(
            f"  '{raw.text[:50]}...' → {labeled.label} "
            f"(dir: {labeled.claimed_direction}/{labeled.actual_direction}, "
            f"score: {labeled.exaggeration_score})"
        )

    click.echo("[5/5] Setup complete.")
    click.echo("\nNext steps:")
    click.echo("  uv run collect --n 100  # Scrape claims")
    click.echo("  uv run serve            # Start API server")


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------

@click.command()
@click.option("-n", "n_per_ticker", default=50, help="Tweets per ticker per day")
@click.option("--days", default=1, type=int,
              help="Number of days to scrape (default: 1). Scrapes each day separately "
                   "for even temporal distribution. Ends 25h ago so all tweets can be enriched.")
@click.option("--tickers", default=None, help="Comma-separated tickers (default: all)")
@click.option("--background", is_flag=True, help="Run in background")
@click.option("--status", "show_status", is_flag=True, help="Check background collection progress")
@click.option("--stop", "do_stop", is_flag=True, help="Stop background collection")
@click.option("--_daemonized", is_flag=True, hidden=True)
def collect(
    n_per_ticker: int,
    days: int,
    tickers: str | None,
    background: bool,
    show_status: bool,
    do_stop: bool,
    _daemonized: bool,
):
    """Scrape and label defense stock claims."""
    _init()

    if show_status:
        _show_status("collect")
        return

    if do_stop:
        _stop_background("collect")
        return

    if days < 1:
        click.echo("Error: --days must be at least 1.", err=True)
        sys.exit(1)

    # Time window: from N days ago until 25h ago (so all tweets can be enriched)
    until = datetime.now(timezone.utc) - timedelta(hours=25)
    since = until - timedelta(days=days)
    daily = days > 1

    from .collector import is_running

    if background and not _daemonized:
        if is_running("collect"):
            click.echo("A collection is already running. Use 'uv run collect --status' to check progress.")
            sys.exit(1)

        import shutil
        import subprocess

        Path("data").mkdir(parents=True, exist_ok=True)
        log_file = open("data/collect.log", "w")

        collect_bin = shutil.which("collect")
        if collect_bin is None:
            collect_bin = str(Path(sys.executable).parent / "collect")

        cmd = [collect_bin, "-n", str(n_per_ticker), "--days", str(days), "--_daemonized"]
        if tickers:
            cmd.extend(["--tickers", tickers])

        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        click.echo(f"Collection started in background (PID {proc.pid})")
        click.echo("  uv run collect --status  — check progress")
        click.echo("  uv run collect --stop    — stop collection")
        return

    # Foreground or daemonized run
    from .collector import read_status, run_collection
    from .data.stocks import get_public_tickers

    ticker_list = tickers.split(",") if tickers else get_public_tickers()

    if not _daemonized:
        parts = [f"Collecting {n_per_ticker}/ticker/day for {days} day{'s' if days != 1 else ''}"]
        parts.append(f"({_format_time(since)} to {_format_time(until)})")
        click.echo(" ".join(parts))

    asyncio.run(run_collection(
        n_per_ticker=n_per_ticker,
        ticker_list=ticker_list,
        database_url=config.database.url,
        since=since,
        until=until,
        daily=daily,
    ))

    if not _daemonized:
        st = read_status("collect")
        if st:
            click.echo(f"Done: {st.labeled} labeled, {st.failed} failed")


# ---------------------------------------------------------------------------
# enrich
# ---------------------------------------------------------------------------

@click.command()
@click.option("--tickers", default=None, help="Comma-separated tickers (default: all)")
@click.option("--days", default=None, type=int,
              help="Only enrich claims from the last N days")
@click.option("--unlabeled", is_flag=True, help="Only enrich claims that haven't been labeled yet")
@click.option("--background", is_flag=True, help="Run in background")
@click.option("--status", "show_status", is_flag=True, help="Check background enrichment progress")
@click.option("--stop", "do_stop", is_flag=True, help="Stop background enrichment")
@click.option("--_daemonized", is_flag=True, hidden=True)
def enrich(
    tickers: str | None,
    days: int | None,
    unlabeled: bool,
    background: bool,
    show_status: bool,
    do_stop: bool,
    _daemonized: bool,
):
    """Re-enrich existing raw claims with fresh price and news data."""
    _init()

    if show_status:
        _show_status("enrich")
        return

    if do_stop:
        _stop_background("enrich")
        return

    if not config.database.url:
        click.echo("Error: DATABASE_URL not set. Enrich reads from the database.", err=True)
        sys.exit(1)

    since = None
    until = None
    if days is not None:
        until = datetime.now(timezone.utc)
        since = until - timedelta(days=days)

    ticker_list = tickers.split(",") if tickers else None

    from .collector import is_running

    if background and not _daemonized:
        if is_running("enrich"):
            click.echo("Enrichment is already running. Use 'uv run enrich --status' to check progress.")
            sys.exit(1)

        import shutil
        import subprocess

        Path("data").mkdir(parents=True, exist_ok=True)
        log_file = open("data/enrich.log", "w")

        enrich_bin = shutil.which("enrich")
        if enrich_bin is None:
            enrich_bin = str(Path(sys.executable).parent / "enrich")

        cmd = [enrich_bin, "--_daemonized"]
        if tickers:
            cmd.extend(["--tickers", tickers])
        if days is not None:
            cmd.extend(["--days", str(days)])
        if unlabeled:
            cmd.append("--unlabeled")

        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        click.echo(f"Enrichment started in background (PID {proc.pid})")
        click.echo("  uv run enrich --status  — check progress")
        click.echo("  uv run enrich --stop    — stop enrichment")
        return

    # Foreground or daemonized run
    from .collector import read_status, run_enrichment

    if not _daemonized:
        parts = ["Re-enriching"]
        if unlabeled:
            parts.append("unlabeled")
        parts.append("claims")
        if ticker_list:
            parts.append(f"for {', '.join(ticker_list)}")
        if days:
            parts.append(f"from last {days} day{'s' if days != 1 else ''}")
        click.echo(" ".join(parts))

    asyncio.run(run_enrichment(
        database_url=config.database.url,
        tickers=ticker_list,
        since=since,
        until=until,
        unlabeled_only=unlabeled,
    ))

    if not _daemonized:
        st = read_status("enrich")
        if st:
            click.echo(f"Done: {st.labeled} labeled, {st.failed} failed")


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------

@click.command()
@click.option("--host", default="0.0.0.0", help="Host to bind")
@click.option("--port", default=None, type=int, help="Port (default from config)")
@click.option("--workers", default=4, type=int, help="Number of gunicorn workers")
@click.option("--dev", is_flag=True, help="Run Flask dev server instead of gunicorn")
def serve(host: str, port: int | None, workers: int, dev: bool):
    """Start the Sentinel API server (gunicorn by default)."""
    _init()

    from .api.app import create_app

    port = port or config.app.port

    if dev:
        app = create_app(config.database.url)
        click.echo(f"Starting Sentinel dev server on {host}:{port}")
        app.run(host=host, port=port, debug=True)
    else:
        click.echo(f"Starting Sentinel API via gunicorn on {host}:{port} ({workers} workers)")
        os.execvp(sys.executable, [
            sys.executable, "-m", "gunicorn",
            "src.api.app:create_app()",
            "--bind", f"{host}:{port}",
            "--workers", str(workers),
            "--access-logfile", "-",
            "--error-logfile", "-",
        ])
