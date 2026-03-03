"""
CLI entry points for Sentinel.

Each function is a standalone click command exposed via pyproject.toml [project.scripts].

Usage:
  uv run setup                        — Initialize DB, sanity check
  uv run collect --n 100              — Scrape and label claims
  uv run collect --n 100 --background — Run collection in background
  uv run status                       — Check collection progress
  uv run stop                         — Stop background collection
  uv run serve                        — Start API server
  uv run train                        — Train ML models
  uv run experiment                   — Run news ablation
"""

import asyncio
import logging
import os
import re
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


def _parse_time(value: str, tz: timezone | ZoneInfo = ET) -> datetime:
    """Parse a time string into a timezone-aware datetime.

    Relative durations are computed from the current time (always UTC-based).
    Absolute times without an explicit offset are interpreted in ``tz``
    (defaults to US Eastern).

    Supports:
      - Relative durations: "30m", "1h", "2d", "1w" (subtracted from now)
      - ISO datetime: "2026-03-02T09:30:00" (interpreted as ``tz``)
      - Date only: "2026-03-02" (midnight in ``tz``)
      - "now" keyword
    """
    value = value.strip()

    if value.lower() == "now":
        return datetime.now(timezone.utc)

    # Relative duration: e.g. "30m", "2h", "1d", "1w"
    match = re.fullmatch(r"(\d+)\s*([mhdw])", value, re.IGNORECASE)
    if match:
        amount = int(match.group(1))
        unit = match.group(2).lower()
        delta = {"m": timedelta(minutes=amount), "h": timedelta(hours=amount),
                 "d": timedelta(days=amount), "w": timedelta(weeks=amount)}[unit]
        return datetime.now(timezone.utc) - delta

    # ISO datetime or date-only — bare values are interpreted in the given tz
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(value, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=tz)
            return dt
        except ValueError:
            continue

    raise click.BadParameter(
        f"Cannot parse '{value}'. Use relative (e.g. 1h, 30m, 2d) "
        f"or ISO format (e.g. 2026-03-02, 2026-03-02T09:30)."
    )


def _format_time(dt: datetime) -> str:
    """Format a datetime for display, showing the timezone abbreviation."""
    # Convert to ET for display so times are always shown in market time
    et_dt = dt.astimezone(ET)
    # %Z gives 'EST' or 'EDT' depending on DST
    return et_dt.strftime("%Y-%m-%d %H:%M %Z")


@click.command()
@click.option("-n", "n_per_ticker", default=50, help="Tweets per ticker")
@click.option("--tickers", default=None, help="Comma-separated tickers (default: all)")
@click.option("--since", "since_str", default=None,
              help="Only scrape tweets after this time. Tweets <25h old are skipped "
                   "during labeling (24h price window hasn't elapsed). "
                   "Relative (1h, 30m, 2d) or ISO in ET (2026-03-02T09:30)")
@click.option("--until", "until_str", default=None,
              help="Only scrape tweets before this time. "
                   "Relative (1h, 30m, 2d) or ISO in ET (2026-03-02T09:30)")
@click.option("--market-open", "market_open_date", default=None,
              help="Scrape tweets around market open (9:00-10:30 AM ET) on DATE. "
                   "Use 'today', 'yesterday', or YYYY-MM-DD.")
@click.option("--background", is_flag=True, help="Run in background")
@click.option("--_daemonized", is_flag=True, hidden=True)
def collect(
    n_per_ticker: int,
    tickers: str | None,
    since_str: str | None,
    until_str: str | None,
    market_open_date: str | None,
    background: bool,
    _daemonized: bool,
):
    """Scrape and label defense stock claims."""
    _init()

    # Parse time window
    since = None
    until = None

    if market_open_date and (since_str or until_str):
        click.echo("Error: --market-open cannot be combined with --since/--until.", err=True)
        sys.exit(1)

    if market_open_date:
        # Resolve the date
        mo = market_open_date.strip().lower()
        if mo == "today":
            target = datetime.now(ET).date()
        elif mo == "yesterday":
            target = (datetime.now(ET) - timedelta(days=1)).date()
        else:
            try:
                target = datetime.strptime(mo, "%Y-%m-%d").date()
            except ValueError:
                click.echo(
                    f"Error: Cannot parse date '{market_open_date}'. "
                    "Use 'today', 'yesterday', or YYYY-MM-DD.", err=True
                )
                sys.exit(1)

        # Market open window: 9:00 AM - 10:30 AM ET
        since = datetime(target.year, target.month, target.day, 9, 0, tzinfo=ET)
        until = datetime(target.year, target.month, target.day, 10, 30, tzinfo=ET)
        click.echo(f"Market-open mode: {_format_time(since)} to {_format_time(until)}")

    if since_str:
        since = _parse_time(since_str)
    if until_str:
        until = _parse_time(until_str)

    if since and until and since >= until:
        click.echo("Error: --since must be before --until.", err=True)
        sys.exit(1)

    from .collector import is_running

    if background and not _daemonized:
        if is_running():
            click.echo("A collection is already running. Use 'uv run status' to check progress.")
            sys.exit(1)

        import shutil
        import subprocess

        Path("data").mkdir(parents=True, exist_ok=True)
        log_file = open("data/collect.log", "w")

        collect_bin = shutil.which("collect")
        if collect_bin is None:
            collect_bin = str(Path(sys.executable).parent / "collect")

        cmd = [collect_bin, "-n", str(n_per_ticker), "--_daemonized"]
        if tickers:
            cmd.extend(["--tickers", tickers])
        if since_str:
            cmd.extend(["--since", since_str])
        if until_str:
            cmd.extend(["--until", until_str])
        if market_open_date:
            cmd.extend(["--market-open", market_open_date])

        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        click.echo(f"Collection started in background (PID {proc.pid})")
        click.echo("  uv run status  — check progress")
        click.echo("  uv run stop    — stop collection")
        return

    # Foreground or daemonized run
    from .collector import read_status, run_collection
    from .data.stocks import get_public_tickers

    ticker_list = tickers.split(",") if tickers else get_public_tickers()

    if not _daemonized:
        parts = [f"Collecting claims for {len(ticker_list)} tickers, {n_per_ticker} each"]
        if since:
            parts.append(f"since {_format_time(since)}")
        if until:
            parts.append(f"until {_format_time(until)}")
        click.echo(" ".join(parts))

    asyncio.run(run_collection(
        n_per_ticker=n_per_ticker,
        ticker_list=ticker_list,
        database_url=config.database.url,
        since=since,
        until=until,
    ))

    if not _daemonized:
        st = read_status()
        if st:
            click.echo(f"Done: {st.labeled} labeled, {st.failed} failed")


@click.command()
def status():
    """Check collection progress."""
    _init()

    from .collector import LOG_FILE, is_running, read_status

    st = read_status()
    if st is None:
        click.echo("No collection has been run yet.")
        return

    running = is_running()

    if running and st.state in ("scraping", "enriching"):
        started = datetime.fromisoformat(st.started_at)
        elapsed = datetime.now() - started
        minutes = int(elapsed.total_seconds() // 60)
        seconds = int(elapsed.total_seconds() % 60)

        click.echo("Sentinel Collection Status")
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

        # Show recent log output
        if LOG_FILE.exists():
            click.echo("")
            click.echo("Recent log output:")
            try:
                lines = LOG_FILE.read_text().splitlines()
                for line in lines[-10:]:
                    click.echo(f"  {line}")
            except Exception:
                pass
    else:
        click.echo("No collection running.")
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


@click.command()
def stop():
    """Stop a background collection."""
    _init()

    from .collector import PID_FILE, _update_status, is_running, read_status

    if not is_running():
        click.echo("No collection is currently running.")
        return

    try:
        pid = int(PID_FILE.read_text().strip())
    except (ValueError, FileNotFoundError):
        click.echo("Could not read PID file.")
        return

    click.echo(f"Stopping collection (PID {pid})...")

    try:
        if sys.platform == "win32":
            os.kill(pid, signal.SIGTERM)
        else:
            os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        click.echo("Process already exited.")
        PID_FILE.unlink(missing_ok=True)
        return

    st = read_status()
    if st and st.state not in ("completed", "failed", "stopped"):
        st.state = "stopped"
        st.finished_at = datetime.now().isoformat()
        _update_status(st)

    PID_FILE.unlink(missing_ok=True)
    click.echo("Collection stopped.")


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


