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
import signal
import sys
from datetime import datetime
from pathlib import Path

import click

from .config import config

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


@click.command()
@click.option("-n", "n_per_ticker", default=50, help="Tweets per ticker")
@click.option("--tickers", default=None, help="Comma-separated tickers (default: all)")
@click.option("--background", is_flag=True, help="Run in background")
@click.option("--_daemonized", is_flag=True, hidden=True)
def collect(n_per_ticker: int, tickers: str | None, background: bool, _daemonized: bool):
    """Scrape and label defense stock claims."""
    _init()

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
        click.echo(f"Collecting claims for {len(ticker_list)} tickers, {n_per_ticker} each")

    asyncio.run(run_collection(
        n_per_ticker=n_per_ticker,
        ticker_list=ticker_list,
        database_url=config.database.url,
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
        import subprocess
        click.echo(f"Starting Sentinel API via gunicorn on {host}:{port} ({workers} workers)")
        subprocess.run([
            sys.executable, "-m", "gunicorn",
            "--factory", "src.api.app:create_app",
            "--bind", f"{host}:{port}",
            "--workers", str(workers),
            "--access-logfile", "-",
            "--error-logfile", "-",
        ], check=True)


@click.command()
@click.option(
    "--model", "model_name",
    type=click.Choice(["all", "baseline", "classical", "neural"]),
    default="all",
    help="Which model to train",
)
def train(model_name: str):
    """Train ML models on labeled data."""
    _init()

    click.echo(f"Training: {model_name}")
    click.echo("Note: Requires labeled data in PostgreSQL. Run 'collect' first.")

    if model_name in ("all", "baseline"):
        click.echo("Baseline: majority class classifier (trained at prediction time)")

    if model_name in ("all", "classical"):
        click.echo("Classical: TF-IDF + LogReg/SVM (requires labeled data)")

    if model_name in ("all", "neural"):
        click.echo("Neural: FinBERT + MiniLM fusion (requires labeled data + GPU recommended)")


@click.command()
def experiment():
    """Run the news feature ablation experiment."""
    _init()

    click.echo("Running news ablation experiment...")
    click.echo("Note: Requires labeled data. Run 'collect' first to build dataset.")
    click.echo("Results will be saved to data/outputs/news_ablation.png")
