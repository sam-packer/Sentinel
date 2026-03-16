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
from .models import MODEL_REGISTRY, MODEL_DIR

# US Eastern timezone (ET) for market hours
try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
except ImportError:
    # Fallback: EST = UTC-5 (ignores DST)
    ET = timezone(timedelta(hours=-5))

logger = logging.getLogger("sentinel.cli")


def _init():
    """Common initialization for all commands."""
    config.setup_logging()


def _launch_background(cmd: list[str], log_path: str) -> int:
    """Launch a command as a detached background process, logging to a file.

    Returns the PID of the spawned process.
    """
    import subprocess

    Path("data").mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return proc.pid


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

        click.echo(f"Sentinel {name.title()} — {minutes}m {seconds}s elapsed (PID {st.pid})")
        click.echo("")

        if st.state == "scraping":
            pct = (st.tickers_scraped / st.tickers_total * 100) if st.tickers_total > 0 else 0
            filled = int(pct / 100 * 20)
            progress_bar = "\u2588" * filled + "\u2591" * (20 - filled)
            click.echo(f"  Scraping  [{progress_bar}] {st.tickers_scraped}/{st.tickers_total} tickers ({pct:.0f}%)")
            click.echo(f"            {st.scrape_tweets_found} tweets found")
            if st.current_ticker:
                click.echo(f"            currently: {st.current_ticker}")

        elif st.phase.startswith("classifying") and st.accounts_total > 0:
            cls_pct = st.accounts_classified / st.accounts_total * 100
            filled = int(cls_pct / 100 * 20)
            progress_bar = "\u2588" * filled + "\u2591" * (20 - filled)
            click.echo(f"  Classify  [{progress_bar}] {st.accounts_classified}/{st.accounts_total} accounts ({cls_pct:.0f}%)")

        else:
            total = st.scraped
            done = st.enriched
            pct = (done / total * 100) if total > 0 else 0
            filled = int(pct / 100 * 20)
            progress_bar = "\u2588" * filled + "\u2591" * (20 - filled)
            click.echo(f"  Enrich    [{progress_bar}] {done}/{total} claims ({pct:.0f}%)")
            click.echo(f"            {st.labeled} labeled, {st.failed} failed")
            if st.current_ticker:
                click.echo(f"            currently: {st.current_ticker}")

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
    for d in ["data/outputs"]:
        Path(d).mkdir(parents=True, exist_ok=True)
    click.echo("[1/4] Directories created")

    # 2. Initialize database
    if config.database.url:
        try:
            from .data.db import SentinelDB
            db = SentinelDB(config.database.url)
            db.connect()
            db.init_schema()
            db.close()
            click.echo("[2/4] Database schema initialized")
        except Exception as e:
            click.echo(f"[2/4] Database setup failed: {e}", err=True)
            click.echo("       Set DATABASE_URL in .env and ensure PostgreSQL is running.")
    else:
        click.echo("[2/4] Skipped (DATABASE_URL not set)")

    # 3. Sanity check with hardcoded examples
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

    click.echo("[3/4] Sanity check:")
    for raw in examples:
        labeled = label_claim(raw)
        click.echo(
            f"  '{raw.text[:50]}...' → {labeled.label} "
            f"(dir: {labeled.claimed_direction}/{labeled.actual_direction}, "
            f"score: {labeled.exaggeration_score})"
        )

    click.echo("[4/4] Setup complete.")
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

        collect_bin = shutil.which("collect")
        if collect_bin is None:
            collect_bin = str(Path(sys.executable).parent / "collect")

        cmd = [collect_bin, "-n", str(n_per_ticker), "--days", str(days), "--_daemonized"]
        if tickers:
            cmd.extend(["--tickers", tickers])

        pid = _launch_background(cmd, "data/collect.log")
        click.echo(f"Collection started in background (PID {pid})")
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
@click.option("--rejudge", is_flag=True, help="Reclassify ALL accounts, not just unclassified ones")
@click.option("--naive", "labeler", flag_value="naive",
              help="Use naive (keyword-based) labeler")
@click.option("--improved", "labeler", flag_value="improved",
              help="Use improved (NLP-enhanced) labeler")
@click.option("--background", is_flag=True, help="Run in background")
@click.option("--status", "show_status", is_flag=True, help="Check background enrichment progress")
@click.option("--stop", "do_stop", is_flag=True, help="Stop background enrichment")
@click.option("--_daemonized", is_flag=True, hidden=True)
@click.option("--_labeler", default=None, hidden=True)
def enrich(
    tickers: str | None,
    days: int | None,
    unlabeled: bool,
    rejudge: bool,
    labeler: str,
    background: bool,
    show_status: bool,
    do_stop: bool,
    _daemonized: bool,
    _labeler: str | None,
):
    """Re-enrich existing raw claims with fresh price and news data.

    Requires --naive or --improved to specify which labeler to use.
    """
    _init()

    # When launched as a background subprocess, use the hidden --_labeler flag
    if _labeler is not None:
        labeler = _labeler

    if show_status:
        _show_status("enrich")
        return

    if do_stop:
        _stop_background("enrich")
        return

    if labeler is None:
        click.echo("Error: specify --naive or --improved to choose a labeler.", err=True)
        sys.exit(1)

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

        enrich_bin = shutil.which("enrich")
        if enrich_bin is None:
            enrich_bin = str(Path(sys.executable).parent / "enrich")

        cmd = [enrich_bin, "--_daemonized", "--_labeler", labeler]
        if tickers:
            cmd.extend(["--tickers", tickers])
        if days is not None:
            cmd.extend(["--days", str(days)])
        if unlabeled:
            cmd.append("--unlabeled")
        if rejudge:
            cmd.append("--rejudge")

        pid = _launch_background(cmd, "data/enrich.log")
        click.echo(f"Enrichment ({labeler} labeler) started in background (PID {pid})")
        click.echo("  uv run enrich --status  — check progress")
        click.echo("  uv run enrich --stop    — stop enrichment")
        return

    # Foreground or daemonized run
    from .collector import read_status, run_enrichment

    if not _daemonized:
        parts = [f"Re-enriching ({labeler} labeler)"]
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
        rejudge=rejudge,
        labeler=labeler,
    ))

    if not _daemonized:
        st = read_status("enrich")
        if st:
            click.echo(f"Done: {st.labeled} labeled, {st.failed} failed")


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------

@click.command()
@click.option("--limit", default=100, type=int, help="Max accounts to classify")
@click.option("--reclassify", is_flag=True, help="Reclassify already-classified accounts")
def classify(limit: int, reclassify: bool):
    """Classify accounts as human or bot using LLM-as-judge.

    Primarily for catch-up classification of existing accounts.
    New accounts are automatically classified during collection."""
    _init()

    if not config.database.url:
        click.echo("Error: DATABASE_URL not set.", err=True)
        sys.exit(1)

    from .bot_detector import classify_accounts_batch
    from .data.db import SentinelDB
    from .data.models import Account

    db = SentinelDB(config.database.url)
    db.connect()

    # Get unclassified accounts with sample tweets
    if reclassify:
        accounts_data = db.get_all_accounts_with_tweets(limit=limit)
        rows = [(a["username"], a["sample_tweets"]) for a in accounts_data]
    else:
        accounts_data = db.get_unclassified_accounts(limit=limit)
        rows = [(a["username"], a["sample_tweets"]) for a in accounts_data]

    if not rows:
        click.echo("No accounts to classify.")
        db.close()
        return

    click.echo(f"Classifying {len(rows)} accounts...")

    # Prepare batch input
    batch = []
    for username, tweets in rows:
        # Take up to 5 sample tweets
        sample = tweets[:5] if isinstance(tweets, list) else [tweets]
        batch.append({"username": username, "sample_tweets": sample})

    # Run classification
    results = classify_accounts_batch(batch)

    # Save results
    classified = 0
    bots_found = 0
    for username, classification in results:
        account = db.get_account(username) or Account(username=username)
        account.account_type = classification.account_type
        account.classification_reason = f"{classification.account_type}: {classification.reason}"
        account.classified_at = datetime.now(timezone.utc)
        db.upsert_account(account)

        if classification.is_filtered:
            bots_found += 1
        classified += 1

        click.echo(
            f"  @{username}: {classification.account_type.upper()} "
            f"({classification.confidence:.0%}) "
            f"— {classification.reason}"
        )

    db.close()
    click.echo(f"\nDone: {classified} classified, {bots_found} filtered (bot/garbage)")


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


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def _get_model(name: str):
    """Import and instantiate a model by name."""
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        click.echo(f"Unknown model '{name}'. Available: {available}", err=True)
        sys.exit(1)

    module_path, class_name = MODEL_REGISTRY[name].rsplit(":", 1)
    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls()


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

@click.command()
@click.argument("model_name")
@click.option("--test-size", default=0.2, type=float, help="Fraction held out for testing (default: 0.2)")
@click.option("--seed", default=42, type=int, help="Random seed for train/test split")
@click.option("--tune", is_flag=True, help="Run Optuna hyperparameter tuning (default: reuse saved params)")
@click.option("--naive", "labels", flag_value="naive",
              help="Train on naive labels")
@click.option("--improved", "labels", flag_value="improved",
              help="Train on improved labels")
def train(model_name: str, test_size: float, seed: int, tune: bool, labels: str):
    """Train a model. Requires --naive or --improved to specify label set."""
    _init()

    if labels is None:
        click.echo("Error: specify --naive or --improved to choose a label set.", err=True)
        sys.exit(1)

    if not config.database.url:
        click.echo("Error: DATABASE_URL not set.", err=True)
        sys.exit(1)

    import json
    from .data.db import SentinelDB
    from .models.data import load_labeled_claims, prepare_split
    from .models.evaluate import compute_metrics, format_metrics, format_report

    label_table = f"{labels}_labeled_claims"
    model = _get_model(model_name)

    # Load data
    click.echo(f"Loading claims from {label_table}...")
    db = SentinelDB(config.database.url)
    db.connect()
    claims = load_labeled_claims(db, label_table=label_table)
    db.close()

    if not claims:
        click.echo(f"No claims found in {label_table}. Run 'uv run enrich --{labels}' first.", err=True)
        sys.exit(1)

    # Split
    split = prepare_split(claims, test_size=test_size, seed=seed)
    click.echo(f"Data: {split.train_size} train, {split.test_size} test")

    # Model directory: models/{name}/{labels}/
    model_dir = MODEL_DIR / model.name / f"{labels}_labeler"

    # Load saved hyperparameters unless --tune is passed
    saved_params = None
    if not tune:
        params_path = model_dir / "best_params.json"
        if params_path.exists():
            with open(params_path) as f:
                saved_params = json.load(f)
            click.echo("Using saved hyperparameters (pass --tune to retune)")
        else:
            click.echo("No saved params found, running Optuna tuning...")

    # Train
    click.echo(f"Training {model.name} on {labels} labels...")
    if saved_params is not None:
        metadata = model.train(split.train_texts, split.train_labels, saved_params=saved_params)
    else:
        metadata = model.train(split.train_texts, split.train_labels)
    for key, value in metadata.items():
        click.echo(f"  {key}: {value}")

    # Save
    model.save(model_dir)
    click.echo(f"Saved to {model_dir}/")

    # Evaluate on test set and save results
    predictions = model.predict_batch(split.test_texts)
    metrics = compute_metrics(predictions, split.test_labels)

    # Collect mispredictions with full context while everything is in memory
    from .models.data import EXCLUDE_LABELS
    import random as _random

    filtered_claims = [c for c in claims if c["label"] not in EXCLUDE_LABELS]
    rng = _random.Random(seed)
    rng.shuffle(filtered_claims)
    split_idx = int(len(filtered_claims) * (1 - test_size))
    test_claims = filtered_claims[split_idx:]

    mispredictions = []
    for i, (pred, actual) in enumerate(zip(predictions, split.test_labels)):
        if pred != actual:
            claim = test_claims[i]
            mispredictions.append({
                "text": claim["text"],
                "ticker": claim["ticker"],
                "username": claim.get("username"),
                "predicted": pred,
                "actual": actual,
                "price_change_pct": claim.get("price_change_pct"),
                "claimed_direction": claim.get("claimed_direction"),
                "actual_direction": claim.get("actual_direction"),
                "has_catalyst": claim.get("has_catalyst"),
                "catalyst_type": claim.get("catalyst_type"),
                "exaggeration_score": claim.get("exaggeration_score"),
            })

    # Save evaluation JSON
    results_path = model_dir / "evaluation.json"
    with open(results_path, "w") as f:
        json.dump({
            "model": model.name,
            "labels": labels,
            "label_table": label_table,
            "test_size": split.test_size,
            "train_size": split.train_size,
            "seed": seed,
            "metrics": metrics,
        }, f, indent=2)

    # Save mispredictions
    errors_path = model_dir / "mispredictions.json"
    with open(errors_path, "w") as f:
        json.dump(mispredictions, f, indent=2, default=str)

    # Write markdown report (includes misprediction examples)
    report = format_report(
        model_name=model.name,
        labels=labels,
        train_size=split.train_size,
        test_size=split.test_size,
        seed=seed,
        metrics=metrics,
        training_meta=metadata,
        mispredictions=mispredictions,
    )
    report_path = model_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")

    click.echo("")
    click.echo(f"Test set results ({labels} labels):")
    click.echo(format_metrics(metrics))
    click.echo(f"\n{len(mispredictions)} mispredictions saved to {errors_path}")
    click.echo(f"Report saved to {report_path}")


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

@click.command()
@click.argument("model_name")
@click.option("--seed", default=42, type=int, help="Random seed (must match training)")
@click.option("--test-size", default=0.2, type=float, help="Test fraction (must match training)")
@click.option("--naive", "labels", flag_value="naive",
              help="Evaluate against naive labels")
@click.option("--improved", "labels", flag_value="improved",
              help="Evaluate against improved labels")
def evaluate(model_name: str, seed: int, test_size: float, labels: str):
    """Evaluate a trained model. Requires --naive or --improved to specify label set."""
    _init()

    if labels is None:
        click.echo("Error: specify --naive or --improved to choose a label set.", err=True)
        sys.exit(1)

    if not config.database.url:
        click.echo("Error: DATABASE_URL not set.", err=True)
        sys.exit(1)

    import json
    from .data.db import SentinelDB
    from .models.data import load_labeled_claims, prepare_split
    from .models.evaluate import compute_metrics, format_metrics

    label_table = f"{labels}_labeled_claims"
    model = _get_model(model_name)

    # Load saved model from the label-specific directory
    model_dir = MODEL_DIR / model.name / f"{labels}_labeler"
    if not (model_dir / "model.json").exists():
        click.echo(
            f"No trained model found at {model_dir}/. "
            f"Run 'uv run train {model_name} --{labels}' first.",
            err=True,
        )
        sys.exit(1)

    model.load(model_dir)

    # Load and split data (same seed = same split)
    click.echo(f"Loading claims from {label_table}...")
    db = SentinelDB(config.database.url)
    db.connect()
    claims = load_labeled_claims(db, label_table=label_table)
    db.close()

    split = prepare_split(claims, test_size=test_size, seed=seed)
    click.echo(f"Evaluating {model.name} on {split.test_size} test samples ({labels} labels)")
    click.echo("")

    predictions = model.predict_batch(split.test_texts)
    metrics = compute_metrics(predictions, split.test_labels)
    click.echo(format_metrics(metrics))

    # Save evaluation results
    results_path = model_dir / "evaluation.json"
    with open(results_path, "w") as f:
        json.dump({
            "model": model.name,
            "labels": labels,
            "label_table": label_table,
            "test_size": split.test_size,
            "train_size": split.train_size,
            "seed": seed,
            "metrics": metrics,
        }, f, indent=2)
    click.echo(f"\nEvaluation saved to {results_path}")


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

@click.command()
@click.argument("model_name")
@click.argument("text")
@click.option("--naive", "labels", flag_value="naive", default="naive",
              help="Use model trained on naive labels")
@click.option("--improved", "labels", flag_value="improved",
              help="Use model trained on improved labels")
def predict(model_name: str, text: str, labels: str):
    """Predict a label for tweet text. Usage: uv run predict baseline "tweet text" """
    _init()

    from .models import load_model

    try:
        model = load_model(model_name, labels=labels)
    except KeyError as e:
        click.echo(str(e), err=True)
        sys.exit(1)
    except FileNotFoundError as e:
        click.echo(str(e), err=True)
        sys.exit(1)

    label = model.predict(text)
    click.echo(f"{label}")
