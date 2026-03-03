"""
CLI entry point for Sentinel.

Commands:
  sentinel setup      — Initialize DB, download models, train, run experiment
  sentinel collect    — Scrape and label defense stock claims
  sentinel serve      — Start the Flask API server
  sentinel train      — Train/retrain ML models
  sentinel experiment — Run news ablation experiment
"""

import asyncio
import logging
import sys

import click

from .config import config

logger = logging.getLogger("sentinel.cli")


@click.group()
def cli():
    """Sentinel — Defense Stock Claim Analyzer."""
    config.setup_logging()


@cli.command()
def setup():
    """Initialize database, download models, and run initial training."""
    from pathlib import Path

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
    from .data.models import RawClaim
    from .data.labeler import label_claim
    from datetime import datetime

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
    click.echo("  uv run python main.py collect --n 100  # Scrape claims")
    click.echo("  uv run python main.py serve            # Start API server")


@cli.command()
@click.option("--n", "n_per_ticker", default=50, help="Tweets per ticker")
@click.option("--tickers", default=None, help="Comma-separated tickers (default: all)")
def collect(n_per_ticker: int, tickers: str | None):
    """Scrape and label defense stock claims."""
    from .scraper import DefenseStockScraper
    from .price_fetcher import PriceFetcher
    from .news_fetcher import fetch_news_for_claim, classify_catalyst
    from .data.labeler import label_claim
    from .data.db import SentinelDB
    from .data.stocks import get_public_tickers

    ticker_list = tickers.split(",") if tickers else get_public_tickers()
    click.echo(f"Collecting claims for {len(ticker_list)} tickers, {n_per_ticker} each")

    async def run():
        scraper = DefenseStockScraper(config.twitter.db_path)
        claims = await scraper.scrape_defense_claims(
            tickers=ticker_list, limit_per_ticker=n_per_ticker,
        )
        click.echo(f"Scraped {len(claims)} raw claims")

        # Enrich with price and news
        pf = PriceFetcher()
        labeled_count = 0

        db = None
        if config.database.url:
            db = SentinelDB(config.database.url)
            db.connect()
            db.init_schema()

        for i, claim in enumerate(claims):
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

                # Label
                labeled = label_claim(claim)
                labeled_count += 1

                if db:
                    db.insert_labeled_claim(labeled)

                if (i + 1) % 10 == 0:
                    click.echo(f"  Processed {i + 1}/{len(claims)} claims")

            except Exception as e:
                logger.warning(f"Failed to process claim {claim.tweet_id}: {e}")

        if db:
            db.close()

        click.echo(f"Done: {labeled_count} claims labeled and stored")

    asyncio.run(run())


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind")
@click.option("--port", default=None, type=int, help="Port (default from config)")
@click.option("--workers", default=4, type=int, help="Number of gunicorn workers")
@click.option("--dev", is_flag=True, help="Run Flask dev server instead of gunicorn")
def serve(host: str, port: int | None, workers: int, dev: bool):
    """Start the Sentinel API server (gunicorn by default)."""
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


@cli.command()
@click.option(
    "--model", "model_name",
    type=click.Choice(["all", "baseline", "classical", "neural"]),
    default="all",
    help="Which model to train",
)
def train(model_name: str):
    """Train ML models on labeled data."""
    click.echo(f"Training: {model_name}")
    click.echo("Note: Requires labeled data in PostgreSQL. Run 'collect' first.")

    # This is a placeholder — full training requires collected data
    if model_name in ("all", "baseline"):
        click.echo("Baseline: majority class classifier (trained at prediction time)")

    if model_name in ("all", "classical"):
        click.echo("Classical: TF-IDF + LogReg/SVM (requires labeled data)")

    if model_name in ("all", "neural"):
        click.echo("Neural: FinBERT + MiniLM fusion (requires labeled data + GPU recommended)")


@cli.command()
def experiment():
    """Run the news feature ablation experiment."""
    click.echo("Running news ablation experiment...")
    click.echo("Note: Requires labeled data. Run 'collect' first to build dataset.")
    click.echo("Results will be saved to data/outputs/news_ablation.png")
