"""
PostgreSQL persistence layer for Sentinel.

Stores raw and labeled claims. The Flask API reads from this DB
for the live feed and stats endpoints.
"""

import json
import logging
from datetime import datetime, timezone

import psycopg

from .models import RawClaim, LabeledClaim

logger = logging.getLogger("sentinel.db")

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS raw_claims (
    tweet_id BIGINT PRIMARY KEY,
    text TEXT NOT NULL,
    username VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    likes INTEGER DEFAULT 0,
    retweets INTEGER DEFAULT 0,
    ticker VARCHAR(10) NOT NULL,
    company_name VARCHAR(255) NOT NULL,
    price_at_tweet DOUBLE PRECISION,
    price_24h_later DOUBLE PRECISION,
    price_change_pct DOUBLE PRECISION,
    news_headlines JSONB DEFAULT '[]',
    has_catalyst BOOLEAN DEFAULT FALSE,
    catalyst_type VARCHAR(20),
    scraped_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS labeled_claims (
    tweet_id BIGINT PRIMARY KEY REFERENCES raw_claims(tweet_id),
    label VARCHAR(20) NOT NULL,
    claimed_direction VARCHAR(10) NOT NULL,
    actual_direction VARCHAR(10) NOT NULL,
    exaggeration_score DOUBLE PRECISION NOT NULL,
    news_summary TEXT DEFAULT '',
    labeled_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_labeled_claims_label ON labeled_claims(label);
CREATE INDEX IF NOT EXISTS idx_labeled_claims_labeled_at ON labeled_claims(labeled_at DESC);
CREATE INDEX IF NOT EXISTS idx_raw_claims_ticker ON raw_claims(ticker);
CREATE INDEX IF NOT EXISTS idx_raw_claims_created_at ON raw_claims(created_at DESC);
"""


class SentinelDB:
    """PostgreSQL database interface for Sentinel."""

    def __init__(self, database_url: str):
        self._url = database_url
        self._conn: psycopg.Connection | None = None

    def connect(self) -> None:
        """Open database connection."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self._url)
            logger.info("Database connected")

    def close(self) -> None:
        """Close database connection."""
        if self._conn and not self._conn.closed:
            self._conn.close()
            logger.info("Database connection closed")

    def _get_conn(self) -> psycopg.Connection:
        """Get an active connection, reconnecting if needed."""
        if self._conn is None or self._conn.closed:
            self.connect()
        return self._conn

    def init_schema(self) -> None:
        """Create tables and indexes if they don't exist."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(_SCHEMA_SQL)
        conn.commit()
        logger.info("Database schema initialized")

    def ping(self) -> bool:
        """Check database connectivity."""
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            return True
        except Exception:
            return False

    def claim_exists(self, tweet_id: int) -> bool:
        """Check if a tweet has already been stored."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM raw_claims WHERE tweet_id = %s", (tweet_id,))
            return cur.fetchone() is not None

    def insert_raw_claim(self, claim: RawClaim) -> None:
        """Insert a raw claim, skipping duplicates."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO raw_claims
                    (tweet_id, text, username, created_at, likes, retweets,
                     ticker, company_name, price_at_tweet, price_24h_later,
                     price_change_pct, news_headlines, has_catalyst, catalyst_type)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (tweet_id) DO NOTHING
                """,
                (
                    claim.tweet_id, claim.text, claim.username,
                    claim.created_at, claim.likes, claim.retweets,
                    claim.ticker, claim.company_name,
                    claim.price_at_tweet, claim.price_24h_later,
                    claim.price_change_pct,
                    json.dumps(claim.news_headlines),
                    claim.has_catalyst, claim.catalyst_type,
                ),
            )
        conn.commit()

    def insert_labeled_claim(self, labeled: LabeledClaim) -> None:
        """Insert a labeled claim (updates raw_claim fields too)."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            # Upsert the raw claim data
            cur.execute(
                """
                INSERT INTO raw_claims
                    (tweet_id, text, username, created_at, likes, retweets,
                     ticker, company_name, price_at_tweet, price_24h_later,
                     price_change_pct, news_headlines, has_catalyst, catalyst_type)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (tweet_id) DO UPDATE SET
                    price_at_tweet = EXCLUDED.price_at_tweet,
                    price_24h_later = EXCLUDED.price_24h_later,
                    price_change_pct = EXCLUDED.price_change_pct,
                    news_headlines = EXCLUDED.news_headlines,
                    has_catalyst = EXCLUDED.has_catalyst,
                    catalyst_type = EXCLUDED.catalyst_type
                """,
                (
                    labeled.tweet_id, labeled.text, labeled.username,
                    labeled.created_at, labeled.likes, labeled.retweets,
                    labeled.ticker, labeled.company_name,
                    labeled.price_at_tweet, labeled.price_24h_later,
                    labeled.price_change_pct,
                    json.dumps(labeled.news_headlines),
                    labeled.has_catalyst, labeled.catalyst_type,
                ),
            )
            # Upsert the label
            cur.execute(
                """
                INSERT INTO labeled_claims
                    (tweet_id, label, claimed_direction, actual_direction,
                     exaggeration_score, news_summary)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (tweet_id) DO UPDATE SET
                    label = EXCLUDED.label,
                    claimed_direction = EXCLUDED.claimed_direction,
                    actual_direction = EXCLUDED.actual_direction,
                    exaggeration_score = EXCLUDED.exaggeration_score,
                    news_summary = EXCLUDED.news_summary,
                    labeled_at = NOW()
                """,
                (
                    labeled.tweet_id, labeled.label,
                    labeled.claimed_direction, labeled.actual_direction,
                    labeled.exaggeration_score, labeled.news_summary,
                ),
            )
        conn.commit()

    def get_feed(
        self,
        limit: int = 50,
        offset: int = 0,
        label: str | None = None,
    ) -> list[dict]:
        """Get paginated labeled claims, newest first.

        Args:
            limit: Max results.
            offset: Pagination offset.
            label: Optional filter ("exaggerated", "accurate", "understated").

        Returns:
            List of claim dicts ready for JSON serialization.
        """
        conn = self._get_conn()
        query = """
            SELECT r.tweet_id, r.text, r.username, r.created_at,
                   r.likes, r.retweets, r.ticker, r.company_name,
                   r.price_at_tweet, r.price_24h_later, r.price_change_pct,
                   r.news_headlines, r.has_catalyst, r.catalyst_type,
                   l.label, l.claimed_direction, l.actual_direction,
                   l.exaggeration_score, l.news_summary, l.labeled_at
            FROM labeled_claims l
            JOIN raw_claims r ON r.tweet_id = l.tweet_id
        """
        params: list = []
        if label:
            query += " WHERE l.label = %s"
            params.append(label)
        query += " ORDER BY l.labeled_at DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])

        with conn.cursor() as cur:
            cur.execute(query, params)
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()

        results = []
        for row in rows:
            d = dict(zip(columns, row))
            # Parse JSON fields
            if isinstance(d.get("news_headlines"), str):
                d["news_headlines"] = json.loads(d["news_headlines"])
            # Serialize datetimes
            for key in ("created_at", "labeled_at"):
                if isinstance(d.get(key), datetime):
                    d[key] = d[key].isoformat()
            results.append(d)

        return results

    def get_stats(self) -> dict:
        """Get aggregate statistics for the /api/stats endpoint."""
        conn = self._get_conn()
        stats = {}

        with conn.cursor() as cur:
            # Total claims
            cur.execute("SELECT COUNT(*) FROM labeled_claims")
            stats["total_claims"] = cur.fetchone()[0]

            # Label distribution
            cur.execute(
                "SELECT label, COUNT(*) FROM labeled_claims GROUP BY label"
            )
            stats["label_distribution"] = dict(cur.fetchall())

            # Catalyst type distribution
            cur.execute(
                """SELECT r.catalyst_type, COUNT(*)
                   FROM raw_claims r
                   JOIN labeled_claims l ON r.tweet_id = l.tweet_id
                   WHERE r.catalyst_type IS NOT NULL
                   GROUP BY r.catalyst_type"""
            )
            stats["catalyst_type_distribution"] = dict(cur.fetchall())

            # Top tickers by claim count
            cur.execute(
                """SELECT r.ticker, COUNT(*) as cnt
                   FROM raw_claims r
                   JOIN labeled_claims l ON r.tweet_id = l.tweet_id
                   GROUP BY r.ticker ORDER BY cnt DESC LIMIT 10"""
            )
            stats["top_tickers"] = [
                {"ticker": row[0], "count": row[1]} for row in cur.fetchall()
            ]

            # Most exaggerated users
            cur.execute(
                """SELECT r.username, COUNT(*) as cnt
                   FROM raw_claims r
                   JOIN labeled_claims l ON r.tweet_id = l.tweet_id
                   WHERE l.label = 'exaggerated'
                   GROUP BY r.username ORDER BY cnt DESC LIMIT 10"""
            )
            stats["most_exaggerated_users"] = [
                {"username": row[0], "count": row[1]} for row in cur.fetchall()
            ]

            # Accuracy by ticker
            cur.execute(
                """SELECT r.ticker,
                          COUNT(*) as total,
                          SUM(CASE WHEN l.label = 'accurate' THEN 1 ELSE 0 END) as accurate,
                          SUM(CASE WHEN l.label = 'exaggerated' THEN 1 ELSE 0 END) as exaggerated,
                          SUM(CASE WHEN l.label = 'understated' THEN 1 ELSE 0 END) as understated
                   FROM raw_claims r
                   JOIN labeled_claims l ON r.tweet_id = l.tweet_id
                   GROUP BY r.ticker ORDER BY total DESC"""
            )
            stats["accuracy_by_ticker"] = [
                {
                    "ticker": row[0], "total": row[1],
                    "accurate": row[2], "exaggerated": row[3], "understated": row[4],
                }
                for row in cur.fetchall()
            ]

        return stats

    def get_latest_claim_id(self) -> int | None:
        """Get the tweet_id of the most recently labeled claim (for SSE polling)."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT tweet_id FROM labeled_claims ORDER BY labeled_at DESC LIMIT 1"
            )
            row = cur.fetchone()
            return row[0] if row else None

    def get_claims_since(self, tweet_id: int) -> list[dict]:
        """Get labeled claims newer than the given tweet_id (for SSE)."""
        conn = self._get_conn()
        query = """
            SELECT r.tweet_id, r.text, r.username, r.created_at,
                   r.likes, r.retweets, r.ticker, r.company_name,
                   r.price_at_tweet, r.price_24h_later, r.price_change_pct,
                   r.news_headlines, r.has_catalyst, r.catalyst_type,
                   l.label, l.claimed_direction, l.actual_direction,
                   l.exaggeration_score, l.news_summary, l.labeled_at
            FROM labeled_claims l
            JOIN raw_claims r ON r.tweet_id = l.tweet_id
            WHERE l.labeled_at > (
                SELECT labeled_at FROM labeled_claims WHERE tweet_id = %s
            )
            ORDER BY l.labeled_at ASC
        """
        with conn.cursor() as cur:
            cur.execute(query, (tweet_id,))
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()

        results = []
        for row in rows:
            d = dict(zip(columns, row))
            if isinstance(d.get("news_headlines"), str):
                d["news_headlines"] = json.loads(d["news_headlines"])
            for key in ("created_at", "labeled_at"):
                if isinstance(d.get(key), datetime):
                    d[key] = d[key].isoformat()
            results.append(d)
        return results
