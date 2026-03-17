"""
PostgreSQL persistence layer for Sentinel.

Stores raw and labeled claims. The Flask API reads from this DB
for the live feed and stats endpoints.
"""

import json
import logging
from datetime import datetime

import psycopg

from .db_accounts import AccountMixin
from .models import RawClaim, LabeledClaim

logger = logging.getLogger("sentinel.db")


def _label_table(labels: str) -> str:
    """Validate the labels parameter and return the corresponding table name."""
    if labels not in ("naive", "improved"):
        raise ValueError(f"Invalid labels: {labels!r} (must be 'naive' or 'improved')")
    return f"{labels}_labeled_claims"

# Exclude stale tweets where the tweet was created more than 90 days
# before it was scraped. These are old results Twitter search returns
# alongside current ones. A tweet from 2019 scraped in 2026 is noise.
FRESHNESS_FILTER = "r.created_at >= r.scraped_at - INTERVAL '90 days'"

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS raw_claims (
    tweet_id BIGINT PRIMARY KEY,
    text TEXT NOT NULL,
    username VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    likes INTEGER DEFAULT 0,
    retweets INTEGER DEFAULT 0,
    replies INTEGER DEFAULT 0,
    views INTEGER,
    hashtags JSONB DEFAULT '[]',
    ticker VARCHAR(10) NOT NULL,
    company_name VARCHAR(255) NOT NULL,
    price_at_tweet DOUBLE PRECISION,
    price_24h_later DOUBLE PRECISION,
    price_change_pct DOUBLE PRECISION,
    news_headlines JSONB DEFAULT '[]',
    has_catalyst BOOLEAN DEFAULT FALSE,
    catalyst_type VARCHAR(20),
    posted_during_market_hours BOOLEAN,
    volume_at_tweet DOUBLE PRECISION,
    scraped_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS naive_labeled_claims (
    tweet_id BIGINT PRIMARY KEY REFERENCES raw_claims(tweet_id),
    label VARCHAR(20) NOT NULL,
    claimed_direction VARCHAR(10) NOT NULL,
    actual_direction VARCHAR(10) NOT NULL,
    exaggeration_score DOUBLE PRECISION NOT NULL,
    news_summary TEXT DEFAULT '',
    labeled_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_naive_labeled_claims_label ON naive_labeled_claims(label);
CREATE INDEX IF NOT EXISTS idx_naive_labeled_claims_labeled_at ON naive_labeled_claims(labeled_at DESC);

CREATE TABLE IF NOT EXISTS improved_labeled_claims (
    tweet_id BIGINT PRIMARY KEY REFERENCES raw_claims(tweet_id),
    label VARCHAR(20) NOT NULL,
    claimed_direction VARCHAR(10) NOT NULL,
    actual_direction VARCHAR(10) NOT NULL,
    exaggeration_score DOUBLE PRECISION NOT NULL,
    news_summary TEXT DEFAULT '',
    labeled_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_improved_labeled_claims_label ON improved_labeled_claims(label);
CREATE INDEX IF NOT EXISTS idx_improved_labeled_claims_labeled_at ON improved_labeled_claims(labeled_at DESC);
CREATE INDEX IF NOT EXISTS idx_raw_claims_ticker ON raw_claims(ticker);
CREATE INDEX IF NOT EXISTS idx_raw_claims_created_at ON raw_claims(created_at DESC);

CREATE TABLE IF NOT EXISTS accounts (
    username VARCHAR(255) PRIMARY KEY,
    account_type VARCHAR(10) DEFAULT 'human',
    classification_reason TEXT,
    -- Naive labeler scores
    naive_total_claims INTEGER DEFAULT 0,
    naive_exaggerated_count INTEGER DEFAULT 0,
    naive_accurate_count INTEGER DEFAULT 0,
    naive_understated_count INTEGER DEFAULT 0,
    naive_grifter_score DOUBLE PRECISION,
    -- Improved labeler scores
    improved_total_claims INTEGER DEFAULT 0,
    improved_exaggerated_count INTEGER DEFAULT 0,
    improved_accurate_count INTEGER DEFAULT 0,
    improved_understated_count INTEGER DEFAULT 0,
    improved_grifter_score DOUBLE PRECISION,
    first_seen TIMESTAMPTZ,
    last_seen TIMESTAMPTZ,
    classified_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_accounts_naive_grifter ON accounts(naive_grifter_score);
CREATE INDEX IF NOT EXISTS idx_accounts_improved_grifter ON accounts(improved_grifter_score);
CREATE INDEX IF NOT EXISTS idx_accounts_account_type ON accounts(account_type);
CREATE INDEX IF NOT EXISTS idx_accounts_naive_total ON accounts(naive_total_claims DESC);
"""


class SentinelDB(AccountMixin):
    """PostgreSQL database interface for Sentinel."""

    def __init__(self, database_url: str):
        self._url = database_url
        self._conn: psycopg.Connection | None = None

    def connect(self) -> None:
        """Open database connection."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self._url, autocommit=True)
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

    def execute_query(self, query: str, params: list | None = None) -> tuple[list[str], list[tuple]]:
        """Execute a read query and return (column_names, rows).

        For use by modules that need custom queries without direct connection access.
        """
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(query, params or [])
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
        return columns, rows

    def get_existing_tweet_ids(self, tweet_ids: list[int]) -> set[int]:
        """Return the subset of tweet_ids that already exist in raw_claims."""
        if not tweet_ids:
            return set()
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT tweet_id FROM raw_claims WHERE tweet_id = ANY(%s)",
                (tweet_ids,),
            )
            return {row[0] for row in cur.fetchall()}

    def get_raw_claims(
        self,
        tickers: list[str] | None = None,
        since: "datetime | None" = None,
        until: "datetime | None" = None,
        unlabeled_only: bool = False,
        labels: str = "naive",
    ) -> list[RawClaim]:
        """Fetch raw claims from the database for re-enrichment.

        Args:
            tickers: Only return claims for these tickers. None = all.
            since: Only return claims created after this time.
            until: Only return claims created before this time.
            unlabeled_only: If True, only return claims without a row in the
                table corresponding to *labels* (``"naive"`` or ``"improved"``).
            labels: Which label set to check for unlabeled filtering.
        """
        conn = self._get_conn()
        query = "SELECT * FROM raw_claims"
        conditions: list[str] = []
        params: list = []

        if unlabeled_only:
            table = _label_table(labels)
            conditions.append(
                f"tweet_id NOT IN (SELECT tweet_id FROM {table})"
            )
        if tickers:
            conditions.append("ticker = ANY(%s)")
            params.append(tickers)
        if since:
            conditions.append("created_at >= %s")
            params.append(since)
        if until:
            conditions.append("created_at <= %s")
            params.append(until)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC"

        with conn.cursor() as cur:
            cur.execute(query, params)
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()

        claims = []
        for row in rows:
            d = dict(zip(columns, row))
            headlines = d.get("news_headlines", [])
            if isinstance(headlines, str):
                headlines = json.loads(headlines)
            hashtags_raw = d.get("hashtags", [])
            if isinstance(hashtags_raw, str):
                hashtags_raw = json.loads(hashtags_raw)
            claims.append(RawClaim(
                tweet_id=d["tweet_id"],
                text=d["text"],
                username=d["username"],
                created_at=d["created_at"],
                likes=d.get("likes", 0),
                retweets=d.get("retweets", 0),
                replies=d.get("replies", 0),
                views=d.get("views"),
                hashtags=hashtags_raw,
                ticker=d["ticker"],
                company_name=d["company_name"],
                price_at_tweet=d.get("price_at_tweet"),
                price_24h_later=d.get("price_24h_later"),
                price_change_pct=d.get("price_change_pct"),
                news_headlines=headlines,
                has_catalyst=d.get("has_catalyst", False),
                catalyst_type=d.get("catalyst_type"),
                posted_during_market_hours=d.get("posted_during_market_hours"),
                volume_at_tweet=d.get("volume_at_tweet"),
            ))
        return claims

    def insert_raw_claim(self, claim: RawClaim) -> None:
        """Insert a raw claim, skipping duplicates."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO raw_claims
                    (tweet_id, text, username, created_at, likes, retweets,
                     replies, views, hashtags,
                     ticker, company_name, price_at_tweet, price_24h_later,
                     price_change_pct, news_headlines, has_catalyst, catalyst_type,
                     posted_during_market_hours, volume_at_tweet)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (tweet_id) DO NOTHING
                """,
                (
                    claim.tweet_id, claim.text, claim.username,
                    claim.created_at, claim.likes, claim.retweets,
                    claim.replies, claim.views,
                    json.dumps(claim.hashtags),
                    claim.ticker, claim.company_name,
                    claim.price_at_tweet, claim.price_24h_later,
                    claim.price_change_pct,
                    json.dumps(claim.news_headlines),
                    claim.has_catalyst, claim.catalyst_type,
                    claim.posted_during_market_hours, claim.volume_at_tweet,
                ),
            )
        conn.commit()

    def insert_labeled_claim(
        self,
        labeled: LabeledClaim,
        label_table: str = "naive_labeled_claims",
    ) -> None:
        """Insert a labeled claim (updates raw_claim fields too).

        Args:
            labeled: The labeled claim to insert.
            label_table: Which label table to write to
                ("naive_labeled_claims" or "improved_labeled_claims").
        """
        if label_table not in ("naive_labeled_claims", "improved_labeled_claims"):
            raise ValueError(f"Invalid label_table: {label_table}")

        conn = self._get_conn()
        with conn.transaction(), conn.cursor() as cur:
            # Upsert the raw claim data
            cur.execute(
                """
                INSERT INTO raw_claims
                    (tweet_id, text, username, created_at, likes, retweets,
                     replies, views, hashtags,
                     ticker, company_name, price_at_tweet, price_24h_later,
                     price_change_pct, news_headlines, has_catalyst, catalyst_type,
                     posted_during_market_hours, volume_at_tweet)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (tweet_id) DO UPDATE SET
                    replies = EXCLUDED.replies,
                    views = EXCLUDED.views,
                    hashtags = EXCLUDED.hashtags,
                    price_at_tweet = EXCLUDED.price_at_tweet,
                    price_24h_later = EXCLUDED.price_24h_later,
                    price_change_pct = EXCLUDED.price_change_pct,
                    news_headlines = EXCLUDED.news_headlines,
                    has_catalyst = EXCLUDED.has_catalyst,
                    catalyst_type = EXCLUDED.catalyst_type,
                    posted_during_market_hours = EXCLUDED.posted_during_market_hours,
                    volume_at_tweet = EXCLUDED.volume_at_tweet
                """,
                (
                    labeled.tweet_id, labeled.text, labeled.username,
                    labeled.created_at, labeled.likes, labeled.retweets,
                    labeled.replies, labeled.views,
                    json.dumps(labeled.hashtags),
                    labeled.ticker, labeled.company_name,
                    labeled.price_at_tweet, labeled.price_24h_later,
                    labeled.price_change_pct,
                    json.dumps(labeled.news_headlines),
                    labeled.has_catalyst, labeled.catalyst_type,
                    labeled.posted_during_market_hours, labeled.volume_at_tweet,
                ),
            )
            # Upsert the label into the specified table
            cur.execute(
                f"""
                INSERT INTO {label_table}
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
        self.update_account_scores(labeled.username, label_table=label_table)

    def get_feed(
        self,
        limit: int = 50,
        offset: int = 0,
        label: str | None = None,
        labels: str = "naive",
    ) -> list[dict]:
        """Get paginated labeled claims, newest first.

        Args:
            limit: Max results.
            offset: Pagination offset.
            label: Optional filter ("exaggerated", "accurate", "understated").
            labels: Which label set to query ("naive" or "improved").

        Returns:
            List of claim dicts ready for JSON serialization.
        """
        label_table = _label_table(labels)
        conn = self._get_conn()
        query = f"""
            SELECT r.tweet_id, r.text, r.username, r.created_at,
                   r.likes, r.retweets, r.replies, r.views, r.hashtags,
                   r.ticker, r.company_name,
                   r.price_at_tweet, r.price_24h_later, r.price_change_pct,
                   r.news_headlines, r.has_catalyst, r.catalyst_type,
                   r.posted_during_market_hours, r.volume_at_tweet,
                   l.label, l.claimed_direction, l.actual_direction,
                   l.exaggeration_score, l.news_summary, l.labeled_at
            FROM {label_table} l
            JOIN raw_claims r ON r.tweet_id = l.tweet_id
        """
        params: list = []
        conditions = [FRESHNESS_FILTER]
        if label:
            conditions.append("l.label = %s")
            params.append(label)
        query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY r.created_at DESC LIMIT %s OFFSET %s"
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
            if isinstance(d.get("hashtags"), str):
                d["hashtags"] = json.loads(d["hashtags"])
            # Serialize datetimes
            for key in ("created_at", "labeled_at"):
                if isinstance(d.get(key), datetime):
                    d[key] = d[key].isoformat()
            results.append(d)

        return results

    def get_stats(self, labels: str = "naive") -> dict:
        """Get aggregate statistics for the /api/stats endpoint.

        Args:
            labels: Which label set to query ("naive" or "improved").
        """
        label_table = _label_table(labels)
        conn = self._get_conn()
        stats = {}

        with conn.cursor() as cur:
            # Total claims
            cur.execute(f"""
                SELECT COUNT(*) FROM raw_claims r
                JOIN {label_table} l ON r.tweet_id = l.tweet_id
                WHERE {FRESHNESS_FILTER}
            """)
            stats["total_claims"] = cur.fetchone()[0]

            # Label distribution
            cur.execute(f"""
                SELECT l.label, COUNT(*) FROM {label_table} l
                JOIN raw_claims r ON r.tweet_id = l.tweet_id
                WHERE {FRESHNESS_FILTER}
                GROUP BY l.label
            """)
            stats["label_distribution"] = dict(cur.fetchall())

            # Catalyst type distribution
            cur.execute(f"""
                SELECT r.catalyst_type, COUNT(*)
                FROM raw_claims r
                JOIN {label_table} l ON r.tweet_id = l.tweet_id
                WHERE r.catalyst_type IS NOT NULL AND {FRESHNESS_FILTER}
                GROUP BY r.catalyst_type
            """)
            stats["catalyst_type_distribution"] = dict(cur.fetchall())

            # Top tickers by claim count
            cur.execute(f"""
                SELECT r.ticker, COUNT(*) as cnt
                FROM raw_claims r
                JOIN {label_table} l ON r.tweet_id = l.tweet_id
                WHERE {FRESHNESS_FILTER}
                GROUP BY r.ticker ORDER BY cnt DESC LIMIT 10
            """)
            stats["top_tickers"] = [
                {"ticker": row[0], "count": row[1]} for row in cur.fetchall()
            ]

            # Most exaggerated users
            cur.execute(f"""
                SELECT r.username, COUNT(*) as cnt
                FROM raw_claims r
                JOIN {label_table} l ON r.tweet_id = l.tweet_id
                WHERE l.label = 'exaggerated' AND {FRESHNESS_FILTER}
                GROUP BY r.username ORDER BY cnt DESC LIMIT 10
            """)
            stats["most_exaggerated_users"] = [
                {"username": row[0], "count": row[1]} for row in cur.fetchall()
            ]

            # Accuracy by ticker
            cur.execute(f"""
                SELECT r.ticker,
                       COUNT(*) as total,
                       SUM(CASE WHEN l.label = 'accurate' THEN 1 ELSE 0 END) as accurate,
                       SUM(CASE WHEN l.label = 'exaggerated' THEN 1 ELSE 0 END) as exaggerated,
                       SUM(CASE WHEN l.label = 'understated' THEN 1 ELSE 0 END) as understated
                FROM raw_claims r
                JOIN {label_table} l ON r.tweet_id = l.tweet_id
                WHERE {FRESHNESS_FILTER}
                GROUP BY r.ticker ORDER BY total DESC
            """)
            stats["accuracy_by_ticker"] = [
                {
                    "ticker": row[0], "total": row[1],
                    "accurate": row[2], "exaggerated": row[3], "understated": row[4],
                }
                for row in cur.fetchall()
            ]

        return stats

    def get_stock_feed(
        self,
        ticker: str,
        limit: int = 50,
        offset: int = 0,
        exclude_bots: bool = True,
        labels: str = "naive",
    ) -> list[dict]:
        """Get paginated labeled claims for a specific ticker.

        Args:
            ticker: Stock ticker to filter by.
            limit: Max results.
            offset: Pagination offset.
            exclude_bots: Whether to exclude bot/garbage accounts.
            labels: Which label set to query ("naive" or "improved").
        """
        label_table = _label_table(labels)
        conn = self._get_conn()
        query = f"""
            SELECT r.tweet_id, r.text, r.username, r.created_at,
                   r.likes, r.retweets, r.replies, r.views, r.hashtags,
                   r.ticker, r.company_name,
                   r.price_at_tweet, r.price_24h_later, r.price_change_pct,
                   r.news_headlines, r.has_catalyst, r.catalyst_type,
                   r.posted_during_market_hours, r.volume_at_tweet,
                   l.label, l.claimed_direction, l.actual_direction,
                   l.exaggeration_score, l.news_summary, l.labeled_at
            FROM {label_table} l
            JOIN raw_claims r ON r.tweet_id = l.tweet_id
        """
        conditions = [FRESHNESS_FILTER, "r.ticker = %s"]
        params: list = [ticker]

        if exclude_bots:
            conditions.append(
                "r.username NOT IN (SELECT username FROM accounts WHERE account_type != 'human')"
            )

        query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY r.created_at DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])

        with conn.cursor() as cur:
            cur.execute(query, params)
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()

        results = []
        for row in rows:
            d = dict(zip(columns, row))
            if isinstance(d.get("news_headlines"), str):
                d["news_headlines"] = json.loads(d["news_headlines"])
            if isinstance(d.get("hashtags"), str):
                d["hashtags"] = json.loads(d["hashtags"])
            for key in ("created_at", "labeled_at"):
                if isinstance(d.get(key), datetime):
                    d[key] = d[key].isoformat()
            results.append(d)

        return results

    def get_stock_stats(self, ticker: str, labels: str = "naive") -> dict:
        """Get aggregate statistics for a specific ticker.

        Args:
            ticker: Stock ticker to get stats for.
            labels: Which label set to query ("naive" or "improved").
        """
        label_table = _label_table(labels)
        conn = self._get_conn()
        stats: dict = {}

        with conn.cursor() as cur:
            # Total claims for this ticker
            cur.execute(f"""
                SELECT COUNT(*) FROM raw_claims r
                JOIN {label_table} l ON r.tweet_id = l.tweet_id
                WHERE r.ticker = %s AND {FRESHNESS_FILTER}
            """, (ticker,))
            stats["total_claims"] = cur.fetchone()[0]

            # Label distribution
            cur.execute(f"""
                SELECT l.label, COUNT(*) FROM {label_table} l
                JOIN raw_claims r ON r.tweet_id = l.tweet_id
                WHERE r.ticker = %s AND {FRESHNESS_FILTER}
                GROUP BY l.label
            """, (ticker,))
            stats["label_distribution"] = dict(cur.fetchall())

            # Average exaggeration score
            cur.execute(f"""
                SELECT AVG(l.exaggeration_score) FROM {label_table} l
                JOIN raw_claims r ON r.tweet_id = l.tweet_id
                WHERE r.ticker = %s AND {FRESHNESS_FILTER}
            """, (ticker,))
            avg = cur.fetchone()[0]
            stats["avg_exaggeration_score"] = float(avg) if avg is not None else None

            # Top claimants for this ticker
            cur.execute(f"""
                SELECT r.username, COUNT(*) as cnt
                FROM raw_claims r
                JOIN {label_table} l ON r.tweet_id = l.tweet_id
                WHERE r.ticker = %s AND {FRESHNESS_FILTER}
                GROUP BY r.username ORDER BY cnt DESC LIMIT 10
            """, (ticker,))
            stats["top_claimants"] = [
                {"username": row[0], "count": row[1]} for row in cur.fetchall()
            ]

        return stats

    def get_latest_claim_id(self, labels: str = "naive") -> int | None:
        """Get the tweet_id of the most recently labeled claim (for SSE polling).

        Args:
            labels: Which label set to query ("naive" or "improved").
        """
        label_table = _label_table(labels)
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT tweet_id FROM {label_table} ORDER BY labeled_at DESC LIMIT 1"
            )
            row = cur.fetchone()
            return row[0] if row else None

    def get_claims_since(self, tweet_id: int, labels: str = "naive") -> list[dict]:
        """Get labeled claims newer than the given tweet_id (for SSE).

        Args:
            tweet_id: The tweet_id to fetch claims after.
            labels: Which label set to query ("naive" or "improved").
        """
        label_table = _label_table(labels)
        conn = self._get_conn()
        query = f"""
            SELECT r.tweet_id, r.text, r.username, r.created_at,
                   r.likes, r.retweets, r.replies, r.views, r.hashtags,
                   r.ticker, r.company_name,
                   r.price_at_tweet, r.price_24h_later, r.price_change_pct,
                   r.news_headlines, r.has_catalyst, r.catalyst_type,
                   r.posted_during_market_hours, r.volume_at_tweet,
                   l.label, l.claimed_direction, l.actual_direction,
                   l.exaggeration_score, l.news_summary, l.labeled_at
            FROM {label_table} l
            JOIN raw_claims r ON r.tweet_id = l.tweet_id
            WHERE l.labeled_at > (
                SELECT labeled_at FROM {label_table} WHERE tweet_id = %s
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
            if isinstance(d.get("hashtags"), str):
                d["hashtags"] = json.loads(d["hashtags"])
            for key in ("created_at", "labeled_at"):
                if isinstance(d.get(key), datetime):
                    d[key] = d[key].isoformat()
            results.append(d)
        return results
