# This file was developed with the assistance of Claude Code and Opus 4.6.

"""Account query methods for SentinelDB.

Extracted as a mixin to keep the main db.py under the pylint
module-line limit. All methods here are mixed into SentinelDB.
"""

from datetime import datetime

from .models import Account


class AccountMixin:
    """Account-related database operations."""

    def upsert_account(self, account: Account) -> None:
        """Insert or update an account record.

        Only updates classification fields (account_type, classification_reason,
        classified_at) and timestamps. Score fields are updated by
        update_account_scores() which targets specific label columns.
        """
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO accounts
                    (username, account_type, classification_reason,
                     first_seen, last_seen, classified_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (username) DO UPDATE SET
                    account_type = EXCLUDED.account_type,
                    classification_reason = EXCLUDED.classification_reason,
                    first_seen = EXCLUDED.first_seen,
                    last_seen = EXCLUDED.last_seen,
                    classified_at = EXCLUDED.classified_at
                """,
                (
                    account.username, account.account_type, account.classification_reason,
                    account.first_seen,
                    account.last_seen, account.classified_at,
                ),
            )
        conn.commit()

    def get_account(self, username: str) -> Account | None:
        """Get a single account by username."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM accounts WHERE username = %s", (username,))
            row = cur.fetchone()
            if row is None:
                return None
            columns = [desc[0] for desc in cur.description]
            d = dict(zip(columns, row))
        return self._row_to_account(d)

    def get_accounts(
        self,
        sort_by: str = "naive_grifter_score",
        order: str = "desc",
        min_claims: int = 0,
        account_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
        labels: str = "naive",
    ) -> list[Account]:
        """Get paginated, filterable, sortable account list."""
        default_sort = f"{labels}_grifter_score"
        allowed_sort = {
            "naive_grifter_score", "improved_grifter_score",
            "naive_total_claims", "improved_total_claims",
            "username", "last_seen",
        }
        if sort_by not in allowed_sort:
            sort_by = default_sort
        order_dir = "ASC" if order.lower() == "asc" else "DESC"

        query = f"SELECT * FROM accounts WHERE {labels}_total_claims >= %s"
        params: list = [min_claims]

        if account_type is not None:
            query += " AND account_type = %s"
            params.append(account_type)

        query += f" ORDER BY {sort_by} {order_dir} NULLS LAST"
        query += " LIMIT %s OFFSET %s"
        params.extend([limit, offset])

        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(query, params)
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()

        return [self._row_to_account(dict(zip(columns, row))) for row in rows]

    @staticmethod
    def _row_to_account(d: dict) -> Account:
        """Build an Account from a row dict."""
        return Account(
            username=d["username"],
            account_type=d.get("account_type", "human"),
            classification_reason=d.get("classification_reason"),
            naive_total_claims=d.get("naive_total_claims", 0),
            naive_exaggerated_count=d.get("naive_exaggerated_count", 0),
            naive_accurate_count=d.get("naive_accurate_count", 0),
            naive_understated_count=d.get("naive_understated_count", 0),
            naive_grifter_score=d.get("naive_grifter_score"),
            improved_total_claims=d.get("improved_total_claims", 0),
            improved_exaggerated_count=d.get("improved_exaggerated_count", 0),
            improved_accurate_count=d.get("improved_accurate_count", 0),
            improved_understated_count=d.get("improved_understated_count", 0),
            improved_grifter_score=d.get("improved_grifter_score"),
            first_seen=d.get("first_seen"),
            last_seen=d.get("last_seen"),
            classified_at=d.get("classified_at"),
        )

    def update_account_scores(
        self,
        username: str,
        label_table: str = "naive_labeled_claims",
    ) -> None:
        """Recalculate account stats from labeled claim data.

        Args:
            username: Twitter handle.
            label_table: Which label table to compute scores from.
        """
        if label_table not in ("naive_labeled_claims", "improved_labeled_claims"):
            raise ValueError(f"Invalid label_table: {label_table}")

        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN l.label = 'exaggerated' THEN 1 ELSE 0 END) as exaggerated,
                    SUM(CASE WHEN l.label = 'accurate' THEN 1 ELSE 0 END) as accurate,
                    SUM(CASE WHEN l.label = 'understated' THEN 1 ELSE 0 END) as understated,
                    MIN(r.created_at) as first_seen,
                    MAX(r.created_at) as last_seen
                FROM raw_claims r
                JOIN {label_table} l ON r.tweet_id = l.tweet_id
                WHERE r.username = %s
                """,
                (username,),
            )
            row = cur.fetchone()

        total = row[0] or 0
        exaggerated = row[1] or 0
        accurate = row[2] or 0
        understated = row[3] or 0
        first_seen = row[4]
        last_seen = row[5]
        grifter_score = exaggerated / total if total >= 5 else None

        # Determine column prefix from table name
        prefix = "naive" if label_table == "naive_labeled_claims" else "improved"

        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO accounts
                    (username, {prefix}_total_claims, {prefix}_exaggerated_count,
                     {prefix}_accurate_count, {prefix}_understated_count,
                     {prefix}_grifter_score, first_seen, last_seen)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (username) DO UPDATE SET
                    {prefix}_total_claims = EXCLUDED.{prefix}_total_claims,
                    {prefix}_exaggerated_count = EXCLUDED.{prefix}_exaggerated_count,
                    {prefix}_accurate_count = EXCLUDED.{prefix}_accurate_count,
                    {prefix}_understated_count = EXCLUDED.{prefix}_understated_count,
                    {prefix}_grifter_score = EXCLUDED.{prefix}_grifter_score,
                    first_seen = EXCLUDED.first_seen,
                    last_seen = EXCLUDED.last_seen
                """,
                (
                    username, total, exaggerated, accurate,
                    understated, grifter_score, first_seen, last_seen,
                ),
            )
        conn.commit()

    def get_account_claims(
        self,
        username: str,
        limit: int = 50,
        offset: int = 0,
        labels: str = "naive",
    ) -> list[dict]:
        """Get paginated labeled claims for a specific account.

        Args:
            username: Twitter handle.
            limit: Max results.
            offset: Pagination offset.
            labels: Which label set to query ("naive" or "improved").
        """
        if labels not in ("naive", "improved"):
            raise ValueError(f"Invalid labels: {labels!r}")
        label_table = f"{labels}_labeled_claims"
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
            WHERE r.username = %s
            ORDER BY l.labeled_at DESC
            LIMIT %s OFFSET %s
        """
        with conn.cursor() as cur:
            cur.execute(query, (username, limit, offset))
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()

        import json
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

    def get_leaderboard(
        self,
        category: str = "grifters",
        labels: str = "naive",
        limit: int = 20,
    ) -> list[Account]:
        """Get account leaderboard by category.

        Args:
            category: "grifters" (worst first) or "signal" (best first).
            labels: "naive" or "improved" — which grifter scores to rank by.
        """
        order_dir = "ASC" if category == "signal" else "DESC"
        prefix = "naive" if labels == "naive" else "improved"

        conn = self._get_conn()
        query = f"""
            SELECT *
            FROM accounts
            WHERE {prefix}_grifter_score IS NOT NULL
              AND {prefix}_total_claims >= 5
              AND account_type = 'human'
            ORDER BY {prefix}_grifter_score {order_dir}
            LIMIT %s
        """
        with conn.cursor() as cur:
            cur.execute(query, (limit,))
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()

        return [self._row_to_account(dict(zip(columns, row))) for row in rows]

    def get_unclassified_accounts(self, limit: int = 100) -> list[dict]:
        """Get accounts that haven't been bot-classified yet, with sample tweets."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            # Find usernames with raw_claims but no classified accounts row
            cur.execute(
                """
                SELECT DISTINCT r.username
                FROM raw_claims r
                LEFT JOIN accounts a ON r.username = a.username
                WHERE a.username IS NULL OR a.classified_at IS NULL
                LIMIT %s
                """,
                (limit,),
            )
            usernames = [row[0] for row in cur.fetchall()]

        if not usernames:
            return []

        results = []
        with conn.cursor() as cur:
            for username in usernames:
                cur.execute(
                    """
                    SELECT text FROM raw_claims
                    WHERE username = %s
                    ORDER BY created_at DESC
                    LIMIT 3
                    """,
                    (username,),
                )
                sample_tweets = [row[0] for row in cur.fetchall()]
                results.append({
                    "username": username,
                    "sample_tweets": sample_tweets,
                })

        return results

    def get_all_accounts_with_tweets(self, limit: int = 100) -> list[dict]:
        """Get all accounts with sample tweets, regardless of classification status.

        Used by the classify --reclassify command.
        """
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT r.username, array_agg(r.text ORDER BY r.created_at DESC)
                FROM raw_claims r
                GROUP BY r.username
                LIMIT %s
            """, (limit,))
            rows = cur.fetchall()
        return [
            {"username": row[0], "sample_tweets": row[1][:5]}
            for row in rows
        ]
