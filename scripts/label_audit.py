"""
Label audit: find contradictions between LLM account classification
and rule-based labeling scores.

Surfaces accounts where the bot classifier and the labeler disagree:
- Garbage/bot accounts with high accuracy (labels say they're right, LLM says they're junk)
- Human accounts with high grifter scores (LLM says they're real, labels say they're wrong)
- Human accounts with suspiciously perfect accuracy
"""

import sys
import os
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.data.db import SentinelDB


def main():
    config.setup_logging()
    db = SentinelDB(config.database.url)
    db.connect()

    print("=" * 90)
    print("GARBAGE/BOT ACCOUNTS WITH HIGH ACCURACY")
    print("LLM says junk, but the labeler scored them as mostly accurate")
    print("=" * 90)

    cols, rows = db.execute_query("""
        SELECT a.username, a.account_type, a.classification_reason,
               a.naive_total_claims, a.naive_accurate_count, a.naive_exaggerated_count,
               a.naive_grifter_score
        FROM accounts a
        WHERE a.account_type IN ('bot', 'garbage')
          AND a.naive_total_claims >= 3
          AND a.naive_accurate_count > 0
        ORDER BY a.naive_accurate_count DESC
        LIMIT 20
    """)

    for row in rows:
        username, acct_type, reason, total, accurate, exaggerated, grifter = row
        acc_rate = accurate / total * 100 if total > 0 else 0
        reason_short = (reason or "")[:120]
        print(f"\n@{username} [{acct_type.upper()}] — {accurate}/{total} accurate ({acc_rate:.0f}%)")
        print(f"  Reason: {reason_short}")

    # Show sample tweets from top offenders
    if rows:
        print("\n--- Sample tweets from top offenders ---")
        for row in rows[:5]:
            username = row[0]
            _, tweet_rows = db.execute_query("""
                SELECT r.text, l.label, l.claimed_direction, r.price_change_pct
                FROM raw_claims r
                JOIN naive_labeled_claims l ON r.tweet_id = l.tweet_id
                WHERE r.username = %s
                ORDER BY r.created_at DESC LIMIT 3
            """, [username])
            print(f"\n@{username}:")
            for tr in tweet_rows:
                text = (tr[0] or "")[:150]
                label = tr[1]
                direction = tr[2]
                pct = f"{tr[3]:+.2f}%" if tr[3] is not None else "N/A"
                print(f"  [{label}] (claimed: {direction}, price: {pct})")
                print(f"    {text}")

    print("\n" + "=" * 90)
    print("HUMAN ACCOUNTS WITH HIGH GRIFTER SCORES (>= 0.6)")
    print("LLM says real human, but the labeler says they're mostly wrong")
    print("=" * 90)

    cols, rows = db.execute_query("""
        SELECT a.username, a.account_type, a.classification_reason,
               a.naive_total_claims, a.naive_accurate_count, a.naive_exaggerated_count,
               a.naive_grifter_score
        FROM accounts a
        WHERE a.account_type = 'human'
          AND a.naive_grifter_score >= 0.6
          AND a.naive_total_claims >= 5
        ORDER BY a.naive_grifter_score DESC
        LIMIT 20
    """)

    for row in rows:
        username, acct_type, reason, total, accurate, exaggerated, grifter = row
        reason_short = (reason or "")[:120]
        print(f"\n@{username} — grifter: {grifter:.2f} ({exaggerated}/{total} exaggerated)")
        print(f"  Reason: {reason_short}")

    if rows:
        print("\n--- Sample tweets from top grifters ---")
        for row in rows[:5]:
            username = row[0]
            _, tweet_rows = db.execute_query("""
                SELECT r.text, l.label, l.claimed_direction, r.price_change_pct,
                       r.has_catalyst, r.catalyst_type
                FROM raw_claims r
                JOIN naive_labeled_claims l ON r.tweet_id = l.tweet_id
                WHERE r.username = %s AND l.label = 'exaggerated'
                ORDER BY r.created_at DESC LIMIT 3
            """, [username])
            print(f"\n@{username}:")
            for tr in tweet_rows:
                text = (tr[0] or "")[:150]
                direction = tr[1]
                claimed = tr[2]
                pct = f"{tr[3]:+.2f}%" if tr[3] is not None else "N/A"
                catalyst = tr[5] or "none"
                print(f"  [exaggerated] (claimed: {claimed}, price: {pct}, catalyst: {catalyst})")
                print(f"    {text}")

    print("\n" + "=" * 90)
    print("HUMAN ACCOUNTS WITH PERFECT ACCURACY (suspicious)")
    print("Every single claim labeled accurate — are these really making claims?")
    print("=" * 90)

    cols, rows = db.execute_query("""
        SELECT a.username, a.naive_total_claims, a.naive_accurate_count,
               a.classification_reason
        FROM accounts a
        WHERE a.account_type = 'human'
          AND a.naive_total_claims >= 5
          AND a.naive_exaggerated_count = 0
          AND a.naive_understated_count = 0
        ORDER BY a.naive_total_claims DESC
        LIMIT 15
    """)

    for row in rows:
        username, total, accurate, reason = row
        reason_short = (reason or "")[:120]
        print(f"\n@{username} — {total} claims, ALL accurate")
        print(f"  Reason: {reason_short}")

    if rows:
        print("\n--- Sample tweets from suspiciously perfect accounts ---")
        for row in rows[:5]:
            username = row[0]
            _, tweet_rows = db.execute_query("""
                SELECT r.text, l.claimed_direction, r.price_change_pct
                FROM raw_claims r
                JOIN naive_labeled_claims l ON r.tweet_id = l.tweet_id
                WHERE r.username = %s
                ORDER BY r.created_at DESC LIMIT 3
            """, [username])
            print(f"\n@{username}:")
            for tr in tweet_rows:
                text = (tr[0] or "")[:150]
                direction = tr[1]
                pct = f"{tr[2]:+.2f}%" if tr[2] is not None else "N/A"
                print(f"  (claimed: {direction}, price: {pct})")
                print(f"    {text}")

    print("\n" + "=" * 90)
    print("SUMMARY STATS")
    print("=" * 90)

    cols, rows = db.execute_query("""
        SELECT
            a.account_type,
            COUNT(*) as accounts,
            SUM(a.naive_total_claims) as total_claims,
            SUM(a.naive_accurate_count) as accurate,
            SUM(a.naive_exaggerated_count) as exaggerated,
            SUM(a.naive_understated_count) as understated,
            AVG(a.naive_grifter_score) FILTER (WHERE a.naive_grifter_score IS NOT NULL) as avg_grifter
        FROM accounts a
        GROUP BY a.account_type
        ORDER BY accounts DESC
    """)

    print(f"\n{'Type':<10} {'Accounts':>10} {'Claims':>10} {'Accurate':>10} {'Exaggerated':>12} {'Avg Grifter':>12}")
    print("-" * 66)
    for row in rows:
        acct_type, accounts, claims, accurate, exaggerated, understated, avg_grifter = row
        grifter_str = f"{avg_grifter:.3f}" if avg_grifter is not None else "—"
        print(f"{acct_type:<10} {accounts:>10} {claims or 0:>10} {accurate or 0:>10} {exaggerated or 0:>12} {grifter_str:>12}")

    db.close()


if __name__ == "__main__":
    main()
