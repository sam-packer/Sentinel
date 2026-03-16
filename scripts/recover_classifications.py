"""
Recover bot classifications from enrich.log after a failed run.

Parses log lines like:
  Classified @username: human (confidence: 0.85) — reason text here

And inserts them into the accounts table.
"""

import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.data.db import SentinelDB
from src.data.models import Account

LOG_PATTERN = re.compile(
    r"Classified @(\S+): (\w+) \(confidence: ([\d.]+)\) — (.+)"
)


def main():
    log_file = Path("data/enrich.log")
    if not log_file.exists():
        print(f"No log file at {log_file}")
        sys.exit(1)

    # Parse classifications from log
    classifications = []
    for line in log_file.read_text(encoding="utf-8").splitlines():
        match = LOG_PATTERN.search(line)
        if match:
            username, account_type, confidence, reason = match.groups()
            classifications.append({
                "username": username,
                "account_type": account_type,
                "confidence": float(confidence),
                "reason": reason.strip(),
            })

    if not classifications:
        print("No classifications found in log.")
        sys.exit(0)

    print(f"Found {len(classifications)} classifications in log.")

    # Check what's already in the DB
    config.setup_logging()
    db = SentinelDB(config.database.url)
    db.connect()

    already_saved = 0
    to_insert = 0

    for c in classifications:
        account = db.get_account(c["username"])
        if account and account.classified_at is not None:
            already_saved += 1
            continue

        account = account or Account(username=c["username"])
        account.account_type = c["account_type"]
        account.classification_reason = f"{c['account_type']}: {c['reason']}"
        account.classified_at = datetime.now(timezone.utc)
        db.upsert_account(account)
        to_insert += 1

    db.close()
    print(f"Already saved: {already_saved}")
    print(f"Recovered: {to_insert}")
    print(f"Total: {already_saved + to_insert}")


if __name__ == "__main__":
    main()
