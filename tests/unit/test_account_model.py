"""Tests for Account model and grifter scoring."""

from datetime import datetime, timezone

from src.data.models import Account


class TestAccount:
    """Tests for Account dataclass."""

    def test_default_values(self):
        account = Account(username="testuser")
        assert account.username == "testuser"
        assert account.account_type == "human"
        assert account.classification_reason is None
        assert account.total_claims == 0
        assert account.exaggerated_count == 0
        assert account.accurate_count == 0
        assert account.understated_count == 0
        assert account.grifter_score is None
        assert account.first_seen is None
        assert account.last_seen is None
        assert account.classified_at is None

    def test_bot_account(self):
        account = Account(
            username="newsbot",
            account_type="bot",
            classification_reason="bot: Verbatim headline reposts",
            classified_at=datetime.now(timezone.utc),
        )
        assert account.account_type == "bot"
        assert "bot" in account.classification_reason

    def test_garbage_account(self):
        account = Account(
            username="cryptospammer",
            account_type="garbage",
            classification_reason="garbage: Crypto scam content",
            classified_at=datetime.now(timezone.utc),
        )
        assert account.account_type == "garbage"
        assert "garbage" in account.classification_reason

    def test_scored_account(self):
        account = Account(
            username="grifter",
            total_claims=10,
            exaggerated_count=8,
            accurate_count=2,
            grifter_score=0.8,
        )
        assert account.grifter_score == 0.8

    def test_grifter_score_none_below_threshold(self):
        """Accounts with < 5 claims should have None grifter_score."""
        account = Account(
            username="newuser",
            total_claims=3,
            exaggerated_count=2,
            accurate_count=1,
        )
        assert account.grifter_score is None
