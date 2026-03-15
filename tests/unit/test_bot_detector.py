"""Tests for bot detection module."""

from unittest.mock import MagicMock, patch

import pytest

from src.bot_detector import classify_account, classify_accounts_batch


class TestClassifyAccount:
    """Tests for single account classification."""

    @patch("src.bot_detector.ANTHROPIC_API_KEY", "test-key")
    @patch("src.bot_detector.anthropic.Anthropic")
    def test_classifies_human(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"classification": "human", "confidence": 0.95, "reason": "Original analysis and opinions"}')]
        mock_client.messages.create.return_value = mock_response

        result = classify_account("realtrader", ["$LMT looking strong after contract win", "Great Q2 earnings from RTX"])

        assert result.is_filtered is False
        assert result.account_type == "human"
        assert result.confidence == 0.95
        assert result.reason == "Original analysis and opinions"

    @patch("src.bot_detector.ANTHROPIC_API_KEY", "test-key")
    @patch("src.bot_detector.anthropic.Anthropic")
    def test_classifies_bot(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"classification": "bot", "confidence": 0.99, "reason": "Verbatim headline reposts"}')]
        mock_client.messages.create.return_value = mock_response

        result = classify_account("DefenseNewsBot", ["LMT: Lockheed Martin wins $2B contract", "RTX: RTX Corporation Q2 earnings beat"])

        assert result.is_filtered is True
        assert result.account_type == "bot"
        assert result.confidence == 0.99

    @patch("src.bot_detector.ANTHROPIC_API_KEY", "test-key")
    @patch("src.bot_detector.anthropic.Anthropic")
    def test_handles_markdown_code_blocks(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='```json\n{"classification": "garbage", "confidence": 0.88, "reason": "Promotional spam"}\n```')]
        mock_client.messages.create.return_value = mock_response

        result = classify_account("spammer123", ["BUY NOW! Free signals!"])

        assert result.is_filtered is True
        assert result.account_type == "garbage"

    @patch("src.bot_detector.ANTHROPIC_API_KEY", "")
    def test_raises_without_api_key(self):
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY not set"):
            classify_account("user", ["tweet"])


class TestClassifyAccountsBatch:
    """Tests for batch classification."""

    @patch("src.bot_detector.ANTHROPIC_API_KEY", "test-key")
    @patch("src.bot_detector.anthropic.Anthropic")
    def test_batch_processes_multiple(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        responses = [
            MagicMock(content=[MagicMock(text='{"classification": "human", "confidence": 0.9, "reason": "Real user"}')]),
            MagicMock(content=[MagicMock(text='{"classification": "bot", "confidence": 0.85, "reason": "Copies tweets"}')]),
        ]
        mock_client.messages.create.side_effect = responses

        accounts = [
            {"username": "user1", "sample_tweets": ["Original take on $LMT"]},
            {"username": "bot1", "sample_tweets": ["RT copy paste content"]},
        ]
        results = classify_accounts_batch(accounts)

        assert len(results) == 2
        assert results[0][0] == "user1"
        assert results[0][1].is_filtered is False
        assert results[1][0] == "bot1"
        assert results[1][1].is_filtered is True

    @patch("src.bot_detector.ANTHROPIC_API_KEY", "test-key")
    @patch("src.bot_detector.anthropic.Anthropic")
    def test_batch_defaults_to_human_on_error(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("API error")

        accounts = [{"username": "user1", "sample_tweets": ["test"]}]
        results = classify_accounts_batch(accounts)

        assert len(results) == 1
        assert results[0][1].is_filtered is False
        assert results[0][1].confidence == 0.0
