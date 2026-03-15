"""
Bot detection using LLM-as-judge (Claude).

Classifies Twitter accounts as human or bot based on their tweet patterns.
Uses Option C (hybrid): LLM classifies each account once during batch
processing, result is cached in the accounts table.
"""

import json
import logging
import os
from dataclasses import dataclass

import anthropic

logger = logging.getLogger("sentinel.bot_detector")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

_SYSTEM_PROMPT = """You are a bot detection classifier for Twitter/X accounts that post about defense stocks.

Given an account's username and sample tweets, classify the account into one of these categories:
- human: A real person sharing opinions, analysis, or commentary about defense stocks
- bot: An automated account — news tickers that repost headlines verbatim, repost bots that copy/rephrase others' tweets, or auto-generated content accounts
- garbage: Accounts posting crypto scams, spam, irrelevant promotional content, or content completely unrelated to defense stocks or financial markets

Respond with JSON only:
{
    "classification": "human" | "bot" | "garbage",
    "confidence": 0.0-1.0,
    "reason": "brief explanation"
}"""


@dataclass
class BotClassification:
    """Result of bot classification for an account."""
    account_type: str  # "human", "bot", "garbage"
    confidence: float
    reason: str

    @property
    def is_filtered(self) -> bool:
        """Whether this account should be filtered from analysis."""
        return self.account_type != "human"


def classify_account(username: str, sample_tweets: list[str], model: str = "claude-haiku-4-5-20251001") -> BotClassification:
    """Classify a single account using Claude.

    Args:
        username: Twitter handle.
        sample_tweets: 3-5 sample tweets from this account.

    Returns:
        BotClassification with account_type and details.

    Raises:
        ValueError: If ANTHROPIC_API_KEY is not set.
        anthropic.APIError: If the API call fails.
    """
    if not ANTHROPIC_API_KEY:
        raise ValueError(
            "ANTHROPIC_API_KEY not set. Required for bot detection."
        )

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    user_message = f"Username: @{username}\n\nSample tweets:\n"
    for i, tweet in enumerate(sample_tweets, 1):
        user_message += f"{i}. {tweet}\n"

    response = client.messages.create(
        model=model,
        max_tokens=256,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    # Parse the JSON response
    response_text = response.content[0].text.strip()
    # Handle markdown code blocks
    if response_text.startswith("```"):
        response_text = response_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    result = json.loads(response_text)

    classification = result["classification"]

    logger.info(
        f"Classified @{username}: {classification} "
        f"(confidence: {result['confidence']:.2f}) — {result['reason']}"
    )

    return BotClassification(
        account_type=classification,
        confidence=result["confidence"],
        reason=result["reason"],
    )


def classify_accounts_batch(
    accounts: list[dict],
    model: str = "claude-haiku-4-5-20251001",
) -> list[tuple[str, BotClassification]]:
    """Classify multiple accounts.

    Args:
        accounts: List of dicts with 'username' and 'sample_tweets' keys.
        model: Anthropic model ID to use for classification.

    Returns:
        List of (username, BotClassification) tuples.
    """
    results = []

    for account in accounts:
        username = account["username"]
        tweets = account["sample_tweets"]

        try:
            classification = classify_account(username, tweets, model=model)
            results.append((username, classification))
        except Exception as e:
            logger.error(f"Failed to classify @{username}: {e}")
            # Default to human on error (don't filter real people)
            results.append((username, BotClassification(
                account_type="human",
                confidence=0.0,
                reason=f"Classification failed: {e}",
            )))

    return results
