"""
tools.py — Tool definitions for the AutoStream agent.

Contains:
  - mock_lead_capture(): backend mock for lead submission
  - IntentClassifier: lightweight rule + LLM-based intent detection
"""

import re
from enum import Enum


# ---------------------------------------------------------------------------
# Intent types
# ---------------------------------------------------------------------------

class Intent(str, Enum):
    GREETING        = "greeting"
    PRODUCT_INQUIRY = "product_inquiry"
    HIGH_INTENT     = "high_intent"
    OTHER           = "other"


# ---------------------------------------------------------------------------
# Mock Lead Capture Tool
# ---------------------------------------------------------------------------

def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """
    Mock backend API for capturing a qualified lead.
    In production this would POST to a CRM or webhook.
    """
    print(f"\n{'='*55}")
    print(f"  LEAD CAPTURED SUCCESSFULLY")
    print(f"  Name     : {name}")
    print(f"  Email    : {email}")
    print(f"  Platform : {platform}")
    print(f"{'='*55}\n")
    return (
        f"Lead captured successfully: {name}, {email}, {platform}"
    )


# ---------------------------------------------------------------------------
# Intent classifier (keyword + pattern heuristics)
# Used as a pre-filter before the LLM makes the final call.
# ---------------------------------------------------------------------------

_GREETING_PATTERNS = re.compile(
    r"\b(hi|hello|hey|good morning|good evening|howdy|greetings|sup|what'?s up)\b",
    re.IGNORECASE
)

_HIGH_INTENT_PATTERNS = re.compile(
    r"\b(sign ?up|sign me up|buy now|purchase|checkout|get started|"
    r"ready to (buy|start)|i'?m ready|let'?s do it|yes start|start now)\b",
    re.IGNORECASE
)

_LOW_COMMITMENT_PATTERNS = re.compile(
    r"\b(explore|consider|thinking|maybe|might|probably|compare|"
    r"deciding|decide|not sure|first|after this|before deciding|"
    r"look at|learn about|tell me about)\b",
    re.IGNORECASE
)

_PRODUCT_PATTERNS = re.compile(
    r"\b(price|pricing|plan|cost|how much|feature|what does|tell me about|"
    r"do you (have|offer|support)|refund|cancel|trial|4k|resolution|video|"
    r"caption|support|upgrade|difference|compare|basic|pro)\b",
    re.IGNORECASE
)


def classify_intent_heuristic(user_message: str) -> Intent:
    """
    Fast heuristic intent classifier.
    The LangGraph agent also gets the LLM to confirm/override this
    in ambiguous cases via the system prompt.
    """
    msg = user_message.strip()

    if _HIGH_INTENT_PATTERNS.search(msg) and not _LOW_COMMITMENT_PATTERNS.search(msg):
        return Intent.HIGH_INTENT

    if _GREETING_PATTERNS.search(msg) and len(msg.split()) <= 8:
        return Intent.GREETING

    if _PRODUCT_PATTERNS.search(msg):
        return Intent.PRODUCT_INQUIRY

    return Intent.OTHER
