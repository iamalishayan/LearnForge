"""Confidence and escalation module for LearnForge assistant.

Provides deterministic, rule-based escalation checks before generation:
1. Rule-based triggers (fraud/security, disputed terms) evaluated directly on user input.
2. Retrieval confidence evaluation based on semantic vector similarity scores.

Note:
# TODO (Task 6 / llm.py): Incorporate LLM self-reported uncertainty / refusal as
# a third confidence layer. In Task 6, generation-time signals (e.g. LLM emitting
# an escalation flag or expressing inability to answer from context) will combine
# with the pre-generation signals defined in this module.
"""

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path for standalone script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.retriever import LOW_CONFIDENCE_THRESHOLD, retrieve
except ImportError:
    from retriever import LOW_CONFIDENCE_THRESHOLD, retrieve

# -----------------------------------------------------------------------------
# Trigger patterns for rule-based escalation
# Based on actual ticket patterns in the KB (TICKET-03, 05, 08, 11, 14):
# -----------------------------------------------------------------------------

# Category: fraud_or_security
# Matches reports of unknown charges, stolen accounts, or unauthorized transactions.
# Phrases matched:
# - "unauthorized", "fraud", "fraudulent", "hacked", "compromised", "suspicious activity"
# - "don't recognize" / "do not recognize" near "charge", "payment", "transaction"
# - "charge" / "payment" near "don't remember" / "didn't remember"
# - "didn't make" / "did not authorize" near "purchase", "charge", "payment"
FRAUD_OR_SECURITY_PATTERNS = [
    r"\b(?:unauthorized|fraudulent|fraud)\b",
    r"\b(?:account|password|login)\s+(?:hacked|compromised|breached|stolen)\b",
    r"\bsuspicious\s+activity\b",
    r"\b(?:don'?t|do\s+not)\s+recognize\b.*?\b(?:charge|transaction|payment|fee)\b",
    r"\b(?:charge|transaction|payment|fee)\b.*?\b(?:don'?t|do\s+not)\s+recognize\b",
    r"\b(?:charge|transaction|payment|fee)\b.*?\b(?:don'?t|didn'?t|do\s+not)\s+remember\b",
    r"\b(?:didn'?t|did\s+not)\s+(?:make|authorize)\b.*?\b(?:charge|purchase|transaction|payment)\b",
    r"\bsomeone\s+else\s+used\s+my\s+(?:card|account)\b",
]

# Category: disputed_policy_terms
# Matches cases where the user claims non-standard terms, promises, or pricing applied to them.
# Phrases matched:
# - "your website said", "the site says", "cancellation page said", "checkout page said"
# - "I was told", "you told me", "support told me", "agent told me"
# - "said/promised/guaranteed [I had] N days"
# - "N-day money-back guarantee"
# - "dispute/disputed policy/terms/charge"
DISPUTED_POLICY_TERMS_PATTERNS = [
    r"\b(?:your\s+website|the\s+site|the\s+website|cancellation\s+page|checkout\s+page|faq)\s+(?:said|says|states|promised|literally\s+says)\b",
    r"\b(?:i\s+was\s+told|you\s+told\s+me|agent\s+told\s+me|support\s+told\s+me)\b",
    r"\b(?:said|says|promised|guaranteed)\s+(?:i\s+had\s+)?\d+\s*days?\b",
    r"\b\d+\s*-?\s*day\s+(?:money\s*-?\s*back\s+guarantee|refund\s+guarantee)\b",
    r"\bdispute\s+(?:the\s+)?(?:policy|terms?|charge)\b",
]


def check_rule_based_escalation(user_message: str) -> Dict[str, Any]:
    """Evaluate raw user message for rule-based escalation triggers before retrieval.

    Args:
        user_message: Raw text input from the user.

    Returns:
        Dictionary:
            - should_escalate (bool): True if any immediate escalation rule triggered.
            - reason (str | None): Human-readable explanation of why escalation fired.
            - category (str | None): "fraud_or_security" | "disputed_policy_terms" | None
    """
    if not user_message or not user_message.strip():
        return {
            "should_escalate": False,
            "reason": None,
            "category": None,
        }

    # 1. Check fraud or account security triggers
    for pattern in FRAUD_OR_SECURITY_PATTERNS:
        if re.search(pattern, user_message, re.IGNORECASE):
            return {
                "should_escalate": True,
                "reason": "Suspected fraudulent charge, unrecognized payment, or compromised account",
                "category": "fraud_or_security",
            }

    # 2. Check disputed terms or conflicting promises
    for pattern in DISPUTED_POLICY_TERMS_PATTERNS:
        if re.search(pattern, user_message, re.IGNORECASE):
            return {
                "should_escalate": True,
                "reason": "User claims specific contradictory or non-standard policy terms",
                "category": "disputed_policy_terms",
            }

    return {
        "should_escalate": False,
        "reason": None,
        "category": None,
    }


def check_retrieval_confidence(retrieval_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate retrieval results against the low-confidence escalation threshold.

    Args:
        retrieval_results: List of retrieved chunks with 'score' field (from retriever.py).

    Returns:
        Dictionary:
            - should_escalate (bool): True if top score is below threshold or results empty.
            - reason (str | None): Explanation for low confidence.
            - top_score (float): Highest similarity score among retrieved results.
    """
    if not retrieval_results:
        return {
            "should_escalate": True,
            "reason": "No relevant documents found in knowledge base",
            "top_score": 0.0,
        }

    top_score = float(retrieval_results[0].get("score", 0.0))

    if top_score < LOW_CONFIDENCE_THRESHOLD:
        return {
            "should_escalate": True,
            "reason": (
                f"Top retrieval similarity score ({top_score:.4f}) is below "
                f"low-confidence threshold ({LOW_CONFIDENCE_THRESHOLD:.2f})"
            ),
            "top_score": top_score,
        }

    return {
        "should_escalate": False,
        "reason": None,
        "top_score": top_score,
    }


def should_escalate(
    user_message: str,
    retrieval_results: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Combined pre-generation escalation decision.

    Runs rule-based checks first. If a rule triggers, escalates immediately without
    requiring retrieval results. Otherwise, evaluates retrieval confidence if provided.

    Args:
        user_message: User's raw query string.
        retrieval_results: Optional list of retrieved chunks from src/retriever.py.

    Returns:
        Dictionary:
            - should_escalate (bool): Final escalation decision.
            - reason (str | None): Reason for escalation.
            - category (str | None): Category of escalation.
            - top_score (float | None): Top retrieval score if available.
    """
    # 1. Rule-based pre-retrieval check
    rule_check = check_rule_based_escalation(user_message)
    if rule_check["should_escalate"]:
        return {
            "should_escalate": True,
            "reason": rule_check["reason"],
            "category": rule_check["category"],
            "top_score": None,
        }

    # 2. Retrieval confidence post-retrieval check (if retrieval results supplied)
    if retrieval_results is not None:
        conf_check = check_retrieval_confidence(retrieval_results)
        if conf_check["should_escalate"]:
            return {
                "should_escalate": True,
                "reason": conf_check["reason"],
                "category": "low_confidence_retrieval",
                "top_score": conf_check["top_score"],
            }
        return {
            "should_escalate": False,
            "reason": None,
            "category": None,
            "top_score": conf_check["top_score"],
        }

    # Neither rule triggered, and no retrieval results provided yet
    return {
        "should_escalate": False,
        "reason": None,
        "category": None,
        "top_score": None,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("CONFIDENCE & ESCALATION MODULE VERIFICATION")
    print(f"LOW_CONFIDENCE_THRESHOLD: {LOW_CONFIDENCE_THRESHOLD}")
    print("=" * 70)

    # Test Case 1: Unrecognized charge
    q1 = "I don't recognize a charge on my card"
    res1 = should_escalate(q1)
    print(f"\n[Case 1] \"{q1}\"")
    print(f"Decision: {res1}")

    # Test Case 2: Charge don't remember with specific dollar amount
    q2 = "There is a $79 charge I don't remember"
    res2 = should_escalate(q2)
    print(f"\n[Case 2] \"{q2}\"")
    print(f"Decision: {res2}")

    # Test Case 3: Disputed terms / refund window
    q3 = "Your website said I had 30 days to refund, it's been 20"
    res3 = should_escalate(q3)
    print(f"\n[Case 3] \"{q3}\"")
    print(f"Decision: {res3}")

    # Test Case 4: Standard supported FAQ (password reset)
    q4 = "How do I reset my password?"
    rule4 = check_rule_based_escalation(q4)
    ret4 = retrieve(q4)
    res4 = should_escalate(q4, ret4)
    print(f"\n[Case 4] \"{q4}\"")
    print(f"Rule-only check:      {rule4}")
    print(f"Combined with search: {res4}")

    # Test Case 5: Completely irrelevant / out-of-scope query
    q5 = "what is the interest rate on financing"
    rule5 = check_rule_based_escalation(q5)
    ret5 = retrieve(q5)
    res5 = should_escalate(q5, ret5)
    print(f"\n[Case 5] \"{q5}\"")
    print(f"Rule-only check:      {rule5}")
    print(f"Combined with search: {res5}")

    print("\n" + "=" * 70)
