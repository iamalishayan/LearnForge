"""Unit tests for the confidence and escalation module.

These tests evaluate the pure, deterministic logic in src/confidence.py:
1. Rule-based pre-retrieval pattern matching (fraud/security, disputed terms).
2. Threshold-based retrieval confidence evaluation against synthetic chunk scores.
No mocking or external services are required.
"""

from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.confidence import (
    check_retrieval_confidence,
    check_rule_based_escalation,
    should_escalate,
)


def test_rule_based_escalation_unrecognized_charge():
    """Verify that reports of unrecognized transactions trigger fraud_or_security escalation."""
    result = check_rule_based_escalation("I don't recognize a charge on my card")
    assert result["should_escalate"] is True
    assert result["category"] == "fraud_or_security"


def test_rule_based_escalation_forgotten_charge():
    """Verify that reports of unfamiliar charges trigger fraud_or_security escalation."""
    result = check_rule_based_escalation("There is a $79 charge I don't remember")
    assert result["should_escalate"] is True
    assert result["category"] == "fraud_or_security"


def test_rule_based_escalation_disputed_policy_terms():
    """Verify that claims of conflicting website guarantees trigger disputed_policy_terms escalation."""
    result = check_rule_based_escalation(
        "Your website said I had 30 days to refund, it's been 20"
    )
    assert result["should_escalate"] is True
    assert result["category"] == "disputed_policy_terms"


def test_rule_based_escalation_standard_inquiry_does_not_escalate():
    """Verify that standard customer support inquiries pass rule checks without escalating."""
    result = check_rule_based_escalation("How do I reset my password?")
    assert result["should_escalate"] is False
    assert result["category"] is None
    assert result["reason"] is None


def test_retrieval_confidence_high_score():
    """Verify that synthetic retrieval results above threshold do not trigger escalation."""
    synthetic_results = [
        {"doc_id": "FAQ-06", "score": 0.7501, "topic": "Certificates"},
        {"doc_id": "TICKET-09", "score": 0.7063, "topic": "Certificate Missing"},
    ]
    result = check_retrieval_confidence(synthetic_results)
    assert result["should_escalate"] is False
    assert result["top_score"] == 0.7501
    assert result["reason"] is None


def test_retrieval_confidence_low_score():
    """Verify that synthetic retrieval results below threshold trigger escalation."""
    synthetic_results = [
        {"doc_id": "TICKET-03", "score": 0.2303, "topic": "Refund Outside Window"},
        {"doc_id": "FAQ-12", "score": 0.2204, "topic": "Course Prices"},
    ]
    result = check_retrieval_confidence(synthetic_results)
    assert result["should_escalate"] is True
    assert result["top_score"] == 0.2303
    assert "below low-confidence threshold" in result["reason"]


def test_retrieval_confidence_empty_results():
    """Verify that empty retrieval results trigger escalation with top_score 0.0."""
    result = check_retrieval_confidence([])
    assert result["should_escalate"] is True
    assert result["top_score"] == 0.0
    assert "No relevant documents found" in result["reason"]


def test_combined_should_escalate_orchestration():
    """Verify combined pre-generation decision logic with synthetic inputs."""
    # 1. Rule trigger short-circuits before retrieval
    rule_res = should_escalate("I don't recognize a charge on my card")
    assert rule_res["should_escalate"] is True
    assert rule_res["category"] == "fraud_or_security"

    # 2. Non-rule query with high retrieval score proceeds normally
    normal_res = should_escalate(
        "How do I get a certificate?",
        retrieval_results=[{"doc_id": "FAQ-06", "score": 0.75}],
    )
    assert normal_res["should_escalate"] is False
    assert normal_res["category"] is None

    # 3. Non-rule query with low retrieval score triggers low_confidence_retrieval
    low_res = should_escalate(
        "what is the interest rate on financing",
        retrieval_results=[{"doc_id": "TICKET-03", "score": 0.23}],
    )
    assert low_res["should_escalate"] is True
    assert low_res["category"] == "low_confidence_retrieval"
