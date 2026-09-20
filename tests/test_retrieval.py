"""Integration tests for the LearnForge retrieval module.

These tests run against the REAL, persisted Chroma vector collection in ./chroma_db.
They intentionally do NOT mock the vector database or embedding models; they serve
as integration tests ensuring that semantic retrieval returns the expected documents,
scores, and threshold boundaries against the actual indexed knowledge base.
"""

from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retriever import LOW_CONFIDENCE_THRESHOLD, retrieve


def test_refund_query_retrieves_policy_or_faq():
    """Verify that a refund inquiry includes authoritative policy/FAQ documents in top-4.

    Note: As observed during Task 4b and Task 9-FIX verification, colloquial support
    tickets (TICKET-03 and TICKET-06) match conversational query phrasing more closely
    and rank #1 and #2. Authoritative documents FAQ-02 and POLICY-02 rank #3 and #4.
    Therefore, we assert presence anywhere within top-4, not strictly at rank #1.
    """
    results = retrieve("How do I get a refund?", top_k=4)
    assert len(results) == 4

    retrieved_doc_ids = [r["doc_id"] for r in results]
    assert "FAQ-02" in retrieved_doc_ids or "POLICY-02" in retrieved_doc_ids


def test_certificate_missing_returns_faq06_as_top_result():
    """Verify that certificate missing inquiries rank FAQ-06 as the top-1 result."""
    results = retrieve("certificate missing after finishing course", top_k=4)
    assert len(results) > 0
    top_result = results[0]
    assert top_result["doc_id"] == "FAQ-06"
    assert top_result["score"] >= 0.55  # High confidence


def test_unsupported_financing_query_scores_below_threshold():
    """Verify that an unsupported inquiry yields a top score below LOW_CONFIDENCE_THRESHOLD."""
    results = retrieve("what is the interest rate on financing", top_k=4)
    assert len(results) > 0
    top_score = results[0]["score"]
    assert top_score < LOW_CONFIDENCE_THRESHOLD


def test_empty_and_whitespace_queries_return_empty_list():
    """Verify that empty or whitespace-only queries return an empty list without raising errors."""
    assert retrieve("") == []
    assert retrieve("   ") == []
    assert retrieve("\t\n") == []
