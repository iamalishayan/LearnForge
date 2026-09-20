"""Conversation and session management module for LearnForge assistant.

Manages in-memory conversation histories per session and orchestrates the complete
turn pipeline:
1. Rule-based pre-retrieval escalation check.
2. Context-aware vector retrieval.
3. Retrieval confidence check.
4. LLM grounded generation with history.
5. Response assembly, history tracking, and structured output.
"""

from datetime import datetime, timezone
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

logger = logging.getLogger(__name__)

# Ensure project root is in sys.path for direct execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.confidence import (
        check_retrieval_confidence,
        check_rule_based_escalation,
    )
    from src.llm import generate_response
    from src.retriever import LOW_CONFIDENCE_THRESHOLD, retrieve
except ImportError:
    from confidence import (
        check_retrieval_confidence,
        check_rule_based_escalation,
    )
    from llm import generate_response
    from retriever import LOW_CONFIDENCE_THRESHOLD, retrieve

# -----------------------------------------------------------------------------
# In-memory session store (prototype scope: module-level dict)
# -----------------------------------------------------------------------------
_sessions: Dict[str, Dict[str, Any]] = {}

# -----------------------------------------------------------------------------
# Varied human escalation messages for non-templated responses
# -----------------------------------------------------------------------------
ESCALATION_MESSAGES = {
    "fraud_or_security": [
        "I've flagged this unfamiliar charge for our security and billing specialists. To protect your account, a team member will review this right away and reach out to assist you.",
        "Because this involves an unrecognized transaction on your payment card, I've escalated this directly to our billing support team to investigate and follow up with you promptly.",
        "Account security is our top priority. I've routed this transaction dispute to our fraud and billing specialists so they can verify the charge and contact you directly.",
    ],
    "disputed_policy_terms": [
        "I understand there appears to be a discrepancy between what was advertised or communicated and our standard terms. I've escalated your case to a support manager who can review your order details and resolve this fairly.",
        "Since your inquiry references specific promotional or policy terms that differ from our standard rules, I've forwarded this to a senior support representative to review and follow up.",
    ],
    "low_confidence_retrieval": [
        "I don't have enough verified information in our knowledge base to answer your question with confidence. I've escalated this to a human support agent who can provide you with the exact details.",
        "I want to make sure you get accurate information, but our documentation does not currently cover this topic. I've connected your question to our support team for personal assistance.",
        "I am unable to find official guidance on this in our knowledge base. A support representative will review your request and get back to you shortly.",
    ],
    "llm_flagged": [
        "I cannot confirm the necessary details to answer this reliably from our official documentation. I have forwarded your inquiry to our support staff to assist you directly.",
        "I want to ensure you receive accurate assistance, so I've escalated your inquiry to a support specialist who can look into this for you.",
    ],
}


def get_escalation_message(category: str) -> str:
    """Return an empathetic, varied human message for the given escalation category."""
    options = ESCALATION_MESSAGES.get(category, ESCALATION_MESSAGES["low_confidence_retrieval"])
    return random.choice(options)


def create_session() -> str:
    """Initialize a new conversation session and return its unique session_id."""
    session_id = uuid.uuid4().hex
    _sessions[session_id] = {
        "history": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return session_id


def get_history(session_id: str) -> List[Dict[str, Any]]:
    """Retrieve full conversation history for a given session.

    Raises:
        ValueError: If session_id does not exist.
    """
    if session_id not in _sessions:
        raise ValueError(
            f"Session '{session_id}' not found. Please create a session first via create_session()."
        )
    return _sessions[session_id]["history"]


def handle_turn(
    session_id: str,
    user_message: str,
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    """Orchestrate one full conversation turn.

    Pipeline:
    a. Validate session existence.
    b. Check rule-based pre-retrieval escalation (short-circuits without LLM).
    c. Retrieve relevant context passages (with conversational follow-up enrichment).
    d. Check retrieval confidence (short-circuits if low confidence/empty).
    e. Call LLM generation with recent conversation history.
    f. Assemble structured final response.
    g. Record user message and assistant turn in session history.

    Args:
        session_id: Valid active session identifier.
        user_message: Current question from the learner.
        provider: Optional LLM provider override ("groq" | "gemini").

    Returns:
        Dictionary with keys:
            - answer (str): Direct response to learner.
            - source_ids (list[str]): Cited document IDs.
            - escalated (bool): Whether query was escalated to human support.
            - escalation_category (str | None): Category if escalated.
            - confidence (str | None): "high" | "medium" | "low" | None.
    """
    # a. Validate session
    if session_id not in _sessions:
        raise ValueError(
            f"Session '{session_id}' not found. Please call create_session() first."
        )

    session = _sessions[session_id]
    history: List[Dict[str, Any]] = session["history"]

    # b. Step 1: Rule-based pre-retrieval escalation check
    rule_check = check_rule_based_escalation(user_message)
    if rule_check["should_escalate"]:
        category = rule_check["category"] or "rule_based"
        answer = get_escalation_message(category)
        response_data = {
            "answer": answer,
            "source_ids": [],
            "escalated": True,
            "escalation_category": category,
            "confidence": None,  # Short-circuited before LLM
        }
        # Update history
        history.append({"role": "user", "content": user_message})
        history.append({
            "role": "assistant",
            "content": answer,
            "sources": [],
        })
        return response_data

    # c. Step 2: Context-aware vector retrieval
    # For conversational follow-ups (e.g. "What about on my laptop specifically?"),
    # if the raw message scores low, enrich the search with the recent user query topic.
    retrieval_results = retrieve(user_message)
    conf_check = check_retrieval_confidence(retrieval_results)

    if conf_check["should_escalate"] and history:
        recent_user_queries = [
            turn["content"] for turn in history if turn.get("role") == "user"
        ]
        if recent_user_queries:
            enriched_query = f"{recent_user_queries[-1]} {user_message}"
            enriched_results = retrieve(enriched_query)
            enriched_conf = check_retrieval_confidence(enriched_results)
            logger.debug(
                "[Trace] Raw query score: %s (< %s) | Enrichment fired with: '%s' -> Enriched score: %s",
                conf_check.get("top_score"),
                LOW_CONFIDENCE_THRESHOLD,
                enriched_query,
                enriched_conf.get("top_score"),
            )
            # Use contextualized results if it resolved the conversational follow-up
            if not enriched_conf["should_escalate"]:
                retrieval_results = enriched_results
                conf_check = enriched_conf
    else:
        logger.debug(
            "[Trace] Raw query score: %s (>= %s or no history) | Raw query used directly.",
            conf_check.get("top_score"),
            LOW_CONFIDENCE_THRESHOLD,
        )

    # d. Step 3: Retrieval confidence check
    if conf_check["should_escalate"]:
        category = "low_confidence_retrieval"
        answer = get_escalation_message(category)
        response_data = {
            "answer": answer,
            "source_ids": [],
            "escalated": True,
            "escalation_category": category,
            "confidence": "low",
        }
        history.append({"role": "user", "content": user_message})
        history.append({
            "role": "assistant",
            "content": answer,
            "sources": [],
        })
        return response_data

    # e. Step 4: LLM grounded generation with conversation history (last 6 messages)
    truncated_history = [
        {"role": entry["role"], "content": entry["content"]}
        for entry in history[-6:]
    ]

    llm_output = generate_response(
        user_message=user_message,
        retrieved_chunks=retrieval_results,
        conversation_history=truncated_history,
        provider=provider,
    )

    # Combine pre-generation confidence with LLM's self-reported confidence
    is_escalated = bool(llm_output.get("should_escalate", False))
    category = "llm_flagged" if is_escalated else None

    # Consistent escalation messaging: Use friendly scripted escalation copy when LLM flags escalation
    if is_escalated:
        answer_text = get_escalation_message("llm_flagged")
        source_ids = []
    else:
        answer_text = llm_output["answer"]
        source_ids = llm_output.get("source_ids", [])

    confidence = llm_output.get("confidence", "medium")

    # f. Step 5: Build final response
    response_data = {
        "answer": answer_text,
        "source_ids": source_ids,
        "escalated": is_escalated,
        "escalation_category": category,
        "confidence": confidence,
    }

    # g. Step 6: Append user and assistant turns to session history
    history.append({"role": "user", "content": user_message})
    history.append({
        "role": "assistant",
        "content": answer_text,
        "sources": source_ids,
    })

    return response_data


if __name__ == "__main__":
    print("=" * 75)
    print("CONVERSATION & SESSION MANAGEMENT VERIFICATION")
    print("=" * 75)

    session_id = create_session()
    print(f"Created session: {session_id}\n")

    test_turns = [
        ("Turn 1", "Can I download courses to watch offline?"),
        ("Turn 2", "What about on my laptop specifically?"),
        ("Turn 3", "I don't recognize a $79 charge though"),
    ]

    for turn_label, query in test_turns:
        print(f"\n--- {turn_label}: \"{query}\" ---")
        response = handle_turn(session_id, query)
        print("Response Dict:")
        for k, v in response.items():
            print(f"  {k}: {v}")

    print("\n" + "=" * 75)
    print("FINAL SESSION HISTORY (get_history):")
    print("=" * 75)
    history_entries = get_history(session_id)
    print(f"Total history entries: {len(history_entries)}")
    for idx, entry in enumerate(history_entries, 1):
        sources = f" | Sources: {entry.get('sources')}" if entry.get("role") == "assistant" else ""
        print(f"[{idx}] {entry['role'].upper()}: {entry['content'][:90]}...{sources}")

    print("\n" + "=" * 75)
