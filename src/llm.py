"""LLM generation module for LearnForge assistant.

Handles multi-provider model invocation (Groq or Gemini), prompt assembly with
grounding and deprecated-content instructions, and robust JSON output parsing with
automatic retry on parse errors.

Note:
# TODO (Task 8 / app.py): This module does NOT call src/confidence.py. It produces
# the LLM's self-reported confidence and escalation decision. Downstream callers
# (e.g. app.py / API layer) are responsible for combining the pre-generation signals
# from src/confidence.py (rule-based + retrieval scores) with this module's output.
"""

import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

# Ensure project root is in sys.path for direct execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load local .env so module works standalone
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# Provider and model defaults
DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
GROQ_DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GEMINI_DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# -----------------------------------------------------------------------------
# System Prompt
# -----------------------------------------------------------------------------
SYSTEM_PROMPT = """You are the official customer support assistant for LearnForge, an online learning platform.
Your task is to provide accurate, grounded answers to learner questions strictly using the provided context passages.

CRITICAL OPERATIONAL RULES:
1. STRICT CONTEXT GROUNDING: Rely ONLY on the provided context passages to answer. Never use unsupported external knowledge or general assumptions about ed-tech platforms.
2. UNCERTAINTY & UNANSWERED QUERIES: If the provided context does not contain enough information to answer the question, clearly state that you do not have this information or cannot confirm it. Do NOT invent or guess policies, timelines, fees, or features. In such cases, set "confidence": "low" and "should_escalate": true.
3. DEPRECATED / STALE CONTENT HANDLING: When a context passage is marked [has_deprecated_content: true], it contains a contrast between an older rule and the current rule. Always state the CURRENT rule as authoritative, and explicitly clarify that older documentation may reference outdated information (e.g. desktop downloads vs mobile-only offline viewing).
4. CITATIONS: In the "source_ids" list, include the exact doc_ids (e.g., ["FAQ-01", "POLICY-02"]) of the passages you actually relied upon. If no passages were relevant, return an empty list [].
5. SENSITIVE DATA PRIVACY: Never request, collect, or repeat back sensitive user data, including full payment card numbers, CVVs, account passwords, PINs, or government identification numbers (per LearnForge security policy).
6. CONCISENESS: Keep answers direct, professional, and concise (typically 2 to 4 sentences, unless detailed multi-step troubleshooting is genuinely necessary).

OUTPUT FORMAT:
You must respond ONLY with a single valid JSON object. Do not include any conversational preamble, markdown formatting outside of JSON, or trailing explanations.
Schema:
{
  "answer": "string containing the direct answer to the user",
  "source_ids": ["string", "string"],
  "confidence": "high" | "medium" | "low",
  "should_escalate": true | false
}"""


def get_chat_model(provider: Optional[str] = None):
    """Instantiate and return the configured chat model (Groq or Gemini).

    Fails with a clear, user-friendly error message if credentials are missing.
    """
    selected_provider = (provider or DEFAULT_LLM_PROVIDER).lower()

    if selected_provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or not api_key.strip():
            raise RuntimeError(
                "Error: GROQ_API_KEY is missing from environment. "
                "Please set GROQ_API_KEY in your .env file or choose another provider."
            )
        try:
            from langchain_groq import ChatGroq
        except ImportError:
            raise RuntimeError("Error: 'langchain-groq' is not installed in the current environment.")

        return ChatGroq(
            model=GROQ_DEFAULT_MODEL,
            groq_api_key=api_key,
            temperature=0.0,
        )

    elif selected_provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or not api_key.strip():
            raise RuntimeError(
                "Error: GEMINI_API_KEY is missing from environment. "
                "Please set GEMINI_API_KEY in your .env file or choose another provider."
            )
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise RuntimeError("Error: 'langchain-google-genai' is not installed in the current environment.")

        return ChatGoogleGenerativeAI(
            model=GEMINI_DEFAULT_MODEL,
            google_api_key=api_key,
            temperature=0.0,
        )

    else:
        raise RuntimeError(
            f"Error: Unsupported LLM_PROVIDER '{selected_provider}'. "
            "Supported providers are 'groq' and 'gemini'."
        )


def format_context_passages(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved passages with metadata headers for LLM prompt injection."""
    if not retrieved_chunks:
        return "No relevant context passages found in knowledge base."

    formatted_blocks = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        doc_id = chunk.get("doc_id", "UNKNOWN")
        doc_type = chunk.get("doc_type", "unknown")
        topic = chunk.get("topic", "")
        has_deprecated = chunk.get("has_deprecated_content", False)
        text = chunk.get("text", "").strip()

        header = (
            f"[Passage {i}] Doc ID: {doc_id} | Type: {doc_type} | "
            f"Topic: {topic} | has_deprecated_content: {has_deprecated}"
        )
        formatted_blocks.append(f"{header}\n{text}")

    return "\n\n---\n\n".join(formatted_blocks)


def clean_json_response(raw_text: str) -> str:
    """Strip markdown code blocks or surrounding text to isolate JSON."""
    text = raw_text.strip()

    # Strip triple-backtick markdown fences (e.g. ```json ... ```)
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Regex extraction if extraneous text wraps the JSON object
    match = re.search(r"(\{[\s\S]*\})", text)
    if match:
        text = match.group(1).strip()

    return text


def parse_and_validate_json(raw_text: str) -> Optional[Dict[str, Any]]:
    """Parse JSON and validate required schema fields."""
    cleaned = clean_json_response(raw_text)
    try:
        data = json.loads(cleaned)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    # Validate and normalize fields
    answer = str(data.get("answer", "")).strip()
    if not answer:
        return None

    raw_sources = data.get("source_ids", [])
    if isinstance(raw_sources, list):
        source_ids = [str(s).strip() for s in raw_sources if str(s).strip()]
    else:
        source_ids = []

    raw_conf = str(data.get("confidence", "")).lower().strip()
    confidence = raw_conf if raw_conf in {"high", "medium", "low"} else "medium"

    raw_escalate = data.get("should_escalate", False)
    if isinstance(raw_escalate, bool):
        should_escalate = raw_escalate
    elif isinstance(raw_escalate, str):
        should_escalate = raw_escalate.lower() in {"true", "1", "yes"}
    else:
        should_escalate = bool(raw_escalate)

    return {
        "answer": answer,
        "source_ids": source_ids,
        "confidence": confidence,
        "should_escalate": should_escalate,
    }


def generate_response(
    user_message: str,
    retrieved_chunks: List[Dict[str, Any]],
    conversation_history: Optional[List[Dict[str, str]]] = None,
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a structured, grounded support response using the selected LLM.

    Args:
        user_message: Current question from the user.
        retrieved_chunks: Top-k knowledge base passages from retriever.py.
        conversation_history: Optional previous turns [{"role": "user"|"assistant", "content": str}].
        provider: Optional LLM provider override ("groq" | "gemini").

    Returns:
        Dictionary with keys:
            - answer (str): Direct answer to the learner.
            - source_ids (list[str]): Cited doc_ids.
            - confidence (str): "high" | "medium" | "low".
            - should_escalate (bool): Model self-reported escalation flag.
            - raw_retrieval_top_score (float | None): Passthrough top score from retriever.
    """
    top_score = (
        float(retrieved_chunks[0].get("score", 0.0))
        if retrieved_chunks
        else None
    )

    # 1. Format context passages
    context_text = format_context_passages(retrieved_chunks)

    # 2. Assemble prompt messages
    messages: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]

    # Include up to last 6 messages from conversation history if provided
    if conversation_history:
        recent_history = conversation_history[-6:]
        for msg in recent_history:
            role = msg.get("role", "").lower()
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

    # Add current query with context passages
    user_prompt = (
        f"Context passages from LearnForge Knowledge Base:\n\n"
        f"{context_text}\n\n"
        f"User Question: {user_message}\n\n"
        f"Respond strictly in the required JSON schema."
    )
    messages.append(HumanMessage(content=user_prompt))

    # 3. Call LLM
    try:
        model = get_chat_model(provider)
        response = model.invoke(messages)
        raw_output = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        # LLM connection / runtime error fallback
        return {
            "answer": (
                "I apologize, but I encountered a technical issue connecting to our support service. "
                "Please hold on while I connect you with a support representative."
            ),
            "source_ids": [],
            "confidence": "low",
            "should_escalate": True,
            "raw_retrieval_top_score": top_score,
            "error": str(e),
        }

    # 4. Parse JSON with single-retry guardrail
    parsed = parse_and_validate_json(raw_output)

    if parsed is None:
        # Retry ONCE with explicit correction prompt
        retry_messages = list(messages)
        retry_messages.append(AIMessage(content=raw_output))
        retry_messages.append(
            HumanMessage(
                content=(
                    "Your previous response could not be parsed as valid JSON. "
                    "Please respond with ONLY a single valid JSON object matching the exact schema: "
                    '{"answer": string, "source_ids": list of strings, "confidence": "high"|"medium"|"low", "should_escalate": boolean}'
                )
            )
        )

        try:
            retry_response = model.invoke(retry_messages)
            retry_raw = (
                retry_response.content
                if hasattr(retry_response, "content")
                else str(retry_response)
            )
            parsed = parse_and_validate_json(retry_raw)
        except Exception:
            parsed = None

    # 5. Fallback if JSON parsing still fails
    if parsed is None:
        return {
            "answer": (
                "I apologize, but I encountered a formatting issue while processing your request. "
                "I will escalate your request to a support specialist to assist you directly."
            ),
            "source_ids": [],
            "confidence": "low",
            "should_escalate": True,
            "raw_retrieval_top_score": top_score,
            "parse_error": True,
        }

    # 6. Return verified response
    return {
        "answer": parsed["answer"],
        "source_ids": parsed["source_ids"],
        "confidence": parsed["confidence"],
        "should_escalate": parsed["should_escalate"],
        "raw_retrieval_top_score": top_score,
    }


if __name__ == "__main__":
    try:
        from src.retriever import retrieve
    except ImportError:
        from retriever import retrieve

    test_queries = [
        "How do I get a refund?",
        "Can I download courses to watch offline on my laptop?",
        "What is the interest rate on financing?",
    ]

    print("=" * 75)
    print("LLM GENERATION MODULE VERIFICATION")
    print(f"Provider: {DEFAULT_LLM_PROVIDER}")
    print("=" * 75)

    for i, q in enumerate(test_queries, 1):
        print(f"\n[Test Case {i}] \"{q}\"")
        chunks = retrieve(q, top_k=4)
        response_data = generate_response(q, retrieved_chunks=chunks)

        print(f"Answer:          {response_data['answer']}")
        print(f"Source IDs:      {response_data['source_ids']}")
        print(f"Confidence:      {response_data['confidence']}")
        print(f"Should Escalate: {response_data['should_escalate']}")
        print(f"Top Retr Score:  {response_data['raw_retrieval_top_score']}")

    print("\n" + "=" * 75)
