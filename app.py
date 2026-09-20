"""LearnForge AI Support Assistant - FastAPI Application.

Exposes REST API endpoints for session initialization, grounded conversational chat,
history inspection, and system health checks.
"""

import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from src.conversation import create_session, get_history, handle_turn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("learnforge.api")

# Initialize FastAPI app with descriptive metadata for OpenAPI /docs
app = FastAPI(
    title="LearnForge Support Assistant API",
    description="Conversational RAG support assistant API with multi-turn context, confidence scoring, and escalation.",
    version="1.0.0",
)

# Enable permissive CORS for local prototype testing (Must be restricted in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Request & Response Schemas
# -----------------------------------------------------------------------------
class SessionResponse(BaseModel):
    """Response payload when initializing a conversation session."""

    session_id: str = Field(..., description="Unique conversation session identifier")


class ChatRequest(BaseModel):
    """Payload for submitting a message to the support assistant."""

    session_id: str = Field(..., min_length=1, description="Active session ID")
    message: str = Field(..., min_length=1, description="User question or query")

    @field_validator("message")
    @classmethod
    def check_non_empty(cls, v: str) -> str:
        """Reject empty or whitespace-only messages."""
        if not v or not v.strip():
            raise ValueError("Message cannot be empty or whitespace-only.")
        return v.strip()


class ChatResponse(BaseModel):
    """Structured response from the support assistant."""

    answer: str = Field(..., description="Grounded answer or escalation message")
    source_ids: List[str] = Field(
        default_factory=list, description="List of cited knowledge base document IDs"
    )
    escalated: bool = Field(
        ..., description="Whether this turn required escalation to a human specialist"
    )
    escalation_category: Optional[str] = Field(
        None, description="Category of escalation if triggered"
    )
    confidence: Optional[str] = Field(
        None, description="Reported confidence level: 'high' | 'medium' | 'low' | None"
    )


class HistoryEntry(BaseModel):
    """A single turn in the conversation history."""

    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message text")
    sources: Optional[List[str]] = Field(
        None, description="Cited document IDs for assistant turns"
    )


class HealthResponse(BaseModel):
    """API health status payload."""

    status: str = Field(default="ok", description="Health check status")


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.get(
    "/",
    response_model=HealthResponse,
    tags=["System"],
    summary="Welcome to Edversity Support Assistant",
)
def home() -> HealthResponse:
    """Welcome to Edversity Support Assistant."""
    return HealthResponse(status="Welcome to Edversity Support Assistant")


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check endpoint",
)
def health_check() -> HealthResponse:
    """Check API server liveness."""
    return HealthResponse(status="ok")


@app.post(
    "/session",
    response_model=SessionResponse,
    tags=["Conversation"],
    summary="Create a new conversation session",
)
def start_session() -> SessionResponse:
    """Initialize a new in-memory conversation session and return its ID."""
    session_id = create_session()
    logger.info("Created new session: %s", session_id)
    return SessionResponse(session_id=session_id)


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["Conversation"],
    summary="Send a message to the assistant",
)
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Process a user message within an active conversation session.

    Executes full pipeline:
    - Rule-based escalation check
    - Semantic retrieval & confidence evaluation
    - Grounded generation with conversation history
    - Automatic fallback and human escalation routing
    """
    try:
        response_dict = handle_turn(
            session_id=request.session_id,
            user_message=request.message,
        )
        return ChatResponse(**response_dict)

    except ValueError as e:
        # Invalid/unknown session_id raised from conversation module
        logger.warning("Session lookup failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        # Catch-all for unexpected internal errors (never leak raw stack traces)
        logger.error("Unhandled error processing chat turn: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while processing your request. Please try again later.",
        )


@app.get(
    "/chat/{session_id}/history",
    response_model=List[HistoryEntry],
    tags=["Conversation"],
    summary="Retrieve session conversation history",
)
def get_session_history(session_id: str) -> List[HistoryEntry]:
    """Return the ordered conversation history for a given session."""
    try:
        raw_history = get_history(session_id)
        return [HistoryEntry(**entry) for entry in raw_history]
    except ValueError as e:
        logger.warning("Session history lookup failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


# -----------------------------------------------------------------------------
# Direct Execution Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
