"""Data structures and schemas for knowledge base documents and chunks."""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Metadata schema for a parsed knowledge base chunk."""

    doc_id: str = Field(description="Unique identifier, e.g., FAQ-01, POLICY-02, TICKET-03")
    doc_type: Literal["faq", "policy", "ticket"] = Field(
        description="Type of knowledge base document"
    )
    topic: str = Field(description="Short topic label derived from the heading")
    has_deprecated_content: bool = Field(
        default=False,
        description="Whether the chunk text contains stale or conflicting policy signals",
    )
    last_reviewed: Optional[str] = Field(
        default=None,
        description="Date of last review or update, if present on policy documents",
    )
    source_file: str = Field(description="Filename of origin, e.g. policies.md")
