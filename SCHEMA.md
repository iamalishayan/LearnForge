# LearnForge Data & Object Schemas

This document defines the data structures, database formats, and API schemas implemented across the LearnForge AI Support Assistant.

---

## 1. Document Chunk Metadata Schema (`src/schema.py`)

Every ingested document chunk is parsed, sanitized, and validated against the `ChunkMetadata` Pydantic model before being indexed into the vector store.

```python
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
```

### Field Specifications & Nullability
| Field | Type | Description | Observed Ingestion Behavior |
| :--- | :--- | :--- | :--- |
| `doc_id` | `str` | Primary key identifier (e.g. `FAQ-02`, `POLICY-01`, `TICKET-14`). | Non-null for all 40 chunks. |
| `doc_type` | `Literal["faq", "policy", "ticket"]` | Source classification. | 15 FAQs, 10 Policies, 15 Tickets. |
| `topic` | `str` | Short topic summary parsed from markdown headings. | Non-null for all 40 chunks. |
| `has_deprecated_content` | `bool` | True if the chunk discusses superseded or conflicting rules. | Flagged `True` on exactly 11 chunks (9 policies, `TICKET-15`, and `FAQ-07`). |
| `last_reviewed` | `Optional[str]` | Plain-text review or effective date string (e.g. `"February 2026"`, `"January 2026"`). | **Important:** Null for all 15 FAQs, null for all 15 Tickets, and null for `POLICY-05`, `POLICY-08`, and `POLICY-10`. Present as extracted month/year plain text on only 7 out of 10 policies. |
| `source_file` | `str` | Source markdown filename (`faqs.md`, `policies.md`, `tickets.md`). | Non-null for all 40 chunks. |

---

## 2. Chroma Vector Database Storage Format

The Chroma vector store is persisted locally in `./chroma_db` under collection `learnforge_kb`.

### Collection Configuration
- **Collection Name**: `learnforge_kb`
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (via `langchain-huggingface`)
- **Vector Dimensionality**: 384 dimensions (float32, unit-normalized $\ell_2 = 1.0$)
- **HNSW Distance Space**: `hnsw:space = "l2"` (squared Euclidean distance)

### Storage Layout
For each of the 40 indexed chunks (example showing actual stored `POLICY-02` document):
```json
{
  "ids": ["POLICY-02"],
  "embeddings": [0.0341, -0.0512, 0.0198, "... 384 float values ..."],
  "documents": ["# POLICY-02 — Cancellation and Refund Policy\n\nYou can cancel an active subscription at any time through Account Settings where self-service cancellation is available. Cancellation normally prevents the next renewal but does not automatically terminate access immediately.\n\nUnless otherwise stated at checkout, users generally retain subscription access until the end of the already-paid billing period. Canceling immediately after a renewal therefore does not necessarily result in an automatic refund.\n\nRefund requests are evaluated separately from cancellation requests. Eligible requests should include the account email and relevant order or transaction information.\n\nFor individual course purchases, LearnForge's standard refund period is generally 14 days, subject to course-consumption restrictions and applicable local law. Certain promotional products, bundles, and third-party purchases may have separate terms.\n\nWhere a technical problem materially prevented access to purchased content, Support may review an exception even when the normal refund period has passed.\n\nOlder help-center documentation referenced a 7-day refund period for all digital products. That article remains accessible in some archived search results but should not be treated as the current standard policy.\n\nRefunds are normally returned to the original payment method. Processing time depends on the payment provider.\n\nEffective date: January 2026."],
  "metadatas": {
    "doc_id": "POLICY-02",
    "doc_type": "policy",
    "topic": "Cancellation and Refund Policy",
    "has_deprecated_content": true,
    "last_reviewed": "January 2026",
    "source_file": "policies.md"
  }
}
```

---

## 3. Retriever Output Schema (`src/retriever.py`)

The `retrieve(query: str, top_k: int = 4)` function queries the vector store and returns a list of dictionaries with normalized scores.

```python
{
    "doc_id": str,                  # e.g. "POLICY-02"
    "doc_type": str,                # "policy" | "faq" | "ticket"
    "topic": str,                   # e.g. "Cancellation and Refund Policy"
    "text": str,                    # Full raw text of the chunk
    "score": float,                 # Normalized cosine similarity: [0.0, 1.0], higher = better
    "has_deprecated_content": bool, # Stale policy flag (True on 11 chunks)
    "last_reviewed": str | None,    # Plain-text review/effective date string or None
    "source_file": str              # e.g. "policies.md"
}
```

### Score Normalization Formula
Chroma returns raw squared Euclidean distance ($d^2 = \|u - v\|_2^2$). Because `all-MiniLM-L6-v2` embeddings are unit-normalized ($\|u\|_2 = 1$ and $\|v\|_2 = 1$):

$$\|u - v\|_2^2 = \|u\|_2^2 + \|v\|_2^2 - 2 \langle u, v \rangle = 1 + 1 - 2 \cos(u, v) = 2(1 - \cos(u, v))$$

Solving for cosine similarity $\cos(u, v)$:

$$\cos(u, v) = 1.0 - \frac{d^2}{2.0}$$

This directly maps Chroma's squared distance to standard cosine similarity:
- Raw distance $0.0 \implies \text{similarity } 1.0$ (identical vectors)
- Raw distance $2.0 \implies \text{similarity } 0.0$ (orthogonal vectors)
- The resulting value is clamped to $[0.0, 1.0]$ and rounded to 4 decimal places.

---

## 4. Session & Conversation History Schema (`src/conversation.py`)

Session state is held in an in-memory dictionary `_sessions: Dict[str, Dict[str, Any]]` keyed by a hex `session_id`.

```python
{
    "0eccdd3550a8466f85caada6c55c7b8b": {
        "created_at": "2026-09-19T04:47:36.823411+00:00",  # ISO 8601 UTC timestamp
        "history": [
            {
                "role": "user",
                "content": "Can I get a refund?",
                "sources": null
            },
            {
                "role": "assistant",
                "content": "Yes. LearnForge generally refunds eligible course purchases made within 14 days of purchase...",
                "sources": ["FAQ-02", "POLICY-02"]
            },
            {
                "role": "user",
                "content": "I don't recognize a $79 charge though",
                "sources": null
            },
            {
                "role": "assistant",
                "content": "Because this involves an unrecognized transaction on your payment card, I've escalated this directly to our billing support team...",
                "sources": []
            }
        ]
    }
}
```

---

## 5. API Request & Response Schemas (`app.py`)

All API interactions are governed by Pydantic v2 models, ensuring automatic OpenAPI specification generation and runtime payload validation:

### 1. `POST /session` -> `SessionResponse`
```json
{
  "session_id": "0eccdd3550a8466f85caada6c55c7b8b"
}
```

### 2. `POST /chat` -> `ChatRequest` (Input)
```json
{
  "session_id": "0eccdd3550a8466f85caada6c55c7b8b",
  "message": "Can I get a refund?"
}
```
*Validation Rules*:
- `session_id`: Required, minimum length 1. Raises HTTP 404 if not found in session registry.
- `message`: Required, minimum length 1. Validated via `@field_validator` to reject empty or whitespace-only inputs with HTTP 422 Unprocessable Entity.

### 3. `POST /chat` -> `ChatResponse` (Output)
```json
{
  "answer": "Yes. LearnForge generally refunds eligible course purchases made within 14 days of purchase...",
  "source_ids": ["FAQ-02", "POLICY-02"],
  "escalated": false,
  "escalation_category": null,
  "confidence": "high"
}
```
*Fields*:
- `answer` (`str`): Grounded response or human escalation notification.
- `source_ids` (`List[str]`): List of cited document IDs (empty when escalated before generation).
- `escalated` (`bool`): Boolean flag indicating whether the turn was handed off to a human agent.
- `escalation_category` (`Optional[str]`): Category of escalation (`"fraud_or_security"`, `"low_confidence_retrieval"`, `"llm_flagged"`, `"disputed_policy_terms"`, or `null`).
- `confidence` (`Optional[str]`): `"high"`, `"medium"`, `"low"`, or `null` (when escalated pre-retrieval).

### 4. `GET /chat/{session_id}/history` -> `List[HistoryEntry]`
```json
[
  {
    "role": "user",
    "content": "Can I get a refund?",
    "sources": null
  },
  {
    "role": "assistant",
    "content": "Yes. LearnForge generally refunds eligible course purchases...",
    "sources": ["FAQ-02", "POLICY-02"]
  }
]
```

### 5. `GET /health` -> `HealthResponse`
```json
{
  "status": "ok"
}
```
