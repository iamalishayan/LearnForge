"""Retriever module for LearnForge knowledge base.

Searches the persisted Chroma vector store using local embeddings and converts
raw distance scores into normalized similarity scores (0.0 to 1.0, higher is better).
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path for direct execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

try:
    from src.ingest import COLLECTION_NAME, DEFAULT_CHROMA_DIR, EMBEDDING_MODEL_NAME
except ImportError:
    from ingest import COLLECTION_NAME, DEFAULT_CHROMA_DIR, EMBEDDING_MODEL_NAME

# Confidence thresholds for downstream confidence and escalation logic (Task 5)
HIGH_CONFIDENCE_THRESHOLD = 0.55
LOW_CONFIDENCE_THRESHOLD = 0.35

# Module-level singletons for lazy-loaded embeddings and vector store
_embeddings: Optional[HuggingFaceEmbeddings] = None
_vector_store: Optional[Chroma] = None


def get_vector_store(
    chroma_dir: Path = DEFAULT_CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    """Load and return the persisted Chroma vector store.

    Fails loudly if the persistence directory does not exist or the collection is empty.
    """
    global _embeddings, _vector_store

    if _vector_store is not None:
        return _vector_store

    # Requirement 5: Fail loudly if ./chroma_db does not exist
    if not chroma_dir.exists() or not chroma_dir.is_dir():
        raise FileNotFoundError(
            f"Chroma persistence directory '{chroma_dir}' not found. "
            "Please run 'python src/ingest.py' to ingest and embed knowledge base documents first."
        )

    client = chromadb.PersistentClient(path=str(chroma_dir))
    existing_collections = [c.name for c in client.list_collections()]

    if collection_name not in existing_collections:
        raise ValueError(
            f"Collection '{collection_name}' does not exist in '{chroma_dir}'. "
            "Please run 'python src/ingest.py' to build the collection first."
        )

    col = client.get_collection(name=collection_name)
    if col.count() == 0:
        raise ValueError(
            f"Collection '{collection_name}' in '{chroma_dir}' is empty (0 documents). "
            "Please run 'python src/ingest.py' to ingest documents first."
        )

    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    _vector_store = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=_embeddings,
    )
    return _vector_store


def normalize_chroma_score(raw_score: float) -> float:
    """Convert Chroma raw distance into a normalized similarity score (0.0 to 1.0).

    Score Normalization Metric Details:
    Chroma's default distance metric for this collection is squared L2 distance (hnsw:space = "l2").
    For unit-normalized vector embeddings (as produced by sentence-transformers/all-MiniLM-L6-v2):
        ||u - v||^2 = ||u||^2 + ||v||^2 - 2 * <u, v> = 1 + 1 - 2 * cos_sim = 2 * (1 - cos_sim)
    Therefore:
        cos_sim = 1.0 - (squared_l2 / 2.0)
    We clamp the result to [0.0, 1.0] and round to 4 decimal places.
    A score of 1.0 means identical, and 0.0 means completely dissimilar / orthogonal / opposite.
    """
    cosine_sim = 1.0 - (raw_score / 2.0)
    return round(max(0.0, min(1.0, cosine_sim)), 4)


def retrieve(query: str, top_k: int = 4) -> List[Dict[str, Any]]:
    """Retrieve top-k relevant knowledge base chunks for a user query.

    Args:
        query: User search or support question string.
        top_k: Number of most similar chunks to return (default: 4).

    Returns:
        List of dictionaries with keys:
            - doc_id: Unique document identifier (e.g., FAQ-01, POLICY-02)
            - doc_type: Type of document ("faq" | "policy" | "ticket")
            - topic: Short topic label
            - text: Full text content of the chunk
            - score: Normalized similarity score in [0.0, 1.0] (higher is better)
            - has_deprecated_content: Boolean flag for stale/conflicting information
            - last_reviewed: Review or effective date string, or None
            - source_file: Origin filename (e.g., policies.md)
    """
    if not query or not query.strip():
        return []

    vector_store = get_vector_store()
    results = vector_store.similarity_search_with_score(query.strip(), k=top_k)

    retrieved_chunks: List[Dict[str, Any]] = []
    for doc, raw_score in results:
        meta = doc.metadata or {}
        similarity_score = normalize_chroma_score(raw_score)

        retrieved_chunks.append(
            {
                "doc_id": meta.get("doc_id", "UNKNOWN"),
                "doc_type": meta.get("doc_type", "unknown"),
                "topic": meta.get("topic", "Unknown"),
                "text": doc.page_content,
                "score": similarity_score,
                "has_deprecated_content": meta.get("has_deprecated_content", False),
                "last_reviewed": meta.get("last_reviewed"),
                "source_file": meta.get("source_file", ""),
            }
        )

    return retrieved_chunks


if __name__ == "__main__":
    test_queries = [
        "How do I get a refund?",
        "certificate missing after finishing course",
        "I don't recognize a charge on my card",
        "can I download courses to watch offline",
        "what is the interest rate on financing",
    ]

    print("=" * 70)
    print("RETRIEVER MANUAL VERIFICATION TEST")
    print(f"HIGH_CONFIDENCE_THRESHOLD: {HIGH_CONFIDENCE_THRESHOLD}")
    print(f"LOW_CONFIDENCE_THRESHOLD:  {LOW_CONFIDENCE_THRESHOLD}")
    print("=" * 70)

    for i, q in enumerate(test_queries, 1):
        print(f"\n[Query {i}] \"{q}\"")
        results = retrieve(q, top_k=4)

        for rank, r in enumerate(results, 1):
            flag_str = " [DEPRECATED]" if r["has_deprecated_content"] else ""
            print(
                f"  #{rank} Score: {r['score']:.4f} | "
                f"Doc ID: {r['doc_id']:<10} | "
                f"Type: {r['doc_type']:<6} | "
                f"Topic: {r['topic']}{flag_str}"
            )

        top_score = results[0]["score"] if results else 0.0
        if top_score >= HIGH_CONFIDENCE_THRESHOLD:
            status = "HIGH CONFIDENCE"
        elif top_score >= LOW_CONFIDENCE_THRESHOLD:
            status = "MEDIUM CONFIDENCE"
        else:
            status = "LOW CONFIDENCE (Escalation Trigger)"
        print(f"  --> Top Score: {top_score:.4f} => Category: {status}")

    print("\n" + "=" * 70)
