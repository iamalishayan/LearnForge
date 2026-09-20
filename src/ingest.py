"""Data ingestion pipeline for LearnForge knowledge base.

Parses markdown knowledge base documents, chunks them by entry, extracts
metadata, generates local embeddings, and stores them in a local Chroma vector store.
"""

import sys
from pathlib import Path

# Ensure project root is in Python path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from collections import Counter
import os
import re
from typing import Dict, List, Optional

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

try:
    from src.schema import ChunkMetadata
except ImportError:
    from schema import ChunkMetadata

# Directory defaults
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "learnforge_kb"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Specific patterns signaling explicit conflict between an older/archived version
# of documentation/policy and current rules (e.g., 'An older version...', 'Older documentation...',
# 'That article appears to contain older instructions', 'no longer a universal requirement', etc.)
DEPRECATED_PATTERNS = [
    r"older\s+(?:version|help|article|documentation|guide|internal|instruction)",
    r"previous\s+(?:version|mobile\s+help)",
    r"archived\s+documentation",
    r"is\s+no\s+longer\s+(?:a\s+universal\s+requirement|offered)",
    r"(?:information|instruction)\s+is\s+obsolete",
    r"has\s+been\s+retired",
    r"instructions\s+are\s+outdated",
    r"outdated\s+wording",
]


def detect_deprecated_content(text: str) -> bool:
    """Check if document text explicitly describes a conflict with an older/stale version."""
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in DEPRECATED_PATTERNS)


def extract_last_reviewed(text: str, doc_type: str) -> Optional[str]:
    """Extract review or effective date if present (primarily on policy documents)."""
    if doc_type != "policy":
        return None

    # Matches phrases like 'Last reviewed: February 2026.', 'Effective date: January 2026.', etc.
    match = re.search(
        r"(?:Last\s+reviewed|Reviewed|Last\s+updated|Updated|Effective\s+date|Effective):\s*([^.\n]+)",
        text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return None


# Header regex: matches '# FAQ-01 — Topic', '# POLICY-02 - Topic', '# TICKET-03 — Topic'
HEADER_PATTERN = re.compile(
    r"^#\s+((FAQ|POLICY|TICKET)-\d+)\s+[-—–]+\s*(.+)$",
    re.MULTILINE,
)


def strip_orphan_section_headers(content: str) -> str:
    """Strip trailing orphan section headers so they don't leak into chunks."""
    return re.sub(r"SECTION\s+\d+.*", "", content, flags=re.IGNORECASE).strip()


def count_headers_in_text(text: str, doc_type: str) -> int:
    """Count raw header occurrences for a given doc_type prefix in text."""
    clean_text = strip_orphan_section_headers(text)
    prefix = doc_type.upper()
    return sum(
        1 for match in HEADER_PATTERN.finditer(clean_text)
        if match.group(2).upper() == prefix
    )


def parse_markdown_file(file_path: Path) -> List[Document]:
    """Parse a knowledge base markdown file into individual entry chunks with metadata.

    Each entry starting with '# FAQ-XX', '# POLICY-XX', or '# TICKET-XX' up to
    the '---' separator is treated as an individual chunk.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Knowledge base file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Requirement: Strip trailing orphan section headers so they don't leak into chunks
    # (e.g. 'SECTION 2 — POLICY / HELP-CENTER DOCUMENT EXCERPTS' at end of faqs.md,
    # and 'SECTION 3 — PAST SUPPORT TICKET TRANSCRIPTS' at end of policies.md)
    content = strip_orphan_section_headers(content)

    # Split document by '---' separators
    raw_sections = [s.strip() for s in re.split(r"(?m)^---+\s*$", content) if s.strip()]

    documents: List[Document] = []

    for section in raw_sections:
        match = HEADER_PATTERN.search(section)
        if not match:
            # Skips file-level title headings (e.g., '# LearnForge FAQs')
            continue

        doc_id = match.group(1).strip()
        doc_type = match.group(2).strip().lower()
        topic = match.group(3).strip()

        # Chunk text starts from the '# DOC_ID' header to the end of the section
        chunk_text = section[match.start() :].strip()

        has_deprecated = detect_deprecated_content(chunk_text)
        last_reviewed = extract_last_reviewed(chunk_text, doc_type)

        # Validate through schema
        metadata_model = ChunkMetadata(
            doc_id=doc_id,
            doc_type=doc_type,
            topic=topic,
            has_deprecated_content=has_deprecated,
            last_reviewed=last_reviewed,
            source_file=file_path.name,
        )

        metadata_dict = metadata_model.model_dump()

        doc = Document(
            page_content=chunk_text,
            metadata=metadata_dict,
        )
        documents.append(doc)

    return documents


def load_all_documents(data_dir: Path = DEFAULT_DATA_DIR) -> List[Document]:
    """Load and chunk all three knowledge base files, validating derived chunk counts."""
    files_to_load = [
        ("faqs.md", "faq"),
        ("policies.md", "policy"),
        ("tickets.md", "ticket"),
    ]

    all_docs: List[Document] = []
    total_expected_count = 0

    for filename, doc_type in files_to_load:
        file_path = data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Knowledge base file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        expected_count = count_headers_in_text(raw_text, doc_type)
        total_expected_count += expected_count

        docs = parse_markdown_file(file_path)
        actual_count = len(docs)

        if actual_count != expected_count:
            raise ValueError(
                f"Unexpected chunk count for {filename}: expected {expected_count}, got {actual_count}"
            )

        all_docs.extend(docs)

    if len(all_docs) != total_expected_count:
        raise ValueError(
            f"Expected {total_expected_count} total chunks across all knowledge base files, "
            f"got {len(all_docs)}"
        )

    return all_docs


def ingest_knowledge_base(
    data_dir: Path = DEFAULT_DATA_DIR,
    chroma_dir: Path = DEFAULT_CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    """Ingest documents into Chroma vector store with idempotency."""
    print("=" * 60)
    print("LEARNFORGE KNOWLEDGE BASE INGESTION")
    print("=" * 60)
    print(f"Data directory:   {data_dir}")
    print(f"Chroma directory: {chroma_dir}")
    print(f"Collection:       {collection_name}")
    print(f"Embedding model:  {EMBEDDING_MODEL_NAME}")
    print("-" * 60)

    # 1. Parse and chunk documents
    docs = load_all_documents(data_dir)
    print(f"Successfully parsed and validated {len(docs)} chunks from knowledge base.")

    # 2. Initialize embeddings model (local HuggingFace sentence-transformers)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # 3. Initialize persistent Chroma client for idempotent ingestion
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))

    # Idempotency: Remove existing collection if present to avoid duplication
    existing_collections = [c.name for c in client.list_collections()]
    if collection_name in existing_collections:
        print(f"Existing collection '{collection_name}' found. Resetting for idempotent ingestion...")
        client.delete_collection(name=collection_name)

    # 4. Create collection and store documents
    vector_store = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    doc_ids = [doc.metadata["doc_id"] for doc in docs]
    print(f"Embedding and storing {len(docs)} chunks into Chroma vector store...")
    vector_store.add_documents(documents=docs, ids=doc_ids)
    print("Ingestion complete.")
    print("-" * 60)

    # 5. Verification output
    print_verification_report(docs, vector_store)

    return vector_store


def print_verification_report(docs: List[Document], vector_store: Chroma) -> None:
    """Print the required verification metrics and flagged deprecated chunks."""
    print("VERIFICATION REPORT")
    print("-" * 60)

    total_count = len(docs)
    counts_by_type = Counter(doc.metadata["doc_type"] for doc in docs)

    print(f"Total chunk count: {total_count} (Expected: 40)")
    print("Counts per doc_type:")
    for doc_type in ["faq", "policy", "ticket"]:
        print(f"  - {doc_type:6s}: {counts_by_type.get(doc_type, 0)}")

    flagged_chunks = [
        doc for doc in docs if doc.metadata.get("has_deprecated_content") is True
    ]

    print(f"\nChunks with has_deprecated_content = True ({len(flagged_chunks)} flagged):")
    print(f"{'Doc ID':<12} {'Type':<8} {'Last Reviewed':<18} {'Topic'}")
    print("-" * 60)
    for doc in flagged_chunks:
        meta = doc.metadata
        rev = meta.get("last_reviewed") or "None"
        print(f"{meta['doc_id']:<12} {meta['doc_type']:<8} {rev:<18} {meta['topic']}")

    print("=" * 60)


if __name__ == "__main__":
    ingest_knowledge_base()
