
"""Evidence retrieval from TALA's persistent Chroma knowledge base."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import chromadb
import ollama

import re

from tala.rag.schemas import (
    ChromaScalar,
    RAGRequest,
    RetrievedEvidence,
    DocumentMetadata,
)


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_KNOWLEDGE_BASE = PROJECT_ROOT / "data" / "knowledge_base"
DEFAULT_DB_DIR = DEFAULT_KNOWLEDGE_BASE / "chroma_db"

DEFAULT_COLLECTION = "tala_nasa_knowledge"
DEFAULT_EMBEDDING_MODEL = "qwen3-embedding:0.6b"


_REFERENCE_LINE_PATTERNS = (
    # Numbered bibliography entries, e.g. "57. Gopalakrishnan, R., et al."
    re.compile(r"^\s*\d{1,3}\.\s+[A-Z][A-Za-z-]+"),

    # Publication details, e.g. journal volume and page references
    re.compile(
        r"\b(19|20)\d{2}\b.*\b(vol|volume|pp|pages?)\.?\b",
        re.IGNORECASE,
    ),

    # DOI and DOI URLs
    re.compile(
        r"\b(doi:|https?://doi\.org/)",
        re.IGNORECASE,
    ),

    # Common journal-title patterns
    re.compile(
        r"\b(journal of|aviation, space|nature reviews|"
        r"european journal|applied physiology|"
        r"clinical journal|scientific reports)\b",
        re.IGNORECASE,
    ),
)


def _embed_query(
    client: ollama.Client,
    query: str,
    model: str,
) -> list[float]:
    """Embed one query using the same model as document ingestion."""
    response = client.embed(
        model=model,
        input=query,
    )

    embeddings = (
        response.get("embeddings")
        if isinstance(response, dict)
        else getattr(response, "embeddings", None)
    )

    if not embeddings or not embeddings[0]:
        raise RuntimeError(
            "Ollama returned no query embedding. Check that the embedding "
            f"model '{model}' is available and the Ollama server is running."
        )

    return embeddings[0]


def _build_where_filter(
    filters: Mapping[str, ChromaScalar],
) -> dict[str, Any] | None:
    """Convert simple equality filters into Chroma's where syntax."""
    if not filters:
        return None

    conditions = [
        {key: {"$eq": value}}
        for key, value in filters.items()
    ]

    if len(conditions) == 1:
        return conditions[0]

    return {"$and": conditions}

def _is_reference_heavy(content: str) -> bool:
    """Identify chunks dominated by bibliography entries."""

    lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip()
    ]

    if len(lines) < 4:
        return False

    reference_header = any(
        re.match(
            r"^(references|reference list|bibliography)\s*$",
            line,
            re.IGNORECASE,
        )
        for line in lines
    )

    if reference_header:
        return True

    reference_lines = sum(
        any(pattern.search(line) for pattern in _REFERENCE_LINE_PATTERNS)
        for line in lines
    )

    return reference_lines / len(lines) >= 0.30


def retrieve_evidence(
    request: RAGRequest,
    persist_dir: Path = DEFAULT_DB_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    client: ollama.Client | None = None,
    exclude_reference_chunks: bool = True,
) -> tuple[RetrievedEvidence, ...]:
    """Retrieve the most relevant evidence passages for a RAG request.

    Chroma returns cosine distances because the collection was created with
    hnsw:space='cosine'. Lower distances indicate closer matches.

    No similarity cutoff is imposed here. We will calibrate abstention using
    retrieval tests rather than inventing an arbitrary threshold.
    """
    if not persist_dir.exists():
        raise FileNotFoundError(
            f"Chroma database directory does not exist: {persist_dir}"
        )

    chroma_client = chromadb.PersistentClient(path=str(persist_dir))

    try:
        collection = chroma_client.get_collection(name=collection_name)
    except Exception as exc:
        raise RuntimeError(
            f"Chroma collection '{collection_name}' was not found. "
            "Run knowledge-base ingestion first."
        ) from exc

    collection_count = collection.count()

    if collection_count == 0:
        LOGGER.warning("Knowledge-base collection is empty.")
        return ()

    ollama_client = client or ollama.Client()

    query_embedding = _embed_query(
        client=ollama_client,
        query=request.question,
        model=embedding_model,
    )

    where = _build_where_filter(request.filters)

    query_args: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": min(request.top_k * 4, collection_count),
        "include": [
            "documents",
            "metadatas",
            "distances",
        ],
    }

    if where is not None:
        query_args["where"] = where

    results = collection.query(**query_args)

    documents = results.get("documents") or [[]]
    metadatas = results.get("metadatas") or [[]]
    distances = results.get("distances") or [[]]

    retrieved: list[RetrievedEvidence] = []

    for content, metadata, distance in zip(
        documents[0],
        metadatas[0],
        distances[0],
    ):
        if not content or not metadata:
            continue

        is_reference = _is_reference_heavy(content)

        print(
            "REFERENCE FILTER:",
            metadata.get("source_id"),
            f"pages {metadata.get('page_start')}-{metadata.get('page_end')}",
            f"excluded={is_reference}",
        )

        if exclude_reference_chunks and is_reference:
            continue

        document_metadata = DocumentMetadata.from_chroma_metadata(metadata)

        retrieved.append(
            RetrievedEvidence(
                content=content,
                metadata=document_metadata,
                distance=float(distance) if distance is not None else None,
            )
        )

    retrieved = retrieved[:request.top_k]

    LOGGER.info(
        "Retrieved %d evidence chunks for query: %s",
        len(retrieved),
        request.question[:120],
    )

    return tuple(retrieved)
