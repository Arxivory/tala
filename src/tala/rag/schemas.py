
"""Shared data schemas for TALA's evidence-grounded RAG pipeline.

These schemas deliberately avoid LangChain-specific types. LangChain and Chroma
are implementation details; the rest of TALA should depend on these interfaces.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


ChromaScalar = str | int | float | bool


@dataclass(frozen=True, slots=True)
class DocumentMetadata:
    """Provenance and classification metadata for a knowledge-base chunk."""

    source_id: str
    title: str
    source_type: str
    domain: str
    population: str = "unspecified"
    environment: str = "unspecified"
    evidence_role: str = "background"
    revision: str = ""
    publication_date: str = ""
    section: str = ""
    page_start: int = 0
    page_end: int = 0
    topics: tuple[str, ...] = ()
    restrictions: tuple[str, ...] = ()
    source_path: str = ""
    chunk_id: str = ""

    def __post_init__(self) -> None:
        required = {
            "source_id": self.source_id,
            "title": self.title,
            "source_type": self.source_type,
            "domain": self.domain,
        }

        for field_name, value in required.items():
            if not value.strip():
                raise ValueError(f"{field_name} must not be empty.")

        if self.page_start < 0 or self.page_end < 0:
            raise ValueError("Page numbers cannot be negative.")

        if self.page_start and self.page_end:
            if self.page_end < self.page_start:
                raise ValueError("page_end cannot precede page_start.")

    def to_chroma_metadata(self) -> dict[str, ChromaScalar]:
        """Convert metadata to Chroma-compatible scalar values.

        Chroma metadata does not accept arbitrary Python tuples or dictionaries,
        so topics and restrictions are stored as JSON strings.
        """
        return {
            "source_id": self.source_id,
            "title": self.title,
            "source_type": self.source_type,
            "domain": self.domain,
            "population": self.population,
            "environment": self.environment,
            "evidence_role": self.evidence_role,
            "revision": self.revision,
            "publication_date": self.publication_date,
            "section": self.section,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "topics": json.dumps(self.topics),
            "restrictions": json.dumps(self.restrictions),
            "source_path": self.source_path,
            "chunk_id": self.chunk_id,
        }

    @classmethod
    def from_chroma_metadata(
        cls,
        metadata: Mapping[str, Any],
    ) -> DocumentMetadata:
        """Reconstruct document metadata returned by Chroma."""
        return cls(
            source_id=str(metadata["source_id"]),
            title=str(metadata["title"]),
            source_type=str(metadata["source_type"]),
            domain=str(metadata["domain"]),
            population=str(metadata.get("population", "unspecified")),
            environment=str(metadata.get("environment", "unspecified")),
            evidence_role=str(metadata.get("evidence_role", "background")),
            revision=str(metadata.get("revision", "")),
            publication_date=str(metadata.get("publication_date", "")),
            section=str(metadata.get("section", "")),
            page_start=int(metadata.get("page_start", 0)),
            page_end=int(metadata.get("page_end", 0)),
            topics=tuple(json.loads(metadata.get("topics", "[]"))),
            restrictions=tuple(
                json.loads(metadata.get("restrictions", "[]"))
            ),
            source_path=str(metadata.get("source_path", "")),
            chunk_id=str(metadata.get("chunk_id", "")),
        )


@dataclass(frozen=True, slots=True)
class RetrievedEvidence:
    """A retrieved passage with its original source provenance."""

    content: str
    metadata: DocumentMetadata
    distance: float | None = None

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("Evidence content must not be empty.")


@dataclass(frozen=True, slots=True)
class RAGRequest:
    """Input to the TALA RAG service."""

    question: str
    assessment_context: Mapping[str, Any] = field(default_factory=dict)
    top_k: int = 5
    filters: Mapping[str, ChromaScalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("Question must not be empty.")

        if self.top_k < 1:
            raise ValueError("top_k must be at least 1.")

        allowed_filter_values = (str, int, float, bool)
        for key, value in self.filters.items():
            if not isinstance(value, allowed_filter_values):
                raise TypeError(
                    f"Filter '{key}' must have a scalar value, "
                    f"not {type(value).__name__}."
                )


@dataclass(frozen=True, slots=True)
class RAGResponse:
    """Structured output from the TALA RAG service.

    The answer explains existing assessment information. It does not create
    or modify assessment flags, diagnoses, or readiness classifications.
    """

    answer: str
    evidence: tuple[RetrievedEvidence, ...] = ()
    limitations: tuple[str, ...] = ()
    abstained: bool = False
    abstention_reason: str = ""

    def __post_init__(self) -> None:
        if self.abstained and not self.abstention_reason:
            raise ValueError(
                "An abstained response must include an abstention_reason."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)
