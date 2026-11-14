
"""Local PDF ingestion, chunking, embedding, and Chroma persistence for TALA."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import chromadb
import ollama
from pypdf import PdfReader

from tala.rag.schemas import DocumentMetadata


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_KNOWLEDGE_BASE = PROJECT_ROOT / "data" / "knowledge_base"
DEFAULT_RAW_DIR = DEFAULT_KNOWLEDGE_BASE / "raw"
DEFAULT_MANIFEST = DEFAULT_KNOWLEDGE_BASE / "source_manifest.json"
DEFAULT_DB_DIR = DEFAULT_KNOWLEDGE_BASE / "chroma_db"

DEFAULT_COLLECTION = "tala_nasa_knowledge"
DEFAULT_EMBEDDING_MODEL = "qwen3-embedding:0.6b"

# Approximate token-to-character conversion. Chunk boundaries are adjusted
# to paragraph boundaries where possible; this is not a tokenizer.
DEFAULT_CHUNK_TOKENS = 550
DEFAULT_OVERLAP_TOKENS = 80
CHARS_PER_TOKEN = 4


@dataclass(frozen=True, slots=True)
class ExtractedPage:
    """Text extracted from one PDF page."""

    page_number: int
    text: str


@dataclass(frozen=True, slots=True)
class TextChunk:
    """A text chunk with its original page range."""

    content: str
    page_start: int
    page_end: int


@dataclass(frozen=True, slots=True)
class IngestionReport:
    """Summary of a knowledge-base ingestion run."""

    documents_seen: int
    documents_ingested: int
    pages_extracted: int
    empty_pages: int
    chunks_created: int
    chunks_upserted: int
    failed_documents: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_manifest(manifest_path: Path = DEFAULT_MANIFEST) -> list[dict[str, Any]]:
    """Load and minimally validate the source manifest."""
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Source manifest not found: {manifest_path}"
        )

    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    documents = manifest.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError(
            "Manifest must contain a non-empty 'documents' list."
        )

    required_fields = {
        "filename",
        "source_id",
        "title",
        "source_type",
        "domain",
    }

    seen_source_ids: set[str] = set()
    seen_filenames: set[str] = set()

    for index, document in enumerate(documents):
        if not isinstance(document, dict):
            raise ValueError(f"Manifest document {index} must be an object.")

        missing = required_fields - document.keys()
        if missing:
            raise ValueError(
                f"Manifest document {index} is missing: {sorted(missing)}"
            )

        filename = str(document["filename"])
        source_id = str(document["source_id"])

        if Path(filename).name != filename:
            raise ValueError(
                f"Use a filename only, not a path: {filename}"
            )

        if source_id in seen_source_ids:
            raise ValueError(f"Duplicate source_id: {source_id}")

        if filename in seen_filenames:
            raise ValueError(f"Duplicate filename: {filename}")

        seen_source_ids.add(source_id)
        seen_filenames.add(filename)

    return documents


def extract_pdf_pages(pdf_path: Path) -> tuple[list[ExtractedPage], list[str]]:
    """Extract text from a PDF while retaining one-based page numbers.

    Returns extracted pages and warnings for pages with no extractable text.
    This does not perform OCR.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))

    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:
            raise ValueError(
                f"PDF is encrypted and could not be opened: {pdf_path.name}"
            ) from exc

    pages: list[ExtractedPage] = []
    warnings: list[str] = []

    for page_index, page in enumerate(reader.pages):
        page_number = page_index + 1

        try:
            text = page.extract_text() or ""
        except Exception as exc:
            warnings.append(
                f"{pdf_path.name}, page {page_number}: extraction failed "
                f"({type(exc).__name__})."
            )
            continue

        normalized = normalize_text(text)

        if not normalized:
            warnings.append(
                f"{pdf_path.name}, page {page_number}: no extractable text; "
                "the page may be scanned or image-based."
            )
            continue

        pages.append(
            ExtractedPage(
                page_number=page_number,
                text=normalized,
            )
        )

    if not pages:
        warnings.append(
            f"{pdf_path.name}: no text was extracted. OCR or manual review "
            "is required before indexing this document."
        )

    return pages, warnings


def normalize_text(text: str) -> str:
    """Normalize extracted PDF text without removing paragraph boundaries."""
    text = text.replace("\x00", " ")
    text = text.replace("\u00ad", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _paragraphs_from_pages(
    pages: Iterable[ExtractedPage],
) -> list[tuple[int, str]]:
    """Convert pages into paragraph units associated with their page number."""
    paragraphs: list[tuple[int, str]] = []

    for page in pages:
        for paragraph in re.split(r"\n\s*\n", page.text):
            cleaned = paragraph.strip()
            if cleaned:
                paragraphs.append((page.page_number, cleaned))

    return paragraphs


def chunk_pages(
    pages: list[ExtractedPage],
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[TextChunk]:
    """Create approximate token-sized chunks while preserving page ranges.

    Paragraphs are kept together when possible. Oversized paragraphs are split
    into smaller character windows. Overlap is retained between chunks.
    """
    if chunk_tokens < 1:
        raise ValueError("chunk_tokens must be positive.")

    if overlap_tokens < 0 or overlap_tokens >= chunk_tokens:
        raise ValueError(
            "overlap_tokens must be non-negative and smaller than chunk_tokens."
        )

    max_chars = chunk_tokens * CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * CHARS_PER_TOKEN

    paragraphs = _paragraphs_from_pages(pages)
    units: list[tuple[int, str]] = []

    for page_number, paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            units.append((page_number, paragraph))
            continue

        # Split long paragraphs at whitespace where possible.
        remaining = paragraph
        while len(remaining) > max_chars:
            split_at = remaining.rfind(" ", 0, max_chars)
            if split_at < max_chars // 2:
                split_at = max_chars

            units.append((page_number, remaining[:split_at].strip()))
            remaining = remaining[split_at:].strip()

        if remaining:
            units.append((page_number, remaining))

    chunks: list[TextChunk] = []
    current_parts: list[tuple[int, str]] = []
    current_length = 0

    for page_number, text in units:
        addition_length = len(text) + (2 if current_parts else 0)

        if current_parts and current_length + addition_length > max_chars:
            chunks.append(
                TextChunk(
                    content="\n\n".join(part for _, part in current_parts),
                    page_start=min(page for page, _ in current_parts),
                    page_end=max(page for page, _ in current_parts),
                )
            )

            # Retain a trailing overlap from the previous chunk.
            previous_text = "\n\n".join(
                part for _, part in current_parts
            )
            previous_page = current_parts[-1][0]
            overlap_text = previous_text[-overlap_chars:] if overlap_chars else ""

            current_parts = (
                [(previous_page, overlap_text)] if overlap_text else []
            )
            current_length = len(overlap_text)

        current_parts.append((page_number, text))
        current_length += addition_length

    if current_parts:
        chunks.append(
            TextChunk(
                content="\n\n".join(part for _, part in current_parts),
                page_start=min(page for page, _ in current_parts),
                page_end=max(page for page, _ in current_parts),
            )
        )

    return [
        chunk for chunk in chunks
        if chunk.content.strip()
    ]


def _build_metadata(
    source: dict[str, Any],
    chunk: TextChunk,
    chunk_id: str,
    pdf_path: Path,
) -> DocumentMetadata:
    """Create provenance metadata for one chunk."""
    return DocumentMetadata(
        source_id=str(source["source_id"]),
        title=str(source["title"]),
        source_type=str(source["source_type"]),
        domain=str(source["domain"]),
        population=str(source.get("population", "unspecified")),
        environment=str(source.get("environment", "unspecified")),
        evidence_role=str(source.get("evidence_role", "background")),
        revision=str(source.get("revision", "")),
        publication_date=str(source.get("publication_date", "")),
        section=str(source.get("section", "")),
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        topics=tuple(source.get("topics", [])),
        restrictions=tuple(source.get("restrictions", [])),
        source_path=str(pdf_path),
        chunk_id=chunk_id,
    )


def _make_chunk_id(
    source_id: str,
    chunk: TextChunk,
) -> str:
    """Generate a deterministic ID for idempotent Chroma upserts."""
    identity = (
        f"{source_id}|{chunk.page_start}|{chunk.page_end}|{chunk.content}"
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return f"{source_id}_{digest[:24]}"


def _embed_texts(
    client: ollama.Client,
    texts: list[str],
    model: str,
) -> list[list[float]]:
    """Generate embeddings through the local Ollama server."""
    response = client.embed(
        model=model,
        input=texts,
    )

    embeddings = (
        response.get("embeddings")
        if isinstance(response, dict)
        else getattr(response, "embeddings", None)
    )
    if embeddings is None:
        raise RuntimeError(
            "Ollama returned no embeddings. Check the embedding model name "
            "and Ollama server."
        )

    if len(embeddings) != len(texts):
        raise RuntimeError(
            "Ollama returned a different number of embeddings than inputs."
        )

    return embeddings


def ingest_knowledge_base(
    raw_dir: Path = DEFAULT_RAW_DIR,
    manifest_path: Path = DEFAULT_MANIFEST,
    persist_dir: Path = DEFAULT_DB_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    batch_size: int = 32,
) -> IngestionReport:
    """Extract, chunk, embed, and upsert all manifest-listed PDFs."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    sources = load_manifest(manifest_path)

    persist_dir.mkdir(parents=True, exist_ok=True)

    chroma_client = chromadb.PersistentClient(path=str(persist_dir))
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    ollama_client = ollama.Client()

    pages_extracted = 0
    empty_pages = 0
    chunks_created = 0
    chunks_upserted = 0
    documents_ingested = 0
    failed_documents: list[str] = []
    all_warnings: list[str] = []

    for source in sources:
        filename = str(source["filename"])
        pdf_path = raw_dir / filename

        LOGGER.info("Processing %s", filename)

        try:
            pages, warnings = extract_pdf_pages(pdf_path)
            all_warnings.extend(warnings)

            # Count empty pages from extraction warnings.
            empty_pages += sum(
                1 for warning in warnings
                if "no extractable text" in warning
            )
            pages_extracted += len(pages)

            if not pages:
                failed_documents.append(filename)
                continue

            chunks = chunk_pages(
                pages,
                chunk_tokens=chunk_tokens,
                overlap_tokens=overlap_tokens,
            )

            if not chunks:
                all_warnings.append(
                    f"{filename}: extraction succeeded but produced no chunks."
                )
                failed_documents.append(filename)
                continue

            chunk_records: list[tuple[str, str, dict[str, Any]]] = []

            for chunk in chunks:
                chunk_id = _make_chunk_id(
                    str(source["source_id"]),
                    chunk,
                )
                metadata = _build_metadata(
                    source,
                    chunk,
                    chunk_id,
                    pdf_path,
                )

                chunk_records.append(
                    (
                        chunk_id,
                        chunk.content,
                        metadata.to_chroma_metadata(),
                    )
                )

            chunks_created += len(chunk_records)

            for start in range(0, len(chunk_records), batch_size):
                batch = chunk_records[start : start + batch_size]

                ids = [record[0] for record in batch]
                texts = [record[1] for record in batch]
                metadatas = [record[2] for record in batch]

                embeddings = _embed_texts(
                    ollama_client,
                    texts,
                    embedding_model,
                )

                collection.upsert(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas,
                    embeddings=embeddings,
                )

                chunks_upserted += len(batch)

            documents_ingested += 1
            LOGGER.info(
                "Ingested %s: %d pages, %d chunks",
                filename,
                len(pages),
                len(chunks),
            )

        except Exception as exc:
            failed_documents.append(filename)
            all_warnings.append(
                f"{filename}: ingestion failed: "
                f"{type(exc).__name__}: {exc}"
            )
            LOGGER.exception("Failed to ingest %s", filename)

    return IngestionReport(
        documents_seen=len(sources),
        documents_ingested=documents_ingested,
        pages_extracted=pages_extracted,
        empty_pages=empty_pages,
        chunks_created=chunks_created,
        chunks_upserted=chunks_upserted,
        failed_documents=tuple(failed_documents),
        warnings=tuple(all_warnings),
    )
