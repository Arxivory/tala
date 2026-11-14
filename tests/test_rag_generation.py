from types import SimpleNamespace
from unittest.mock import Mock

from google.genai import types

from tala.rag import generation
from tala.rag.schemas import (
    DocumentMetadata,
    RAGRequest,
    RetrievedEvidence,
)


def test_default_client_uses_bounded_generation_timeout(monkeypatch):
    evidence = RetrievedEvidence(
        content="Exercise helps preserve muscle strength in microgravity.",
        metadata=DocumentMetadata(
            source_id="source-1",
            title="Exercise in Space",
            source_type="research",
            domain="spaceflight",
            page_start=1,
            page_end=1,
        ),
    )
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(
        text="Exercise can help preserve strength. [Exercise in Space, p. 1]"
    )
    client_factory = Mock(return_value=client)

    monkeypatch.setattr(
        generation,
        "retrieve_evidence",
        Mock(return_value=(evidence,)),
    )
    monkeypatch.setattr(generation.genai, "Client", client_factory)

    result = generation.generate_response(
        RAGRequest(question="Why is exercise important in microgravity?"),
        api_key="test-key",
    )

    assert not result.abstained
    client_factory.assert_called_once_with(
        api_key="test-key",
        http_options=types.HttpOptions(
            timeout=generation.GENERATION_TIMEOUT_MS,
        ),
    )
