"""Tests for configurable Gemini embeddings and query-time fallback in agents/rag.py."""

from __future__ import annotations

import pytest

from agents.rag import DEFAULT_DOCS_DIR, LocalTfidfEmbeddings, MaintenanceManualRAG


def test_rag_uses_configured_gemini_embedding_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """_init_embeddings should read GEMINI_EMBEDDING_MODEL when GOOGLE_API_KEY is set."""
    captured: dict[str, str] = {}

    class DummyGoogleEmbeddings:
        def __init__(self, model: str, google_api_key: str) -> None:
            captured["model"] = model
            captured["api_key"] = google_api_key

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

        def embed_query(self, text: str) -> list[float]:
            return [1.0, 0.0]

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_EMBEDDING_MODEL", "models/custom-embedding-test")
    monkeypatch.setattr("agents.rag.GoogleGenerativeAIEmbeddings", DummyGoogleEmbeddings)

    rag = MaintenanceManualRAG(docs_dir=DEFAULT_DOCS_DIR, prefer_google_embeddings=True)
    assert captured["model"] == "models/custom-embedding-test"
    assert captured["api_key"] == "test-key"
    assert isinstance(rag.embeddings, DummyGoogleEmbeddings)


def test_rag_falls_back_to_local_embeddings_on_query_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If vector_store.similarity_search raises at query time, fall back to LocalTfidf."""
    rag = MaintenanceManualRAG(docs_dir=DEFAULT_DOCS_DIR, prefer_google_embeddings=False)
    original_store = rag.vector_store

    def failing_similarity_search(*args: object, **kwargs: object) -> list[object]:
        raise RuntimeError("Simulated embedding API failure at query time")

    monkeypatch.setattr(original_store, "similarity_search", failing_similarity_search)

    chunks = rag.search("high bearing vibration and shaft misalignment", top_k=3)
    assert len(chunks) == 3
    assert isinstance(rag.embeddings, LocalTfidfEmbeddings)
    assert rag.vector_store is not original_store
    assert any(chunk.source == "bearing_vibration.md" for chunk in chunks)


def test_rag_falls_back_to_local_embeddings_on_init_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If GoogleGenerativeAIEmbeddings raises during init, fall back to LocalTfidf."""

    def failing_google_embeddings(*args: object, **kwargs: object) -> object:
        raise RuntimeError("Simulated embedding model initialization failure")

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_EMBEDDING_MODEL", "models/nonexistent-embedding")
    monkeypatch.setattr("agents.rag.GoogleGenerativeAIEmbeddings", failing_google_embeddings)

    rag = MaintenanceManualRAG(docs_dir=DEFAULT_DOCS_DIR, prefer_google_embeddings=True)
    assert isinstance(rag.embeddings, LocalTfidfEmbeddings)
    chunks = rag.search("overheating coolant thermal shutdown", top_k=2)
    assert len(chunks) == 2
