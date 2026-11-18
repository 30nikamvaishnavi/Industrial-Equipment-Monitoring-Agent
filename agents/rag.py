"""Hybrid FAISS + BM25 RAG retriever over maintenance manual markdown docs."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DOCS_DIR = Path(os.environ.get("DOCS_DIR", str(BASE_DIR / "docs")))
# Check Google AI Studio for the current Gemini embedding model name.
DEFAULT_EMBEDDING_MODEL = "models/text-embedding-004"


@dataclass(frozen=True)
class RetrievedChunk:
    """A retrieved maintenance manual section with source metadata and RRF score."""

    source: str
    section: str
    score: float
    content: str


class LocalTfidfEmbeddings(Embeddings):
    """Deterministic local TF-IDF dense embeddings implementing LangChain Embeddings."""

    def __init__(self, corpus: list[str], max_features: int = 256) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),
        )
        self.vectorizer.fit(corpus)
        self._dim = len(self.vectorizer.get_feature_names_out())

    def _transform(self, texts: list[str]) -> list[list[float]]:
        if self._dim == 0:
            return [[0.0] for _ in texts]
        matrix = self.vectorizer.transform(texts).toarray().astype(np.float32)
        normed = normalize(matrix, norm="l2")
        return normed.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document chunks."""
        return self._transform(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query."""
        return self._transform([text])[0]


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization for BM25."""
    return re.findall(r"[a-z0-9]+", text.lower())


def load_markdown_chunks(docs_dir: Path = DEFAULT_DOCS_DIR) -> list[Document]:
    """Load markdown files from docs_dir and split by level-2 headings.

    Args:
        docs_dir: Directory containing maintenance manual markdown files.

    Returns:
        List of LangChain Document objects with 'source' and 'section' metadata.

    Raises:
        FileNotFoundError: If docs_dir does not exist or has no markdown files.
    """
    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs directory not found at {docs_dir}.")

    md_files = sorted(docs_dir.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No markdown files found in {docs_dir}.")

    documents: list[Document] = []
    for path in md_files:
        text = path.read_text(encoding="utf-8").strip()
        lines = text.splitlines()
        doc_title = lines[0].lstrip("# ").strip() if lines else path.stem

        sections = re.split(r"\n(?=##\s+)", text)
        for sec in sections:
            sec_clean = sec.strip()
            if not sec_clean:
                continue
            first_line = sec_clean.splitlines()[0].lstrip("# ").strip()
            section_name = (
                f"{doc_title} - {first_line}" if sec_clean.startswith("##") else doc_title
            )
            documents.append(
                Document(
                    page_content=sec_clean,
                    metadata={"source": path.name, "section": section_name},
                )
            )
    return documents


class MaintenanceManualRAG:
    """Hybrid dense (FAISS) + sparse (BM25) retriever over docs/ manuals."""

    def __init__(
        self,
        docs_dir: Path = DEFAULT_DOCS_DIR,
        prefer_google_embeddings: bool = True,
    ) -> None:
        self.docs_dir = docs_dir
        self.documents = load_markdown_chunks(docs_dir)
        self.corpus_texts = [doc.page_content for doc in self.documents]

        # Sparse BM25 index
        tokenized_corpus = [_tokenize(text) for text in self.corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Dense FAISS index with automatic fallback to local TF-IDF on any error
        try:
            self.embeddings = self._init_embeddings(prefer_google_embeddings)
            self.vector_store = FAISS.from_documents(self.documents, self.embeddings)
        except Exception:
            self._switch_to_local_embeddings()

    def _init_embeddings(self, prefer_google_embeddings: bool) -> Embeddings:
        """Select GoogleGenerativeAIEmbeddings if GOOGLE_API_KEY is available."""
        api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
        embedding_model = (
            os.environ.get("GEMINI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip()
            or DEFAULT_EMBEDDING_MODEL
        )
        if prefer_google_embeddings and api_key:
            return GoogleGenerativeAIEmbeddings(
                model=embedding_model,
                google_api_key=api_key,
            )
        return LocalTfidfEmbeddings(self.corpus_texts)

    def _switch_to_local_embeddings(self) -> None:
        """Rebuild the FAISS index using deterministic local TF-IDF embeddings."""
        self.embeddings = LocalTfidfEmbeddings(self.corpus_texts)
        self.vector_store = FAISS.from_documents(self.documents, self.embeddings)

    def search(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        """Perform hybrid FAISS + BM25 search combined via Reciprocal Rank Fusion.

        Args:
            query: Natural-language symptom or diagnostic query.
            top_k: Maximum number of chunks to return.

        Returns:
            Ranked list of RetrievedChunk results.
        """
        if not query or not query.strip():
            return []

        k = min(max(1, top_k), len(self.documents))
        rrf_k = 60.0
        scores: dict[int, float] = {i: 0.0 for i in range(len(self.documents))}

        # FAISS dense retrieval (falls back to local TF-IDF on any query-time error)
        content_to_idx: dict[str, int] = {
            doc.page_content: idx for idx, doc in enumerate(self.documents)
        }
        try:
            faiss_docs = self.vector_store.similarity_search(query, k=len(self.documents))
        except Exception:
            self._switch_to_local_embeddings()
            faiss_docs = self.vector_store.similarity_search(query, k=len(self.documents))

        for rank, doc in enumerate(faiss_docs):
            idx = content_to_idx.get(doc.page_content)
            if idx is not None:
                scores[idx] += 1.0 / (rrf_k + rank + 1.0)

        # BM25 sparse retrieval
        query_tokens = _tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_ranking = np.argsort(bm25_scores)[::-1]
        for rank, idx in enumerate(bm25_ranking):
            scores[int(idx)] += 1.0 / (rrf_k + rank + 1.0)

        ranked_indices = sorted(scores, key=lambda idx: scores[idx], reverse=True)[:k]
        results: list[RetrievedChunk] = []
        for idx in ranked_indices:
            doc = self.documents[idx]
            results.append(
                RetrievedChunk(
                    source=str(doc.metadata.get("source", "unknown")),
                    section=str(doc.metadata.get("section", "Overview")),
                    score=round(float(scores[idx]), 5),
                    content=doc.page_content,
                )
            )
        return results


def query_manuals(query: str, top_k: int = 3, docs_dir: Path | None = None) -> list[dict[str, Any]]:
    """Convenience function returning serializable manual retrieval results."""
    rag = MaintenanceManualRAG(docs_dir=docs_dir or DEFAULT_DOCS_DIR)
    chunks = rag.search(query=query, top_k=top_k)
    return [
        {
            "source": chunk.source,
            "section": chunk.section,
            "score": chunk.score,
            "content": chunk.content,
        }
        for chunk in chunks
    ]
