from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from rag_pipeline import Page, RAGError, RAGSystem, split_text


class FakeEmbeddingModel:
    def encode(self, texts, **_kwargs):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    float(lowered.count("alpha")),
                    float(lowered.count("beta")),
                    0.1,
                ]
            )
        return np.asarray(vectors, dtype="float32")


class FakeIndex:
    def __init__(self, _dimension):
        self.vectors = None

    def add(self, vectors):
        self.vectors = vectors

    def search(self, query, limit):
        scores = query @ self.vectors.T
        indices = np.argsort(-scores, axis=1)[:, :limit]
        ordered_scores = np.take_along_axis(scores, indices, axis=1)
        return ordered_scores, indices


def test_split_text_preserves_overlap_and_validates_configuration():
    chunks = split_text("abcdefghij", chunk_size=6, chunk_overlap=2)
    assert chunks == ["abcdef", "efghij", "ij"]
    with pytest.raises(ValueError):
        split_text("text", chunk_size=5, chunk_overlap=5)


def test_add_pdf_stores_each_page_once():
    fake_pages = [
        SimpleNamespace(extract_text=lambda: "First page"),
        SimpleNamespace(extract_text=lambda: "Second page"),
    ]
    with patch("rag_pipeline.PdfReader", return_value=SimpleNamespace(pages=fake_pages)):
        rag = RAGSystem()
        assert rag.add_pdf(object(), "report.pdf") == 2

    assert rag.page_count == 2
    assert [page.text for page in rag.pages] == ["First page", "Second page"]
    assert [page.page_number for page in rag.pages] == [1, 2]


def test_empty_or_scanned_pdf_has_clear_error():
    fake_pages = [SimpleNamespace(extract_text=lambda: "")]
    with patch("rag_pipeline.PdfReader", return_value=SimpleNamespace(pages=fake_pages)):
        with pytest.raises(RAGError, match="require OCR"):
            RAGSystem().add_pdf(object(), "scan.pdf")


def test_retrieval_returns_relevant_chunk_with_metadata():
    rag = RAGSystem(
        chunk_size=100,
        chunk_overlap=10,
        embedding_model=FakeEmbeddingModel(),
        index_factory=FakeIndex,
    )
    rag.pages = [
        Page("Alpha is the primary subject.", "alpha.pdf", 2),
        Page("Beta is discussed separately.", "beta.pdf", 4),
    ]

    assert rag.embed_chunks() == 2
    result = rag.retrieve("Tell me about alpha", top_k=1)[0]

    assert result.chunk.source == "alpha.pdf"
    assert result.chunk.page_number == 2
    assert result.score > 0


def test_indexing_requires_documents():
    rag = RAGSystem(
        embedding_model=FakeEmbeddingModel(),
        index_factory=FakeIndex,
    )
    with pytest.raises(RAGError, match="Upload at least one"):
        rag.embed_chunks()
