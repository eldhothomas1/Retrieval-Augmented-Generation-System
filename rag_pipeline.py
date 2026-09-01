"""Local retrieval-augmented generation pipeline for PDF documents."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
from pypdf import PdfReader

LOGGER = logging.getLogger(__name__)


class RAGError(RuntimeError):
    """Raised when the document pipeline cannot complete a user operation."""


@dataclass(frozen=True)
class Page:
    text: str
    source: str
    page_number: int


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    page_number: int
    chunk_number: int


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Answer:
    text: str
    sources: tuple[SearchResult, ...]


def split_text(text: str, chunk_size: int = 700, chunk_overlap: int = 100) -> list[str]:
    """Split text into overlapping, whitespace-trimmed character chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be between zero and chunk_size")

    normalized = " ".join(text.split())
    if not normalized:
        return []

    step = chunk_size - chunk_overlap
    return [
        normalized[start : start + chunk_size].strip()
        for start in range(0, len(normalized), step)
        if normalized[start : start + chunk_size].strip()
    ]


class RAGSystem:
    """Ingest PDFs, build a FAISS index, and answer grounded questions."""

    def __init__(
        self,
        *,
        model_name: str | None = None,
        embedding_model_name: str | None = None,
        chunk_size: int = 700,
        chunk_overlap: int = 100,
        top_k: int = 5,
        embedding_model: object | None = None,
        tokenizer: object | None = None,
        model: object | None = None,
        index_factory: Callable[[int], object] | None = None,
    ) -> None:
        self.model_name = model_name or os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
        self.embedding_model_name = embedding_model_name or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.pages: list[Page] = []
        self.chunks: list[Chunk] = []
        self.index: object | None = None
        self._embedding_model = embedding_model
        self._tokenizer = tokenizer
        self._model = model
        self._index_factory = index_factory

    @property
    def document_count(self) -> int:
        return len({page.source for page in self.pages})

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def ready(self) -> bool:
        return self.index is not None and bool(self.chunks)

    def clear(self) -> None:
        """Remove all loaded documents and the current vector index."""
        self.pages.clear()
        self.chunks.clear()
        self.index = None

    def add_pdf(self, file: str | Path | BinaryIO, title: str | None = None) -> int:
        """Extract text from a PDF and return the number of usable pages added."""
        source = title or getattr(file, "name", None) or "document.pdf"
        source = Path(str(source)).name
        try:
            reader = PdfReader(file)
        except Exception as exc:
            raise RAGError(f"Could not read {source} as a PDF.") from exc

        added = 0
        for page_number, pdf_page in enumerate(reader.pages, start=1):
            text = (pdf_page.extract_text() or "").strip()
            if text:
                self.pages.append(Page(text=text, source=source, page_number=page_number))
                added += 1

        if not added:
            raise RAGError(f"{source} contains no extractable text. Scanned PDFs require OCR.")
        LOGGER.info("Loaded %s usable pages from %s", added, source)
        return added

    def load_documents_from_folder(self, folder_path: str | Path) -> int:
        """Load every PDF in a folder, raising a clear error for empty input."""
        folder = Path(folder_path)
        if not folder.exists():
            raise RAGError(f"Document folder does not exist: {folder}")
        pdf_paths = sorted(folder.glob("*.pdf"))
        if not pdf_paths:
            raise RAGError(f"No PDF files were found in {folder}.")

        pages_before = self.page_count
        for path in pdf_paths:
            self.add_pdf(path, path.name)
        return self.page_count - pages_before

    def _create_chunks(self) -> list[Chunk]:
        chunks: list[Chunk] = []
        for page in self.pages:
            page_chunks = split_text(page.text, self.chunk_size, self.chunk_overlap)
            chunks.extend(
                Chunk(text, page.source, page.page_number, number)
                for number, text in enumerate(page_chunks, start=1)
            )
        return chunks

    def _get_embedding_model(self) -> object:
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer

            LOGGER.info("Loading embedding model %s", self.embedding_model_name)
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model

    def _new_index(self, dimension: int) -> object:
        if self._index_factory is not None:
            return self._index_factory(dimension)
        import faiss

        return faiss.IndexFlatIP(dimension)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype="float32")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def embed_chunks(self) -> int:
        """Create chunks, encode them, and build a cosine-similarity FAISS index."""
        if not self.pages:
            raise RAGError("Upload at least one text-based PDF before indexing.")
        self.chunks = self._create_chunks()
        if not self.chunks:
            raise RAGError("The uploaded PDFs did not produce searchable text chunks.")

        embeddings = self._get_embedding_model().encode(
            [chunk.text for chunk in self.chunks],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        embeddings = self._normalize(embeddings)
        self.index = self._new_index(embeddings.shape[1])
        self.index.add(embeddings)
        LOGGER.info("Indexed %s chunks", len(self.chunks))
        return len(self.chunks)

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Return the most similar indexed chunks for a query."""
        query = query.strip()
        if not query:
            raise RAGError("Enter a question before searching.")
        if not self.ready:
            raise RAGError("Upload and index at least one PDF before asking a question.")

        limit = min(top_k or self.top_k, len(self.chunks))
        query_vector = self._get_embedding_model().encode(
            [query], convert_to_numpy=True, show_progress_bar=False
        )
        scores, indices = self.index.search(self._normalize(query_vector), limit)
        return [
            SearchResult(self.chunks[int(index)], float(score))
            for score, index in zip(scores[0], indices[0], strict=True)
            if index >= 0
        ]

    def _load_generator(self) -> tuple[object, object]:
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            LOGGER.info("Loading generation model %s", self.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            options: dict[str, object] = {"low_cpu_mem_usage": True}
            if torch.cuda.is_available():
                options.update(device_map="auto", torch_dtype=torch.float16)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **options)
        except Exception as exc:
            raise RAGError(
                f"Could not load {self.model_name}. Check disk space, memory, and "
                "your Hugging Face connection."
            ) from exc
        return self._tokenizer, self._model

    @staticmethod
    def _format_prompt(tokenizer: object, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return f"System: {system}\n\nUser: {user}\n\nAssistant:"

    def _generate(self, system: str, user: str, max_new_tokens: int = 300) -> str:
        import torch

        tokenizer, model = self._load_generator()
        prompt = self._format_prompt(tokenizer, system, user)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072)
        device = getattr(model, "device", torch.device("cpu"))
        inputs = {name: value.to(device) for name, value in inputs.items()}
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = output[0, inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        if not text:
            raise RAGError("The local model returned an empty response.")
        return text

    def generate_answer(self, question: str, top_k: int | None = None) -> Answer:
        """Answer a question using retrieved passages and return its sources."""
        sources = tuple(self.retrieve(question, top_k))
        context = "\n\n".join(
            f"[{number}] {result.chunk.source}, page {result.chunk.page_number}\n"
            f"{result.chunk.text}"
            for number, result in enumerate(sources, start=1)
        )
        system = (
            "Answer only from the supplied document excerpts. Cite supporting excerpts "
            "with bracketed source numbers such as [1]. If the excerpts do not contain "
            "the answer, say: 'I could not find that in the indexed documents.'"
        )
        user = f"Document excerpts:\n\n{context}\n\nQuestion: {question.strip()}"
        return Answer(self._generate(system, user), sources)

    def summarize_documents(self) -> str:
        """Generate a concise overview from the currently indexed content."""
        if not self.ready:
            raise RAGError("Upload and index at least one PDF before summarizing.")
        selected = self.chunks[: min(12, len(self.chunks))]
        context = "\n\n".join(
            f"{chunk.source}, page {chunk.page_number}: {chunk.text}" for chunk in selected
        )
        return self._generate(
            "Summarize only the supplied document text. State the major themes and key "
            "facts concisely, and do not add outside information.",
            context,
            max_new_tokens=400,
        )


_shared_rag: RAGSystem | None = None


def get_rag_instance() -> RAGSystem:
    global _shared_rag
    if _shared_rag is None:
        _shared_rag = RAGSystem()
    return _shared_rag
