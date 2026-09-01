"""Flask interface for the local PDF RAG assistant."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

from flask import Flask, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from rag_pipeline import Answer, RAGError, RAGSystem, get_rag_instance

ALLOWED_EXTENSIONS = {"pdf"}
LOGGER = logging.getLogger(__name__)


def _is_pdf(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def create_app(rag_provider: Callable[[], RAGSystem] | None = None) -> Flask:
    """Create the web application with an injectable RAG service."""
    app = Flask(__name__)
    app.config.update(
        MAX_CONTENT_LENGTH=20 * 1024 * 1024,
        SECRET_KEY=os.getenv("FLASK_SECRET_KEY", "local-development-only"),
    )
    provider = rag_provider or get_rag_instance

    @app.get("/")
    def index():
        rag = provider()
        return render_template("index.html", rag=rag)

    @app.post("/documents")
    def upload_documents():
        rag = provider()
        files = [file for file in request.files.getlist("documents") if file.filename]
        if not files:
            flash("Choose at least one PDF.", "error")
            return redirect(url_for("index"))

        invalid = [file.filename for file in files if not _is_pdf(file.filename)]
        if invalid:
            flash("Only PDF files are supported.", "error")
            return redirect(url_for("index"))

        rag.clear()
        try:
            for file in files:
                filename = secure_filename(file.filename) or "document.pdf"
                rag.add_pdf(file.stream, filename)
            rag.embed_chunks()
        except RAGError as exc:
            rag.clear()
            flash(str(exc), "error")
        else:
            flash(
                f"Indexed {rag.document_count} document(s), {rag.page_count} page(s), "
                f"and {rag.chunk_count} chunk(s).",
                "success",
            )
        return redirect(url_for("index"))

    @app.post("/ask")
    def ask():
        rag = provider()
        question = request.form.get("query", "").strip()
        try:
            answer: Answer = rag.generate_answer(question)
        except RAGError as exc:
            flash(str(exc), "error")
            return render_template("index.html", rag=rag, query=question), 400
        return render_template(
            "index.html",
            rag=rag,
            query=question,
            answer=answer.text,
            sources=answer.sources,
        )

    @app.post("/summarize")
    def summarize():
        rag = provider()
        try:
            summary = rag.summarize_documents()
        except RAGError as exc:
            flash(str(exc), "error")
            return render_template("index.html", rag=rag), 400
        return render_template("index.html", rag=rag, summary=summary)

    @app.errorhandler(413)
    def upload_too_large(_error):
        flash("Upload is too large. The combined limit is 20 MB.", "error")
        return redirect(url_for("index"))

    return app


app = create_app()

if __name__ == "__main__":
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    app.run(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8888")),
        debug=os.getenv("FLASK_DEBUG", "").lower() == "true",
    )
