# Local PDF RAG Assistant

A local document question-answering application that converts PDFs into a searchable vector index, retrieves relevant passages, and generates answers with inspectable filename and page citations.

Built with Python, Flask, SentenceTransformers, FAISS, Hugging Face Transformers, and Qwen. No hosted LLM API or API key is required.

## Project highlights

- Complete pipeline from PDF ingestion to grounded answer generation
- Overlapping, metadata-aware chunks that retain filename and page provenance
- Cosine-similarity retrieval through a normalized FAISS index
- Local Qwen inference with lazy model loading
- Source excerpts displayed beside every generated answer
- Explicit abstention instruction for unsupported questions
- Multi-file upload, controlled errors, and a responsive Flask interface
- Model-free unit and route tests suitable for continuous integration

## Architecture

~~~mermaid
flowchart TD
    A["PDF uploads"] --> B["Text + page metadata"]
    B --> C["Overlapping chunks"]
    C --> D["SentenceTransformer embeddings"]
    D --> E["Normalized FAISS index"]
    Q["User question"] --> F["Query embedding"]
    F --> E
    E --> G["Top-k passages"]
    G --> H["Local Qwen model"]
    H --> I["Answer + source excerpts"]
~~~

The embedding vectors are L2-normalized before being stored in a FAISS inner-product index. This makes the inner product equivalent to cosine similarity and keeps retrieval fast without discarding document provenance.

## Application flow

1. Upload one or more text-based PDFs.
2. The application extracts each usable page and creates overlapping chunks.
3. SentenceTransformers encodes the chunks and FAISS indexes the normalized vectors.
4. A question is embedded with the same model.
5. The top matching passages are provided to the local language model.
6. The UI returns the answer alongside the retrieved filename, page number, and text.

## Tech stack

| Layer | Technology |
|---|---|
| Web application | Flask, Jinja, HTML, CSS |
| PDF extraction | pypdf |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector search | FAISS |
| Local generation | Qwen/Qwen2.5-0.5B-Instruct |
| Runtime | PyTorch, Hugging Face Transformers |
| Quality | pytest, Ruff, GitHub Actions |

## Quick start

### Requirements

- Python 3.11 or 3.12
- Git
- Internet access on the first run to download the embedding and generation models
- Enough local memory and disk space for PyTorch and the selected models

The default 0.5B-parameter model is chosen to make local CPU use more practical. Generation will still be faster with a compatible CUDA GPU.

### Install

~~~bash
git clone https://github.com/eldhothomas1/Retrieval-Augmented-Generation-System.git
cd Retrieval-Augmented-Generation-System

python -m venv .venv
~~~

Activate the environment:

~~~bash
# macOS or Linux
source .venv/bin/activate
~~~

~~~powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
~~~

Install the runtime dependencies:

~~~bash
python -m pip install --upgrade pip
pip install -r requirements.txt
~~~

### Run

~~~bash
python app.py
~~~

Open [http://127.0.0.1:8888](http://127.0.0.1:8888), upload one or more PDFs, and wait for the index confirmation before asking a question.

The first indexing request loads the embedding model. The first question or summary loads the generation model. Later requests reuse both models for the life of the server process.

## Configuration

Copy .env.example as a reference and export any settings you want to override:

| Variable | Default | Purpose |
|---|---|---|
| MODEL_NAME | Qwen/Qwen2.5-0.5B-Instruct | Hugging Face generation model |
| EMBEDDING_MODEL | sentence-transformers/all-MiniLM-L6-v2 | Sentence embedding model |
| HOST | 127.0.0.1 | Flask bind address |
| PORT | 8888 | Local server port |
| LOG_LEVEL | INFO | Python logging level |
| FLASK_DEBUG | unset | Set to true for local debugging |

Example:

~~~bash
MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct PORT=5000 python app.py
~~~

PowerShell:

~~~powershell
$env:MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
$env:PORT = "5000"
python app.py
~~~

## Testing

The test suite uses dependency injection and lightweight fakes, so it validates ingestion, chunking, retrieval, Flask routes, and error handling without downloading an LLM.

~~~bash
pip install -r requirements-dev.txt
pytest -q
ruff check .
~~~

GitHub Actions runs the same model-free checks on every push and pull request.

## Project structure

~~~text
.
├── .github/workflows/test.yml  # Continuous integration
├── static/styles.css           # Responsive interface styles
├── templates/index.html        # Flask/Jinja interface
├── tests/
│   ├── test_app.py             # Route and upload behavior
│   └── test_pipeline.py        # Ingestion, chunking, and retrieval
├── .env.example                # Configuration reference
├── app.py                      # Web routes and application factory
├── rag_pipeline.py             # RAG ingestion, retrieval, and generation
├── requirements-dev.txt
└── requirements.txt
~~~

## Reliability and privacy

- Uploaded documents are read in memory and are not written to the repository.
- The active document index is process-local and is replaced by each new upload batch.
- The application rejects unsupported file extensions and limits each request to 20 MB.
- Scanned or image-only PDFs receive a clear OCR-related error.
- Model-loading failures are surfaced rather than silently falling back to a weaker model.
- Answers include the passages sent to the generator so users can inspect the evidence.

This is a local development project, not a hardened multi-user service. Do not expose the Flask development server directly to the public internet.

## Design decisions

**Why FAISS instead of a database?**

The project focuses on an in-memory, single-user local workflow. FAISS provides efficient vector search without requiring a separate service.

**Why a smaller default Qwen model?**

The default should be feasible for more reviewers to run. MODEL_NAME remains configurable for users with more capable hardware.

**Why character chunks?**

Character chunking keeps the implementation easy to inspect and test. The overlap reduces context loss across chunk boundaries. Token-aware or sentence-aware splitting would be a useful next experiment.

**Why show retrieved text even when an answer is generated?**

Retrieval-augmented generation is easier to evaluate when the evidence is visible. Source passages let users distinguish retrieval failures from generation failures.

## Current limitations

- pypdf does not perform OCR, so scanned PDFs are not supported.
- The FAISS index is in memory and is rebuilt when documents are uploaded.
- Summarization uses a bounded selection of chunks rather than a hierarchical full-document summarizer.
- Local generation latency depends on the selected model and hardware.
- Prompt-based grounding reduces hallucinations but does not guarantee factual correctness; important answers should be checked against the displayed passages.

## Next improvements

- Add OCR for scanned documents.
- Persist indexes and document metadata between sessions.
- Stream generated tokens to the interface.
- Add a small, versioned evaluation set for retrieval hit rate and grounded-answer quality.
- Compare character, sentence, and token-aware chunking strategies.
