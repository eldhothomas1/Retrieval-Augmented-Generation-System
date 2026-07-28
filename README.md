# Retrieval-Augmented Generation System

A local RAG pipeline for document summarization and Q&A, built with:

- Retrieval-Augmented Generation (RAG)
- Hugging Face Transformers & SentenceTransformers
- FAISS vector indexing
- Qwen1.5-1.8B language model (locally loaded)
- Flask web interface

Upload PDF documents, generate summaries, and ask questions based on their content — designed to run locally without external APIs.

## Features

- Summarize multiple PDF documents
- Answer questions based on indexed content
- FAISS-powered semantic search
- Clean Flask-based web UI
- Local model loading (no API needed)
- Optimized for low-GPU-memory systems (4-bit quantization supported)

## Project Structure

```
├── rag_pipeline.py     # Core logic: document parsing, embedding, QA
├── app.py              # Flask web server
├── templates/           # HTML templates for the UI
├── requirements.txt     # Python dependencies
├── .gitignore
└── README.md
```

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/eldhothomas1/Retrieval-Augmented-Generation-System.git
cd Retrieval-Augmented-Generation-System
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Model setup (Qwen1.5-1.8B)

This project uses the Qwen/Qwen1.5-1.8B model from Hugging Face. Due to its size, it's excluded from the repo — download it manually:

```bash
mkdir -p models/Qwen1.5-1.8B
git lfs install
git clone https://huggingface.co/Qwen/Qwen1.5-1.8B models/Qwen1.5-1.8B
```

### 4. Run the app

```bash
python app.py
```

Then go to `http://127.0.0.1:5000`

## Files Not Included

To keep the repo lightweight, the following are excluded via `.gitignore`:
- `models/` (language models)
- `*.pdf` files
- `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`
- FAISS indexes (`faiss_index/`, `faiss_multi_index/`)

You'll need to regenerate these locally.
