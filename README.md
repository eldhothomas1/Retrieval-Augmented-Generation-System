# 🧠 AI Document Summarizer + QA System

This project is an interactive AI-powered summarization and question-answering system built with:

 Retrieval-Augmented Generation (RAG)
 Hugging Face Transformers & SentenceTransformers
 FAISS vector indexing
 Qwen1.5-1.8B language model (locally loaded)
 Flask web interface

The system allows users to upload and process documents (PDFs), generate summaries, and ask questions based on the content. Designed for local use without relying on external APIs or internet access during runtime.

---

##  Features

 Summarize multiple PDF documents  
 Answer questions based on indexed content  
 FAISS-powered semantic search  
 Clean Flask-based web UI  
 Local model loading (no API needed)  
 Optimized for low-GPU-memory systems (4-bit quantization supported)

---

## Project Structure

├── AIConversionTest.py # Core logic: document parsing, embedding, QA
├── app.py # Flask web server
├── templates/ # HTML templates for the UI
├── notebooks/ # Optional experimentation notebooks
├── requirements.txt # Python dependencies
├── .gitignore # Prevents large files from being pushed
└── README.md


---

## 💻 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/eldhothomas1/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

### 2. Install Dependencies

pip install -r requirements.txt

### 3. Model Setup (Qwen1.5-1.8B)

This project uses the Qwen/Qwen1.5-1.8B model from Hugging Face.
Due to its large size, it is excluded from the repository.

You must manually download and place the model into the models/ directory before running the app.

mkdir -p models/Qwen1.5-1.8B
git lfs install
git clone https://huggingface.co/Qwen/Qwen1.5-1.8B models/Qwen1.5-1.8B

FILES NOT INCLUDED:

To keep the repo clean and lightweight, the following are excluded using .gitignore:
models/ (all language models)
Any .pdf files
.pyc, __pycache__, .ipynb_checkpoints, etc.
FAISS indexes (faiss_index/, faiss_multi_index/)
Make sure to recreate your own models and FAISS index locally as needed.

HOW TO RUN:

1. python app.py
2. Go to: http://127.0.0.1:5000
