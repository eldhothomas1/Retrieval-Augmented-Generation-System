import os
import torch
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader

class Document:
    def __init__(self, content, title=None):
        self.content = content
        self.title = title or "Untitled"


class RAGSystem:


    def __init__(self):
        self.documents = []
        self.chunks = []
        self.embeddings = []
        self.index = None

        print("Loading SentenceTransformer for embeddings...")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)

        print("Loading Qwen model WITHOUT quantization (CPU fallback)...")
        try:
            if torch.cuda.is_available():
                print("CUDA is available. Loading model on GPU with float16...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    "Qwen/Qwen3-1.7B",
                    device_map="auto",
                    torch_dtype="auto",
                    trust_remote_code=True,
                )
            else:
                print("CUDA not available. Loading model on CPU...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    "Qwen/Qwen-7B-Chat",
                    trust_remote_code=True
                )
        except Exception as e:
            print("Failed to load the Qwen model. Falling back to a smaller model like GPT2.")
            from transformers import GPT2Tokenizer, GPT2LMHeadModel
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.model = GPT2LMHeadModel.from_pretrained("gpt2")

        print("Model and tokenizer successfully loaded.")



    def __init__(self):
        self.documents = []
        self.chunks = []
        self.embeddings = []
        self.index = None

        print("Loading SentenceTransformer for embeddings...")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Set model name once and reuse
        model_name = "Qwen/Qwen3-1.7B"

        print(f"Loading tokenizer for {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        print(f"Loading {model_name} model...")
        try:
            if torch.cuda.is_available():
                print("CUDA is available. Loading model on GPU with float16...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                )
            else:
                print("CUDA not available. Loading model on CPU (may be slower)...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
        except Exception as e:
            print(f"Failed to load {model_name} due to: {e}")
            print("Falling back to GPT-2...")
            from transformers import GPT2Tokenizer, GPT2LMHeadModel
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.model = GPT2LMHeadModel.from_pretrained("gpt2")

        print("Model and tokenizer successfully loaded.")

    def load_documents_from_folder(self, folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith(".pdf"):
                path = os.path.join(folder_path, filename)
                with open(path, "rb") as f:
                    reader = PdfReader(f)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() or ""
                        self.documents.append(Document(text, title=filename))

    def load_pdf(self, filepath):
        """Load PDF and return text."""
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text


    def add_document(self, filepath, title=None):
        """Load and store document."""
        text = self.load_pdf(filepath)
        doc = Document(text, title or os.path.basename(filepath))
        self.documents.append(doc)


    def split_text(self, text, chunk_size=300):
        """Split text into fixed-size chunks."""
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


    def embed_chunks(self):
        """Chunk all documents and embed them."""
        self.chunks = []
        for doc in self.documents:
            chunks = self.split_text(doc.content)
            for chunk in chunks:
                self.chunks.append((chunk, doc.title))

        texts = [chunk[0] for chunk in self.chunks]
        embeddings = self.embedding_model.encode(texts, convert_to_tensor=True).cpu().numpy()

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)


    def retrieve(self, query, top_k=5):
        """Retrieve top-k chunks relevant to the query."""
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=True).cpu().numpy()
        D, I = self.index.search(query_embedding, top_k)
        results = [self.chunks[i][0] for i in I[0]]
        return results


    def summarize_documents(self):
        """Summarize all loaded documents."""
        summaries = []
        for doc in self.documents:
            prompt = f"Summarize the following document:\n{doc.content}\nSummary:"
            input_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).input_ids.to(self.model.device)

            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_new_tokens=1000,
                    do_sample=False,
                    temperature=0.7,
                    top_k=50
                )

            summary = self.tokenizer.decode(output[0], skip_special_tokens=True)
            summaries.append(f"Summary for {doc.title}:\n{summary.strip()}")
        return "\n\n".join(summaries)


    def generate_answer(self, question, top_k=5):
        """Answer a question using retrieved context."""
        retrieved_chunks = self.retrieve(question, top_k=top_k)
        context = "\n".join(retrieved_chunks)

        prompt = f"Answer the question based on the context below:\n\n{context}\n\nQuestion: {question}\nAnswer:"
        input_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).input_ids.to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=1000,
                do_sample=False,
                temperature=0.7,
                top_k=50
            )

        answer = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return answer.strip()

rag = None

def get_rag_instance():
    global rag
    if rag is None:
        print("Initiating shared RAGSystem Instance")
        rag = RAGSystem()
    return rag

