from io import BytesIO
from types import SimpleNamespace

from app import create_app
from rag_pipeline import Answer, RAGError


class FakeRAG:
    def __init__(self):
        self.document_count = 0
        self.page_count = 0
        self.chunk_count = 0
        self.ready = False

    def clear(self):
        self.__init__()

    def add_pdf(self, _file, _filename):
        self.document_count += 1
        self.page_count += 2

    def embed_chunks(self):
        self.chunk_count = 3
        self.ready = True

    def generate_answer(self, question):
        if not self.ready:
            raise RAGError("Upload and index at least one PDF before asking a question.")
        source = SimpleNamespace(
            chunk=SimpleNamespace(
                source="sample.pdf",
                page_number=1,
                text="A supporting source passage.",
            )
        )
        return Answer(f"Answer for: {question}", (source,))

    def summarize_documents(self):
        if not self.ready:
            raise RAGError("Upload and index at least one PDF before summarizing.")
        return "A concise summary."


def make_client():
    rag = FakeRAG()
    app = create_app(lambda: rag)
    app.config.update(TESTING=True, SECRET_KEY="test")
    return app.test_client(), rag


def test_home_loads_without_initializing_models():
    client, _rag = make_client()
    response = client.get("/")
    assert response.status_code == 200
    assert b"Ask better questions of your PDFs" in response.data


def test_upload_indexes_documents_and_reports_counts():
    client, rag = make_client()
    response = client.post(
        "/documents",
        data={"documents": (BytesIO(b"%PDF-test"), "sample.pdf")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert rag.ready
    assert b"Indexed 1 document(s), 2 page(s), and 3 chunk(s)." in response.data


def test_question_displays_answer_and_source():
    client, rag = make_client()
    rag.ready = True
    rag.document_count = rag.page_count = rag.chunk_count = 1
    response = client.post("/ask", data={"query": "What happened?"})
    assert response.status_code == 200
    assert b"Answer for: What happened?" in response.data
    assert b"sample.pdf" in response.data
    assert b"Page 1" in response.data


def test_question_before_upload_is_controlled_error():
    client, _rag = make_client()
    response = client.post("/ask", data={"query": "What happened?"})
    assert response.status_code == 400
    assert b"Upload and index at least one PDF" in response.data
