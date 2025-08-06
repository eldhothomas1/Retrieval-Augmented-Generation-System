from flask import Flask, render_template, request
from AIConversionTest import get_rag_instance

app = Flask(__name__)
rag = get_rag_instance()
rag.load_documents_from_folder("docs")  # your folder with PDFs
#rag.chunk_documents()
rag.embed_chunks()

@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    summary = ""
    if request.method == "POST":
        if request.form.get("query"):
            query = request.form["query"]
            answer = rag.generate_answer(query)
        if request.form.get("summarize"):
            summary = rag.summarize_documents()
    return render_template("index.html", answer=answer, summary=summary)

if __name__ == "__main__":
    app.run(debug=False,port=8888)

