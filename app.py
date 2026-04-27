from flask import Flask, render_template, request, jsonify
import ollama
import os
import pdfplumber

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------- PDF Reader ----------
def read_pdf(filepath):
    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    model = request.form["model"]
    question = request.form["message"]
    file = request.files.get("file")

    if not file:
        return jsonify({"reply": "Please upload a PDF file."})

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    context = read_pdf(filepath)

    if context.strip() == "":
        return jsonify({"reply": "PDF text could not be extracted."})

    # Limit context size (important for Ollama)
    context = context[:4000]

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "user", "content": f"""
You are a helpful assistant.

Use ONLY the following PDF content to answer.

PDF Content:
{context}

Question:
{question}

If the answer is not in the PDF, say:
'Answer not found in document.'
"""}
        ]
    )

    return jsonify({"reply": response["message"]["content"]})

if __name__ == "__main__":
    app.run(debug=True)