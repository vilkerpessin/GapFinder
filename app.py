import io
import os

import pandas as pd
from flask import Flask, render_template, request, send_file, session
from flask_session import Session
from cachelib.file import FileSystemCache

from pdf_extractor import extract_metadata, extract_text_from_pdf
from gap_analyzer import analyze_text


app = Flask(__name__)

app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "cachelib"
app.config["SESSION_CACHELIB"] = FileSystemCache(cache_dir="flask_session")
Session(app)

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB per file
MAX_TOTAL_SIZE = 50 * 1024 * 1024  # 50MB total


@app.route('/', methods=['GET', 'POST'])
def index():
    session.clear()
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process_files():
    files = request.files.getlist('pdf_files')
    data = []
    total_size = 0

    for file in files:
        if not file.filename.endswith('.pdf'):
            continue

        file_content = file.read()
        file_size = len(file_content)

        if file_size > MAX_FILE_SIZE:
            return (
                f"Error: File '{file.filename}' is too large "
                f"({file_size / 1024 / 1024:.1f}MB). "
                f"Maximum size is {MAX_FILE_SIZE / 1024 / 1024}MB per file.",
                400
            )

        total_size += file_size
        if total_size > MAX_TOTAL_SIZE:
            return (
                f"Error: Total file size exceeds "
                f"{MAX_TOTAL_SIZE / 1024 / 1024}MB limit. "
                "Please upload fewer or smaller files.",
                400
            )

        file_stream = io.BytesIO(file_content)

        try:
            text = extract_text_from_pdf(file_stream)
        except Exception:
            continue

        file_stream.seek(0)
        try:
            metadata = extract_metadata(file_stream)
        except Exception:
            continue

        paragraphs = analyze_text(text)

        for page, paragraph, insight in paragraphs:
            data.append({
                "file": file.filename,
                "doi": metadata["doi"],
                "author": metadata["author"],
                "title": metadata["title"],
                "keywords": metadata.get("keywords", ""),
                "page": page,
                "paragraph": paragraph,
                "insight": insight
            })

    session['resultados'] = data
    return render_template('result.html', data=data)


@app.route('/download')
def download():
    if 'resultados' not in session:
        return "No results available for download. Please process your files first.", 400

    df = pd.DataFrame(session['resultados'])
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    output.seek(0)

    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name='results.xlsx'
    )


if __name__ == '__main__':
    debug_mode = os.environ.get("FLASK_ENV") != "production"
    app.run(debug=debug_mode)
