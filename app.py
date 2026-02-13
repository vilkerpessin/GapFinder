import gc
import io
import os
import tempfile
import uuid

import pandas as pd
from flask import Flask, render_template, request, send_file

from pdf_extractor import extract_metadata, extract_text_by_page
from gap_analyzer import analyze_pages


app = Flask(__name__)

app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")

RESULTS_DIR = os.path.join(tempfile.gettempdir(), "gapfinder_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB per file
MAX_TOTAL_SIZE = 50 * 1024 * 1024  # 50MB total


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process_files():
    files = request.files.getlist('pdf_files')
    data = []
    total_size = 0
    files_analyzed = 0

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
            metadata = extract_metadata(file_stream)
        except Exception:
            continue

        file_stream.seek(0)
        try:
            pages = extract_text_by_page(file_stream)
            paragraphs = analyze_pages(pages)
        except Exception:
            continue

        files_analyzed += 1

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

        del file_content, file_stream, paragraphs

    gc.collect()
    files_with_gaps = len(set(row["file"] for row in data))

    result_id = uuid.uuid4().hex
    if data:
        filepath = os.path.join(RESULTS_DIR, f"{result_id}.xlsx")
        df = pd.DataFrame(data)
        with pd.ExcelWriter(filepath, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')

    return render_template(
        'result.html',
        data=data,
        files_analyzed=files_analyzed,
        files_with_gaps=files_with_gaps,
        result_id=result_id
    )


@app.route('/download/<result_id>')
def download(result_id):
    if not all(c in '0123456789abcdef' for c in result_id) or len(result_id) != 32:
        return "Invalid download link.", 400

    filepath = os.path.join(RESULTS_DIR, f"{result_id}.xlsx")
    if not os.path.isfile(filepath):
        return "No results available for download. Please process your files first.", 400

    return send_file(
        filepath,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name='results.xlsx'
    )


if __name__ == '__main__':
    debug_mode = os.environ.get("FLASK_ENV") != "production"
    app.run(host="0.0.0.0", port=7860, debug=debug_mode)
