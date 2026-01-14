from flask import Flask, render_template, request, send_file, session, redirect, url_for
from flask_session import Session
from pypdf import PdfReader
from pdfminer.high_level import extract_text
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from cachelib.file import FileSystemCache
import re
import io
import pandas as pd
import os

app = Flask(__name__)

app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "cachelib"
app.config["SESSION_CACHELIB"] = FileSystemCache(cache_dir="flask_session")
Session(app)

def extract_doi(text):
    pattern_doi = r"10\.\d{4,9}/[-._;()/:A-Z0-9]+"
    match = re.search(pattern_doi, text, re.IGNORECASE)
    return match.group(0) if match else None

def extract_metadata(stream):
    stream.seek(0)
    reader = PdfReader(stream)
    metadata = reader.metadata
    num_pages = len(reader.pages)
    metadata_text = ' '.join(filter(None, [metadata.get(key) for key in ['/Author', '/Title', '/Subject']]))
    doi = extract_doi(metadata_text)
    keywords = metadata.get('/Keywords', None)

    return {
        "author": metadata.get('/Author'),
        "title": metadata.get('/Title'),
        "doi": doi,
        "keywords": keywords,
        "num_pages": num_pages
    }

_sentiment_analyzer = SentimentIntensityAnalyzer()

_GAP_KEYWORDS = [
    "limitation", "limitação", "research gap", "lacuna", "gap", "shortage", "insufficiency", "lack", "deficiency",
    "inadequacy", "unexplored", "under-researched", "insufficiently studied", "neglected", "unexamined", "sparse",
    "incomplete", "under-theorized", "unaddressed", "overlooked", "underestimated", "uncharted", "knowledge gap"
]

_keyword_pattern = re.compile(
    r'\b(' + '|'.join([re.escape(kw) for kw in _GAP_KEYWORDS]) + r')\b',
    re.IGNORECASE
)

def analyze_sentiment_vader(text):
    sentiment = _sentiment_analyzer.polarity_scores(text)
    return sentiment['compound']  # Retorna o score composto que é um resumo do sentimento

def find_keywords_and_extract_paragraphs(text):
    pages = text.split('\f')
    results = []
    for page_num, page in enumerate(pages):
        for paragraph in page.split('\n\n'):
            if _keyword_pattern.search(paragraph):
                insight_score = analyze_sentiment_vader(paragraph)
                results.append((page_num + 1, paragraph, insight_score))
    return results

@app.route('/', methods=['GET', 'POST'])
def index():
    session.clear()  # Clear session on new GET request
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_files():
    files = request.files.getlist('pdf_files')
    data = []
    
    MAX_FILE_SIZE = 20 * 1024 * 1024
    MAX_TOTAL_SIZE = 50 * 1024 * 1024
    total_size = 0

    for file in files:
        if not file.filename.endswith('.pdf'):
            continue

        file_content = file.read()
        file_size = len(file_content)
        
        if file_size > MAX_FILE_SIZE:
            return f"Error: File '{file.filename}' is too large ({file_size / 1024 / 1024:.1f}MB). Maximum size is {MAX_FILE_SIZE / 1024 / 1024}MB per file.", 400
        
        total_size += file_size
        if total_size > MAX_TOTAL_SIZE:
            return f"Error: Total file size exceeds {MAX_TOTAL_SIZE / 1024 / 1024}MB limit. Please upload fewer or smaller files.", 400
        
        file_stream = io.BytesIO(file_content)

        try:
            text = extract_text(file_stream)
        except Exception as e:
            continue

        file_stream.seek(0)
        try:
            metadata = extract_metadata(file_stream)
        except Exception as e:
            continue

        paragraphs = find_keywords_and_extract_paragraphs(text)

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
    return send_file(output, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', as_attachment=True, download_name='results.xlsx')

if __name__ == '__main__':
    debug_mode = os.environ.get("FLASK_ENV") != "production"
    app.run(debug=debug_mode)
