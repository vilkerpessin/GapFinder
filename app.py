from flask import Flask, render_template, request, send_file, session, redirect, url_for
from flask_session import Session
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text
from textblob import TextBlob
import re
import io
import pandas as pd

app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

def extract_doi(text):
    pattern_doi = r"10.\d{4,9}/[-._;()/:A-Z0-9]+"
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

def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

def find_keywords_and_extract_paragraphs(text, keywords):
    pages = text.split('\f')
    results = []
    for page_num, page in enumerate(pages):
        for paragraph in page.split('\n\n'):
            if any(keyword.lower() in paragraph.lower() for keyword in keywords):
                insight_score = analyze_sentiment(paragraph)
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
    keywords = [
        "limitation", "limitação", "research gap", "lacuna", "gap", "shortage", "insufficiency", "lack", "deficiency",
        "inadequacy", "unexplored", "under-researched", "insufficiently studied", "neglected", "unexamined", "sparse",
        "incomplete", "under-theorized", "unaddressed", "overlooked", "underestimated", "uncharted", "knowledge gap"
    ]

    for file in files:
        if not file.filename.endswith('.pdf'):
            continue
        file_stream = io.BytesIO(file.read())
        text = extract_text(file_stream)
        metadata = extract_metadata(file_stream)
        paragraphs = find_keywords_and_extract_paragraphs(text, keywords)

        for page, paragraph, insight in paragraphs:
            data.append({
                "file": file.filename,
                "doi": metadata["doi"],
                "author": metadata["author"],
                "title": metadata["title"],
                "keywords": metadata.get("keywords", ""),
                "page": page,
                "paragraph": paragraph,
                "insight": insight  # Renamed Score to Insight
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
        writer.book.close()
    output.seek(0)
    return send_file(output, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', as_attachment=True, download_name='results.xlsx')

if __name__ == '__main__':
    app.run(debug=True)
