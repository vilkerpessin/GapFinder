import gc
import io
import json
import os
import tempfile
import threading
import time
import uuid

import pandas as pd
from flask import Flask, Response, jsonify, render_template, request, send_file

from pdf_extractor import extract_metadata, extract_text_by_page
from gap_analyzer import analyze_pages


app = Flask(__name__)

app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")

RESULTS_DIR = os.path.join(tempfile.gettempdir(), "gapfinder_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB per file
MAX_TOTAL_SIZE = 50 * 1024 * 1024  # 50MB total

# SSE job state — shared between request threads and background processing threads
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _is_valid_hex_id(hex_id: str) -> bool:
    return len(hex_id) == 32 and all(c in '0123456789abcdef' for c in hex_id)


def _push_progress(job_id: str, message: str, event: str = "progress") -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id]["progress"].append({"event": event, "data": message})


def _cleanup_stale_jobs(max_age: int = 3600) -> None:
    now = time.time()
    with _jobs_lock:
        stale = [jid for jid, j in _jobs.items() if now - j["created_at"] > max_age]
        for jid in stale:
            for path in _jobs[jid].get("files", []):
                try:
                    os.remove(path)
                except OSError:
                    pass
            job_dir = _jobs[jid].get("job_dir")
            if job_dir:
                try:
                    os.rmdir(job_dir)
                except OSError:
                    pass
            del _jobs[jid]


def _process_job(job_id: str, saved_files: list[tuple[str, str]]) -> None:
    """Process uploaded PDFs in background, pushing SSE progress events."""
    data = []
    files_analyzed = 0
    total_files = len(saved_files)

    try:
        for file_idx, (filename, filepath) in enumerate(saved_files, 1):
            _push_progress(job_id, f"Processing file {file_idx} of {total_files}: {filename}")

            with open(filepath, 'rb') as f:
                file_content = f.read()
            file_stream = io.BytesIO(file_content)

            try:
                metadata = extract_metadata(file_stream)
            except Exception:
                _push_progress(job_id, f"Skipping {filename}: could not read metadata")
                continue

            file_stream.seek(0)
            _push_progress(job_id, "Extracting text...")

            try:
                pages = extract_text_by_page(file_stream)
                paragraphs = analyze_pages(
                    pages,
                    progress_callback=lambda msg: _push_progress(job_id, msg)
                )
            except Exception:
                _push_progress(job_id, f"Skipping {filename}: analysis failed")
                continue

            files_analyzed += 1

            for page, paragraph, insight in paragraphs:
                data.append({
                    "file": filename,
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

        result_id = uuid.uuid4().hex
        files_with_gaps = len(set(row["file"] for row in data))

        if data:
            xlsx_path = os.path.join(RESULTS_DIR, f"{result_id}.xlsx")
            df = pd.DataFrame(data)
            with pd.ExcelWriter(xlsx_path, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Sheet1')

        # Save result data as JSON for the /result page
        json_path = os.path.join(RESULTS_DIR, f"{result_id}.json")
        with open(json_path, 'w') as f:
            json.dump({
                "data": data,
                "files_analyzed": files_analyzed,
                "files_with_gaps": files_with_gaps,
            }, f)

        with _jobs_lock:
            _jobs[job_id]["status"] = "complete"
            _jobs[job_id]["result_id"] = result_id

        _push_progress(job_id, json.dumps({"result_id": result_id}), event="complete")

    except Exception as e:
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error_message"] = str(e)
        _push_progress(job_id, str(e), event="error")

    finally:
        # Clean up uploaded temp files
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job:
                for path in job.get("files", []):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                job_dir = job.get("job_dir")
                if job_dir:
                    try:
                        os.rmdir(job_dir)
                    except OSError:
                        pass


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_files():
    _cleanup_stale_jobs()

    files = request.files.getlist('pdf_files')
    total_size = 0
    saved_files = []

    job_id = uuid.uuid4().hex
    job_dir = os.path.join(tempfile.gettempdir(), f"gapfinder_upload_{job_id}")
    os.makedirs(job_dir, exist_ok=True)

    for file in files:
        if not file.filename.endswith('.pdf'):
            continue

        file_content = file.read()
        file_size = len(file_content)

        if file_size > MAX_FILE_SIZE:
            return jsonify({
                "error": f"File '{file.filename}' is too large "
                         f"({file_size / 1024 / 1024:.1f}MB). "
                         f"Maximum size is {MAX_FILE_SIZE / 1024 / 1024:.0f}MB per file."
            }), 400

        total_size += file_size
        if total_size > MAX_TOTAL_SIZE:
            return jsonify({
                "error": f"Total file size exceeds "
                         f"{MAX_TOTAL_SIZE / 1024 / 1024:.0f}MB limit. "
                         "Please upload fewer or smaller files."
            }), 400

        filepath = os.path.join(job_dir, file.filename)
        with open(filepath, 'wb') as f:
            f.write(file_content)
        saved_files.append((file.filename, filepath))

    if not saved_files:
        return jsonify({"error": "No valid PDF files uploaded."}), 400

    with _jobs_lock:
        _jobs[job_id] = {
            "status": "processing",
            "progress": [],
            "result_id": None,
            "error_message": None,
            "files": [fp for _, fp in saved_files],
            "job_dir": job_dir,
            "created_at": time.time(),
        }

    thread = threading.Thread(target=_process_job, args=(job_id, saved_files), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route('/progress/<job_id>')
def progress_stream(job_id):
    if not _is_valid_hex_id(job_id):
        return "Invalid job ID.", 400

    with _jobs_lock:
        if job_id not in _jobs:
            return "Job not found.", 404

    def generate():
        cursor = 0
        while True:
            with _jobs_lock:
                job = _jobs.get(job_id)
                if not job:
                    break
                events = job["progress"][cursor:]
                cursor += len(events)
                status = job["status"]

            for evt in events:
                yield f"event: {evt['event']}\ndata: {evt['data']}\n\n"

            if status in ("complete", "error"):
                break

            time.sleep(0.5)

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        }
    )


@app.route('/result/<result_id>')
def show_result(result_id):
    if not _is_valid_hex_id(result_id):
        return "Invalid result ID.", 400

    json_path = os.path.join(RESULTS_DIR, f"{result_id}.json")
    if not os.path.isfile(json_path):
        return "Results not found. They may have expired.", 404

    with open(json_path, 'r') as f:
        result = json.load(f)

    return render_template(
        'result.html',
        data=result["data"],
        files_analyzed=result["files_analyzed"],
        files_with_gaps=result["files_with_gaps"],
        result_id=result_id
    )


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
    if not _is_valid_hex_id(result_id):
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
