import io
import json
import os
import tempfile

import pytest
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app, RESULTS_DIR


class TestFlaskApp:
    """Basic tests for Flask application routes"""

    @pytest.fixture
    def client(self):
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client

    def test_index_returns_200(self, client):
        response = client.get('/')
        assert response.status_code == 200


class TestUploadRoute:

    @pytest.fixture
    def client(self):
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client

    def test_upload_rejects_no_files(self, client):
        response = client.post('/upload', content_type='multipart/form-data', data={})
        assert response.status_code == 400
        assert "No valid PDF" in response.json["error"]

    def test_upload_rejects_non_pdf(self, client):
        data = {'pdf_files': (io.BytesIO(b'not a pdf'), 'test.txt')}
        response = client.post('/upload', data=data, content_type='multipart/form-data')
        assert response.status_code == 400

    def test_upload_rejects_oversized_file(self, client):
        big_content = b'0' * (21 * 1024 * 1024)
        data = {'pdf_files': (io.BytesIO(big_content), 'big.pdf')}
        response = client.post('/upload', data=data, content_type='multipart/form-data')
        assert response.status_code == 400
        assert "too large" in response.json["error"]


class TestResultRoute:

    @pytest.fixture
    def client(self):
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client

    def test_result_invalid_id_returns_400(self, client):
        response = client.get('/result/not-valid')
        assert response.status_code == 400

    def test_result_missing_returns_404(self, client):
        response = client.get('/result/' + 'a' * 32)
        assert response.status_code == 404

    def test_result_renders_with_valid_json(self, client):
        result_id = 'b' * 32
        json_path = os.path.join(RESULTS_DIR, f"{result_id}.json")
        with open(json_path, 'w') as f:
            json.dump({
                "data": [],
                "files_analyzed": 1,
                "files_with_gaps": 0,
            }, f)
        try:
            response = client.get(f'/result/{result_id}')
            assert response.status_code == 200
        finally:
            os.remove(json_path)


class TestProgressRoute:

    @pytest.fixture
    def client(self):
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client

    def test_progress_invalid_id_returns_400(self, client):
        response = client.get('/progress/not-valid')
        assert response.status_code == 400

    def test_progress_missing_job_returns_404(self, client):
        response = client.get('/progress/' + 'c' * 32)
        assert response.status_code == 404


class TestDownloadRoute:

    @pytest.fixture
    def client(self):
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client

    def test_download_invalid_id_returns_400(self, client):
        response = client.get('/download/not-valid')
        assert response.status_code == 400

    def test_download_missing_returns_400(self, client):
        response = client.get('/download/' + 'd' * 32)
        assert response.status_code == 400
