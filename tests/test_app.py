import io
import json
import os
import tempfile
from unittest.mock import patch

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


class TestAnalyzeLlmRoute:

    @pytest.fixture
    def client(self):
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client

    def test_rejects_invalid_result_id(self, client):
        response = client.post('/analyze-llm/invalid',
                               json={"api_key": "test"})
        assert response.status_code == 400

    def test_rejects_missing_api_key(self, client):
        result_id = 'e' * 32
        json_path = os.path.join(RESULTS_DIR, f"{result_id}.json")
        with open(json_path, 'w') as f:
            json.dump({
                "data": [{"paragraph": "text", "insight": 0.5, "page": 1,
                          "file": "a.pdf", "doi": None, "author": "X",
                          "title": "Y", "keywords": ""}],
                "files_analyzed": 1,             }, f)
        try:
            response = client.post(f'/analyze-llm/{result_id}', json={"api_key": ""})
            assert response.status_code == 400
            assert "required" in response.json["error"].lower()
        finally:
            os.remove(json_path)

    def test_returns_404_for_missing_results(self, client):
        result_id = 'f' * 32
        response = client.post(f'/analyze-llm/{result_id}',
                               json={"api_key": "test"})
        assert response.status_code == 404

    def _setup_result_file(self, result_id):
        json_path = os.path.join(RESULTS_DIR, f"{result_id}.json")
        with open(json_path, 'w') as f:
            json.dump({
                "data": [{"paragraph": "text", "insight": 0.5, "page": 1,
                          "file": "a.pdf", "doi": None, "author": "X",
                          "title": "Y", "keywords": ""}],
                "files_analyzed": 1,             }, f)
        return json_path

    @patch("app.analyze_gaps")
    def test_invalid_key_returns_401(self, mock_verify, client):
        from google.genai import errors as genai_errors
        mock_verify.side_effect = genai_errors.ClientError(
            401, {"error": {"message": "API key not valid", "status": "UNAUTHENTICATED"}})

        result_id = 'b' * 32
        json_path = self._setup_result_file(result_id)
        try:
            response = client.post(f'/analyze-llm/{result_id}', json={"api_key": "bad-key"})
            assert response.status_code == 401
            assert "Invalid API key" in response.json["error"]
        finally:
            os.remove(json_path)

    @patch("app.analyze_gaps")
    def test_rate_limit_returns_429(self, mock_verify, client):
        from google.genai import errors as genai_errors
        mock_verify.side_effect = genai_errors.ClientError(
            429, {"error": {"message": "Resource exhausted", "status": "RESOURCE_EXHAUSTED"}})

        result_id = 'c' * 32
        json_path = self._setup_result_file(result_id)
        try:
            response = client.post(f'/analyze-llm/{result_id}', json={"api_key": "key"})
            assert response.status_code == 429
            assert "Rate limit" in response.json["error"]
        finally:
            os.remove(json_path)

    @patch("app.analyze_gaps")
    def test_server_error_returns_502(self, mock_verify, client):
        from google.genai import errors as genai_errors
        mock_verify.side_effect = genai_errors.ServerError(
            500, {"error": {"message": "Internal error", "status": "INTERNAL"}})

        result_id = 'd' * 32
        json_path = self._setup_result_file(result_id)
        try:
            response = client.post(f'/analyze-llm/{result_id}', json={"api_key": "key"})
            assert response.status_code == 502
            assert "temporarily unavailable" in response.json["error"]
        finally:
            os.remove(json_path)

    @patch("app.analyze_gaps")
    def test_unknown_error_returns_500(self, mock_verify, client):
        mock_verify.side_effect = RuntimeError("unexpected failure")

        result_id = 'a1' * 16
        json_path = self._setup_result_file(result_id)
        try:
            response = client.post(f'/analyze-llm/{result_id}', json={"api_key": "key"})
            assert response.status_code == 500
            assert "unexpected failure" in response.json["error"]
        finally:
            os.remove(json_path)


class TestDownloadReportRoute:

    @pytest.fixture
    def client(self):
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client

    def test_rejects_invalid_id(self, client):
        response = client.get('/download-report/invalid')
        assert response.status_code == 400

    def test_returns_400_when_no_llm_analysis(self, client):
        result_id = 'a1' * 16
        json_path = os.path.join(RESULTS_DIR, f"{result_id}.json")
        with open(json_path, 'w') as f:
            json.dump({
                "data": [], "files_analyzed": 0,             }, f)
        try:
            response = client.get(f'/download-report/{result_id}')
            assert response.status_code == 400
        finally:
            os.remove(json_path)
