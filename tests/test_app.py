import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app


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