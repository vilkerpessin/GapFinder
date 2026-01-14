import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app, extract_doi, analyze_sentiment_vader, find_keywords_and_extract_paragraphs


class TestExtractDOI:
    """Test DOI extraction functionality"""

    def test_extract_doi_with_standard_format(self):
        """Validate DOI regex pattern works with standard format"""
        text = "This paper DOI: 10.1234/example.5678 is important"
        result = extract_doi(text)
        assert result == "10.1234/example.5678"

    def test_extract_doi_not_found(self):
        """Ensure function returns None when no DOI is present"""
        text = "This text has no DOI"
        result = extract_doi(text)
        assert result is None


class TestSentimentAnalysis:
    """Test sentiment analysis integration"""

    def test_analyze_sentiment_returns_valid_score(self):
        """Validate VADER integration returns correct type and range"""
        text = "This research presents interesting findings"
        score = analyze_sentiment_vader(text)
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0, "Sentiment score should be between -1 and 1"

    def test_vader_sentiment_positive_text(self):
        """Validate VADER detects positive sentiment correctly"""
        text = "This is excellent and amazing research with outstanding results!"
        score = analyze_sentiment_vader(text)
        assert score > 0.5, f"Expected positive score > 0.5, got {score}"

    def test_vader_sentiment_negative_text(self):
        """Validate VADER detects negative sentiment correctly"""
        text = "This research is terrible, flawed, and completely worthless."
        score = analyze_sentiment_vader(text)
        assert score < -0.5, f"Expected negative score < -0.5, got {score}"

    def test_vader_sentiment_neutral_text(self):
        """Validate VADER handles neutral text appropriately"""
        text = "The study examined data from multiple sources."
        score = analyze_sentiment_vader(text)
        assert -0.3 <= score <= 0.3, f"Expected neutral score near 0, got {score}"


class TestKeywordExtraction:
    """Test core gap detection functionality"""

    def test_find_keywords_detects_limitation(self):
        """Validate core functionality: detecting research gaps"""
        text = "This research has a significant limitation in the methodology."
        results = find_keywords_and_extract_paragraphs(text)
        assert len(results) > 0, "Should detect 'limitation' keyword"
        assert results[0][0] == 1, "Should be on page 1"
        assert "limitation" in results[0][1].lower()

    def test_find_keywords_no_matches(self):
        """Ensure function handles text without gaps gracefully"""
        text = "This text contains no relevant keywords at all."
        results = find_keywords_and_extract_paragraphs(text)
        assert len(results) == 0, "Should return empty list when no keywords found"


class TestFlaskApp:
    """Test Flask application routes"""

    @pytest.fixture
    def client(self):
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client

    def test_index_route(self, client):
        """Sanity check: main page loads correctly"""
        response = client.get('/')
        assert response.status_code == 200
