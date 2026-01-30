import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gap_analyzer import analyze_sentiment, analyze_text


class TestSentimentAnalysis:
    """VADER sentiment integration tests"""

    def test_returns_valid_score_range(self):
        score = analyze_sentiment("This research presents interesting findings")
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0

    def test_detects_strongly_positive(self):
        score = analyze_sentiment("This is excellent and amazing research with outstanding results!")
        assert score > 0.5, score

    def test_detects_strongly_negative(self):
        score = analyze_sentiment("This research is terrible, flawed, and completely worthless.")
        assert score < -0.5, score

    def test_handles_neutral_text(self):
        score = analyze_sentiment("The study examined data from multiple sources.")
        assert -0.3 <= score <= 0.3, score


class TestGapDetection:
    """Keyword-based research gap detection tests"""

    def test_finds_limitation_keyword(self):
        results = analyze_text("This research has a significant limitation in the methodology.")
        assert len(results) > 0
        assert results[0][0] == 1
        assert "limitation" in results[0][1].lower()

    def test_returns_empty_for_no_keywords(self):
        results = analyze_text("This text contains no relevant keywords at all.")
        assert len(results) == 0