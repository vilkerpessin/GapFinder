import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gap_analyzer import (
    analyze_sentiment,
    analyze_pages,
    _join_paragraphs,
    _is_incomplete
)


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
        text = "This research has a significant limitation in the methodology."
        # (page_num, page_text, trailing_text, leading_text)
        pages = iter([(1, text, text, text)])
        results = analyze_pages(pages)
        assert len(results) > 0
        assert results[0][0] == 1
        assert "limitation" in results[0][1].lower()

    def test_returns_empty_for_no_keywords(self):
        text = "This text contains no relevant keywords at all."
        pages = iter([(1, text, text, text)])
        results = analyze_pages(pages)
        assert len(results) == 0


class TestCrossPageParagraphs:
    """Tests for paragraph continuation across pages using visual positions"""

    def test_joins_split_paragraph_using_visual_positions(self):
        # Page 1: trailing_text is incomplete
        # Page 2: leading_text is the continuation
        pages = iter([
            (1, "Some other text.\n\nThis study has a significant limitation in the",
             "This study has a significant limitation in the",  # trailing (visually last)
             "Some other text."),  # leading (visually first)
            (2, "Other content.\n\nmethodology that affects all results.",
             "Other content.",  # trailing
             "methodology that affects all results.")  # leading (continuation)
        ])
        results = analyze_pages(pages)
        # Should find the joined paragraph with "limitation"
        found = any("limitation" in r[1].lower() and "methodology" in r[1].lower() for r in results)
        assert found, f"Expected joined paragraph not found. Results: {results}"

    def test_handles_hyphenated_word_across_pages(self):
        pages = iter([
            (1, "This study has a significant limita-",
             "This study has a significant limita-",
             "This study has a significant limita-"),
            (2, "tion in the methodology.",
             "tion in the methodology.",
             "tion in the methodology.")
        ])
        results = analyze_pages(pages)
        assert len(results) == 1
        assert "limitation" in results[0][1]
        assert "limita-" not in results[0][1]

    def test_preserves_complete_paragraphs(self):
        page1_text = "This paragraph is complete.\n\nThis has a limitation."
        pages = iter([
            (1, page1_text, "This has a limitation.", "This paragraph is complete."),
            (2, "Another complete paragraph.", "Another complete paragraph.", "Another complete paragraph.")
        ])
        results = analyze_pages(pages)
        assert len(results) == 1
        assert results[0][0] == 1


class TestHelperFunctions:
    """Tests for internal helper functions"""

    def test_join_paragraphs_with_hyphen(self):
        result = _join_paragraphs("limita-", "tion")
        assert result == "limitation"

    def test_join_paragraphs_without_hyphen(self):
        result = _join_paragraphs("some text", "more text")
        assert result == "some text more text"

    def test_is_incomplete_without_ending(self):
        assert _is_incomplete("This text continues") is True

    def test_is_incomplete_with_period(self):
        assert _is_incomplete("This text ends.") is False

    def test_is_incomplete_with_question(self):
        assert _is_incomplete("Is this complete?") is False

    def test_is_incomplete_with_quote(self):
        assert _is_incomplete('He said "done"') is False
