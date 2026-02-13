import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gap_analyzer import (
    calculate_similarity_score,
    analyze_pages,
    _join_paragraphs,
    _is_incomplete
)


class TestSemanticSimilarity:
    """Sentence-transformers semantic similarity tests"""

    def test_returns_valid_score_range(self):
        score = calculate_similarity_score("This research presents interesting findings")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_high_similarity_for_gap_text(self):
        text = "This research reveals a significant gap in understanding the underlying mechanisms"
        score = calculate_similarity_score(text)
        assert score >= 0.6, f"Expected score >= 0.6 for gap text, got {score}"

    def test_low_similarity_for_unrelated_text(self):
        text = "The weather was sunny and warm today with clear blue skies"
        score = calculate_similarity_score(text)
        assert score < 0.5, f"Expected score < 0.5 for unrelated text, got {score}"

    def test_model_loads_lazily(self):
        # This test just verifies the model loads without error
        score = calculate_similarity_score("A limitation of this study")
        assert score > 0.0


class TestThresholdFiltering:
    """Tests for threshold and top-K filtering"""

    def test_respects_threshold(self):
        pages = iter([
            (1, "This has a major limitation.\n\nThis has a gap.\n\nThis has a shortage.",
             None, None)
        ])
        # Use a high threshold that filters out low-scoring matches
        results = analyze_pages(pages, threshold=0.5, top_k=100)
        assert all(r[2] >= 0.5 for r in results), f"Found scores below threshold: {[r[2] for r in results]}"

    def test_returns_top_k(self):
        # Create many paragraphs with gaps
        gaps_text = "\n\n".join([f"This study has a significant limitation in aspect {i}." for i in range(30)])
        pages = iter([(1, gaps_text, None, None)])
        results = analyze_pages(pages, threshold=0.0, top_k=5)
        assert len(results) <= 5, f"Expected at most 5 results, got {len(results)}"

    def test_handles_unrelated_input(self):
        pages = iter([(1, "The cat sat on the warm windowsill.", None, None)])
        results = analyze_pages(pages)
        assert len(results) == 0


class TestGapDetection:
    """Semantic research gap detection tests"""

    def test_detects_gap_with_keywords(self):
        text = "This research has a significant limitation in the methodology used to evaluate the proposed framework."
        pages = iter([(1, text, text, text)])
        results = analyze_pages(pages)
        assert len(results) > 0
        assert results[0][0] == 1
        assert "limitation" in results[0][1].lower()

    def test_detects_gap_without_keywords(self):
        text = "Current models fail to account for temporal dynamics in cross-cultural contexts."
        pages = iter([(1, text, text, text)])
        results = analyze_pages(pages, threshold=0.3)
        assert len(results) > 0, f"Expected semantic gap detection without keywords, got no results"

    def test_returns_empty_for_unrelated_text(self):
        text = "The weather was sunny and warm today with clear blue skies."
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
            (2, "tion in the methodology used to evaluate the proposed framework and its implications.",
             "tion in the methodology used to evaluate the proposed framework and its implications.",
             "tion in the methodology used to evaluate the proposed framework and its implications.")
        ])
        results = analyze_pages(pages)
        assert len(results) == 1
        assert "limitation" in results[0][1]
        assert "limita-" not in results[0][1]

    def test_preserves_complete_paragraphs(self):
        gap_text = "This study has a significant limitation in the methodology used to evaluate the proposed framework."
        page1_text = f"This paragraph is complete and does not contain anything relevant to research.\n\n{gap_text}"
        pages = iter([
            (1, page1_text, gap_text, "This paragraph is complete and does not contain anything relevant to research."),
            (2, "Another complete paragraph about unrelated topics.", "Another complete paragraph about unrelated topics.", "Another complete paragraph about unrelated topics.")
        ])
        results = analyze_pages(pages, threshold=0.4)
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


class TestProgressCallback:
    """Tests for analyze_pages progress_callback support"""

    def test_callback_receives_scoring_messages(self):
        messages = []
        text = "This study has a significant limitation in the methodology used to evaluate the proposed framework."
        pages = iter([(1, text, text, text)])
        analyze_pages(pages, threshold=0.0, progress_callback=messages.append)
        assert any("Scored" in m for m in messages)
        assert any("Scoring" in m for m in messages)

    def test_works_without_callback(self):
        text = "This study has a significant limitation in the methodology used to evaluate the proposed framework."
        pages = iter([(1, text, text, text)])
        results = analyze_pages(pages, threshold=0.0)
        assert isinstance(results, list)
