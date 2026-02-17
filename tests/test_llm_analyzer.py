import sys
import os
import json
import pytest
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm_analyzer import analyze_gaps, generate_report, GapComment, BATCH_SIZE


def _make_gap(paragraph="This area remains poorly understood in the literature.",
              insight=0.65, page=3, file="paper.pdf"):
    return {
        "file": file, "doi": None, "author": "Smith",
        "title": "Test Paper", "keywords": "",
        "page": page, "paragraph": paragraph, "insight": insight,
    }


class TestAnalyzeGaps:

    def test_empty_input_returns_empty(self):
        assert analyze_gaps("fake-key", []) == []

    @patch("llm_analyzer.genai.Client")
    def test_merges_comments(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.parsed = [
            GapComment(index=0, comment="Identifies understudied area"),
        ]
        mock_client.models.generate_content.return_value = mock_response

        gaps = [_make_gap()]
        result = analyze_gaps("test-key", gaps)

        assert len(result) == 1
        assert result[0]["llm_comment"] == "Identifies understudied area"
        assert result[0]["paragraph"] == gaps[0]["paragraph"]

    @patch("llm_analyzer.genai.Client")
    def test_batching_splits_large_input(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.parsed = []
        mock_client.models.generate_content.return_value = mock_response

        gaps = [_make_gap(page=i) for i in range(15)]
        analyze_gaps("test-key", gaps)

        assert mock_client.models.generate_content.call_count == 2

    @patch("llm_analyzer.genai.Client")
    def test_missing_index_leaves_empty_comment(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.parsed = [
            GapComment(index=0, comment="Clear gap identified"),
            # index=1 missing — LLM skipped it
        ]
        mock_client.models.generate_content.return_value = mock_response

        gaps = [_make_gap(), _make_gap(paragraph="Another paragraph.")]
        result = analyze_gaps("test-key", gaps)

        assert result[0]["llm_comment"] == "Clear gap identified"
        assert result[1]["llm_comment"] == ""

    @patch("llm_analyzer.genai.Client")
    def test_all_batches_fail_raises_error(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_client.models.generate_content.side_effect = RuntimeError("API error")

        gaps = [_make_gap()]
        with pytest.raises(RuntimeError, match="API error"):
            analyze_gaps("test-key", gaps)

    @patch("llm_analyzer.genai.Client")
    def test_partial_batch_failure_returns_results(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        good_response = MagicMock()
        good_response.parsed = [
            GapComment(index=i, comment="Gap noted")
            for i in range(BATCH_SIZE)
        ]

        mock_client.models.generate_content.side_effect = [
            good_response,
            RuntimeError("Rate limit"),
        ]

        gaps = [_make_gap(page=i) for i in range(BATCH_SIZE + 2)]
        result = analyze_gaps("test-key", gaps)

        assert result[0]["llm_comment"] == "Gap noted"
        assert result[BATCH_SIZE]["llm_comment"] == ""


class TestGenerateReport:

    @patch("llm_analyzer.genai.Client")
    def test_generates_report(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.text = "# Report\n\nAnalysis of gaps."
        mock_client.models.generate_content.return_value = mock_response

        gaps = [{**_make_gap(), "llm_comment": "Clear gap"}]
        metadata = [{"title": "Test Paper", "author": "Smith", "doi": None}]

        report = generate_report("test-key", gaps, metadata)
        assert "Report" in report
        mock_client.models.generate_content.assert_called_once()

    @patch("llm_analyzer.genai.Client")
    def test_report_includes_comments_in_prompt(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.text = "# Report"
        mock_client.models.generate_content.return_value = mock_response

        gaps = [{**_make_gap(), "llm_comment": "Methodological limitation"}]

        generate_report("test-key", gaps, [])

        call_args = mock_client.models.generate_content.call_args
        prompt = call_args.kwargs.get("contents") or call_args[1].get("contents")
        assert "Methodological limitation" in prompt
