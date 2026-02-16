import sys
import os
import json
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm_analyzer import verify_gaps, generate_report, GapVerdict, VERIFY_BATCH_SIZE


def _make_gap(paragraph="This area remains poorly understood in the literature.",
              insight=0.65, page=3, file="paper.pdf"):
    return {
        "file": file, "doi": None, "author": "Smith",
        "title": "Test Paper", "keywords": "",
        "page": page, "paragraph": paragraph, "insight": insight,
    }


class TestVerifyGaps:

    def test_empty_input_returns_empty(self):
        assert verify_gaps("fake-key", []) == []

    @patch("llm_analyzer.genai.Client")
    def test_merges_verdicts_with_evidence(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.parsed = [
            GapVerdict(
                index=0, verdict="confirmed_gap",
                reason="Explicitly identifies understudied area",
                evidence_quote="remains poorly understood",
            ),
        ]
        mock_client.models.generate_content.return_value = mock_response

        gaps = [_make_gap()]
        result = verify_gaps("test-key", gaps)

        assert len(result) == 1
        assert result[0]["verdict"] == "confirmed_gap"
        assert result[0]["evidence_quote"] == "remains poorly understood"
        assert result[0]["paragraph"] == gaps[0]["paragraph"]

    @patch("llm_analyzer.genai.Client")
    def test_downgrades_when_quote_not_in_source(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.parsed = [
            GapVerdict(
                index=0, verdict="confirmed_gap",
                reason="Identifies a gap",
                evidence_quote="words that do not exist in the paragraph",
            ),
        ]
        mock_client.models.generate_content.return_value = mock_response

        gaps = [_make_gap()]
        result = verify_gaps("test-key", gaps)

        assert result[0]["verdict"] == "uncertain"
        assert "not found" in result[0]["reason"].lower()

    @patch("llm_analyzer.genai.Client")
    def test_downgrades_when_quote_is_empty(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.parsed = [
            GapVerdict(
                index=0, verdict="confirmed_gap",
                reason="Some reason", evidence_quote="",
            ),
        ]
        mock_client.models.generate_content.return_value = mock_response

        gaps = [_make_gap()]
        result = verify_gaps("test-key", gaps)

        assert result[0]["verdict"] == "uncertain"

    @patch("llm_analyzer.genai.Client")
    def test_batching_splits_large_input(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.parsed = []
        mock_client.models.generate_content.return_value = mock_response

        gaps = [_make_gap(page=i) for i in range(15)]
        verify_gaps("test-key", gaps)

        assert mock_client.models.generate_content.call_count == 2

    @patch("llm_analyzer.genai.Client")
    def test_missing_index_falls_back_to_uncertain(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.parsed = [
            GapVerdict(
                index=0, verdict="confirmed_gap",
                reason="Clear gap",
                evidence_quote="remains poorly understood",
            ),
            # index=1 is missing — the LLM skipped it
        ]
        mock_client.models.generate_content.return_value = mock_response

        gaps = [_make_gap(), _make_gap(paragraph="Another paragraph about something.")]
        result = verify_gaps("test-key", gaps)

        assert result[0]["verdict"] == "confirmed_gap"
        assert result[1]["verdict"] == "uncertain"

    @patch("llm_analyzer.genai.Client")
    def test_batch_api_error_marks_batch_uncertain(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_client.models.generate_content.side_effect = RuntimeError("API error")

        gaps = [_make_gap()]
        result = verify_gaps("test-key", gaps)

        assert result[0]["verdict"] == "uncertain"


class TestGenerateReport:

    @patch("llm_analyzer.genai.Client")
    def test_generates_report_for_confirmed_gaps(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.text = "# Report\n\nAnalysis of confirmed gaps."
        mock_client.models.generate_content.return_value = mock_response

        gaps = [
            {**_make_gap(), "verdict": "confirmed_gap",
             "reason": "Clear gap", "evidence_quote": "remains poorly understood"},
        ]
        metadata = [{"title": "Test Paper", "author": "Smith", "doi": None}]

        report = generate_report("test-key", gaps, metadata)
        assert "Report" in report
        mock_client.models.generate_content.assert_called_once()

    def test_no_confirmed_gaps_skips_api_call(self):
        gaps = [
            {**_make_gap(), "verdict": "false_positive",
             "reason": "Literature review", "evidence_quote": "studies have shown"},
        ]

        report = generate_report("test-key", gaps, [])
        assert "No confirmed research gaps" in report

    @patch("llm_analyzer.genai.Client")
    def test_report_includes_evidence_in_prompt(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.text = "# Report"
        mock_client.models.generate_content.return_value = mock_response

        gaps = [
            {**_make_gap(), "verdict": "confirmed_gap",
             "reason": "Gap found", "evidence_quote": "remains poorly understood"},
        ]

        generate_report("test-key", gaps, [])

        call_args = mock_client.models.generate_content.call_args
        prompt = call_args.kwargs.get("contents") or call_args[1].get("contents")
        assert "remains poorly understood" in prompt
