import json
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from rag_engine import GapFinderAI, _parse_gaps_json, _parse_retry_delay


# ── _parse_gaps_json ─────────────────────────────────────────────────────────

class TestParseGapsJson:

    def test_parses_clean_json_array(self):
        raw = json.dumps([{
            "type": "Methodological",
            "description": "Small sample size",
            "evidence": "n=12 participants",
            "suggestion": "Increase sample to 100+",
        }])
        result = _parse_gaps_json(raw)
        assert len(result) == 1
        assert result[0]["type"] == "Methodological"

    def test_handles_markdown_fences(self):
        raw = '```json\n[{"type": "Empirical", "description": "d", "evidence": "e", "suggestion": "s"}]\n```'
        result = _parse_gaps_json(raw)
        assert len(result) == 1

    def test_handles_preamble_text(self):
        raw = 'Here are the gaps:\n[{"type": "Theoretical", "description": "d", "evidence": "e", "suggestion": "s"}]'
        result = _parse_gaps_json(raw)
        assert len(result) == 1

    def test_returns_empty_for_no_json(self):
        assert _parse_gaps_json("No gaps found in this text.") == []

    def test_returns_empty_for_invalid_json(self):
        assert _parse_gaps_json("[{broken json}]") == []

    def test_skips_items_missing_required_fields(self):
        raw = json.dumps([
            {"type": "Empirical", "description": "d"},
            {"type": "Empirical", "description": "d", "evidence": "e", "suggestion": "s"},
        ])
        result = _parse_gaps_json(raw)
        assert len(result) == 1

    def test_defaults_unknown_type_to_empirical(self):
        raw = json.dumps([{
            "type": "Unknown",
            "description": "d",
            "evidence": "e",
            "suggestion": "s",
        }])
        result = _parse_gaps_json(raw)
        assert result[0]["type"] == "Empirical"

    def test_returns_empty_for_empty_array(self):
        assert _parse_gaps_json("[]") == []

    def test_skips_non_dict_items(self):
        raw = json.dumps(["not a dict", {"type": "Empirical", "description": "d", "evidence": "e", "suggestion": "s"}])
        result = _parse_gaps_json(raw)
        assert len(result) == 1


# ── GapFinderAI init ─────────────────────────────────────────────────────────

class TestGapFinderAIInit:

    def test_rejects_invalid_mode(self):
        with pytest.raises(ValueError, match="Invalid mode"):
            GapFinderAI(mode="invalid")

    def test_cloud_mode_requires_api_key(self):
        with pytest.raises(ValueError, match="API key"):
            GapFinderAI(mode="cloud")

    @patch("rag_engine.HuggingFaceEmbeddings")
    @patch("rag_engine.genai.Client")
    def test_cloud_mode_initializes(self, mock_client, mock_embeddings):
        engine = GapFinderAI(mode="cloud", api_key="test-key")
        assert engine.mode == "cloud"
        mock_client.assert_called_once_with(api_key="test-key")

    @patch("rag_engine.HuggingFaceEmbeddings")
    @patch("rag_engine.Path.exists", return_value=False)
    def test_local_mode_raises_without_model(self, mock_exists, mock_embeddings):
        with pytest.raises(FileNotFoundError, match="GGUF model not found"):
            GapFinderAI(mode="local")


# ── Ingestion ────────────────────────────────────────────────────────────────

class TestIngestPdf:

    @patch("rag_engine.HuggingFaceEmbeddings")
    @patch("rag_engine.genai.Client")
    @patch("rag_engine.Chroma")
    @patch("rag_engine.PyMuPDFLoader")
    def test_ingest_creates_vectorstore(self, mock_loader_cls, mock_chroma, mock_client, mock_embeddings):
        mock_doc = MagicMock()
        mock_doc.page_content = "Some text about research limitations."
        mock_doc.metadata = {"page": 1}
        mock_loader = MagicMock()
        mock_loader.load.return_value = [mock_doc]
        mock_loader_cls.return_value = mock_loader

        mock_chroma.from_documents.return_value = MagicMock()

        engine = GapFinderAI(mode="cloud", api_key="test-key")
        count = engine.ingest_pdf(b"%PDF-1.4 fake content", "test.pdf")

        assert count > 0
        mock_chroma.from_documents.assert_called_once()

    @patch("rag_engine.HuggingFaceEmbeddings")
    @patch("rag_engine.genai.Client")
    @patch("rag_engine.PyMuPDFLoader")
    def test_ingest_returns_zero_for_empty_pdf(self, mock_loader_cls, mock_client, mock_embeddings):
        mock_loader = MagicMock()
        mock_loader.load.return_value = []
        mock_loader_cls.return_value = mock_loader

        engine = GapFinderAI(mode="cloud", api_key="test-key")
        count = engine.ingest_pdf(b"%PDF-1.4 empty", "empty.pdf")
        assert count == 0


# ── Analysis ─────────────────────────────────────────────────────────────────

class TestAnalyzeGaps:

    @patch("rag_engine.HuggingFaceEmbeddings")
    @patch("rag_engine.genai.Client")
    def test_raises_without_ingestion(self, mock_client, mock_embeddings):
        engine = GapFinderAI(mode="cloud", api_key="test-key")
        with pytest.raises(RuntimeError, match="No PDF ingested"):
            engine.analyze_gaps()

    @patch("rag_engine.HuggingFaceEmbeddings")
    @patch("rag_engine.genai.Client")
    def test_cloud_analysis_calls_gemini(self, mock_client_cls, mock_embeddings):
        engine = GapFinderAI(mode="cloud", api_key="test-key")

        mock_doc = MagicMock()
        mock_doc.page_content = "This study has methodological limitations."
        mock_doc.metadata = {"page": 1}

        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = [mock_doc]
        engine._vectorstore = mock_vs

        mock_response = MagicMock()
        mock_response.text = json.dumps([{
            "type": "Methodological",
            "description": "Small sample",
            "evidence": "n=12",
            "suggestion": "Increase sample",
        }])
        engine._gemini_client.models.generate_content.return_value = mock_response

        gaps = engine.analyze_gaps()
        assert len(gaps) == 1
        assert gaps[0]["type"] == "Methodological"
        engine._gemini_client.models.generate_content.assert_called_once()

    @patch("rag_engine.HuggingFaceEmbeddings")
    @patch("rag_engine.genai.Client")
    def test_returns_empty_when_no_chunks_retrieved(self, mock_client_cls, mock_embeddings):
        engine = GapFinderAI(mode="cloud", api_key="test-key")

        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = []
        engine._vectorstore = mock_vs

        gaps = engine.analyze_gaps()
        assert gaps == []


# ── Retry logic ─────────────────────────────────────────────────────────────

class TestRetryLogic:

    def test_parse_retry_delay_extracts_seconds(self):
        msg = "Please retry in 38.316348874s."
        assert _parse_retry_delay(msg) == 38

    def test_parse_retry_delay_default_on_missing(self):
        assert _parse_retry_delay("some error") == 40

    @patch("rag_engine.time.sleep")
    @patch("rag_engine.HuggingFaceEmbeddings")
    @patch("rag_engine.genai.Client")
    def test_gemini_retries_on_429(self, mock_client_cls, mock_embeddings, mock_sleep):
        from google.genai import errors as genai_errors

        engine = GapFinderAI(mode="cloud", api_key="test-key")

        mock_doc = MagicMock()
        mock_doc.page_content = "Study has limitations."
        mock_doc.metadata = {"page": 1}

        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = [mock_doc]
        engine._vectorstore = mock_vs

        error_429 = genai_errors.ClientError(
            429, {"error": {"message": "retry in 38s"}})
        success_response = MagicMock()
        success_response.text = json.dumps([{
            "type": "Empirical", "description": "d",
            "evidence": "e", "suggestion": "s",
        }])

        engine._gemini_client.models.generate_content.side_effect = [
            error_429, success_response,
        ]

        callback_msgs = []
        gaps = engine.analyze_gaps(progress_callback=callback_msgs.append)

        assert len(gaps) == 1
        mock_sleep.assert_called_once_with(38)
        assert any("Rate limited" in msg for msg in callback_msgs)


# ── Cleanup ──────────────────────────────────────────────────────────────────

class TestCleanup:

    @patch("rag_engine.HuggingFaceEmbeddings")
    @patch("rag_engine.genai.Client")
    def test_cleanup_deletes_collection(self, mock_client, mock_embeddings):
        engine = GapFinderAI(mode="cloud", api_key="test-key")
        mock_vs = MagicMock()
        engine._vectorstore = mock_vs

        engine.cleanup()
        mock_vs.delete_collection.assert_called_once()
        assert engine._vectorstore is None

    @patch("rag_engine.HuggingFaceEmbeddings")
    @patch("rag_engine.genai.Client")
    def test_cleanup_noop_without_vectorstore(self, mock_client, mock_embeddings):
        engine = GapFinderAI(mode="cloud", api_key="test-key")
        engine.cleanup()
        assert engine._vectorstore is None
