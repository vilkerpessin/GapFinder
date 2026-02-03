import io
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pdf_extractor import extract_doi, extract_metadata, extract_text_by_page


class TestModuleImports:
    """Verify all PDF extraction dependencies are available."""

    def test_fitz_import(self):
        import fitz
        assert hasattr(fitz, 'open')


class TestExtractDOI:
    """Tests for DOI extraction via regex"""

    def test_matches_standard_doi(self):
        text = "This paper DOI: 10.1234/example.5678 is important"
        assert extract_doi(text) == "10.1234/example.5678"

    def test_returns_none_when_no_doi(self):
        text = "This text has no DOI"
        assert extract_doi(text) is None