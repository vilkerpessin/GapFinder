"""Smoke tests for Streamlit app — verifies imports and constants."""


def test_app_module_imports():
    from app import GAP_TYPE_COLORS
    assert "Methodological" in GAP_TYPE_COLORS
    assert "Theoretical" in GAP_TYPE_COLORS
    assert "Contextual" in GAP_TYPE_COLORS
    assert "Empirical" in GAP_TYPE_COLORS


def test_rag_engine_importable():
    from rag_engine import GapFinderAI, _parse_gaps_json
    assert callable(GapFinderAI)
    assert callable(_parse_gaps_json)


def test_pdf_extractor_importable():
    from pdf_extractor import extract_metadata, extract_text_by_page, extract_doi
    assert callable(extract_metadata)
    assert callable(extract_text_by_page)
    assert callable(extract_doi)
