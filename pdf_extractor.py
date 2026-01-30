"""PDF text and metadata extraction utilities."""

import io
import re

from pypdf import PdfReader
from pdfminer.high_level import extract_text


def extract_doi(text: str) -> str | None:
    """Extract the first DOI found in text using the common regex pattern.

    Looks for strings starting with '10.' followed by a valid DOI suffix.
    Case insensitive.
    """
    pattern_doi = r"10\.\d{4,9}/[-._;()/:A-Z0-9]+"
    match = re.search(pattern_doi, text, re.IGNORECASE)
    return match.group(0) if match else None


def extract_metadata(stream: io.BytesIO) -> dict:
    """Return basic metadata dict from PDF: author, title, doi, keywords, num_pages.

    DOI is heuristically extracted from Author/Title/Subject metadata fields.
    """
    stream.seek(0)
    reader = PdfReader(stream)
    metadata = reader.metadata
    num_pages = len(reader.pages)

    metadata_text = ' '.join(filter(None, [
        metadata.get(key) for key in ['/Author', '/Title', '/Subject']
    ]))
    doi = extract_doi(metadata_text)
    keywords = metadata.get('/Keywords', None)

    return {
        "author": metadata.get('/Author'),
        "title": metadata.get('/Title'),
        "doi": doi,
        "keywords": keywords,
        "num_pages": num_pages
    }


def extract_text_from_pdf(stream: io.BytesIO) -> str:
    """Extract all text from the PDF using pdfminer.six."""
    stream.seek(0)
    return extract_text(stream)
