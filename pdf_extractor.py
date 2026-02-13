"""PDF text and metadata extraction using PyMuPDF with column-aware layout."""

import gc
import io
import re
from collections.abc import Iterator

import fitz


def extract_doi(text: str) -> str | None:
    """Extract the first DOI found in text using the common regex pattern."""
    pattern_doi = r"10\.\d{4,9}/[-._;()/:A-Z0-9]+"
    match = re.search(pattern_doi, text, re.IGNORECASE)
    return match.group(0) if match else None


def extract_metadata(stream: io.BytesIO) -> dict:
    """Return basic metadata dict from PDF: author, title, doi, keywords, num_pages."""
    stream.seek(0)
    with fitz.open(stream=stream, filetype="pdf") as doc:
        metadata = doc.metadata
        num_pages = len(doc)

        metadata_text = ' '.join(filter(None, [
            metadata.get("author"),
            metadata.get("title"),
            metadata.get("subject")
        ]))
        doi = extract_doi(metadata_text)

        return {
            "author": metadata.get("author"),
            "title": metadata.get("title"),
            "doi": doi,
            "keywords": metadata.get("keywords"),
            "num_pages": num_pages
        }


HEADER_MARGIN = 60  # points (~0.8 inch) from top
FOOTER_MARGIN = 60  # points (~0.8 inch) from bottom


def _extract_blocks_column_aware(page: fitz.Page) -> tuple[str, str | None, str | None]:
    """Extract text from page respecting two-column layout.

    Filters out header/footer areas and reads left column before right.
    Returns (page_text, trailing_text, leading_text) where trailing/leading
    are the visually-last/first blocks for cross-page continuation.
    """
    blocks = page.get_text("blocks")
    text_blocks = [b for b in blocks if b[6] == 0]

    if not text_blocks:
        return "", None, None

    page_width = page.rect.width
    page_height = page.rect.height
    midpoint = page_width / 2

    header_limit = HEADER_MARGIN
    footer_limit = page_height - FOOTER_MARGIN

    left_blocks = []
    right_blocks = []
    all_blocks_with_y = []  # Track (y0, text) for all valid blocks

    for block in text_blocks:
        x0, y0, x1, y1, text, block_no, block_type = block

        # Skip blocks entirely within header/footer margins
        if y1 < header_limit or y0 > footer_limit:
            continue

        all_blocks_with_y.append((y0, text))
        block_center_x = (x0 + x1) / 2

        if block_center_x < midpoint:
            left_blocks.append((y0, text))
        else:
            right_blocks.append((y0, text))

    if not all_blocks_with_y:
        return "", None, None

    left_blocks.sort(key=lambda b: b[0])
    right_blocks.sort(key=lambda b: b[0])

    ordered_texts = [text for _, text in left_blocks] + [text for _, text in right_blocks]

    # Find visually-first (lowest y0) and visually-last (highest y0) blocks
    all_blocks_with_y.sort(key=lambda b: b[0])
    leading_text = all_blocks_with_y[0][1] if all_blocks_with_y else None
    trailing_text = all_blocks_with_y[-1][1] if all_blocks_with_y else None

    return "\n\n".join(ordered_texts), trailing_text, leading_text


def extract_text_by_page(stream: io.BytesIO) -> Iterator[tuple[int, str, str | None, str | None]]:
    """Yield (page_number, page_text, trailing_text, leading_text) tuples.

    Memory-efficient: only one page loaded at a time.
    Column-aware: reads left column before right in two-column layouts.

    trailing_text: text from visually-last block (for cross-page continuation)
    leading_text: text from visually-first block (for joining with previous page)
    """
    stream.seek(0)
    doc = fitz.open(stream=stream, filetype="pdf")

    try:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text, trailing_text, leading_text = _extract_blocks_column_aware(page)
            yield (page_num + 1, page_text, trailing_text, leading_text)
            del page
    finally:
        doc.close()
        gc.collect()
