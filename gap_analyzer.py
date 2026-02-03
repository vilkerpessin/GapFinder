"""Detect potential research gaps via keyword presence and sentiment polarity."""

import re
from collections.abc import Iterator

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


GAP_KEYWORDS = [
    "limitation", "limitação", "research gap", "lacuna", "gap", "shortage",
    "insufficiency", "lack", "deficiency", "inadequacy", "unexplored",
    "under-researched", "insufficiently studied", "neglected", "unexamined",
    "sparse", "incomplete", "under-theorized", "unaddressed", "overlooked",
    "underestimated", "uncharted", "knowledge gap"
]

_sentiment_analyzer = SentimentIntensityAnalyzer()

_keyword_pattern = re.compile(
    r'\b(' + '|'.join([re.escape(kw) for kw in GAP_KEYWORDS]) + r')\b',
    re.IGNORECASE
)

SENTENCE_ENDINGS = ('.', '!', '?', '"', ')', ']')


def analyze_sentiment(text: str) -> float:
    """Return VADER compound sentiment score (-1.0 to 1.0)."""
    sentiment = _sentiment_analyzer.polarity_scores(text)
    return sentiment['compound']


def _join_paragraphs(previous: str, current: str) -> str:
    """Join two paragraph fragments, handling hyphenation."""
    previous = previous.rstrip()
    current = current.lstrip()

    if previous.endswith('-'):
        return previous[:-1] + current
    else:
        return previous + " " + current


def _is_incomplete(text: str) -> bool:
    """Check if paragraph appears to continue on next page."""
    stripped = text.rstrip()
    if not stripped:
        return False
    return not stripped.endswith(SENTENCE_ENDINGS)


def _normalize_for_comparison(text: str) -> str:
    """Normalize text for comparison by removing extra whitespace."""
    return ' '.join(text.split())


def analyze_pages(pages: Iterator[tuple[int, str, str | None, str | None]]) -> list[tuple[int, str, float]]:
    """Find paragraphs that mention research gaps and score their sentiment.

    Handles paragraphs split across pages using visual position information
    (trailing_text from visually-last block, leading_text from visually-first block).

    Args:
        pages: Iterator of (page_num, page_text, trailing_text, leading_text) tuples

    Returns:
        List of (page_number, paragraph_text, compound_sentiment)
        for paragraphs containing any gap-related keyword.
    """
    results = []
    pending_visual = ""
    pending_page = 0

    for page_num, page_text, trailing_text, leading_text in pages:
        paragraphs = [p.strip() for p in page_text.split('\n\n') if p.strip()]

        if not paragraphs:
            # Even with no paragraphs, process any pending visual continuation
            if pending_visual and trailing_text:
                joined = _join_paragraphs(pending_visual, trailing_text)
                if _keyword_pattern.search(joined):
                    page_attr = pending_page if _keyword_pattern.search(pending_visual) else page_num
                    results.append((page_attr, joined, analyze_sentiment(joined)))
            pending_visual = ""
            pending_page = 0
            continue

        skip_leading = False

        # Handle cross-page continuation using visual positions
        if pending_visual and leading_text:
            joined = _join_paragraphs(pending_visual, leading_text)
            if _keyword_pattern.search(joined):
                page_attr = pending_page if _keyword_pattern.search(pending_visual) else page_num
                results.append((page_attr, joined, analyze_sentiment(joined)))
            skip_leading = True

        pending_visual = ""
        pending_page = 0

        # Analyze paragraphs, skipping any that match the leading_text (already processed)
        leading_normalized = _normalize_for_comparison(leading_text) if leading_text else ""
        trailing_normalized = _normalize_for_comparison(trailing_text) if trailing_text else ""

        for paragraph in paragraphs:
            para_normalized = _normalize_for_comparison(paragraph)

            # Skip if this paragraph starts with leading_text (already processed in continuation)
            if skip_leading and leading_normalized and para_normalized.startswith(leading_normalized[:100]):
                skip_leading = False  # Only skip once
                continue

            # Skip if this paragraph matches trailing_text (will be processed as continuation)
            if trailing_text and _is_incomplete(trailing_text):
                if para_normalized == trailing_normalized or paragraph.strip() == trailing_text.strip():
                    continue

            if _keyword_pattern.search(paragraph):
                results.append((page_num, paragraph, analyze_sentiment(paragraph)))

        # Store visually-last block if incomplete (for next page continuation)
        if trailing_text and _is_incomplete(trailing_text):
            pending_visual = trailing_text
            pending_page = page_num

    # Process any remaining pending visual paragraph
    if pending_visual and _keyword_pattern.search(pending_visual):
        results.append((pending_page, pending_visual, analyze_sentiment(pending_visual)))

    return results
