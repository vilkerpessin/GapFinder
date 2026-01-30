"""Detect potential research gaps via keyword presence and sentiment polarity."""

import re

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


def analyze_sentiment(text: str) -> float:
    """Return VADER compound sentiment score (-1.0 to 1.0)."""
    sentiment = _sentiment_analyzer.polarity_scores(text)
    return sentiment['compound']


def analyze_text(text: str) -> list[tuple[int, str, float]]:
    """Find paragraphs that mention research gaps and score their sentiment.

    Splits text by form-feed (\\f) for pages and double-newline for paragraphs.

    Returns:
        List of (page_number 1-based, paragraph_text, compound_sentiment)
        for paragraphs containing any gap-related keyword.
    """
    pages = text.split('\f')
    results = []

    for page_num, page in enumerate(pages):
        for paragraph in page.split('\n\n'):
            if _keyword_pattern.search(paragraph):
                insight_score = analyze_sentiment(paragraph)
                results.append((page_num + 1, paragraph, insight_score))

    return results
