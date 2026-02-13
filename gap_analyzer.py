"""Detect potential research gaps via semantic similarity."""

from collections.abc import Iterator

from sentence_transformers import SentenceTransformer, util


ANCHOR_PHRASES = [
    "This research reveals a significant gap in understanding how these mechanisms interact.",
    "A major limitation of existing studies is the lack of comprehensive data on this phenomenon.",
    "Further investigation is needed to address this unexplored area of research.",
    "Current literature provides insufficient evidence regarding the underlying causes.",
    "Previous research has neglected to examine these factors in real-world settings.",
    "The relationship between these variables remains poorly understood and under-theorized.",
    "Existing studies show conflicting results, highlighting the need for more rigorous analysis.",
    "Recent developments have created new research opportunities that remain unaddressed.",
    "Most studies focus on narrow contexts, leaving broader applications unexplored.",
    "The intersection of these fields has received inadequate scholarly attention."
]

_semantic_model = None
_anchor_embeddings = None

SENTENCE_ENDINGS = ('.', '!', '?', '"', ')', ']')
MIN_PARAGRAPH_LENGTH = 80


def _get_model() -> SentenceTransformer:
    """Lazy load multilingual MiniLM model (~500MB runtime memory, 50+ languages)."""
    global _semantic_model
    if _semantic_model is None:
        _semantic_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return _semantic_model


def _get_anchor_embeddings():
    """Compute and cache anchor phrase embeddings (computed once on first use)."""
    global _anchor_embeddings
    if _anchor_embeddings is None:
        model = _get_model()
        # Encode anchors and keep as tensor (batch of 10 x 384)
        _anchor_embeddings = model.encode(ANCHOR_PHRASES, convert_to_tensor=True)
    return _anchor_embeddings


def calculate_similarity_score(text: str) -> float:
    """Calculate semantic similarity between text and anchor gap phrases.

    Returns max cosine similarity across all anchor phrases (0.0 to 1.0).
    Higher score indicates stronger similarity to known research gaps.
    """
    model = _get_model()
    anchor_embeddings = _get_anchor_embeddings()

    # Encode text as tensor (shape: 384)
    text_embedding = model.encode(text, convert_to_tensor=True)

    # Cosine similarity against all anchors (returns tensor of shape: 10)
    similarities = util.cos_sim(text_embedding, anchor_embeddings)[0]

    return float(similarities.max())


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


def _is_scorable_paragraph(text: str) -> bool:
    """Check if text is a real paragraph worth scoring (not a TOC entry, table, etc)."""
    if len(text) < MIN_PARAGRAPH_LENGTH:
        return False

    # TOC entries: leader dots pattern (". . ." or "...")
    if '. . .' in text or '…' in text:
        return False

    non_space = text.replace(' ', '')
    if not non_space:
        return False
    alpha_count = sum(c.isalpha() for c in non_space)
    alpha_ratio = alpha_count / len(non_space)

    # Tables and data rows have low alphabetic ratio
    if alpha_ratio < 0.6:
        return False

    return True


def _normalize_for_comparison(text: str) -> str:
    """Normalize text for comparison by removing extra whitespace."""
    return ' '.join(text.split())


def analyze_pages(
    pages: Iterator[tuple[int, str, str | None, str | None]],
    threshold: float = 0.4,
    top_k: int = 20
) -> list[tuple[int, str, float]]:
    """Find paragraphs semantically similar to known research gaps.

    Scores every paragraph against anchor phrases using cosine similarity.
    Handles paragraphs split across pages using visual position information.

    Each page tuple: (page_num, page_text, trailing_text, leading_text).
    Returns top-K (page_number, paragraph_text, score) tuples above threshold.
    """
    candidates = []
    pending_visual = ""
    pending_page = 0

    for page_num, page_text, trailing_text, leading_text in pages:
        paragraphs = [p.strip() for p in page_text.split('\n\n') if p.strip()]

        if not paragraphs:
            # Even with no paragraphs, process any pending visual continuation
            if pending_visual and trailing_text:
                joined = _join_paragraphs(pending_visual, trailing_text)
                if _is_scorable_paragraph(joined):
                    candidates.append((pending_page, joined))
            pending_visual = ""
            pending_page = 0
            continue

        skip_leading = False

        # Handle cross-page continuation using visual positions
        if pending_visual and leading_text:
            joined = _join_paragraphs(pending_visual, leading_text)
            if _is_scorable_paragraph(joined):
                candidates.append((pending_page, joined))
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

            if _is_scorable_paragraph(paragraph):
                candidates.append((page_num, paragraph))

        # Store visually-last block if incomplete (for next page continuation)
        if trailing_text and _is_incomplete(trailing_text):
            pending_visual = trailing_text
            pending_page = page_num

    if pending_visual and _is_scorable_paragraph(pending_visual):
        candidates.append((pending_page, pending_visual))

    if not candidates:
        return []

    # Batch encode all paragraphs in one call instead of one-by-one
    model = _get_model()
    anchor_embeddings = _get_anchor_embeddings()
    texts = [text for _, text in candidates]
    text_embeddings = model.encode(texts, batch_size=32, convert_to_tensor=True)

    # Batch cosine similarity: (N x 384) vs (10 x 384) -> (N x 10)
    all_similarities = util.cos_sim(text_embeddings, anchor_embeddings)
    max_scores = all_similarities.max(dim=1).values

    results = [
        (page_num, text, float(score))
        for (page_num, text), score in zip(candidates, max_scores)
        if float(score) >= threshold
    ]

    results.sort(key=lambda x: x[2], reverse=True)
    return results[:top_k]
