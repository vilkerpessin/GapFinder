"""LLM-based gap analysis and report generation using Gemini API."""

import logging

from pydantic import BaseModel

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

BATCH_SIZE = 10
PARAGRAPH_TRUNCATE = 300
MODEL = "gemini-2.5-flash-lite"

COMMENT_SYSTEM_PROMPT = """\
You are an academic research reviewer. For each paragraph, write a single short \
comment (1 sentence, max 20 words) assessing its relevance as a research gap.

Examples of good comments:
- "Methodological limitation — small sample size limits generalization"
- "Identifies understudied geographic region in existing literature"
- "Literature review summary, not a gap — describes existing findings"
- "Suggests conflicting results that need further investigation"
- "General conclusion language, no specific research opportunity identified"

Be concise and specific. Focus on what makes it a gap or why it is not one."""

REPORT_SYSTEM_PROMPT = """\
You are an expert academic research advisor. Based on the paragraphs flagged as \
potential research gaps in the analyzed papers, write a concise research analysis.

The report should:
1. Identify the most promising research gaps and opportunities
2. Group related gaps by theme if applicable
3. Suggest specific research directions
4. Note which paragraphs are likely false positives (literature reviews, general statements)

Write in clear academic English. Use Markdown formatting with headers and bullet \
points. Keep the report between 150–300 words. Be specific and actionable."""


class GapComment(BaseModel):
    index: int
    comment: str


def _analyze_batch(
    client: genai.Client,
    batch: list[dict],
) -> list[GapComment]:
    """Send one batch of paragraphs to Gemini for comment."""
    numbered = []
    for i, gap in enumerate(batch):
        numbered.append(
            f"[{i}] (Score: {gap['insight']:.2f})\n"
            f"{gap['paragraph'][:PARAGRAPH_TRUNCATE]}"
        )

    user_prompt = (
        f"Comment on each of these {len(batch)} paragraphs. "
        f"They were flagged by a semantic similarity model as potential research gaps.\n\n"
        + "\n\n".join(numbered)
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=COMMENT_SYSTEM_PROMPT,
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=list[GapComment],
        ),
    )

    return response.parsed


def analyze_gaps(api_key: str, gaps: list[dict]) -> list[dict]:
    """Add a short LLM comment to each scored paragraph."""
    if not gaps:
        return []

    client = genai.Client(api_key=api_key)
    enriched = [{**g, "llm_comment": ""} for g in gaps]

    last_error = None
    batches_failed = 0
    total_batches = 0

    for batch_start in range(0, len(gaps), BATCH_SIZE):
        batch = gaps[batch_start:batch_start + BATCH_SIZE]
        total_batches += 1

        try:
            comments = _analyze_batch(client, batch)
        except Exception as exc:
            logger.exception("Batch analysis failed (offset %d)", batch_start)
            last_error = exc
            batches_failed += 1
            continue

        comment_map = {c.index: c for c in comments}

        for i in range(len(batch)):
            c = comment_map.get(i)
            if c is not None:
                enriched[batch_start + i]["llm_comment"] = c.comment

    if batches_failed == total_batches and last_error is not None:
        raise last_error

    return enriched


def generate_report(
    api_key: str,
    analyzed_gaps: list[dict],
    paper_metadata: list[dict],
) -> str:
    """Generate a Markdown research analysis report from all scored paragraphs."""
    client = genai.Client(api_key=api_key)

    papers_section = ""
    if paper_metadata:
        papers_section = "Papers analyzed:\n"
        for meta in paper_metadata:
            title = meta.get("title") or "Unknown"
            author = meta.get("author") or "Unknown"
            papers_section += f"- {title} by {author}\n"
        papers_section += "\n"

    gaps_section = ""
    for i, gap in enumerate(analyzed_gaps, 1):
        comment = gap.get("llm_comment", "")
        gaps_section += (
            f"Paragraph {i} (page {gap['page']}, score: {gap['insight']:.2f}):\n"
            f"{gap['paragraph'][:PARAGRAPH_TRUNCATE]}\n"
            f"AI comment: {comment}\n\n"
        )

    user_prompt = (
        f"{papers_section}"
        f"Scored paragraphs ({len(analyzed_gaps)} total):\n\n"
        f"{gaps_section}"
        f"Write a research analysis report identifying the most promising gaps "
        f"and suggested research directions."
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=REPORT_SYSTEM_PROMPT,
            temperature=0.7,
        ),
    )

    return response.text
