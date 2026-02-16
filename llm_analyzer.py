"""LLM-based gap verification and report generation using Gemini API."""

import json
import logging

from pydantic import BaseModel

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

VERIFY_BATCH_SIZE = 10
MODEL = "gemini-2.5-flash"

VERIFICATION_SYSTEM_PROMPT = """\
You are an expert academic research methodology reviewer.

TASK: Classify each numbered paragraph as a genuine research gap or a false positive.

A RESEARCH GAP is where authors EXPLICITLY identify:
- An area that has NOT been studied or is understudied
- A limitation of existing research that creates an opening for new work
- Conflicting findings that need resolution
- A methodology weakness that future research should address

CRITICAL — These are FALSE POSITIVES, not gaps:
- Literature review sentences: "Studies have investigated...", "Several authors have \
shown...", "Research has demonstrated...", "Previous work has examined...", \
"It has been widely reported..."
- Descriptions of the authors' own methodology or results
- General academic language that superficially resembles gap language
- Summaries of existing findings without identifying what is MISSING

Literature reviews DESCRIBE what IS known. Gaps identify what is NOT known. \
High semantic similarity to gap language does NOT mean it is a gap.

For evidence_quote: extract the EXACT 5–10 consecutive words from the paragraph \
that most clearly signal this is a gap (e.g., "remains poorly understood", \
"no study has examined", "further research is needed"). Copy the paragraph's \
exact wording — do not paraphrase. If classifying as false_positive, quote the \
words that reveal it is a review/description, not a gap."""

REPORT_SYSTEM_PROMPT = """\
You are an expert academic research advisor. Based on the confirmed research gaps \
found in the analyzed papers, write a concise research analysis report.

The report should:
1. Summarize the key research gaps identified across the papers
2. Group related gaps by theme if applicable
3. Suggest specific research directions that could address these gaps
4. Note any patterns (e.g., multiple papers identifying similar limitations)

Write in clear academic English. Use Markdown formatting with headers and bullet \
points. Keep the report between 300–800 words. Be specific and actionable. \
Ground your analysis in the evidence quotes provided."""


class GapVerdict(BaseModel):
    index: int
    verdict: str
    reason: str
    evidence_quote: str


def _validate_evidence(verdict: GapVerdict, source_paragraph: str) -> dict:
    """Check that evidence_quote exists in the source text. Downgrade if not."""
    result = {
        "verdict": verdict.verdict,
        "reason": verdict.reason,
        "evidence_quote": verdict.evidence_quote,
    }

    if not verdict.evidence_quote or verdict.evidence_quote not in source_paragraph:
        # Quote not found in source — can't trust the classification
        result["verdict"] = "uncertain"
        result["reason"] = "Evidence quote not found in source text"

    return result


def _verify_batch(
    client: genai.Client,
    batch: list[dict],
    index_offset: int,
) -> list[GapVerdict]:
    """Send one batch of paragraphs to Gemini for classification."""
    numbered = []
    for i, gap in enumerate(batch):
        numbered.append(
            f"[{i}] (Score: {gap['insight']:.2f}, File: {gap['file']}, Page: {gap['page']})\n"
            f"{gap['paragraph'][:1000]}"
        )

    user_prompt = (
        f"Classify each of these {len(batch)} paragraphs. "
        f"They were flagged by a semantic similarity model as potential research gaps.\n\n"
        + "\n\n".join(numbered)
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=VERIFICATION_SYSTEM_PROMPT,
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=list[GapVerdict],
        ),
    )

    return response.parsed


def verify_gaps(api_key: str, gaps: list[dict]) -> list[dict]:
    """Classify semantic-scored paragraphs as confirmed gaps or false positives.

    Sends paragraphs in batches of 10. Validates that evidence_quote exists
    in the source text — downgrades to 'uncertain' if not found.
    """
    if not gaps:
        return []

    client = genai.Client(api_key=api_key)
    enriched = [{**g, "verdict": "uncertain", "reason": "", "evidence_quote": ""} for g in gaps]

    for batch_start in range(0, len(gaps), VERIFY_BATCH_SIZE):
        batch = gaps[batch_start:batch_start + VERIFY_BATCH_SIZE]

        try:
            verdicts = _verify_batch(client, batch, batch_start)
        except Exception:
            logger.exception("Batch verification failed (offset %d)", batch_start)
            continue

        verdict_map = {v.index: v for v in verdicts}

        for i, gap in enumerate(batch):
            verdict = verdict_map.get(i)
            if verdict is None:
                continue

            validated = _validate_evidence(verdict, gap["paragraph"])
            idx = batch_start + i
            enriched[idx]["verdict"] = validated["verdict"]
            enriched[idx]["reason"] = validated["reason"]
            enriched[idx]["evidence_quote"] = validated["evidence_quote"]

    return enriched


def generate_report(
    api_key: str,
    verified_gaps: list[dict],
    paper_metadata: list[dict],
) -> str:
    """Generate a Markdown research analysis report from confirmed gaps."""
    confirmed = [g for g in verified_gaps if g["verdict"] == "confirmed_gap"]

    if not confirmed:
        return (
            "# Research Gap Analysis Report\n\n"
            "No confirmed research gaps were found after AI verification. "
            "The paragraphs flagged by semantic similarity were classified as "
            "false positives or uncertain matches.\n"
        )

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
    for i, gap in enumerate(confirmed, 1):
        gaps_section += (
            f"Gap {i} (from {gap['file']}, page {gap['page']}, "
            f"similarity score: {gap['insight']:.2f}):\n"
            f"{gap['paragraph'][:500]}\n"
            f"Evidence: \"{gap['evidence_quote']}\"\n\n"
        )

    user_prompt = (
        f"{papers_section}"
        f"Confirmed research gaps ({len(confirmed)} found):\n\n"
        f"{gaps_section}"
        f"Write a research analysis report based on these confirmed gaps."
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
