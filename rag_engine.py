"""Hybrid RAG engine for research gap detection."""

import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Callable

import requests

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
GEMINI_MODEL = "gemini-2.5-flash-lite"

_MODAL_QUERIES = [
    "limitations future research shortcomings unanswered questions",
    "methodological constraints data collection sample bias measurement issues",
    "theoretical framework gaps conflicting findings unresolved debates",
]
_CLOUD_QUERIES = [
    "limitations, future research, shortcomings, unanswered questions",
]

GAP_ANALYSIS_PROMPT = """\
You are an expert academic research reviewer. Analyze the following text chunks \
extracted from academic papers and identify concrete research gaps.

For each gap you find, return a JSON object with these fields:
- "type": one of "Methodological", "Theoretical", "Contextual", "Empirical"
- "description": a clear explanation of the gap (1-2 sentences)
- "evidence": a direct quote from the text that supports this gap
- "suggestion": how a researcher could address this gap (1 sentence)

Return ONLY a JSON array of gap objects. No markdown, no explanation, just the array.
If no gaps are found, return an empty array: []

Text chunks:
{chunks}"""


class GapFinderAI:
    """RAG-based research gap detector using Modal (Qwen 2.5-7B) or Gemini."""

    def __init__(self, mode: str = "modal", api_key: str | None = None, embeddings=None):
        if mode not in ("modal", "cloud"):
            raise ValueError(f"Invalid mode: {mode!r}. Use 'modal' or 'cloud'.")
        if mode == "cloud" and not api_key:
            raise ValueError("Cloud mode requires a Gemini API key.")

        self.mode = mode
        self._embeddings = embeddings or HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self._vectorstore: Chroma | None = None

        if mode == "modal":
            modal_url = os.environ.get("MODAL_INFERENCE_URL")
            if not modal_url:
                raise ValueError(
                    "Modal mode requires MODAL_INFERENCE_URL environment variable."
                )
            self._modal_url = modal_url
        else:
            self._gemini_client = genai.Client(api_key=api_key)

    def ingest_pdf(self, pdf_bytes: bytes, filename: str = "upload.pdf") -> int:
        """Load PDF, chunk it, and store in ephemeral vector store.

        Returns the number of chunks created.
        """
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        try:
            loader = PyMuPDFLoader(tmp_path)
            documents = loader.load()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = splitter.split_documents(documents)

        if not chunks:
            return 0

        # Ephemeral store — no persist_directory means in-memory only
        self._vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self._embeddings,
        )
        return len(chunks)

    def analyze_gaps(
        self,
        top_k: int = 8,
        progress_callback: Callable[[str], None] | None = None,
    ) -> list[dict]:
        """Retrieve relevant chunks and ask the LLM to identify research gaps."""
        if self._vectorstore is None:
            raise RuntimeError("No PDF ingested. Call ingest_pdf() first.")

        queries = _MODAL_QUERIES if self.mode == "modal" else _CLOUD_QUERIES
        retrieved = []
        for query in queries:
            results = self._vectorstore.similarity_search(query, k=top_k)
            retrieved.extend(results)

        # Deduplicate by page content
        seen = set()
        unique_chunks = []
        for doc in retrieved:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_chunks.append(doc)

        if not unique_chunks:
            return []

        chunks_text = "\n\n---\n\n".join(
            f"[Chunk {i+1}] (page {doc.metadata.get('page', '?')}):\n{doc.page_content}"
            for i, doc in enumerate(unique_chunks[:top_k * len(queries)])
        )

        prompt = GAP_ANALYSIS_PROMPT.format(chunks=chunks_text)

        if self.mode == "modal":
            raw = self._call_modal(prompt)
        else:
            raw = self._call_gemini(prompt, progress_callback)

        return _parse_gaps_json(raw)

    def _call_modal(self, prompt: str) -> str:
        """Call Modal inference endpoint for Qwen 2.5-7B generation."""
        response = requests.post(
            self._modal_url,
            json={"prompt": prompt, "max_new_tokens": 2048},
            timeout=300,
        )
        response.raise_for_status()
        return response.json()["text"]

    def _call_gemini(
        self,
        prompt: str,
        progress_callback: Callable[[str], None] | None = None,
    ) -> str:
        """Call Gemini API with one retry on rate limit (429)."""
        try:
            response = self._gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.1),
            )
            return response.text
        except genai_errors.ClientError as e:
            if e.code != 429:
                raise

            # Extract retry delay from error message
            wait = _parse_retry_delay(str(e))
            if progress_callback:
                progress_callback(f"Rate limited by Gemini. Retrying in {wait}s...")
            logger.info("Gemini 429 — retrying in %ds", wait)
            time.sleep(wait)

            response = self._gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.1),
            )
            return response.text

    def cleanup(self):
        """Release vector store resources."""
        if self._vectorstore is not None:
            try:
                self._vectorstore.delete_collection()
            except Exception:
                pass
            self._vectorstore = None


def _parse_retry_delay(error_msg: str, default: int = 40) -> int:
    """Extract retry delay in seconds from a Gemini 429 error message."""
    match = re.search(r"retry in (\d+)", error_msg, re.IGNORECASE)
    return int(match.group(1)) if match else default


def _parse_gaps_json(raw: str) -> list[dict]:
    """Extract JSON array from LLM response, tolerating markdown fences."""
    logger.info("Parsing LLM output (%d chars)", len(raw))
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()

    # Find the JSON array
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1:
        logger.warning("No JSON array found in LLM response")
        return []

    try:
        gaps = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM response as JSON. Raw output:\n%s", raw)
        return []

    valid_types = {"Methodological", "Theoretical", "Contextual", "Empirical"}
    validated = []
    for gap in gaps:
        if not isinstance(gap, dict):
            continue
        if not all(k in gap for k in ("type", "description", "evidence", "suggestion")):
            continue
        if gap["type"] not in valid_types:
            gap["type"] = "Empirical"
        validated.append(gap)

    return validated
