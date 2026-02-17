"""Hybrid RAG engine for research gap detection."""

import json
import logging
import re
import tempfile
import time
from pathlib import Path
from typing import Callable

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

logger = logging.getLogger(__name__)

LOCAL_MODEL_PATH = "/app/models/qwen2.5-3b-instruct-q4_k_m.gguf"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
GEMINI_MODEL = "gemini-2.5-flash-lite"

RETRIEVAL_QUERIES = [
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
    """RAG-based research gap detector using local or cloud LLM."""

    def __init__(self, mode: str = "local", api_key: str | None = None):
        if mode not in ("local", "cloud"):
            raise ValueError(f"Invalid mode: {mode!r}. Use 'local' or 'cloud'.")
        if mode == "cloud" and not api_key:
            raise ValueError("Cloud mode requires a Gemini API key.")

        self.mode = mode
        self._embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self._vectorstore: Chroma | None = None

        if mode == "local":
            model_path = LOCAL_MODEL_PATH
            # Fall back to local dev path if HF Spaces path doesn't exist
            if not Path(model_path).exists():
                local_path = Path("models/qwen2.5-3b-instruct-q4_k_m.gguf")
                if local_path.exists():
                    model_path = str(local_path)
                else:
                    raise FileNotFoundError(
                        f"GGUF model not found at {LOCAL_MODEL_PATH} or {local_path}"
                    )
            self._llm = LlamaCpp(
                model_path=model_path,
                n_gpu_layers=0,
                n_ctx=4096,
                max_tokens=1024,
                verbose=False,
            )
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

        retrieved = []
        for query in RETRIEVAL_QUERIES:
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
            for i, doc in enumerate(unique_chunks[:top_k])
        )

        prompt = GAP_ANALYSIS_PROMPT.format(chunks=chunks_text)

        if self.mode == "local":
            raw = self._llm.invoke(prompt)
        else:
            raw = self._call_gemini(prompt, progress_callback)

        return _parse_gaps_json(raw)

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
