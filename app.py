"""GapFinder — RAG-powered research gap detector."""

import io
import os
import threading

import pandas as pd
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings

from pdf_extractor import extract_metadata
from rag_engine import EMBEDDING_MODEL, GapFinderAI

st.set_page_config(page_title="GapFinder", page_icon="🔍", layout="wide")

GAP_TYPE_COLORS = {
    "Methodological": "red",
    "Theoretical": "orange",
    "Contextual": "blue",
    "Empirical": "green",
}


@st.cache_resource
def _load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@st.cache_resource
def _concurrency_state():
    return {"count": 0, "lock": threading.Lock()}


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("GapFinder")
    st.caption("AI-powered research gap detector · **Beta** · [Report issues](https://github.com/vilkerpessin/GapFinder/issues)")

    st.divider()

    _modal_configured = bool(os.environ.get("MODAL_INFERENCE_URL"))

    mode = st.radio(
        "Analysis Mode",
        options=["Cloud (Gemini)", "Modal (Qwen 2.5-7B)"],
        help="Cloud requires a free Gemini API key. Modal runs Qwen 2.5-7B on GPU (~10-30s per PDF, no key needed).",
    )

    api_key = ""
    if mode == "Cloud (Gemini)":
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="Enter your API key",
        )
        st.caption("Don't have a key? [Get one free at Google AI Studio](https://aistudio.google.com/app/apikey)")
    elif not _modal_configured:
        st.warning("MODAL_INFERENCE_URL not set. Modal mode unavailable.")

    st.divider()
    st.subheader("System Status")
    if mode == "Cloud (Gemini)":
        if api_key:
            st.info("Model: Gemini 2.5 Flash Lite (Cloud)")
        else:
            st.warning("Waiting for API key...")
    elif _modal_configured:
        st.info("Model: Qwen 2.5-7B (Modal)")
    else:
        st.error("MODAL_INFERENCE_URL not configured")


# ── Main Interface ───────────────────────────────────────────────────────────

st.header("Upload Papers")
uploaded_files = st.file_uploader(
    "Upload one or more academic PDFs",
    type=["pdf"],
    accept_multiple_files=True,
)

if "analyzing" not in st.session_state:
    st.session_state.analyzing = False

analyze_clicked = st.button(
    "Analyze Papers",
    type="primary",
    disabled=st.session_state.analyzing
    or not uploaded_files
    or (mode == "Cloud (Gemini)" and not api_key)
    or (mode == "Modal (Qwen 2.5-7B)" and not _modal_configured),
    on_click=lambda: setattr(st.session_state, "analyzing", True),
)

if analyze_clicked:
    engine_mode = "cloud" if mode == "Cloud (Gemini)" else "modal"

    _state = _concurrency_state()
    with _state["lock"]:
        _state["count"] += 1
        current_count = _state["count"]

    try:
        if current_count > 1:
            st.warning(
                f"{current_count} analyses running simultaneously — "
                "processing may be slower than usual."
            )

        with st.spinner("Initializing model..."):
            try:
                engine = GapFinderAI(
                    mode=engine_mode,
                    api_key=api_key if engine_mode == "cloud" else None,
                    embeddings=_load_embeddings(),
                )
            except (ValueError, FileNotFoundError) as e:
                st.error(str(e))
                st.stop()

        all_results: dict[str, dict] = {}

        for uploaded_file in uploaded_files:
            pdf_bytes = uploaded_file.read()
            filename = uploaded_file.name

            with st.status(f"Analyzing: {filename}", expanded=True) as status:
                st.write("Extracting metadata...")
                try:
                    metadata = extract_metadata(io.BytesIO(pdf_bytes))
                except Exception:
                    st.error(f"Could not read {filename}")
                    status.update(label=f"{filename}: failed", state="error")
                    continue

                st.write("Ingesting PDF and building vector store...")
                try:
                    num_chunks = engine.ingest_pdf(pdf_bytes, filename)
                except Exception as e:
                    if getattr(e, "code", None) == 429:
                        st.error("Google API rate limit exceeded. Please wait a minute and try again.")
                    else:
                        st.error(f"Could not process {filename}. Please try again.")
                    status.update(label=f"{filename}: failed", state="error")
                    continue
                if num_chunks == 0:
                    st.warning(f"No text extracted from {filename}")
                    status.update(label=f"{filename}: no text found", state="error")
                    continue

                st.write(f"Retrieving context from {num_chunks} chunks...")
                st.write("Generating insights with LLM...")
                try:
                    gaps = engine.analyze_gaps(progress_callback=st.write)
                except Exception as e:
                    if getattr(e, "code", None) == 429:
                        st.error("Google API rate limit exceeded. Please wait a minute and try again.")
                    else:
                        st.error(f"Analysis failed for {filename}. Please try again.")
                    status.update(label=f"{filename}: analysis error", state="error")
                    continue
                finally:
                    engine.cleanup()

                all_results[filename] = {
                    "metadata": metadata,
                    "gaps": gaps,
                }
                status.update(
                    label=f"{filename}: {len(gaps)} gap(s) found",
                    state="complete",
                )

        # Store results in session state for persistence across reruns
        if all_results:
            st.session_state["results"] = all_results

    finally:
        with _state["lock"]:
            _state["count"] -= 1
        st.session_state.analyzing = False


# ── Results Display ──────────────────────────────────────────────────────────

if "results" in st.session_state and st.session_state["results"]:
    results = st.session_state["results"]

    total_gaps = sum(len(r["gaps"]) for r in results.values())

    st.divider()
    st.header(f"Results — {total_gaps} gap(s) across {len(results)} paper(s)")

    export_rows = []

    for idx, (filename, result) in enumerate(results.items()):
        gaps = result["gaps"]
        metadata = result["metadata"]

        if idx > 0:
            st.divider()

        st.subheader(metadata.get("title") or filename)
        col1, col2 = st.columns(2)
        col1.metric("Gaps Found", len(gaps))
        col2.metric("DOI", metadata.get("doi") or "N/A")

        if not gaps:
            st.info("No research gaps identified in this paper.")
            continue

        for i, gap in enumerate(gaps, 1):
            gap_type = gap.get("type", "Unknown")
            color = GAP_TYPE_COLORS.get(gap_type, "gray")

            with st.expander(f"Gap {i}: {gap.get('description', '')[:80]}"):
                st.markdown(f":{color}[**{gap_type}**]")
                st.markdown(f"**Description:** {gap['description']}")
                st.markdown(f"> *\"{gap['evidence']}\"*")
                st.markdown(f"**Suggestion:** {gap['suggestion']}")

            export_rows.append({
                "file": filename,
                "title": metadata.get("title", ""),
                "doi": metadata.get("doi", ""),
                "type": gap_type,
                "description": gap.get("description", ""),
                "evidence": gap.get("evidence", ""),
                "suggestion": gap.get("suggestion", ""),
            })

    # ── Export ────────────────────────────────────────────────────────────

    if export_rows:
        st.divider()
        df = pd.DataFrame(export_rows)

        col_csv, col_xlsx = st.columns(2)

        with col_csv:
            st.download_button(
                label="Download CSV",
                data=df.to_csv(index=False),
                file_name="gapfinder_results.csv",
                mime="text/csv",
            )

        with col_xlsx:
            xlsx_buffer = io.BytesIO()
            with pd.ExcelWriter(xlsx_buffer, engine="xlsxwriter") as writer:
                df.to_excel(writer, index=False, sheet_name="Gaps")
            st.download_button(
                label="Download Excel",
                data=xlsx_buffer.getvalue(),
                file_name="gapfinder_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
