"""GapFinder 2.0 — RAG-powered research gap detector."""

import io

import pandas as pd
import streamlit as st

from pdf_extractor import extract_metadata
from rag_engine import GapFinderAI

st.set_page_config(page_title="GapFinder", page_icon="🔍", layout="wide")

GAP_TYPE_COLORS = {
    "Methodological": "red",
    "Theoretical": "orange",
    "Contextual": "blue",
    "Empirical": "green",
}


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("GapFinder")
    st.caption("AI-powered research gap detector")

    st.divider()

    mode = st.radio(
        "Analysis Mode",
        options=["Cloud (Gemini)", "Local LLM (Experimental)"],
        help="Cloud requires a free Gemini API key. Local runs Qwen 2.5-3B on CPU (~5-10 min per PDF).",
    )

    api_key = ""
    if mode == "Cloud (Gemini)":
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="Enter your API key",
        )

    st.divider()
    st.subheader("System Status")
    if mode == "Cloud (Gemini)":
        if api_key:
            st.info("Model: Gemini 2.5 Flash Lite (Cloud)")
        else:
            st.warning("Waiting for API key...")
    else:
        st.warning("Model: Qwen 2.5-3B (Local CPU — Slow)")


# ── Main Interface ───────────────────────────────────────────────────────────

st.header("Upload Papers")
uploaded_files = st.file_uploader(
    "Upload one or more academic PDFs",
    type=["pdf"],
    accept_multiple_files=True,
)

analyze_clicked = st.button(
    "Analyze Papers",
    type="primary",
    disabled=not uploaded_files or (mode == "Cloud (Gemini)" and not api_key),
)

if analyze_clicked:
    engine_mode = "cloud" if mode == "Cloud (Gemini)" else "local"

    try:
        engine = GapFinderAI(
            mode=engine_mode,
            api_key=api_key if engine_mode == "cloud" else None,
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
            num_chunks = engine.ingest_pdf(pdf_bytes, filename)
            if num_chunks == 0:
                st.warning(f"No text extracted from {filename}")
                status.update(label=f"{filename}: no text found", state="error")
                continue

            st.write(f"Retrieving context from {num_chunks} chunks...")
            st.write("Generating insights with LLM...")
            try:
                gaps = engine.analyze_gaps(progress_callback=st.write)
            except Exception as e:
                st.error(f"Analysis failed: {e}")
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


# ── Results Display ──────────────────────────────────────────────────────────

if "results" in st.session_state and st.session_state["results"]:
    results = st.session_state["results"]

    st.divider()
    st.header("Results")

    tabs = st.tabs(list(results.keys()))

    export_rows = []

    for tab, (filename, result) in zip(tabs, results.items()):
        gaps = result["gaps"]
        metadata = result["metadata"]

        with tab:
            col1, col2, col3 = st.columns(3)
            col1.metric("Gaps Found", len(gaps))
            col2.metric("Pages", metadata.get("num_pages", "?"))
            col3.metric("DOI", metadata.get("doi") or "N/A")

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
