"""
Streamlit UI: a thin client over the FastAPI backend.

Save as: sec-rag-project/ui/streamlit_app.py

Run with:
    streamlit run ui/streamlit_app.py

Requires the API server running first:
    uvicorn src.api.main:app --host 127.0.0.1 --port 8000

--- Why this holds no pipeline logic of its own ---
Every question goes through requests calls to the same /ask/stream, /health,
/tickers endpoints src/api/static/index.html already uses -- guardrails,
caching, confidence gating, retrieval, generation, numeric/citation
verification all happen exactly once, in the FastAPI server (see
src/generation/rag_pipeline.py). This file only renders what comes back.
Two UIs (this one, and the plain HTML/JS one) can coexist safely because
neither one is a second implementation of the pipeline -- they're two
skins over the same API.

--- Why Streamlit's chat components instead of the hand-rolled HTML/JS ones ---
st.chat_message/st.chat_input/st.write_stream give conversation history,
streaming token-by-token display, and consistent styling for free, in pure
Python -- no hand-written SSE parsing, no manual DOM manipulation. The
trade-off is a heavier runtime (a whole Streamlit server, reruns the script
top-to-bottom on every interaction) versus the static page's near-zero
footprint; for an interactive local chat UI, that trade is worth it.
"""

import json
import sys
from pathlib import Path

import requests
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import API_BASE_URL

st.set_page_config(page_title="SEC 10-K RAG", page_icon="📊", layout="wide")

STAGES = [
    {"n": 1, "name": "Ingestion", "desc": "PDF → structured Markdown via Docling (layout-aware: preserves headings and tables), pdfplumber as a plain-text fallback.", "tech": ["Docling", "pdfplumber"]},
    {"n": 2, "name": "Cleaning", "desc": "Strips repeated running headers and page-number lines, conservatively — never touches text that merely looks repetitive.", "tech": ["custom rules"]},
    {"n": 3, "name": "Metadata & sections", "desc": "Extracts ticker/company/fiscal year and every Item-N section boundary. Cross-checked against SEC EDGAR's own registry, with a Markdown-heading fallback for filers (e.g. Intel) that don't use standard Item labels.", "tech": ["SEC EDGAR API", "regex + scoring heuristics"]},
    {"n": 4, "name": "Chunking", "desc": "Section-bounded so a chunk never mixes two topics. Tables extracted as standalone, captioned chunks; oversized tables split only at row boundaries so numbers are never corrupted.", "tech": ["custom recursive splitter"]},
    {"n": 5, "name": "Embeddings", "desc": "Every chunk embedded into a 384-dim vector for semantic search. Runs on CPU via OpenVINO (Intel-optimized), cached locally after first export.", "tech": ["BAAI/bge-small-en-v1.5", "OpenVINO", "sentence-transformers"]},
    {"n": 6, "name": "Vector storage", "desc": "All vectors + metadata stored in a local Qdrant collection with HNSW indexing, running in Docker.", "tech": ["Qdrant", "Docker"]},
    {"n": 7, "name": "Retrieval", "desc": "Hybrid search: dense (embedding similarity) + BM25 (keyword) results merged via Reciprocal Rank Fusion, with metadata filtering by ticker/year/content-type.", "tech": ["Qdrant search", "rank-bm25", "RRF"]},
    {"n": 8, "name": "Reranking", "desc": "A cross-encoder re-scores the fused candidates for final relevance ordering before anything reaches the LLM.", "tech": ["FlashRank · ms-marco-MiniLM-L-12-v2"]},
    {"n": 9, "name": "Query transform & generation", "desc": "Optional rewrite / multi-query / decomposition / HyDE before retrieval; the final answer is generated strictly from retrieved sources, with automatic checks for invented numbers and missing/fabricated citations.", "tech": ["Qwen2.5 7B via Ollama", "numeric + citation grounding checks"]},
    {"n": 10, "name": "API & guardrails", "desc": "FastAPI server exposing the pipeline over HTTP, with input/output guardrails, a confidence gate that refuses weakly-grounded answers, a response cache, and structured request tracing.", "tech": ["FastAPI", "SSE streaming", "SQLite cache", "JSONL tracing"]},
]


@st.cache_data(ttl=15)
def get_health():
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return r.json()
    except requests.exceptions.RequestException:
        return {"qdrant": False, "ollama": False}


@st.cache_data(ttl=60)
def get_tickers():
    try:
        r = requests.get(f"{API_BASE_URL}/tickers", timeout=5)
        return r.json()
    except requests.exceptions.RequestException:
        return []


def stream_answer(question: str, filters: dict, technique: str, result_holder: dict):
    """
    Yields answer text chunks as they stream in from /ask/stream (for
    st.write_stream to render live), and stores the final "done" event's
    metadata (sources, numeric_check, citation_check, cached, blocked,
    low_confidence) into result_holder as a side effect -- a generator can
    only yield the text, so anything else it learns has to escape through
    a mutable object the caller already holds a reference to.
    """
    payload = {"question": question, "technique": technique, **filters}
    with requests.post(f"{API_BASE_URL}/ask/stream", json=payload, stream=True, timeout=(10, 600)) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            if event["type"] == "token":
                yield event["content"]
            elif event["type"] == "done":
                result_holder.update(event)


def render_result_extras(result: dict):
    """Renders the badges/warnings/sources that come after the answer text
    itself -- shared by both the live-streaming render and history replay."""
    badges = []
    if result.get("cached"):
        badges.append("🟢 from cache")
    if result.get("blocked"):
        badges.append("🔴 blocked by guardrail")
    if result.get("low_confidence"):
        badges.append("🟠 low retrieval confidence")
    if badges:
        st.caption(" · ".join(badges))

    numeric_check = result.get("numeric_check") or {}
    if numeric_check.get("unverified"):
        st.warning(f"Numbers in this answer weren't found in any retrieved source "
                   f"(possible fabrication): {', '.join(numeric_check['unverified'])}")

    citation_check = result.get("citation_check") or {}
    if citation_check.get("missing_citation"):
        st.warning("This answer doesn't cite any source.")
    if citation_check.get("out_of_range_citations"):
        st.warning(f"This answer cites a source number that doesn't exist: "
                   f"{citation_check['out_of_range_citations']}")

    sources = result.get("sources") or []
    if sources:
        with st.expander(f"{len(sources)} source(s)"):
            for i, s in enumerate(sources, start=1):
                label = f"**[{i}]** {s.get('company_name') or s.get('ticker')} " \
                        f"FY{s.get('fiscal_year')} · {s.get('section_item')} · {s.get('chunk_type')} " \
                        f"· score {s.get('score', 0):.3f}"
                st.markdown(label)
                st.caption(s.get("text", "")[:300] + "…")


# --- Sidebar ---
with st.sidebar:
    st.header("SEC 10-K RAG")
    health = get_health()
    st.write(("🟢" if health.get("qdrant") else "🔴") + " Qdrant")
    st.write(("🟢" if health.get("ollama") else "🔴") + " Ollama")

    st.divider()
    st.subheader("Filters")
    tickers = get_tickers()
    ticker = st.selectbox("Company", ["Any"] + tickers)
    fiscal_year = st.number_input("Fiscal year", min_value=0, max_value=2100, value=0, step=1,
                                   help="0 = any fiscal year")
    chunk_type = st.selectbox("Content type", ["Any", "text", "table"])
    technique = st.selectbox(
        "Query transformation", ["none", "rewrite", "step-back", "multi-query", "decompose", "hyde"],
        help="Each technique adds an extra LLM call before generation, roughly doubling response time.",
    )

    st.divider()
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

filters = {}
if ticker != "Any":
    filters["ticker"] = ticker
if fiscal_year:
    filters["fiscal_year"] = fiscal_year
if chunk_type != "Any":
    filters["chunk_type"] = chunk_type

# --- Main area ---
tab_ask, tab_overview = st.tabs(["💬 Ask a question", "🏗️ Pipeline overview"])

with tab_overview:
    st.caption(
        "A retrieval-augmented question-answering system over 10 companies' SEC 10-K filings "
        "(5 fiscal years each), running entirely on this machine with open-source models — "
        "no cloud LLM, no API keys, nothing leaves this computer."
    )
    cols = st.columns(5)
    for col, (value, label) in zip(cols, [
        ("50", "filings"), ("10", "companies"), ("29,886", "chunks"),
        ("384", "embedding dims"), ("10", "pipeline stages"),
    ]):
        col.metric(label, value)

    for stage in STAGES:
        with st.container(border=True):
            st.markdown(f"**{stage['n']}. {stage['name']}**")
            st.caption(stage["desc"])
            st.write(" ".join(f"`{t}`" for t in stage["tech"]))

with tab_ask:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("result"):
                render_result_extras(message["result"])

    if question := st.chat_input("Ask a question about the SEC filings..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            if not (health.get("qdrant") and health.get("ollama")):
                st.error("Qdrant and/or Ollama aren't reachable — check the status in the sidebar.")
            else:
                result_holder = {}
                with st.spinner("Retrieving and generating — this typically takes a few minutes on this CPU..."):
                    full_answer = st.write_stream(stream_answer(question, filters, technique, result_holder))
                render_result_extras(result_holder)
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_answer, "result": result_holder}
                )
