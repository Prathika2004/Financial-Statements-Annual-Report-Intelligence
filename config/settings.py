"""
Centralized configuration. Every module should import from here rather than
hardcoding paths or magic numbers -- this is the one place you change things
when moving from your laptop to a different machine, or tuning parameters.

Save this file as: sec-rag-project/config/settings.py
"""

import os
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_PDF_DIR = PROJECT_ROOT / "data" / "raw_pdfs"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EVAL_SET_DIR = PROJECT_ROOT / "data" / "eval_set"

# ---------------------------------------------------------------------------
# Ingestion / Parsing (Stage 1)
# ---------------------------------------------------------------------------

# If a page's extracted text length falls below this, it's a candidate for
# the OCR fallback branch (not implemented yet -- this dataset is all
# native-text PDFs, but the threshold is defined now so the branch is a
# clean plug-in later rather than a rewrite).
OCR_FALLBACK_TEXT_THRESHOLD = 20  # characters

# ---------------------------------------------------------------------------
# Cleaning (Stage 2)
# ---------------------------------------------------------------------------

# Running headers that repeat on nearly every page of a 10-K and carry no
# semantic value -- confirmed by frequency analysis on a real filing.
BOILERPLATE_LINES = [
    "table of contents",
]

# ---------------------------------------------------------------------------
# Metadata extraction (Stage 1)
# ---------------------------------------------------------------------------

# 10-K "Item" sections are reliably rendered in the filing body as markdown
# headers (e.g. "## Item 1. Business" or "## ITEM 1. BUSINESS").
# The # prefix is added by Docling; the ITEM/number/title format is consistent.
# Confirmed against real Tesla and Amazon 10-K filings.
ITEM_HEADING_PATTERN = r"^#+\s*ITEM\s+(\d{1,2}[A]?)\.\s+(.+)$"

# Pattern for fiscal year on cover page
FISCAL_YEAR_PATTERN = r"fiscal year ended\s+\w+\s+\d{1,2},\s+(\d{4})"

# ---------------------------------------------------------------------------
# Company name / CIK / filing metadata patterns
# ---------------------------------------------------------------------------

# Typical cover-page company name line: "Amazon.com, Inc." or "Tesla, Inc."
COMPANY_NAME_PATTERN = r"^((?:Co|Inc|Ltd|Corp)\.\s+)?([A-Z][A-Za-z\s]{3,}\b(?:Inc|LLC|Corp|Ltd))$"

# CIK number pattern (10-digit, often appears as "CIK: 0001018724" or "1018724")
CIK_PATTERN = re.compile(r"CIK\s*:\s*(\d{9,10})|Central Index Key\s*:\s*(\d{9,10})", re.IGNORECASE)

# Filing date pattern: "Filed by date: YYYY-MM-DD" or "Date filed: MM/DD/YYYY"
FILING_DATE_PATTERN = re.compile(
    r"(?:Filed|Date\s+filed)\s+by\s+date\s*:\s*(\d{4}-\d{2}-\d{2})|"
    r"(?:Date\s+filed)\s*:\s*(\d{2}/\d{2}/\d{4})",
    re.IGNORECASE
)

# Reporting period end pattern: "for the fiscal year ended December 31, 2021"
# or "period ended December 31, 2021" — distinguishes from fiscal_year derived from filename
REPORTING_PERIOD_PATTERN = re.compile(
    r"(?:for\s+the\s+fiscal\s+year\s+ended|period\s+ended)\s+(\w+)\s+(\d{1,2}),\s+(\d{4})",
    re.IGNORECASE
)

# Filing type pattern
FILING_TYPE_PATTERN = re.compile(r"10-K", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Chunking (Stage 3 -- not built yet, params defined now for later use)
# ---------------------------------------------------------------------------

CHUNK_SIZE_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 50

# ---------------------------------------------------------------------------
# Embeddings (Stage 5)
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384  # fixed output size of the model above; used to sanity-check vectors
EMBEDDING_BATCH_SIZE = 32

# ---------------------------------------------------------------------------
# Vector DB (Stage 6)
# ---------------------------------------------------------------------------

QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION_NAME = "sec_10k_filings"
QDRANT_UPSERT_BATCH_SIZE = 256

# ---------------------------------------------------------------------------
# Retrieval (Stage 7)
# ---------------------------------------------------------------------------

# How many candidates each individual retriever (dense, BM25) pulls per
# query, before fusion. Deliberately larger than RETRIEVAL_TOP_K: a chunk
# that ranks #30 in dense but #2 in BM25 should still be eligible to win
# after fusion -- cutting each retriever to the final top-k first would
# throw it away before fusion ever sees it.
DENSE_CANDIDATE_K = 30
BM25_CANDIDATE_K = 30

# Final number of chunks returned after hybrid fusion.
RETRIEVAL_TOP_K = 10

# Reciprocal Rank Fusion's rank-damping constant. 60 is the value from the
# original RRF paper (Cormack et al.) and is the de facto default used by
# most hybrid search implementations -- large enough that a #1-ranked
# result doesn't completely dominate a fused list, small enough that rank
# order still matters more than which retriever found it.
RRF_K = 60

# Simple whitespace/punctuation tokenizer shared by BM25 indexing and query
# tokenization -- both sides must tokenize identically or BM25 term matching
# silently degrades.
BM25_TOKEN_PATTERN = r"[a-z0-9]+"

# ---------------------------------------------------------------------------
# Reranking (Stage 8)
# ---------------------------------------------------------------------------

# FlashRank's model zoo (see .venv/Lib/site-packages/flashrank/Config.py)
# offers several ONNX cross-encoders, already CPU-quantized. MiniLM-L-12 is
# the standard, most widely validated reranker in this size class for
# passage reranking -- see src/reranking/flashrank_reranker.py's module
# docstring for the full comparison against the zoo's other models.
RERANKER_MODEL_NAME = "ms-marco-MiniLM-L-12-v2"

# Hybrid retrieval pulls this many fused candidates for the reranker to
# re-score -- larger than RETRIEVAL_TOP_K (which is for retrieval used
# stand-alone, e.g. scripts/run_retrieval.py) because reranking is exactly
# what recovers a relevant chunk that RRF fusion alone ranked too low to
# survive a smaller cut.
RERANK_CANDIDATE_K = 25

# Final number of chunks handed to the LLM after reranking.
RERANK_TOP_K = 5

# ---------------------------------------------------------------------------
# Generation (Stage 9)
# ---------------------------------------------------------------------------

# Served locally via Ollama (already running on this machine). Chosen for
# CPU-only inference: 7B is the largest size that stays comfortably usable
# on this machine's RAM alongside Qdrant/Docker and the embedding/reranking
# models, at 4-bit quantization Ollama pulls by default.
GENERATION_MODEL_NAME = "qwen2.5:7b"
OLLAMA_URL = "http://localhost:11434"

# Low temperature: this is a grounded-answer RAG system over financial
# filings, not a creative-writing task -- answers should be as
# deterministic and literal as possible given the same retrieved context.
GENERATION_TEMPERATURE = 0.1

# ---------------------------------------------------------------------------
# Evaluation (RAGAS)
# ---------------------------------------------------------------------------

# A different model than GENERATION_MODEL_NAME (different lineage: Meta vs
# Alibaba) so the pipeline isn't scored by the same model that answered.
# gpt-oss:20b was tried first (already pulled, unused elsewhere) but its
# MoE architecture hits a real Ollama-level bug on this machine --
# "tensor blk.0.ffn_down_exps.weight size overflow" -- reproduced via a
# raw curl to Ollama directly (not our code) both before AND after a full
# fresh re-pull, and gpt-oss:120b shares the same MoE tensor layout so
# isn't a safe fallback either. Llama 3.1 8B is a dense (non-MoE)
# architecture, avoiding that whole bug class. Reached via Ollama's
# OpenAI-compatible endpoint since RAGAS's llm_factory expects an
# OpenAI-shaped client.
RAGAS_JUDGE_MODEL_NAME = "llama3.1:8b"
OLLAMA_OPENAI_COMPATIBLE_URL = "http://localhost:11434/v1"

# ---------------------------------------------------------------------------
# Hallucination prevention: confidence gating (Module 19)
# ---------------------------------------------------------------------------

# Below this rerank score, the top retrieved chunk is judged too weak to
# ground an answer -- refuse rather than force a guess from marginal
# context. This is a starting point, not a tuned value: FlashRank scores
# observed for genuinely relevant chunks throughout this project were
# consistently 0.98-0.997 (see Stage 8/9 testing), so 0.3 is intentionally
# conservative (errs toward answering) given there's no labeled negative
# (irrelevant-chunk) score sample yet to calibrate a tighter cutoff against.
# Revisit once the eval set grows enough to include known-irrelevant cases.
CONFIDENCE_THRESHOLD = 0.3

# ---------------------------------------------------------------------------
# Caching (Module 25)
# ---------------------------------------------------------------------------

# SQLite, not Redis: a single-user local tool doesn't benefit from a
# separate cache server, and it would be one more process (like Qdrant)
# that can silently stop running between sessions (see note #14 in
# PIPELINE_TECHNICAL_NOTES.md). One file, no server, is the right amount of
# infrastructure here.
RESPONSE_CACHE_DB = PROJECT_ROOT / "data" / "cache" / "response_cache.db"

# Filings don't change; the only reason a cached answer should ever go
# stale is the pipeline itself changing (a prompt fix, a model swap). A
# week is long enough to matter, short enough that a stale entry from a
# fixed bug doesn't linger indefinitely.
RESPONSE_CACHE_TTL_SECONDS = 7 * 24 * 3600

# ---------------------------------------------------------------------------
# Observability (Module 26)
# ---------------------------------------------------------------------------

# Plain JSON-lines, not LangSmith/Phoenix/OpenTelemetry: those are built for
# a team debugging a deployed, multi-user service with a hosted dashboard --
# overkill for a single-user local tool. A grep/jq-able local file covers
# the actual need (was this request slow, what did it retrieve, did it hit
# cache, did a guardrail fire) without a new service or account.
TRACE_LOG_PATH = PROJECT_ROOT / "data" / "logs" / "traces.jsonl"

# ---------------------------------------------------------------------------
# API security (Module 22)
# ---------------------------------------------------------------------------

# None (disabled) by default -- this API is bound to localhost only, for
# one user, on one machine. Set the SEC_RAG_API_KEY environment variable to
# require an "Authorization: Bearer <key>" header on every request; this is
# the one piece of Module 22 that's actually proportionate here (the moment
# this server is reachable from anywhere but localhost, *something* should
# gate it) -- OAuth/JWT/SSO/RBAC would be solving a multi-user problem this
# project doesn't have.
API_KEY = os.environ.get("SEC_RAG_API_KEY")

# ---------------------------------------------------------------------------
# UI (ui/streamlit_app.py)
# ---------------------------------------------------------------------------

# The Streamlit app is a thin client over the FastAPI server -- it holds no
# pipeline logic of its own, only calls these same HTTP endpoints (see
# src/api/routes.py), so there is exactly one implementation of retrieval/
# guardrails/caching/generation to keep correct, not two.
API_BASE_URL = "http://127.0.0.1:8000"