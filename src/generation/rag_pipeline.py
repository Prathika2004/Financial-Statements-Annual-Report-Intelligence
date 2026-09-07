"""
Full RAG pipeline: guardrails -> cache -> retrieve -> rerank -> confidence
gate -> generate -> verify -> citation check -> cache write -> trace.

Save as: sec-rag-project/src/generation/rag_pipeline.py

--- Why this is split into screen_and_prepare() / finalize_answer(), not
one function that also calls the LLM ---
/ask (blocking) and /ask/stream (SSE) need to share every cross-cutting
concern here (guardrails, cache, confidence gating, citation/numeric
checks) but differ in exactly one place: how the answer text is produced
(one blocking chat() call vs. streaming tokens as they arrive). Splitting
the pipeline at that seam means both endpoints call the exact same
pre-generation logic and the exact same post-generation logic -- neither
duplicates it, and neither has to fake streaming through a function that
wasn't built for it. answer_question() below is the blocking convenience
wrapper most callers (scripts, tests) actually want; src/api/routes.py
calls screen_and_prepare()/finalize_answer() directly so it can stream the
middle part.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.settings import CONFIDENCE_THRESHOLD
from src.retrieval.retriever import RetrievalContext, apply_technique_and_retrieve
from src.generation.prompt_templates import assemble_messages
from src.generation.llm_router import chat, generate
from src.generation.numeric_verifier import verify_numbers
from src.guardrails.input_guardrails import screen_input
from src.guardrails.output_guardrails import check_citations
from src.caching.response_cache import get_cached, set_cached
from src.observability.tracer import log_trace

logger = logging.getLogger(__name__)

LOW_CONFIDENCE_MESSAGE = (
    "The retrieved filing excerpts don't appear closely related enough to this question "
    "to answer confidently, so no answer was generated. Try rephrasing the question or "
    "narrowing it to a specific company/fiscal year."
)
BLOCKED_MESSAGE = "This question could not be processed."


def check_confidence(sources: List[dict], threshold: float = CONFIDENCE_THRESHOLD) -> Dict:
    """
    Context Validation / Confidence Threshold (Module 19): the reranker's
    score for the top result is a direct, already-computed signal of how
    relevant the best available context actually is -- if even the best
    match is weak, forcing a generation call produces a fluent-sounding
    answer built on a shaky foundation. Gating on it costs nothing extra
    (the score already exists) and catches this before an expensive
    generation call, not after.
    """
    if not sources:
        return {"confident": False, "top_score": None}
    top_score = sources[0]["score"]
    return {"confident": top_score >= threshold, "top_score": top_score}


def screen_and_prepare(question: str, context: RetrievalContext, ranker,
                        filters: Optional[Dict] = None, technique: str = "none",
                        use_cache: bool = True) -> Dict:
    """
    Runs every pre-generation step: input guardrails, cache lookup,
    retrieval+reranking, confidence gate. Returns one of:
      {"status": "blocked", "reasons": [...]}
      {"status": "cached", "result": {...}}
      {"status": "low_confidence", "sources": [...], "top_score": float}
      {"status": "ready", "sources": [...]}
    Never calls the LLM for generation -- callers branch on "status" and,
    for "ready", proceed to generate an answer themselves (blocking or
    streamed) and then call finalize_answer().
    """
    input_check = screen_input(question)
    if input_check["blocked"]:
        logger.warning("Blocked question (guardrail): %s", input_check["reasons"])
        return {"status": "blocked", "reasons": input_check["reasons"]}

    if use_cache:
        cached = get_cached(question, filters, technique)
        if cached is not None:
            return {"status": "cached", "result": cached}

    sources = apply_technique_and_retrieve(question, context, ranker, generate,
                                            technique=technique, filters=filters)

    confidence = check_confidence(sources)
    if not confidence["confident"]:
        return {"status": "low_confidence", "sources": sources, "top_score": confidence["top_score"]}

    return {"status": "ready", "sources": sources}


def finalize_answer(question: str, sources: List[dict], answer: str,
                     filters: Optional[Dict] = None, technique: str = "none",
                     use_cache: bool = True) -> Dict:
    """
    Runs every post-generation step once a complete answer string exists:
    numeric grounding check, citation enforcement, cache write. Shared by
    both the blocking and streaming endpoints once they each have an answer.
    """
    numeric_check = verify_numbers(answer, sources)
    citation_check = check_citations(answer, len(sources))

    result = {
        "answer": answer,
        "sources": sources,
        "numeric_check": numeric_check,
        "citation_check": citation_check,
    }
    if use_cache:
        set_cached(question, filters, technique, result)
    return result


def answer_question(question: str, context: RetrievalContext, ranker,
                     filters: Optional[Dict] = None, technique: str = "none",
                     use_cache: bool = True) -> Dict:
    """
    Blocking convenience wrapper running the complete pipeline for one
    question -- the composition root for scripts/tests that just want one
    call. src/api/routes.py does NOT call this for /ask/stream (it needs to
    stream the generation step), but calls screen_and_prepare()/
    finalize_answer() directly instead, with identical behavior otherwise.
    """
    t0 = time.time()

    prep = screen_and_prepare(question, context, ranker, filters, technique, use_cache)

    if prep["status"] == "blocked":
        log_trace({"question": question, "filters": filters, "technique": technique,
                   "status": "blocked", "reasons": prep["reasons"], "elapsed_seconds": time.time() - t0})
        return {
            "answer": BLOCKED_MESSAGE, "sources": [],
            "numeric_check": {"checked": [], "unverified": []}, "citation_check": None,
            "blocked": True, "cached": False, "low_confidence": False,
        }

    if prep["status"] == "cached":
        log_trace({"question": question, "filters": filters, "technique": technique,
                   "status": "cached", "elapsed_seconds": time.time() - t0})
        return {**prep["result"], "blocked": False, "cached": True, "low_confidence": False}

    if prep["status"] == "low_confidence":
        log_trace({"question": question, "filters": filters, "technique": technique,
                   "status": "low_confidence", "top_score": prep["top_score"],
                   "elapsed_seconds": time.time() - t0})
        return {
            "answer": LOW_CONFIDENCE_MESSAGE, "sources": prep["sources"],
            "numeric_check": {"checked": [], "unverified": []}, "citation_check": None,
            "blocked": False, "cached": False, "low_confidence": True,
        }

    sources = prep["sources"]
    messages = assemble_messages(question, sources)
    answer = chat(messages)
    result = finalize_answer(question, sources, answer, filters, technique, use_cache)

    log_trace({
        "question": question, "filters": filters, "technique": technique, "status": "answered",
        "num_sources": len(sources), "top_score": sources[0]["score"] if sources else None,
        "unverified_numbers": result["numeric_check"]["unverified"],
        "citation_check": result["citation_check"], "elapsed_seconds": time.time() - t0,
    })
    return {**result, "blocked": False, "cached": False, "low_confidence": False}
