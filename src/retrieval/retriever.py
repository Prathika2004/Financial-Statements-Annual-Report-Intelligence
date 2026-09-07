"""
Top-level hybrid retrieval entry point: dense + BM25 + RRF fusion.

Save as: sec-rag-project/src/retrieval/retriever.py

--- Why retrieve() takes a list of queries, not one string ---
This is the one design decision in Stage 7 made specifically for a stage
that doesn't exist yet: Query Transformation. Multi-query expansion and
decomposition both turn one user question into several query variants, and
each variant needs its own dense+BM25 search before all results get fused
together. retrieve() already runs dense+BM25 for every string in `queries`
and RRF-fuses every resulting list -- so adding query transformation later
means "generate a list of query strings upstream and call retrieve() with
it", not "change this function's signature and every one of its callers".
Every caller today just passes a single-item list.

HyDE doesn't fit this same shape (it embeds a hypothetical *answer*
document, not literal query text) -- a HyDE strategy should call
dense_retriever.dense_search() directly with its own pre-computed vector
and feed the result into hybrid_fusion.reciprocal_rank_fusion() alongside
whatever else it runs, bypassing retrieve() entirely.

--- Why RetrievalContext exists ---
The embedding model, Qdrant client, and BM25 index/corpus are all expensive
to build and cheap to reuse. build_retrieval_context() constructs all of
them once; retrieve() takes the resulting RetrievalContext as a parameter
rather than rebuilding any of it per call. A future FastAPI server builds
one RetrievalContext at startup and reuses it across every request.
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.settings import (
    DENSE_CANDIDATE_K, BM25_CANDIDATE_K, RETRIEVAL_TOP_K, RERANK_CANDIDATE_K, RERANK_TOP_K,
)
from src.retrieval.dense_retriever import dense_search_from_text
from src.retrieval.bm25_retriever import load_all_chunks, build_bm25_index, bm25_search
from src.retrieval.hybrid_fusion import reciprocal_rank_fusion

logger = logging.getLogger(__name__)


@dataclass
class RetrievalContext:
    """Everything hybrid retrieval needs, built once and reused across
    queries. See build_retrieval_context()."""
    embedding_model: object
    qdrant_client: object
    bm25_chunks: list
    bm25_index: object


def build_retrieval_context() -> RetrievalContext:
    from src.embeddings.embedder import load_embedding_model
    from src.vectorstore.qdrant_client import get_client

    logger.info("Loading embedding model...")
    embedding_model = load_embedding_model()

    logger.info("Connecting to Qdrant...")
    qdrant_client = get_client()

    logger.info("Loading chunks and building BM25 index...")
    bm25_chunks = load_all_chunks()
    bm25_index = build_bm25_index(bm25_chunks)

    logger.info("Retrieval context ready (%d chunks indexed for BM25).", len(bm25_chunks))
    return RetrievalContext(embedding_model, qdrant_client, bm25_chunks, bm25_index)


def retrieve(queries: List[str], context: RetrievalContext, filters: Optional[Dict] = None,
             top_k: int = RETRIEVAL_TOP_K) -> List[dict]:
    """
    Runs dense + BM25 retrieval for every query string in `queries`, then
    RRF-fuses all resulting ranked lists into one. See module docstring for
    why `queries` is a list even for a single plain question.
    """
    ranked_lists = []
    for query_text in queries:
        ranked_lists.append(dense_search_from_text(
            query_text, context.embedding_model, context.qdrant_client,
            top_k=DENSE_CANDIDATE_K, filters=filters,
        ))
        ranked_lists.append(bm25_search(
            query_text, context.bm25_chunks, context.bm25_index,
            top_k=BM25_CANDIDATE_K, filters=filters,
        ))

    return reciprocal_rank_fusion(ranked_lists, top_k=top_k)


def retrieve_and_rerank(queries: List[str], context: RetrievalContext, ranker,
                         filters: Optional[Dict] = None, original_query: Optional[str] = None,
                         candidate_k: int = RERANK_CANDIDATE_K, final_k: int = RERANK_TOP_K) -> List[dict]:
    """
    Stage 7 + Stage 8 composed: hybrid-retrieve a larger candidate pool
    (candidate_k, default RERANK_CANDIDATE_K) across every query variant,
    then cross-encoder rerank that pool down to final_k.

    `original_query` is what the reranker scores candidates against --
    it defaults to queries[0]. This split matters once Query Transformation
    exists: multi-query/decomposition variants in `queries` are retrieval
    aids meant to widen recall, but the reranking judgment should be made
    against what the user actually asked, not against a generated variant.
    A HyDE strategy should not use this function at all -- see
    dense_retriever.py's module docstring for why.
    """
    from src.reranking.flashrank_reranker import rerank

    candidates = retrieve(queries, context, filters=filters, top_k=candidate_k)
    query_for_reranking = original_query or queries[0]
    return rerank(query_for_reranking, candidates, ranker, top_k=final_k)


TECHNIQUES = ("none", "rewrite", "step-back", "multi-query", "decompose", "hyde")


def apply_technique_and_retrieve(question: str, context: RetrievalContext, ranker, llm_generate,
                                  technique: str = "none", filters: Optional[Dict] = None,
                                  candidate_k: int = RERANK_CANDIDATE_K, final_k: int = RERANK_TOP_K) -> List[dict]:
    """
    The single place a caller (the API) selects a query-transformation
    technique ahead of hybrid retrieval + reranking. See
    src/query_transformation/'s module docstrings for what each technique
    does and why it's shaped the way it is; this function only wires them
    into retrieve_and_rerank() (or, for "hyde", the separate path its
    different mechanics require -- see hyde.py's module docstring).

    "step-back" retrieves with BOTH the original question and its
    generalized form (not the generalized form alone) -- the module's own
    purpose is to add broader context *alongside* the specific answer, not
    replace the specific query that actually answers what was asked.
    """
    if technique in (None, "none"):
        return retrieve_and_rerank([question], context, ranker, filters=filters,
                                    candidate_k=candidate_k, final_k=final_k)

    if technique == "rewrite":
        from src.query_transformation.query_rewriting import rewrite_query
        queries = [rewrite_query(question, llm_generate)]

    elif technique == "step-back":
        from src.query_transformation.query_rewriting import step_back_query
        queries = [question, step_back_query(question, llm_generate)]

    elif technique == "multi-query":
        from src.query_transformation.multi_query import generate_multi_queries
        queries = generate_multi_queries(question, llm_generate)

    elif technique == "decompose":
        from src.query_transformation.decomposition import decompose_query
        queries = decompose_query(question, llm_generate)

    elif technique == "hyde":
        from src.query_transformation.hyde import hyde_search
        from src.reranking.flashrank_reranker import rerank
        candidates = hyde_search(question, llm_generate, context.embedding_model, context.qdrant_client,
                                  top_k=candidate_k, filters=filters)
        return rerank(question, candidates, ranker, top_k=final_k)

    else:
        raise ValueError(f"Unknown technique {technique!r} -- expected one of {TECHNIQUES}")

    return retrieve_and_rerank(queries, context, ranker, filters=filters, original_query=question,
                                candidate_k=candidate_k, final_k=final_k)
