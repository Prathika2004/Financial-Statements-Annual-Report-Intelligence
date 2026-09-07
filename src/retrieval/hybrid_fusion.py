"""
Merges multiple ranked retrieval result lists into one via Reciprocal Rank
Fusion (RRF).

Save as: sec-rag-project/src/retrieval/hybrid_fusion.py

--- Why RRF instead of combining raw scores ---
A cosine similarity (dense) and a BM25 score are on completely different,
non-comparable scales -- averaging or summing them directly would let
whichever retriever happens to produce larger numbers dominate regardless
of actual relevance. RRF sidesteps this by using each result's *rank
position* within its own list rather than its raw score, so no
normalization step or per-retriever score calibration is needed.

--- Why this takes a list of lists, not exactly two lists ---
Hybrid search today means "one dense list + one BM25 list", but once Query
Transformation (multi-query expansion, HyDE) exists, a single user question
can produce several query variants, each contributing its own dense+BM25
pair. reciprocal_rank_fusion() already accepts an arbitrary number of ranked
lists for exactly that reason -- the query-transformation layer will call
the dense/BM25 retrievers once per variant and hand ALL resulting lists to
this same function, unchanged.
"""

import logging
import sys
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.settings import RRF_K, RETRIEVAL_TOP_K

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(ranked_lists: List[List[dict]], k: int = RRF_K,
                            top_k: int = RETRIEVAL_TOP_K) -> List[dict]:
    """
    Fuses any number of ranked chunk lists (each already sorted best-first,
    e.g. from dense_search / bm25_search) into one ranked list.

    Each chunk is identified by chunk_id -- the same chunk appearing in
    multiple input lists (e.g. found by both dense and BM25) has its RRF
    contributions summed, rewarding chunks multiple retrievers agree on.
    The returned dicts carry the original chunk payload from whichever
    occurrence was seen first, plus "rrf_score" and "found_by" (the list of
    sources -- e.g. ["dense", "bm25"] -- that contributed to it).
    """
    rrf_scores = {}
    chunk_data = {}
    found_by = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            chunk_id = chunk["chunk_id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = chunk
            found_by.setdefault(chunk_id, [])
            source = chunk.get("source", "unknown")
            if source not in found_by[chunk_id]:
                found_by[chunk_id].append(source)

    ranked_chunk_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

    fused = []
    for chunk_id in ranked_chunk_ids[:top_k]:
        hit = dict(chunk_data[chunk_id])
        hit["rrf_score"] = rrf_scores[chunk_id]
        hit["found_by"] = found_by[chunk_id]
        fused.append(hit)
    return fused
