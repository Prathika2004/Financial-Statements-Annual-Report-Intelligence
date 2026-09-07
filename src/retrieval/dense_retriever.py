"""
Dense (embedding-based) retrieval against the Qdrant collection.

Save as: sec-rag-project/src/retrieval/dense_retriever.py

--- Why dense_search() takes a vector, not a query string ---
dense_search() is the low-level primitive; dense_search_from_text() is the
convenience wrapper most callers actually want. They're kept separate for
Query Transformation (a later stage, not built yet, but this split exists
specifically for it): HyDE generates a hypothetical *answer* document and
embeds that instead of the user's literal question, and that embedding
should NOT get BGE's query-instruction prefix (see embedder.py's docstring
-- the prefix is for queries, not passage-like text). Because
dense_search_from_text() is a thin wrapper over dense_search(), HyDE (or
any future transform) can call embed_texts(..., is_query=False) itself and
hand the resulting vector straight to dense_search() without this module
changing at all.

--- Filters are a plain dict here, not a Qdrant Filter object ---
{"ticker": "TSLA", "fiscal_year": 2023} rather than qdrant_client.models
types, so the retrieval layer above this (hybrid fusion, and eventually the
API layer) doesn't need to import Qdrant-specific types just to ask for a
filtered search. bm25_retriever.py accepts the same dict shape and applies
it as an in-memory predicate instead -- one filter vocabulary, two
backends.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.settings import QDRANT_COLLECTION_NAME, DENSE_CANDIDATE_K

logger = logging.getLogger(__name__)


def build_qdrant_filter(filters: Optional[Dict]):
    """Translates a plain {"field": value} dict into a Qdrant Filter with
    an equality match per field, ANDed together. Returns None if filters is
    empty/None, so callers can pass it straight through without a branch."""
    if not filters:
        return None

    from qdrant_client.models import Filter, FieldCondition, MatchValue

    return Filter(must=[
        FieldCondition(key=field, match=MatchValue(value=value))
        for field, value in filters.items()
    ])


def dense_search(query_vector: np.ndarray, client, top_k: int = DENSE_CANDIDATE_K,
                  filters: Optional[Dict] = None,
                  collection_name: str = QDRANT_COLLECTION_NAME) -> List[dict]:
    """
    Searches Qdrant with an already-embedded query vector. Returns a list of
    chunk payload dicts, each with "score" (Qdrant's cosine similarity) and
    "source": "dense" added, ordered best-first.
    """
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector.tolist(),
        query_filter=build_qdrant_filter(filters),
        limit=top_k,
    ).points

    hits = []
    for r in results:
        hit = dict(r.payload)
        hit["score"] = r.score
        hit["source"] = "dense"
        hits.append(hit)
    return hits


def dense_search_from_text(query_text: str, model, client, top_k: int = DENSE_CANDIDATE_K,
                            filters: Optional[Dict] = None,
                            collection_name: str = QDRANT_COLLECTION_NAME) -> List[dict]:
    """Convenience wrapper: embeds query_text as a query (BGE instruction
    prefix applied) then searches. This is what a plain user question uses;
    see the module docstring for why HyDE-style callers should use
    dense_search() directly instead."""
    from src.embeddings.embedder import embed_query

    query_vector = embed_query(query_text, model)
    return dense_search(query_vector, client, top_k=top_k, filters=filters, collection_name=collection_name)
