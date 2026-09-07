"""
Sparse (keyword-based) retrieval over all chunks using BM25.

Save as: sec-rag-project/src/retrieval/bm25_retriever.py

--- Why build the index in-process instead of persisting it ---
rank-bm25 has no built-in save/load format, and this corpus (~30k chunks) is
small enough that building the index from chunks.json takes a fraction of a
second (see the module's __main__ smoke test for a measured number). A
caller that runs many queries (a retrieval CLI, or later a FastAPI server)
should call build_bm25_index() once at startup and reuse the returned index
across requests -- do not rebuild it per query.

--- Filtering happens after scoring, not before ---
BM25Okapi.get_scores() scores every document in the corpus regardless of
which ones you actually want (there's no way to score a subset only), so
computing full corpus scores and then keeping only the ones matching
`filters` costs nothing extra over restricting the corpus up front, and
avoids needing a separate BM25 index per filter combination.

--- Tokenization must match on both sides ---
The same tokenize() function is used to build the index and to tokenize an
incoming query; BM25 term matching is exact-token, so any mismatch here
(e.g. one side lowercasing and the other not) silently drops recall.
"""

import glob
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.settings import PROCESSED_DIR, BM25_CANDIDATE_K, BM25_TOKEN_PATTERN

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(BM25_TOKEN_PATTERN)


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def load_all_chunks(processed_dir: Path = PROCESSED_DIR) -> List[dict]:
    """Reads every chunks.json under data/processed/ into one flat list, in
    the same folder-then-in-file order every run (sorted glob) so BM25
    index/chunk alignment is reproducible."""
    chunks = []
    for path in sorted(glob.glob(str(processed_dir / "*" / "chunks.json"))):
        with open(path, encoding="utf-8") as f:
            chunks.extend(json.load(f))
    return chunks


def build_bm25_index(chunks: List[dict]):
    """Builds a BM25Okapi index over chunks, in the given order. The caller
    must pass this same `chunks` list (same order) to bm25_search()."""
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [tokenize(c["text"]) for c in chunks]
    return BM25Okapi(tokenized_corpus)


def matches_filters(chunk: dict, filters: Optional[Dict]) -> bool:
    """Shared filter predicate: every field in `filters` must equal the
    chunk's value for that field. Same {"field": value} shape dense_retriever
    accepts, so both retrievers filter identically."""
    if not filters:
        return True
    return all(chunk.get(field) == value for field, value in filters.items())


def bm25_search(query_text: str, chunks: List[dict], bm25_index, top_k: int = BM25_CANDIDATE_K,
                 filters: Optional[Dict] = None) -> List[dict]:
    """
    Scores query_text against every chunk in the index, keeps only chunks
    matching `filters`, and returns the top_k as a list of chunk payload
    dicts with "score" (raw BM25 score) and "source": "bm25" added,
    ordered best-first. `chunks` must be the exact list build_bm25_index()
    was built from (same order).
    """
    # Not filtered by score sign: BM25's IDF term goes negative for a query
    # term that appears in more than half the corpus (log((N-df+0.5)/(df+0.5))
    # < 0 when df > N/2) -- common in SEC filings, which share a lot of
    # boilerplate vocabulary ("Item", "fiscal year", "the Company"). A
    # negative score can still be the most relevant chunk available; only
    # matches_filters should exclude a candidate here.
    scores = bm25_index.get_scores(tokenize(query_text))

    scored = [
        (score, chunk) for score, chunk in zip(scores, chunks)
        if matches_filters(chunk, filters)
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)

    hits = []
    for score, chunk in scored[:top_k]:
        hit = dict(chunk)
        hit["score"] = float(score)
        hit["source"] = "bm25"
        hits.append(hit)
    return hits


if __name__ == "__main__":
    import time

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    t0 = time.time()
    chunks = load_all_chunks()
    print(f"Loaded {len(chunks)} chunks in {time.time() - t0:.2f}s")

    t0 = time.time()
    index = build_bm25_index(chunks)
    print(f"Built BM25 index in {time.time() - t0:.2f}s")

    t0 = time.time()
    results = bm25_search("risk factors related to cybersecurity", chunks, index)
    print(f"Searched in {time.time() - t0:.3f}s -- top result:")
    print(f"  {results[0]['ticker']} FY{results[0]['fiscal_year']} {results[0]['section_item']} "
          f"score={results[0]['score']:.2f}: {results[0]['text'][:100]!r}")
