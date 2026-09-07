"""
Cross-encoder reranking over hybrid retrieval's fused candidates, via FlashRank.

Save as: sec-rag-project/src/reranking/flashrank_reranker.py

--- Why FlashRank, and why ms-marco-MiniLM-L-12-v2 specifically ---
FlashRank ships several ONNX cross-encoders that are already quantized for
CPU inference -- no backend work needed here the way embedder.py needed an
explicit OpenVINO export, FlashRank does the equivalent up front. Its model
zoo (see .venv/Lib/site-packages/flashrank/Config.py's model_file_map):

  - ms-marco-TinyBERT-L-2-v2 (default, ~4MB, 2 transformer layers):
    fastest, but a 2-layer cross-encoder measurably under-ranks compared to
    a deeper one -- fine when reranking hundreds of candidates under a
    tight latency budget, which isn't this project's situation.
  - ms-marco-MiniLM-L-12-v2 (~34MB, 12 layers, int8-quantized ONNX):
    the standard, most widely validated reranker in this size class for
    exactly this task (trained on MS MARCO passage ranking) -- chosen here.
  - rank-T5-flan: a seq2seq (T5) reranker, typically the best quality in
    this zoo but meaningfully heavier per query-passage pair than a pure
    BERT-encoder cross-encoder -- an easy one-line upgrade (just change
    RERANKER_MODEL_NAME) if quality is later found lacking.
  - ce-esci-MiniLM-L12-v2: trained on Amazon product-search relevance
    (ESCI) -- a domain mismatch for financial filing text.
  - rank_zephyr_7b_v1_full: a 7B LLM-based listwise reranker -- the same
    weight-class problem an LLM would have for generation; unnecessary
    here and the wrong tool on a CPU-only laptop at this stage.

Reranking only ever scores RERANK_CANDIDATE_K (25) already-hybrid-retrieved
candidates per query, not hundreds -- so MiniLM-L-12's extra layers over
TinyBERT cost single-digit milliseconds in practice. There is no real
latency reason to take the quality hit of the smaller model here.

--- "score" vs "rrf_score" after reranking ---
Chunks entering rerank() already carry a "score" field from whichever
retriever found them (dense cosine similarity or raw BM25) and an
"rrf_score" from hybrid_fusion.reciprocal_rank_fusion(). FlashRank
overwrites "score" in its own returned dicts with its cross-encoder
relevance score -- which becomes the correct, final ranking signal
post-rerank. "rrf_score" is left untouched so the pre-rerank fusion rank
stays inspectable for debugging. rerank() does not mutate the caller's
input list -- it reranks copies.

--- Local model cache ---
FlashRank's own default cache_dir is "/tmp", which isn't a meaningful
project-local path on Windows. Caching under PROJECT_ROOT/models/flashrank
instead keeps every downloaded/converted model artifact in one place (next
to the OpenVINO/ONNX embedding model cache from Stage 5) rather than
scattered into a temp directory that may not persist or even resolve.
"""

import logging
import sys
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.settings import PROJECT_ROOT, RERANKER_MODEL_NAME, RERANK_TOP_K

logger = logging.getLogger(__name__)

_RERANKER_CACHE_DIR = PROJECT_ROOT / "models" / "flashrank"


def load_reranker(model_name: str = RERANKER_MODEL_NAME):
    from flashrank import Ranker

    _RERANKER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ranker = Ranker(model_name=model_name, cache_dir=str(_RERANKER_CACHE_DIR))
    logger.info("Loaded FlashRank reranker '%s' (cache: %s)", model_name, _RERANKER_CACHE_DIR)
    return ranker


def rerank(query: str, chunks: List[dict], ranker, top_k: int = RERANK_TOP_K) -> List[dict]:
    """
    Re-scores `chunks` (already hybrid-retrieved, e.g. from
    retriever.retrieve()) against `query` with a cross-encoder, and returns
    the top_k best-scoring ones, best-first. Does not mutate `chunks`.
    """
    from flashrank import RerankRequest

    if not chunks:
        return []

    passages = [dict(c, id=c["chunk_id"]) for c in chunks]
    results = ranker.rerank(RerankRequest(query=query, passages=passages))

    # FlashRank assigns "score" as a numpy float32 (see Ranker.rerank()'s
    # sigmoid/softmax computation), not a native Python float. Pydantic
    # response models silently coerce this, which is why the bug wasn't
    # visible through the API's normal JSON responses -- but a raw
    # json.dumps() call (observability's tracer, the streaming endpoint's
    # inline events) has no such coercion and raises
    # "Object of type float32 is not JSON serializable". Cast here, once,
    # at the source, so every downstream consumer always gets a plain float.
    for r in results:
        r["score"] = float(r["score"])

    return results[:top_k]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ranker = load_reranker()
    sample_chunks = [
        {"chunk_id": "1", "text": "The company's revenue grew 12% year over year driven by cloud services."},
        {"chunk_id": "2", "text": "Item 1A. Risk Factors. We face risks related to cybersecurity threats."},
        {"chunk_id": "3", "text": "The Board of Directors approved a quarterly cash dividend."},
    ]
    results = rerank("What are the company's cybersecurity risks?", sample_chunks, ranker, top_k=2)
    for r in results:
        print(f"score={r['score']:.4f} | {r['text'][:80]!r}")
