"""
Tests for Stage 8 reranking logic.

Uses a fake ranker instead of loading the real ~34MB FlashRank model, so
this suite stays fast and offline -- load_reranker() itself (model
download/caching) is exercised by running src/reranking/flashrank_reranker.py
directly as a manual smoke test instead, same pattern as test_embedder.py.

Run with: pytest tests/test_reranker.py -v
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.reranking.flashrank_reranker import rerank


class FakeRanker:
    """Mimics FlashRank's Ranker.rerank(): scores each passage (here, by
    text length -- a nonsense but deterministic stand-in for a real
    cross-encoder) and returns passages sorted best-first."""

    def rerank(self, request):
        for passage in request.passages:
            passage["score"] = float(len(passage["text"]))
        return sorted(request.passages, key=lambda p: p["score"], reverse=True)


def make_chunk(chunk_id, text):
    return {"chunk_id": chunk_id, "text": text, "rrf_score": 0.01, "found_by": ["dense"]}


def test_rerank_orders_by_cross_encoder_score():
    chunks = [make_chunk("a", "short"), make_chunk("b", "a much longer piece of text")]
    results = rerank("query", chunks, FakeRanker(), top_k=10)
    assert [r["chunk_id"] for r in results] == ["b", "a"]


def test_rerank_truncates_to_top_k():
    chunks = [make_chunk(str(i), "x" * i) for i in range(1, 6)]
    results = rerank("query", chunks, FakeRanker(), top_k=2)
    assert len(results) == 2


def test_rerank_empty_input_returns_empty_without_calling_ranker():
    class ExplodingRanker:
        def rerank(self, request):
            raise AssertionError("should not be called for empty input")

    assert rerank("query", [], ExplodingRanker(), top_k=5) == []


def test_rerank_does_not_mutate_caller_chunks():
    chunks = [make_chunk("a", "some text")]
    rerank("query", chunks, FakeRanker(), top_k=10)
    assert "score" not in chunks[0]  # rerank() must copy, not mutate in place


def test_rerank_preserves_rrf_score_and_found_by_alongside_new_score():
    chunks = [make_chunk("a", "some text")]
    results = rerank("query", chunks, FakeRanker(), top_k=10)
    assert results[0]["rrf_score"] == 0.01
    assert results[0]["found_by"] == ["dense"]
    assert results[0]["score"] == float(len("some text"))


def test_rerank_score_is_a_native_float_not_a_numpy_scalar():
    """
    Real FlashRank assigns "score" as numpy.float32 (from its sigmoid/
    softmax computation over ONNX logits) -- confirmed the hard way: this
    passed through Pydantic-validated API responses silently (Pydantic
    coerces it) but crashed a raw json.dumps() call in the observability
    tracer and the streaming endpoint's inline events with "Object of type
    float32 is not JSON serializable". FakeRanker elsewhere in this file
    already returns a native float and would never have caught this --
    this test specifically mimics FlashRank's actual numpy return type.
    """
    class NumpyScoreRanker:
        def rerank(self, request):
            for passage in request.passages:
                passage["score"] = np.float32(0.87)
            return request.passages

    results = rerank("query", [make_chunk("a", "text")], NumpyScoreRanker(), top_k=10)
    assert isinstance(results[0]["score"], float)
    assert not isinstance(results[0]["score"], np.floating)
    json.dumps(results[0])  # must not raise


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
