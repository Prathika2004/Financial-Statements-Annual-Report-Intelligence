"""
Tests for apply_technique_and_retrieve()'s dispatch logic: which technique
calls which query_transformation function, and with what queries.

Uses monkeypatching to intercept retrieve_and_rerank()/hyde internals
rather than hitting live Qdrant/BM25/a reranker -- this is purely about
verifying the routing logic, not re-testing retrieval/reranking themselves
(already covered elsewhere).

Run with: pytest tests/test_retriever_technique.py -v
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.append(str(Path(__file__).resolve().parent.parent))

import src.retrieval.retriever as retriever_module
from src.retrieval.retriever import apply_technique_and_retrieve, TECHNIQUES
from src.query_transformation.llm_interface import FakeLLM


def test_none_technique_retrieves_with_bare_question(monkeypatch):
    captured = {}

    def fake_retrieve_and_rerank(queries, context, ranker, filters=None, original_query=None,
                                  candidate_k=None, final_k=None):
        captured["queries"] = queries
        captured["original_query"] = original_query
        return ["result"]

    monkeypatch.setattr(retriever_module, "retrieve_and_rerank", fake_retrieve_and_rerank)

    result = apply_technique_and_retrieve("What was revenue?", context=None, ranker=None,
                                           llm_generate=FakeLLM(), technique="none")

    assert result == ["result"]
    assert captured["queries"] == ["What was revenue?"]


def test_rewrite_technique_uses_rewritten_query(monkeypatch):
    captured = {}

    def fake_retrieve_and_rerank(queries, context, ranker, filters=None, original_query=None,
                                  candidate_k=None, final_k=None):
        captured["queries"] = queries
        captured["original_query"] = original_query
        return []

    monkeypatch.setattr(retriever_module, "retrieve_and_rerank", fake_retrieve_and_rerank)

    llm = FakeLLM(default="Explain artificial intelligence.")
    apply_technique_and_retrieve("Explain AI", context=None, ranker=None,
                                  llm_generate=llm, technique="rewrite")

    assert captured["queries"] == ["Explain artificial intelligence."]
    assert captured["original_query"] == "Explain AI"  # reranking still judges the literal question


def test_step_back_technique_includes_original_and_generalized(monkeypatch):
    captured = {}

    def fake_retrieve_and_rerank(queries, context, ranker, filters=None, original_query=None,
                                  candidate_k=None, final_k=None):
        captured["queries"] = queries
        return []

    monkeypatch.setattr(retriever_module, "retrieve_and_rerank", fake_retrieve_and_rerank)

    llm = FakeLLM(default="How has revenue trended over time?")
    apply_technique_and_retrieve("What was Q3 2023 revenue?", context=None, ranker=None,
                                  llm_generate=llm, technique="step-back")
    assert captured["queries"] == ["What was Q3 2023 revenue?", "How has revenue trended over time?"]


def test_multi_query_technique_passes_through_generated_list(monkeypatch):
    captured = {}

    def fake_retrieve_and_rerank(queries, context, ranker, filters=None, original_query=None,
                                  candidate_k=None, final_k=None):
        captured["queries"] = queries
        return []

    monkeypatch.setattr(retriever_module, "retrieve_and_rerank", fake_retrieve_and_rerank)

    llm = FakeLLM(default="variant one\nvariant two")
    apply_technique_and_retrieve("original question", context=None, ranker=None,
                                  llm_generate=llm, technique="multi-query")

    assert captured["queries"] == ["original question", "variant one", "variant two"]


def test_hyde_technique_bypasses_retrieve_and_rerank(monkeypatch):
    """HyDE has structurally different mechanics (embeds a hypothetical
    document, not a query string) -- it must NOT go through
    retrieve_and_rerank() at all."""
    def exploding_retrieve_and_rerank(*args, **kwargs):
        raise AssertionError("hyde must not call retrieve_and_rerank")

    monkeypatch.setattr(retriever_module, "retrieve_and_rerank", exploding_retrieve_and_rerank)

    import src.query_transformation.hyde as hyde_module
    import src.reranking.flashrank_reranker as reranker_module

    fake_hyde_results = [{"chunk_id": "a", "text": "t"}]
    monkeypatch.setattr(hyde_module, "hyde_search", lambda *a, **k: fake_hyde_results)
    monkeypatch.setattr(reranker_module, "rerank", lambda query, chunks, ranker, top_k: chunks)

    # embedding_model/qdrant_client are only ever passed through to the
    # (patched) hyde_search call, never dereferenced -- None stand-ins are
    # fine, but the attributes must exist since they're evaluated as call
    # arguments before dispatch happens.
    fake_context = SimpleNamespace(embedding_model=None, qdrant_client=None)
    result = apply_technique_and_retrieve("question", context=fake_context, ranker=None,
                                           llm_generate=FakeLLM(), technique="hyde")
    assert result == fake_hyde_results


def test_unknown_technique_raises():
    try:
        apply_technique_and_retrieve("q", context=None, ranker=None, llm_generate=FakeLLM(),
                                      technique="not-a-real-technique")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "not-a-real-technique" in str(e)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
