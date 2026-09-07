"""
Tests for Stage 7 retrieval logic: RRF fusion and BM25 tokenization/filtering.

Does not require a running Qdrant instance or the real embedding model --
dense_retriever.py's Qdrant-dependent functions are exercised manually via
scripts/run_retrieval.py instead, consistent with how load_embedding_model()
and the real Docling parse are tested elsewhere in this project.

Run with: pytest tests/test_retrieval.py -v
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.retrieval.hybrid_fusion import reciprocal_rank_fusion
from src.retrieval.bm25_retriever import tokenize, matches_filters, build_bm25_index, bm25_search


def make_chunk(chunk_id, text="filler text", **extra):
    return {"chunk_id": chunk_id, "text": text, **extra}


class TestReciprocalRankFusion:

    def test_chunk_found_by_both_retrievers_outranks_single_source_chunk(self):
        dense = [make_chunk("a", source="dense"), make_chunk("b", source="dense")]
        bm25 = [make_chunk("b", source="bm25"), make_chunk("c", source="bm25")]

        fused = reciprocal_rank_fusion([dense, bm25], top_k=10)

        assert fused[0]["chunk_id"] == "b"  # appears in both lists -- summed RRF score
        assert fused[0]["found_by"] == ["dense", "bm25"]

    def test_top_k_truncates_result_count(self):
        dense = [make_chunk(str(i)) for i in range(5)]
        fused = reciprocal_rank_fusion([dense], top_k=2)
        assert len(fused) == 2

    def test_empty_ranked_lists_return_empty_result(self):
        assert reciprocal_rank_fusion([[], []], top_k=10) == []

    def test_single_list_preserves_original_rank_order(self):
        dense = [make_chunk("a"), make_chunk("b"), make_chunk("c")]
        fused = reciprocal_rank_fusion([dense], top_k=10)
        assert [c["chunk_id"] for c in fused] == ["a", "b", "c"]

    def test_scales_to_more_than_two_lists(self):
        """
        Not exercised anywhere else yet, but this is exactly the shape
        Query Transformation (multi-query, several variants each producing
        a dense+BM25 pair) will call this with later.
        """
        lists = [
            [make_chunk("x"), make_chunk("y")],
            [make_chunk("y"), make_chunk("z")],
            [make_chunk("y"), make_chunk("x")],
        ]
        fused = reciprocal_rank_fusion(lists, top_k=10)
        assert fused[0]["chunk_id"] == "y"  # present in all three lists


class TestBM25Retriever:

    def test_tokenize_lowercases_and_strips_punctuation(self):
        assert tokenize("Risk Factors: Cybersecurity!") == ["risk", "factors", "cybersecurity"]

    def test_matches_filters_requires_all_fields_to_match(self):
        chunk = make_chunk("a", ticker="TSLA", fiscal_year=2023)
        assert matches_filters(chunk, {"ticker": "TSLA"}) is True
        assert matches_filters(chunk, {"ticker": "TSLA", "fiscal_year": 2023}) is True
        assert matches_filters(chunk, {"ticker": "TSLA", "fiscal_year": 2022}) is False

    def test_matches_filters_none_or_empty_matches_everything(self):
        chunk = make_chunk("a", ticker="TSLA")
        assert matches_filters(chunk, None) is True
        assert matches_filters(chunk, {}) is True

    def test_search_finds_relevant_chunk_and_applies_filter(self):
        # A realistic-ish spread of documents matters here: BM25's IDF term
        # goes negative for a query term appearing in most of the corpus, so
        # a 2-3 document corpus where the query terms are shared by every
        # "relevant" doc and absent from just one "irrelevant" doc produces
        # nonsense statistics (df > N/2). Enough unrelated filler documents
        # keeps the cybersecurity-related terms rare, like they'd actually
        # be against the real ~30k-chunk corpus.
        chunks = [
            make_chunk("a", text="revenue grew due to strong cloud demand", ticker="AMZN"),
            make_chunk("b", text="risk factors include cybersecurity threats", ticker="AMZN"),
            make_chunk("c", text="risk factors include cybersecurity threats", ticker="TSLA"),
            make_chunk("d", text="the company manufactures automobiles and batteries", ticker="TSLA"),
            make_chunk("e", text="quarterly dividends were paid to shareholders", ticker="AMZN"),
            make_chunk("f", text="supply chain disruptions affected manufacturing output", ticker="TSLA"),
        ]
        index = build_bm25_index(chunks)

        results = bm25_search("cybersecurity risk", chunks, index, top_k=2)
        assert {r["chunk_id"] for r in results} == {"b", "c"}

        filtered = bm25_search("cybersecurity risk", chunks, index, top_k=10, filters={"ticker": "TSLA"})
        assert filtered[0]["chunk_id"] == "c"

    def test_search_result_carries_score_and_source(self):
        chunks = [
            make_chunk("a", text="quarterly revenue increased sharply"),
            make_chunk("b", text="the board approved a new share buyback program"),
            make_chunk("c", text="supply chain costs rose due to inflation"),
        ]
        index = build_bm25_index(chunks)
        results = bm25_search("revenue", chunks, index, top_k=10)
        assert results[0]["chunk_id"] == "a"
        assert results[0]["source"] == "bm25"
        assert isinstance(results[0]["score"], float)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
