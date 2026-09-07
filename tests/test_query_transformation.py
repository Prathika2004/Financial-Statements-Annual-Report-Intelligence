"""
Tests for Stage "Query Transformation": rewriting, multi-query,
decomposition, and HyDE. All exercised against FakeLLM (see
llm_interface.py) -- no real LLM provider exists yet (Stage 9), and every
function in this package is deliberately written to accept any
`llm_generate` callable for exactly this reason.

Run with: pytest tests/test_query_transformation.py -v
"""

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import EMBEDDING_DIM
from src.query_transformation.llm_interface import FakeLLM
from src.query_transformation.query_rewriting import rewrite_query, step_back_query
from src.query_transformation.multi_query import generate_multi_queries
from src.query_transformation.decomposition import decompose_query
from src.query_transformation.hyde import generate_hypothetical_document, hyde_search


class TestQueryRewriting:

    def test_rewrite_query_returns_llm_output(self):
        llm = FakeLLM(default="Explain Artificial Intelligence")
        assert rewrite_query("Explain AI", llm) == "Explain Artificial Intelligence"
        assert "Explain AI" in llm.calls[0]  # the prompt included the original question

    def test_rewrite_query_falls_back_to_original_on_empty_output(self):
        llm = FakeLLM(default="   ")
        assert rewrite_query("Explain AI", llm) == "Explain AI"

    def test_step_back_query_returns_llm_output(self):
        llm = FakeLLM(default="How has revenue trended over time?")
        result = step_back_query("What was Q3 2023 revenue?", llm)
        assert result == "How has revenue trended over time?"


class TestMultiQuery:

    def test_parses_newline_separated_variants_and_includes_original_first(self):
        llm = FakeLLM(default="What is the firm's revenue growth?\nHow much did sales increase?")
        result = generate_multi_queries("What was revenue growth?", llm, num_variants=2)
        assert result[0] == "What was revenue growth?"
        assert len(result) == 3

    def test_strips_leading_numbering_the_llm_added_anyway(self):
        llm = FakeLLM(default="1. First variant\n2) Second variant\n- Third variant")
        result = generate_multi_queries("original", llm, num_variants=3)
        assert result == ["original", "First variant", "Second variant", "Third variant"]

    def test_deduplicates_a_variant_identical_to_the_original(self):
        llm = FakeLLM(default="original\nsomething new")
        result = generate_multi_queries("Original", llm, num_variants=2)
        assert result == ["Original", "something new"]

    def test_falls_back_to_original_only_when_output_unparseable(self):
        llm = FakeLLM(default="   \n   ")
        assert generate_multi_queries("original", llm) == ["original"]


class TestDecomposition:

    def test_splits_compound_question_into_sub_questions(self):
        llm = FakeLLM(default="How did revenue change in 2023?\nHow did headcount change in 2023?")
        result = decompose_query("How did revenue and headcount both change in 2023?", llm)
        assert result == ["How did revenue change in 2023?", "How did headcount change in 2023?"]

    def test_passes_through_simple_question_unchanged(self):
        llm = FakeLLM(default="What was 2023 revenue?")
        result = decompose_query("What was 2023 revenue?", llm)
        assert result == ["What was 2023 revenue?"]

    def test_falls_back_to_original_when_output_unparseable(self):
        llm = FakeLLM(default="")
        assert decompose_query("original question", llm) == ["original question"]


class TestHyDE:

    def test_generate_hypothetical_document_returns_llm_output(self):
        llm = FakeLLM(default="Revenue for fiscal 2023 increased 12% year over year...")
        result = generate_hypothetical_document("What was revenue growth?", llm)
        assert result == "Revenue for fiscal 2023 increased 12% year over year..."

    def test_generate_hypothetical_document_falls_back_to_question(self):
        llm = FakeLLM(default="")
        assert generate_hypothetical_document("original question", llm) == "original question"

    def test_hyde_search_embeds_hypothetical_doc_as_passage_not_query(self):
        """The whole point of HyDE: the hypothetical answer must be embedded
        WITHOUT the BGE query-instruction prefix (is_query=False)."""
        llm = FakeLLM(default="A hypothetical filing passage answering the question.")

        class FakeEmbeddingModel:
            def encode(self, texts, batch_size, normalize_embeddings, show_progress_bar, convert_to_numpy):
                assert texts == ["A hypothetical filing passage answering the question."]
                return np.ones((1, EMBEDDING_DIM), dtype=np.float32)

        class FakePoint:
            score = 0.9
            payload = {"chunk_id": "a", "text": "matched chunk"}

        class FakeResponse:
            points = [FakePoint()]

        class FakeQdrantClient:
            def query_points(self, collection_name, query, query_filter, limit):
                self.received_query = query
                return FakeResponse()

        client = FakeQdrantClient()
        results = hyde_search("What was revenue growth?", llm, FakeEmbeddingModel(), client, top_k=5)

        assert results[0]["chunk_id"] == "a"
        assert results[0]["source"] == "dense"
        assert len(client.received_query) == EMBEDDING_DIM


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
