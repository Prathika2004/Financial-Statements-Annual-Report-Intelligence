"""
Tests for Stage 5 embedding logic.

Uses a fake model instead of loading the real ~130MB BGE model, so this
suite stays fast and offline like the rest of tests/ -- load_embedding_model()
itself (backend selection, disk caching) is exercised by running
src/embeddings/embedder.py directly as a manual smoke test instead.

Run with: pytest tests/test_embedder.py -v
"""

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import EMBEDDING_DIM
from src.embeddings.embedder import embed_texts, embed_query, BGE_QUERY_INSTRUCTION


class FakeModel:
    """Records what it was called with and returns a deterministic, already
    L2-normalized vector per input text (unit vector along a rotating axis)."""

    def __init__(self):
        self.last_call = None

    def encode(self, texts, batch_size, normalize_embeddings, show_progress_bar, convert_to_numpy):
        self.last_call = {"texts": texts, "batch_size": batch_size,
                           "normalize_embeddings": normalize_embeddings}
        vectors = np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)
        for i in range(len(texts)):
            vectors[i, i % EMBEDDING_DIM] = 1.0
        return vectors


def test_passage_text_is_not_prefixed():
    model = FakeModel()
    embed_texts(["Revenue grew 12%."], model, is_query=False)
    assert model.last_call["texts"] == ["Revenue grew 12%."]


def test_query_text_gets_bge_instruction_prefix():
    model = FakeModel()
    embed_texts(["What was revenue growth?"], model, is_query=True)
    assert model.last_call["texts"] == [BGE_QUERY_INSTRUCTION + "What was revenue growth?"]


def test_embed_query_returns_single_vector_not_batch():
    model = FakeModel()
    vector = embed_query("What was revenue growth?", model)
    assert vector.shape == (EMBEDDING_DIM,)


def test_empty_input_returns_empty_array_without_calling_model():
    model = FakeModel()
    vectors = embed_texts([], model)
    assert vectors.shape == (0, EMBEDDING_DIM)
    assert model.last_call is None  # never invoked -- nothing to encode


def test_output_is_float32_and_requests_normalization():
    model = FakeModel()
    vectors = embed_texts(["some chunk text"], model)
    assert vectors.dtype == np.float32
    assert model.last_call["normalize_embeddings"] is True


def test_dimension_mismatch_raises_instead_of_silently_passing_through():
    class WrongDimModel:
        def encode(self, texts, **kwargs):
            return np.zeros((len(texts), EMBEDDING_DIM + 1), dtype=np.float32)

    try:
        embed_texts(["text"], WrongDimModel())
        assert False, "expected ValueError for dimension mismatch"
    except ValueError:
        pass


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
