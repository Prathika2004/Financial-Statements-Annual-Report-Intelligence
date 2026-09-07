"""
Turns chunk text into dense vectors using BAAI/bge-small-en-v1.5.

Save as: sec-rag-project/src/embeddings/embedder.py

--- Backend selection (why this isn't just SentenceTransformer(model_name)) ---
This machine has no GPU. Plain PyTorch CPU inference works but leaves speed
on the table: sentence-transformers (>=3.2) can load the same model weights
through an OpenVINO or ONNX Runtime backend instead, which run meaningfully
faster for encoder-only models like this one. OpenVINO is Intel's own
inference runtime and is tried first since this project runs on an Intel
CPU; ONNX Runtime is the second choice; plain PyTorch is the last-resort
fallback if neither optional backend package is installed, so the pipeline
is never blocked on an optional dependency (optimum-intel / optimum).

--- BGE usage convention (not obvious from the API, easy to get wrong) ---
- Passages/chunks are embedded as-is.
- Queries get an instruction prefix prepended, because BGE was trained
  asymmetrically -- queries and passages are different distributions at
  training time. Skipping this for queries measurably hurts retrieval
  quality. Never add it to chunk/passage text.
- Vectors are L2-normalized at encode time so cosine similarity (and a dot
  product, which is what Qdrant's Cosine distance computes under the hood)
  behaves correctly.
"""

import logging
import sys
from pathlib import Path
from typing import List

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.settings import PROJECT_ROOT, EMBEDDING_MODEL_NAME, EMBEDDING_DIM, EMBEDDING_BATCH_SIZE

logger = logging.getLogger(__name__)

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Tried in order; None means sentence-transformers' own default (PyTorch).
# OpenVINO is preferred on this project's Intel CPU; ONNX Runtime is the
# portable second choice; PyTorch always works so it's the final fallback.
_BACKEND_PREFERENCE = ("openvino", "onnx", None)

# BAAI/bge-small-en-v1.5 on the HF Hub ships PyTorch weights only -- an
# OpenVINO/ONNX backend has to convert them first. That conversion takes
# real wall-clock time and, without a save destination, sentence-transformers
# repeats it on every single process start (confirmed empirically: the
# "Exporting the model to..." warning fires on every load, not just the
# first). Caching the converted model to disk once turns every later load
# into a fast local read instead of a re-export.
_MODEL_CACHE_DIR = PROJECT_ROOT / "models"


def load_embedding_model(model_name: str = EMBEDDING_MODEL_NAME):
    """
    Loads the embedding model through the fastest available CPU backend.
    Never raises just because an optional backend package (optimum-intel,
    optimum) isn't installed -- only if the model can't be loaded at all.
    """
    from sentence_transformers import SentenceTransformer

    last_error = None
    for backend in _BACKEND_PREFERENCE:
        cache_path = _MODEL_CACHE_DIR / f"{model_name.replace('/', '__')}-{backend}"
        try:
            if backend is None:
                model = SentenceTransformer(model_name)
            elif cache_path.exists():
                model = SentenceTransformer(str(cache_path), backend=backend)
                logger.info("Loaded '%s' from local %s cache (no re-export needed).", model_name, backend)
                return model
            else:
                model = SentenceTransformer(model_name, backend=backend)
                model.save(str(cache_path))
                logger.info("Exported '%s' to %s and cached it at %s.", model_name, backend, cache_path)

            logger.info("Loaded '%s' using backend=%s", model_name, backend or "torch (default)")
            return model
        except Exception as e:
            last_error = e
            logger.info("Backend '%s' unavailable for '%s' (%s) -- trying next.", backend, model_name, e)

    raise RuntimeError(f"Could not load embedding model '{model_name}' with any backend") from last_error


def embed_texts(texts: List[str], model, batch_size: int = EMBEDDING_BATCH_SIZE,
                 is_query: bool = False) -> np.ndarray:
    """
    Encodes a list of texts into L2-normalized float32 vectors, shape
    (len(texts), EMBEDDING_DIM).

    is_query=True prepends BGE's required query instruction prefix -- pass
    this for a user's search query, never for chunk/passage text.
    """
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

    if is_query:
        texts = [BGE_QUERY_INSTRUCTION + t for t in texts]

    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    vectors = np.asarray(vectors, dtype=np.float32)

    if vectors.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"Model produced {vectors.shape[1]}-dim vectors, expected {EMBEDDING_DIM} "
            f"(EMBEDDING_DIM in config/settings.py is out of sync with EMBEDDING_MODEL_NAME)."
        )
    return vectors


def embed_query(text: str, model) -> np.ndarray:
    """Convenience wrapper for embedding a single search query at retrieval time."""
    return embed_texts([text], model, batch_size=1, is_query=True)[0]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    model = load_embedding_model()
    sample_chunks = ["Revenue increased 12% year over year.", "Item 1A. Risk Factors."]
    vectors = embed_texts(sample_chunks, model)
    query_vector = embed_query("What was the revenue growth?", model)

    print(f"Chunk vectors: {vectors.shape}, dtype={vectors.dtype}")
    print(f"Query vector: {query_vector.shape}")
    print(f"Chunk 0 norm: {np.linalg.norm(vectors[0]):.4f} (should be ~1.0)")
