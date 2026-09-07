"""
Qdrant vector store: collection lifecycle + chunk upsert.

Save as: sec-rag-project/src/vectorstore/qdrant_client.py

--- Point ID scheme ---
Qdrant point IDs must be an unsigned integer or a UUID -- chunk_id strings
like "AMZN_2020_amzn_10K_2021_0001" (see src/chunking/section_chunker.py)
aren't valid IDs on their own. Each chunk_id is deterministically mapped to
a UUID5 (same input always produces the same UUID, from a fixed namespace
constant), so re-running the upload script is idempotent: re-upserting a
chunk overwrites the same point instead of creating a duplicate. The
original chunk_id string is kept in the payload for filtering/debugging.

--- Distance metric ---
Cosine, matching how Stage 5 produced the embeddings (L2-normalized vectors
-- see src/embeddings/embedder.py's module docstring). Qdrant's Cosine
distance normalizes internally regardless, but keeping this explicit and
consistent avoids a silent mismatch if a future embedding model isn't
pre-normalized.
"""

import logging
import sys
import uuid
from pathlib import Path
from typing import List

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.settings import QDRANT_URL, QDRANT_COLLECTION_NAME, QDRANT_UPSERT_BATCH_SIZE, EMBEDDING_DIM

logger = logging.getLogger(__name__)

# Fixed and arbitrary -- only needs to be stable across runs so the same
# chunk_id always maps to the same point id.
_CHUNK_ID_NAMESPACE = uuid.UUID("6f1c9c1a-1a4e-4f2e-9c9a-2a6b6f1c9c1a")


def chunk_id_to_point_id(chunk_id: str) -> str:
    """Maps a chunk_id string to a deterministic UUID5 point id."""
    return str(uuid.uuid5(_CHUNK_ID_NAMESPACE, chunk_id))


def get_client(url: str = QDRANT_URL):
    from qdrant_client import QdrantClient
    return QdrantClient(url=url)


def ensure_collection(client, collection_name: str = QDRANT_COLLECTION_NAME, vector_size: int = EMBEDDING_DIM):
    """
    Creates the collection if it doesn't already exist. Never recreates an
    existing collection -- that would silently discard previously uploaded
    vectors. Call drop_collection() explicitly first for a fresh start.
    """
    from qdrant_client.models import Distance, VectorParams

    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        logger.info("Collection '%s' already exists -- leaving it as-is.", collection_name)
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    logger.info("Created collection '%s' (dim=%d, distance=Cosine).", collection_name, vector_size)


def drop_collection(client, collection_name: str = QDRANT_COLLECTION_NAME):
    client.delete_collection(collection_name)
    logger.info("Dropped collection '%s'.", collection_name)


def upsert_chunks(client, chunks: List[dict], vectors: np.ndarray,
                   collection_name: str = QDRANT_COLLECTION_NAME,
                   batch_size: int = QDRANT_UPSERT_BATCH_SIZE) -> int:
    """
    Upserts one filing's chunks + their vectors into Qdrant. `vectors` must
    be row-aligned with `chunks` (row i's vector belongs to chunks[i]) --
    the same contract embeddings.npy/embeddings_meta.json establish for
    Stage 5. The full chunk dict (including its text) is stored as the
    point's payload so retrieval can return the actual passage, not just a
    vector id.
    """
    from qdrant_client.models import PointStruct

    if len(chunks) != len(vectors):
        raise ValueError(f"chunks ({len(chunks)}) and vectors ({len(vectors)}) count mismatch")

    points = [
        PointStruct(
            id=chunk_id_to_point_id(chunk["chunk_id"]),
            vector=vectors[i].tolist(),
            payload=chunk,
        )
        for i, chunk in enumerate(chunks)
    ]

    for start in range(0, len(points), batch_size):
        client.upsert(collection_name=collection_name, points=points[start:start + batch_size])

    return len(points)


def delete_by_source_filename(client, source_filename: str, collection_name: str = QDRANT_COLLECTION_NAME):
    """
    Deletes every point belonging to one source PDF, identified by the
    source_filename payload field every chunk carries.

    Why this needs to exist at all: chunk_id is a per-document sequential
    counter (see section_chunker.py), so if a document's chunk *count*
    changes between two chunking runs -- e.g. an oversized table's caption
    text changes length, which shifts how many row-split parts it needs
    (see table_chunker.py's table_budget calculation) -- every chunk_id
    after that point in the document shifts too. Upserting the new chunks
    only adds/overwrites points at the NEW chunk_ids; it never removes a
    point sitting at an OLD chunk_id that no longer exists in the new
    output. Without deleting by source_filename first, a re-chunked
    document leaves orphaned stale points behind silently.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue, FilterSelector

    client.delete(
        collection_name=collection_name,
        points_selector=FilterSelector(
            filter=Filter(must=[FieldCondition(key="source_filename", match=MatchValue(value=source_filename))])
        ),
    )


def count_points(client, collection_name: str = QDRANT_COLLECTION_NAME) -> int:
    return client.count(collection_name=collection_name, exact=True).count
