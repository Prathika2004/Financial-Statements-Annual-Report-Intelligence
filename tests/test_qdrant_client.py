"""
Tests for Stage 6 Qdrant point-id logic. Does not require a running Qdrant
instance -- chunk_id_to_point_id is pure string/UUID logic.

Run with: pytest tests/test_qdrant_client.py -v
"""

import sys
import uuid
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.vectorstore.qdrant_client import chunk_id_to_point_id


def test_same_chunk_id_always_maps_to_same_point_id():
    """Re-running the upload script must overwrite the same point, not
    create a duplicate -- this is what makes re-uploads idempotent."""
    a = chunk_id_to_point_id("AMZN_2020_amzn_10K_2021_0001")
    b = chunk_id_to_point_id("AMZN_2020_amzn_10K_2021_0001")
    assert a == b


def test_different_chunk_ids_map_to_different_point_ids():
    a = chunk_id_to_point_id("AMZN_2020_amzn_10K_2021_0001")
    b = chunk_id_to_point_id("AMZN_2020_amzn_10K_2021_0002")
    assert a != b


def test_point_id_is_a_valid_uuid():
    point_id = chunk_id_to_point_id("TSLA_2023_tesla_10K_2024_0042")
    uuid.UUID(point_id)  # raises ValueError if not a valid UUID string


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
