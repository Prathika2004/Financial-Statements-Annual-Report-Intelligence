"""
Tests for the SQLite-backed response cache.

Run with: pytest tests/test_response_cache.py -v
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.caching.response_cache import get_cached, set_cached, clear_cache, _cache_key


def test_miss_on_empty_cache(tmp_path):
    db = tmp_path / "cache.db"
    assert get_cached("What was revenue?", None, "none", db_path=db) is None


def test_set_then_get_round_trips(tmp_path):
    db = tmp_path / "cache.db"
    result = {"answer": "Revenue was $1B.", "sources": []}
    set_cached("What was revenue?", {"ticker": "TSLA"}, "none", result, db_path=db)

    cached = get_cached("What was revenue?", {"ticker": "TSLA"}, "none", db_path=db)
    assert cached == result


def test_different_filters_are_different_cache_entries(tmp_path):
    db = tmp_path / "cache.db"
    set_cached("q", {"ticker": "TSLA"}, "none", {"answer": "tesla answer"}, db_path=db)
    set_cached("q", {"ticker": "AAPL"}, "none", {"answer": "apple answer"}, db_path=db)

    assert get_cached("q", {"ticker": "TSLA"}, "none", db_path=db)["answer"] == "tesla answer"
    assert get_cached("q", {"ticker": "AAPL"}, "none", db_path=db)["answer"] == "apple answer"


def test_different_technique_is_a_different_cache_entry(tmp_path):
    db = tmp_path / "cache.db"
    set_cached("q", None, "none", {"answer": "plain"}, db_path=db)
    assert get_cached("q", None, "hyde", db_path=db) is None


def test_expired_entry_is_a_miss(tmp_path, monkeypatch):
    db = tmp_path / "cache.db"
    set_cached("q", None, "none", {"answer": "old"}, db_path=db)

    import src.caching.response_cache as cache_module
    monkeypatch.setattr(cache_module, "RESPONSE_CACHE_TTL_SECONDS", 0)
    time.sleep(0.01)
    assert get_cached("q", None, "none", db_path=db) is None


def test_clear_cache_removes_everything(tmp_path):
    db = tmp_path / "cache.db"
    set_cached("q1", None, "none", {"answer": "a"}, db_path=db)
    set_cached("q2", None, "none", {"answer": "b"}, db_path=db)

    removed = clear_cache(db_path=db)
    assert removed == 2
    assert get_cached("q1", None, "none", db_path=db) is None


def test_cache_key_is_order_independent_for_filter_dict():
    key_a = _cache_key("q", {"ticker": "TSLA", "fiscal_year": 2023}, "none")
    key_b = _cache_key("q", {"fiscal_year": 2023, "ticker": "TSLA"}, "none")
    assert key_a == key_b


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
