"""
SQLite-backed response cache: an identical (question, filters, technique)
request is served from a prior run instead of re-paying for retrieval +
reranking + generation + verification.

Save as: sec-rag-project/src/caching/response_cache.py

--- Why this is a real win for THIS project specifically, not a generic checkbox ---
Every cache miss costs several minutes of Qwen2.5 generation time on this
CPU (see llm_router.py's docstring), and this exact project has already
re-run the identical "cybersecurity_risk_factors_general" question multiple
times across different testing sessions, each time paying the full cost
again. A cache turns a repeated question into an instant response.

--- Cache key ---
A hash of (question, filters, technique) -- the exact inputs that change
apply_technique_and_retrieve()'s behavior. Anything that could change the
answer must change the key.
"""

import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, Optional

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.settings import RESPONSE_CACHE_DB, RESPONSE_CACHE_TTL_SECONDS


def _cache_key(question: str, filters: Optional[Dict], technique: str) -> str:
    payload = json.dumps({"question": question, "filters": filters or {}, "technique": technique}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _get_connection(db_path: Path = RESPONSE_CACHE_DB) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS response_cache (
            cache_key TEXT PRIMARY KEY,
            question TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at REAL NOT NULL
        )
    """)
    return conn


def get_cached(question: str, filters: Optional[Dict], technique: str,
               db_path: Path = RESPONSE_CACHE_DB) -> Optional[Dict]:
    """Returns the cached result dict, or None on a miss or an expired entry."""
    key = _cache_key(question, filters, technique)
    conn = _get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT result_json, created_at FROM response_cache WHERE cache_key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        result_json, created_at = row
        if time.time() - created_at > RESPONSE_CACHE_TTL_SECONDS:
            conn.execute("DELETE FROM response_cache WHERE cache_key = ?", (key,))
            conn.commit()
            return None
        return json.loads(result_json)
    finally:
        conn.close()


def set_cached(question: str, filters: Optional[Dict], technique: str, result: Dict,
               db_path: Path = RESPONSE_CACHE_DB) -> None:
    key = _cache_key(question, filters, technique)
    conn = _get_connection(db_path)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO response_cache (cache_key, question, result_json, created_at) "
            "VALUES (?, ?, ?, ?)",
            (key, question, json.dumps(result, ensure_ascii=False), time.time()),
        )
        conn.commit()
    finally:
        conn.close()


def clear_cache(db_path: Path = RESPONSE_CACHE_DB) -> int:
    """Deletes every cached entry. Returns the number of rows removed."""
    conn = _get_connection(db_path)
    try:
        cursor = conn.execute("DELETE FROM response_cache")
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()
