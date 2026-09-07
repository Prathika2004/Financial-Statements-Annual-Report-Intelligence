"""
Tests for the API layer's pure logic (filter-building). The routes
themselves need a live Qdrant + Ollama + loaded models, so they're
exercised via a real running server instead (see PIPELINE_TECHNICAL_NOTES.md).

Run with: pytest tests/test_api_routes.py -v
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.api.schemas import AskRequest
from src.api.routes import _build_filters, load_available_filings


def test_build_filters_uppercases_ticker():
    req = AskRequest(question="q", ticker="tsla")
    assert _build_filters(req) == {"ticker": "TSLA"}


def test_build_filters_combines_all_fields():
    req = AskRequest(question="q", ticker="tsla", fiscal_year=2023, chunk_type="table")
    assert _build_filters(req) == {"ticker": "TSLA", "fiscal_year": 2023, "chunk_type": "table"}


def test_build_filters_returns_none_when_nothing_set():
    req = AskRequest(question="q")
    assert _build_filters(req) is None


def _write_metadata(folder: Path, status="ok", **document_fields):
    folder.mkdir()
    document = {"status": status, "ticker": None, "fiscal_year": None, "company_name": None}
    document.update(document_fields)
    (folder / "metadata.json").write_text(json.dumps({"document": document}), encoding="utf-8")


def test_load_available_filings_skips_non_ok_status(tmp_path):
    _write_metadata(tmp_path / "amazon__amzn_10K_2021", status="ok", ticker="AMZN", fiscal_year=2020,
                     company_name="AMAZON.COM, INC.")
    _write_metadata(tmp_path / "broken__x_10K_2021", status="skipped_empty", ticker=None, fiscal_year=None)

    filings = load_available_filings(tmp_path)

    assert filings == [{"ticker": "AMZN", "fiscal_year": 2020, "company_name": "AMAZON.COM, INC."}]


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
