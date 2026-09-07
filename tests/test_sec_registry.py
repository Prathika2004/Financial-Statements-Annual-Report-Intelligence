"""
Tests for sec_registry.py's filing lookup.

Run with: pytest tests/test_sec_registry.py -v
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.ingestion.sec_registry import lookup_filing


JNJ_REGISTRY = {
    "JNJ": {
        "company_name": "Johnson & Johnson",
        "cik": "0000200406",
        "filings": [
            {"filing_date": "2024-02-16", "reporting_period_end": "2023-12-31", "accession_number": "a1"},
            {"filing_date": "2023-02-16", "reporting_period_end": "2023-01-01", "accession_number": "a2"},
            {"filing_date": "2022-02-17", "reporting_period_end": "2022-01-02", "accession_number": "a3"},
        ],
    }
}


def test_exact_reporting_period_end_match_disambiguates_same_fiscal_year():
    """
    J&J's 52/53-week fiscal calendar produces two filings that both derive to
    fiscal_year 2022 under a naive year-of-date rule (2023-01-01 -> 2022 after
    the early-January shift, but 2022-01-02 also starts with "2022"). Matching
    on the exact in-document reporting_period_end must pick the correct one
    instead of grabbing whichever registry entry happens to share that year.
    """
    result = lookup_filing("JNJ", 2022, JNJ_REGISTRY, reporting_period_end="2023-01-01")
    assert result["filing_date"] == "2023-02-16"
    assert result["reporting_period_end"] == "2023-01-01"


def test_falls_back_to_fiscal_year_prefix_when_no_reporting_period_end():
    result = lookup_filing("JNJ", 2023, JNJ_REGISTRY, reporting_period_end=None)
    assert result["filing_date"] == "2024-02-16"
    assert result["reporting_period_end"] == "2023-12-31"


def test_falls_back_to_fiscal_year_prefix_when_exact_date_not_found():
    result = lookup_filing("JNJ", 2023, JNJ_REGISTRY, reporting_period_end="1999-09-09")
    assert result["filing_date"] == "2024-02-16"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
