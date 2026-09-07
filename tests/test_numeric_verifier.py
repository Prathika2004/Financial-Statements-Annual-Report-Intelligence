"""
Tests for Stage 9's numeric grounding check.

Run with: pytest tests/test_numeric_verifier.py -v
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.generation.numeric_verifier import extract_numbers, verify_numbers


def test_extract_numbers_handles_currency_percentage_and_scale_words():
    text = "Revenue was $1,234.5 million, up 12.5% from a $987 million base."
    numbers = extract_numbers(text)
    assert "$1234.5million" in numbers
    assert "12.5%" in numbers
    assert "$987million" in numbers


def test_extract_numbers_ignores_short_bare_integers():
    text = "See Item 7 for 5 examples."
    assert extract_numbers(text) == []


def test_verify_numbers_flags_figure_absent_from_sources():
    answer = "Revenue grew to $500 million in fiscal 2023 [Source 1]."
    sources = [{"text": "Revenue for fiscal 2023 was $450 million, up from $400 million."}]

    result = verify_numbers(answer, sources)

    assert "$500million" in result["checked"]
    assert "$500million" in result["unverified"]
    assert "$450million" not in result["unverified"]


def test_verify_numbers_passes_figure_present_in_sources():
    answer = "Revenue was $450 million [Source 1]."
    sources = [{"text": "Revenue for fiscal 2023 was $450 million."}]

    result = verify_numbers(answer, sources)

    assert result["unverified"] == []
    assert "$450million" in result["checked"]


def test_verify_numbers_matches_regardless_of_comma_formatting():
    answer = "The company had 1,234,567 shares outstanding."
    sources = [{"text": "Shares outstanding: 1234567."}]

    result = verify_numbers(answer, sources)

    assert result["unverified"] == []


def test_verify_numbers_with_no_numeric_claims_returns_empty_lists():
    answer = "The filing discusses risk factors qualitatively [Source 1]."
    sources = [{"text": "We face various risks including cybersecurity threats."}]

    result = verify_numbers(answer, sources)

    assert result == {"checked": [], "unverified": []}


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
