"""
Tests for Stage 1 metadata/section extraction.

Save as: sec-rag-project/tests/test_metadata_extraction.py

Run with: pytest tests/test_metadata_extraction.py -v

These deliberately construct synthetic filing text rather than requiring a
real PDF/Docling run, so they run fast, offline, and catch regressions in
the regex/scoring logic directly.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.ingestion.metadata_extractor import (
    extract_sections,
    extract_ticker_from_text,
    extract_company_name_from_text,
    extract_reporting_period_end,
    extract_identity_from_text,
    fiscal_year_from_reporting_period_end,
)


def make_synthetic_filing(item_numbers, markdown_decorated=False, body_content_len=200):
    """
    Builds a minimal synthetic 10-K: a Table of Contents block (Title Case,
    trailing page numbers) followed by body sections (ALL CAPS headings,
    each with `body_content_len` chars of filler text).
    """
    toc_lines = [f"Item {n}. Title For Item {n} {10 + i}" for i, n in enumerate(item_numbers)]
    toc_block = "\n".join(toc_lines)

    body_blocks = []
    for n in item_numbers:
        heading = f"ITEM {n}. TITLE FOR ITEM {n}"
        if markdown_decorated:
            heading = "## " + heading
        filler = ("Lorem ipsum content. " * 20)[:body_content_len]
        body_blocks.append(f"{heading}\n{filler}")

    return toc_block + "\n\n" + "\n\n".join(body_blocks)


STANDARD_ITEMS = ["1", "1A", "1B", "2", "3", "4", "5", "6", "7", "7A", "8",
                   "9", "9A", "9B", "10", "11", "12", "13", "14", "15", "16"]


class TestSectionDetection:

    def test_plain_text_all_sections_found(self):
        """Baseline: no markdown decoration, should find every section."""
        text = make_synthetic_filing(STANDARD_ITEMS, markdown_decorated=False)
        sections = extract_sections(text)
        found_items = {s.item.replace("Item ", "") for s in sections}
        assert found_items == set(STANDARD_ITEMS), f"Missing: {set(STANDARD_ITEMS) - found_items}"

    def test_markdown_decorated_sections_still_found(self):
        """
        The actual bug being fixed: Docling wraps headings in '## '.
        This must not cause num_sections to drop to 0.
        """
        text = make_synthetic_filing(STANDARD_ITEMS, markdown_decorated=True)
        sections = extract_sections(text)
        found_items = {s.item.replace("Item ", "") for s in sections}
        assert found_items == set(STANDARD_ITEMS), f"Missing: {set(STANDARD_ITEMS) - found_items}"
        assert len(sections) == len(STANDARD_ITEMS)

    def test_toc_entries_excluded_from_sections(self):
        """TOC entries (Title Case + trailing page number) must not be
        mistaken for real section boundaries."""
        text = make_synthetic_filing(["1", "1A"], markdown_decorated=False)
        sections = extract_sections(text)
        # Should find exactly 2 real sections, not 4 (2 TOC + 2 body)
        assert len(sections) == 2

    def test_short_real_sections_not_misclassified_as_toc(self):
        """
        Regression test for the Items 10-15 bug: short "incorporated by
        reference" sections sitting close together must still be detected
        as real body sections, not accidentally clustered with the TOC.
        """
        text = make_synthetic_filing(["10", "11", "12", "13", "14", "15"],
                                      markdown_decorated=False, body_content_len=15)
        sections = extract_sections(text)
        found_items = {s.item.replace("Item ", "") for s in sections}
        assert found_items == {"10", "11", "12", "13", "14", "15"}

    def test_empty_text_returns_empty_list_not_crash(self):
        assert extract_sections("") == []

    def test_no_item_headings_returns_empty_list(self):
        assert extract_sections("Just some prose with no structure at all.") == []

    def test_markdown_headings_are_used_for_nontraditional_filings(self):
        text = (
            "## Fundamentals of Our Business\n"
            + ("Body text for the first nontraditional section. " * 6)
            + "\n\n"
            "## Management's Discussion and Analysis\n"
            + ("Body text for the second nontraditional section. " * 6)
        )
        sections = extract_sections(text)
        assert [section.item for section in sections] == ["Heading 1", "Heading 2"]
        assert [section.title for section in sections] == [
            "Fundamentals of Our Business",
            "Management's Discussion and Analysis",
        ]

    def test_sections_have_correct_char_offsets(self):
        text = make_synthetic_filing(["1", "1A"], markdown_decorated=False)
        sections = extract_sections(text)
        for s in sections:
            # the text between start_position and end_position should equal
            # what's stored, and should actually contain the item's heading
            substring = text[s.start_position:s.end_position]
            assert substring.strip() == s.text.strip()
            assert s.item.split(" ")[1] in substring[:60]  # heading appears early in slice

    def test_wrapped_toc_line_edge_case(self):
        """
        A TOC entry whose title is too long and wraps to a second line
        (losing its trailing page number on that line) must still be
        correctly classified as TOC, not kept as a duplicate/false section.
        This mirrors the real Item 5 case found in a live Tesla 10-K.
        """
        text = (
            "Item 5. This Is A Very Long Title That Wraps\n"
            "Onto A Second Line Before The Page Number 42\n"
            "\n"
            "ITEM 5. THIS IS A VERY LONG TITLE THAT WRAPS\n"
            "Real body content goes here for quite a while so this section has bulk. " * 5
        )
        sections = extract_sections(text)
        matching = [s for s in sections if s.item == "Item 5"]
        assert len(matching) == 1, f"Expected exactly one Item 5 section, got {len(matching)}"
        # the kept one should be the real (larger) body section, not the wrapped TOC fragment
        assert matching[0].text_char_count > 100


class TestIdentityExtraction:

    def test_ticker_extraction_plain_line(self):
        text = "Some preamble text.\nCommon Stock TSLA The Nasdaq Global Select Market\nMore text."
        assert extract_ticker_from_text(text) == "TSLA"

    def test_ticker_extraction_markdown_table_row(self):
        text = (
            "| Title of each class | Trading Symbol(s) | Name of exchange |\n"
            "| --- | --- | --- |\n"
            "| Common Stock | AMZN | Nasdaq |\n"
        )
        assert extract_ticker_from_text(text) == "AMZN"

    def test_ticker_extraction_returns_none_when_absent(self):
        assert extract_ticker_from_text("No ticker info in this text at all.") is None

    def test_company_name_extraction(self):
        text = (
            "UNITED STATES SECURITIES AND EXCHANGE COMMISSION\n"
            "Tesla, Inc.\n"
            "(Exact name of registrant as specified in its charter)\n"
        )
        assert extract_company_name_from_text(text) == "Tesla, Inc."

    def test_company_name_strips_leaked_markdown_heading_marker(self):
        """
        Docling renders the company name as a markdown heading (e.g.
        "## AMAZON.COM, INC."). The leading '#'s must not leak into
        company_name -- they'd pollute chunk metadata/embeddings downstream.
        """
        text = (
            "UNITED STATES SECURITIES AND EXCHANGE COMMISSION\n"
            "## AMAZON.COM, INC.\n"
            "(Exact name of registrant as specified in its charter)\n"
        )
        assert extract_company_name_from_text(text) == "AMAZON.COM, INC."

    def test_company_name_rejects_docling_image_placeholder(self):
        """
        Some filers (Apple, Intel, Visa confirmed) render their cover-page
        company name as a logo image rather than text. Docling represents
        that as an HTML comment placeholder, which must not be accepted as
        the company name -- it should fall through to None (triggering the
        SEC registry fallback in run_ingestion.py) rather than return
        "<!-- image -->" verbatim.
        """
        text = (
            "001-36743\n"
            "\n"
            "<!-- image -->\n"
            "\n"
            "(Exact name of registrant as specified in its charter)\n"
        )
        assert extract_company_name_from_text(text) is None

    def test_reporting_period_end_parses_to_iso_date(self):
        text = "For the fiscal year ended December 31, 2023\nmore text"
        assert extract_reporting_period_end(text) == "2023-12-31"

    def test_reporting_period_end_none_when_absent(self):
        assert extract_reporting_period_end("No fiscal year statement here.") is None

    def test_fiscal_year_derived_from_reporting_period_not_filename(self):
        """
        Explicit requirement: fiscal_year must come from the document's
        reporting period, never from a filename-based guess.
        """
        text = (
            "Tesla, Inc.\n"
            "(Exact name of registrant as specified in its charter)\n"
            "For the fiscal year ended December 31, 2021\n"
            "Common Stock TSLA The Nasdaq Global Select Market\n"
        )
        identity = extract_identity_from_text(text)
        assert identity.fiscal_year == 2021  # NOT e.g. 2024 even if filename said "2024"

    def test_fiscal_year_shifts_back_for_early_january_period_end(self):
        """
        A 52/53-week fiscal calendar landing in the first few days of January
        (e.g. J&J's) is labeled by the prior calendar year -- confirmed
        against J&J's own filing text, which calls its period ended
        2023-01-01 "fiscal year 2022".
        """
        assert fiscal_year_from_reporting_period_end("2023-01-01") == 2022
        assert fiscal_year_from_reporting_period_end("2021-01-03") == 2020

    def test_fiscal_year_not_shifted_for_late_january_period_end(self):
        """
        NVIDIA's fiscal year-end lands in the last few days of January and is
        labeled by that same calendar year (fiscal 2024 ended 2024-01-28) --
        the early-January shift must not misfire on this case.
        """
        assert fiscal_year_from_reporting_period_end("2024-01-28") == 2024

    def test_fiscal_year_not_shifted_outside_january(self):
        assert fiscal_year_from_reporting_period_end("2023-12-31") == 2023


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
