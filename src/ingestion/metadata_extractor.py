"""
Extracts structural + identity metadata from a parsed 10-K filing.

Save as: sec-rag-project/src/ingestion/metadata_extractor.py

--- Why the previous version failed (root cause) ---
The old regex required an exact `^ITEM` anchor at the very start of a line.
Docling detects section headings via its layout model and renders them with
markdown heading syntax (e.g. "## ITEM 1A. RISK FACTORS"), which breaks that
anchor -- so on Docling-parsed text, num_sections came back 0 even though
the headings were clearly present in the document. This version strips
markdown decoration before matching and was validated against real filing
text (including a synthetic Docling-style transformation) before being
shipped -- see tests/test_metadata_extraction.py.

--- How TOC entries are told apart from real section headings ---
Two independent signals, combined as a score (more robust than either alone):
  1. TOC entries end with a trailing page number ("Risk Factors 14");
     real body headings don't.
  2. Real body headings are typed in ALL CAPS in the source PDF
     ("ITEM 1A. RISK FACTORS"); TOC entries are Title Case ("Item 1A. Risk
     Factors"). This holds because Docling/pdfplumber both extract literal
     PDF text content, not a re-typeset version.
When an Item number matches multiple times (e.g. a TOC line that wraps
across two lines, splitting off its page number), the occurrence with the
most negative score (strongest "real body heading" signal) wins.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

ITEM_LINE_PATTERN = re.compile(
    r"^(ITEM)\s+(\d{1,2}[A-C]?)\b[\.\)\-\u2014:]*\s*(.*?)\s*$",
    re.IGNORECASE,
)
MARKDOWN_HEADING_PATTERN = re.compile(r"^#{1,6}\s+(.+?)\s*$")
TRAILING_PAGE_NUM_PATTERN = re.compile(r"[\s\.]{1,6}\d{1,4}\s*$")
MIN_MARKDOWN_SECTION_CHARS = 200

# How far into the document to search for cover-page identity fields. Some
# filers (e.g. NVIDIA) have enough front matter that the "fiscal year ended"
# declaration lands past 5000 chars -- a too-small window silently produced
# null fiscal_year/reporting_period_end for those filings even though the
# text was present and well-formed.
COVER_PAGE_SEARCH_CHARS = 8000

# Docling's placeholder for an image it couldn't extract text from. Some
# filers (confirmed: Apple, Intel, Visa) render their cover-page company
# name as a stylized logo image rather than plain text -- without this
# check, that placeholder gets accepted as a "valid" company_name candidate
# verbatim.
IMAGE_PLACEHOLDER_PATTERN = re.compile(r"^<!--.*-->$")

TICKER_LINE_PATTERN = re.compile(r"^Common\s+Stock\s+([A-Z]{1,6})\b", re.IGNORECASE)
TICKER_TABLE_ROW_PATTERN = re.compile(
    r"\|\s*Common\s+Stock\s*\|\s*([A-Z]{1,6})\s*\|", re.IGNORECASE
)
REGISTRANT_NAME_ANCHOR = re.compile(
    r"\(Exact name of [Rr]egistrant as specified in its charter\)"
)
FISCAL_YEAR_FULL_DATE_PATTERN = re.compile(
    r"fiscal year ended\s+(\w+)\s+(\d{1,2}),\s*(\d{4})", re.IGNORECASE
)

MONTH_NAME_TO_NUM = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}

# Standard SEC Form 10-K item titles (per SEC regulation, same for every
# filer) -- used ONLY as a display-cleanup fallback when the parsed title is
# empty or clearly truncated, never as the primary detection mechanism.
CANONICAL_ITEM_TITLES = {
    "1": "Business", "1A": "Risk Factors", "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity", "2": "Properties", "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity, Related Stockholder Matters and Issuer Purchases of Equity Securities",
    "6": "[Reserved]",
    "7": "Management's Discussion and Analysis of Financial Condition and Results of Operations",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants on Accounting and Financial Disclosure",
    "9A": "Controls and Procedures", "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership of Certain Beneficial Owners and Management and Related Stockholder Matters",
    "13": "Certain Relationships and Related Transactions, and Director Independence",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}

def _matches_canonical_title(item_number: str, title: str) -> bool:
    canonical = CANONICAL_ITEM_TITLES.get(item_number)
    if not canonical:
        return False

    title_tokens = re.findall(r"[a-z0-9]+", title.casefold())
    canonical_tokens = re.findall(r"[a-z0-9]+", canonical.casefold())

    # Item 1 has the one-word title "Business"; most others use two words.
    prefix_length = min(2, len(canonical_tokens))
    return title_tokens[:prefix_length] == canonical_tokens[:prefix_length]
# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Section:
    item: str               # "Item 1", "Item 1A", etc. (matches required schema)
    title: str
    start_position: int
    end_position: int
    text: str = field(repr=False, default="")

    @property
    def text_char_count(self) -> int:
        return len(self.text)

    def to_dict(self) -> dict:
        return {
            "item": self.item,
            "title": self.title,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "text_char_count": self.text_char_count,
        }


@dataclass
class DocumentIdentity:
    company_name: Optional[str] = None
    ticker: Optional[str] = None
    cik: Optional[str] = None
    filing_date: Optional[str] = None
    reporting_period_end: Optional[str] = None
    fiscal_year: Optional[int] = None


# ---------------------------------------------------------------------------
# Section / heading detection
# ---------------------------------------------------------------------------

def _clean_heading_line(line: str) -> str:
    """Strips markdown decoration Docling may add, without altering the
    underlying text content (casing is preserved -- it matters for scoring)."""
    stripped = line.strip()
    stripped = re.sub(r"^#{1,6}\s*", "", stripped)   # markdown heading markers
    stripped = re.sub(r"^\*+\s*", "", stripped)       # leading bold/italic
    stripped = re.sub(r"\*+$", "", stripped)           # trailing bold/italic
    stripped = re.sub(r"^[-\u2022>]\s*", "", stripped)  # bullets / blockquote
    return stripped.strip()


def _find_item_candidates(text: str) -> List[tuple]:
    """
    Scans line by line, tracking character offsets, and returns every line
    that looks like an "Item N[letter]. Title" heading as
    (offset, item_number, title, is_toc, score) tuples.

    score: negative = confidently a real body heading, positive = confidently
    a Table of Contents entry. Two independent signals contribute:
      - trailing page number  -> +2 (TOC signal)
      - ALL CAPS "ITEM" keyword -> -2 (body signal), else +1 (TOC signal)
    """
    candidates = []
    offset = 0
    for line in text.split("\n"):
        line_len = len(line) + 1  # account for the '\n' split removed
        cleaned = _clean_heading_line(line)
        m = ITEM_LINE_PATTERN.match(cleaned)
        if m:
            keyword_raw = m.group(1)
            item_number = m.group(2).upper()
            title_raw = m.group(3).strip()

            score = 0
            if TRAILING_PAGE_NUM_PATTERN.search(title_raw):
                score += 2
            if _matches_canonical_title(item_number, title_raw):
                score -= 2    
            if keyword_raw.isupper():
                score -= 2
            else:
                score += 1

            is_toc = score > 0
            candidates.append((offset, item_number, title_raw, is_toc, score))
        offset += line_len
    return candidates


def _clean_title(item_number: str, raw_title: str) -> str:
    """Strips a trailing page number if one slipped through, and falls back
    to the canonical SEC title if the parsed title is empty or too short to
    be useful (e.g. a stray heading fragment)."""
    title = TRAILING_PAGE_NUM_PATTERN.sub("", raw_title).strip(" .")
    if len(title) < 3:
        canonical = CANONICAL_ITEM_TITLES.get(item_number)
        if canonical:
            logger.debug(
                "Title for Item %s was empty/too short ('%s') -- using canonical fallback.",
                item_number, raw_title,
            )
            return canonical
    return title.title() if title.isupper() else title


def _extract_markdown_heading_sections(text: str) -> List[Section]:
    """Fallback for nontraditional 10-Ks that do not label body sections as
    ``Item N``. Docling preserves their semantic headings as Markdown headings.

    These sections deliberately use ``Heading N`` rather than inventing SEC
    Item numbers. The source filing remains authoritative about its structure.
    """
    candidates = []
    offset = 0
    for line in text.split("\n"):
        match = MARKDOWN_HEADING_PATTERN.match(line.strip())
        if match:
            title = match.group(1).strip()
            # Ignore empty/very long layout artifacts; ordinary document
            # headings remain useful boundaries even when they are nested.
            if 3 <= len(title) <= 200:
                candidates.append((offset, title))
        offset += len(line) + 1

    # Intel-style filings put their standard cover pages and table of contents
    # before this conventional body heading. Use it when available, but do not
    # require it for other nontraditional filing formats.
    for index, (_, title) in enumerate(candidates):
        if title.casefold() == "forward-looking statements":
            candidates = candidates[index:]
            break

    substantive_candidates = []
    for index, (offset, title) in enumerate(candidates):
        next_offset = candidates[index + 1][0] if index + 1 < len(candidates) else len(text)
        if next_offset - offset >= MIN_MARKDOWN_SECTION_CHARS:
            substantive_candidates.append((offset, title))

    sections = []
    for index, (offset, title) in enumerate(substantive_candidates):
        end_offset = (
            substantive_candidates[index + 1][0]
            if index + 1 < len(substantive_candidates)
            else len(text)
        )
        section_text = text[offset:end_offset].strip()
        sections.append(
            Section(
                item=f"Heading {index + 1}",
                title=title,
                start_position=offset,
                end_position=end_offset,
                text=section_text,
            )
        )

    return sections


def extract_sections(text: str) -> List[Section]:
    """
    Finds every real Item-section heading in the document body and returns
    Section objects with populated start/end character offsets and text.
    Logs a warning (not a silent failure) if zero sections are found or if
    the count looks abnormally low for a 10-K.
    """
    candidates = _find_item_candidates(text)

    if not candidates:
        heading_sections = _extract_markdown_heading_sections(text)
        if heading_sections:
            logger.info(
                "No standard SEC ITEM headings found; using %d Markdown heading "
                "sections from this nontraditional filing.",
                len(heading_sections),
            )
            return heading_sections

        logger.warning(
            "No ITEM or Markdown heading candidates found at all -- inspect "
            "raw_docling.md manually."
        )
        return []

    body_candidates = [c for c in candidates if not c[3]]

    if not body_candidates:
        logger.warning(
            "%d ITEM candidates found but ALL were classified as Table-of-Contents "
            "entries -- this usually means the document has no distinguishable "
            "body headings (e.g. an unusual filing format). Falling back to "
            "treating every candidate as a body heading.",
            len(candidates),
        )
        body_candidates = candidates

    # Dedup: keep the occurrence with the most negative (most body-like) score
    best_by_item = {}
    for c in body_candidates:
        item_number = c[1]
        if item_number not in best_by_item or c[4] < best_by_item[item_number][4]:
            best_by_item[item_number] = c

    deduped = sorted(best_by_item.values(), key=lambda c: c[0])

    sections = []
    for i, (offset, item_number, title_raw, _, _) in enumerate(deduped):
        end_offset = deduped[i + 1][0] if i + 1 < len(deduped) else len(text)
        section_text = text[offset:end_offset].strip()
        sections.append(
            Section(
                item=f"Item {item_number}",
                title=_clean_title(item_number, title_raw),
                start_position=offset,
                end_position=end_offset,
                text=section_text,
            )
        )

    if len(sections) < 5:
        logger.warning(
            "Only %d sections detected -- a standard 10-K has 15-20+ Item "
            "sections. This document may have an unusual structure, or "
            "detection may have failed partially. Inspect manually.",
            len(sections),
        )

    return sections


# ---------------------------------------------------------------------------
# Identity field extraction (ticker, company name, dates)
# ---------------------------------------------------------------------------

def extract_ticker_from_text(text: str) -> Optional[str]:
    """
    Tries two patterns against the raw text: a plain line match (works on
    pdfplumber-style flat text) and a markdown-table-row match (works if
    Docling rendered the cover-page securities table as a proper Markdown
    table). Returns None (with a log message) if neither matches -- caller
    should fall back to the SEC registry lookup rather than guess.
    """
    for line in text[:6000].split("\n"):  # cover-page table is always near the top
        m = TICKER_LINE_PATTERN.match(line.strip())
        if m:
            return m.group(1).upper()
        m = TICKER_TABLE_ROW_PATTERN.search(line)
        if m:
            return m.group(1).upper()

    logger.info(
        "Could not find ticker via in-document text patterns "
        "(checked plain-line and markdown-table-row formats). "
        "Will rely on the SEC registry / folder-hint fallback instead."
    )
    return None


def extract_company_name_from_text(text: str) -> Optional[str]:
    """
    Every 10-K cover page has the fixed SEC-mandated phrase "(Exact name of
    Registrant as specified in its charter)" directly below the company
    name -- this is generic across all filers, not company-specific.
    """
    lines = text[:COVER_PAGE_SEARCH_CHARS].split("\n")
    for i, line in enumerate(lines):
        if REGISTRANT_NAME_ANCHOR.search(line):
            for back in range(1, 4):
                # Docling renders the company name as a markdown heading
                # (e.g. "## AMAZON.COM, INC.") -- strip that decoration so it
                # doesn't leak into every chunk's company_name metadata.
                candidate = _clean_heading_line(lines[i - back])
                if (candidate and len(candidate) > 2 and not candidate.startswith("(")
                        and not IMAGE_PLACEHOLDER_PATTERN.match(candidate)):
                    return candidate
    logger.info("Could not find company name via the 'Exact name of Registrant' anchor phrase.")
    return None


def extract_reporting_period_end(text: str) -> Optional[str]:
    """Returns an ISO date string (YYYY-MM-DD) parsed from the cover-page
    'For the fiscal year ended <Month> <Day>, <Year>' declaration."""
    m = FISCAL_YEAR_FULL_DATE_PATTERN.search(text[:COVER_PAGE_SEARCH_CHARS])
    if not m:
        logger.info("Could not find 'fiscal year ended' declaration on the cover page.")
        return None

    month_name, day, year = m.group(1).lower(), int(m.group(2)), int(m.group(3))
    month_num = MONTH_NAME_TO_NUM.get(month_name)
    if not month_num:
        logger.warning("Unrecognized month name '%s' in fiscal year declaration.", month_name)
        return None

    try:
        return datetime(year, month_num, day).strftime("%Y-%m-%d")
    except ValueError as e:
        logger.warning("Invalid date in fiscal year declaration: %s", e)
        return None


# A 52/53-week fiscal calendar (e.g. Johnson & Johnson's, which ends on the
# Sunday nearest December 31) can land in the first few days of January.
# Filers label that period by the calendar year the bulk of it falls in, not
# the year its last day happens to land in -- confirmed against J&J's own
# 10-K text for the period ended 2023-01-01, which repeatedly calls itself
# "fiscal year 2022" ("In the fiscal year 2022, the Company recorded...")
# and refers to "fiscal year 2023" only as a future event. A fiscal year-end
# late in January (e.g. NVIDIA's, in the last few days of the month) is NOT
# shifted: NVIDIA labels its period ended 2024-01-28 as "fiscal year 2024",
# the same calendar year the date falls in. The cutoff below separates the
# two cases without needing a per-company table.
EARLY_JANUARY_FISCAL_YEAR_SHIFT_CUTOFF_DAY = 7


def fiscal_year_from_reporting_period_end(reporting_period_end: str) -> int:
    year, month, day = (int(part) for part in reporting_period_end.split("-"))
    if month == 1 and day <= EARLY_JANUARY_FISCAL_YEAR_SHIFT_CUTOFF_DAY:
        return year - 1
    return year


def extract_identity_from_text(text: str) -> DocumentIdentity:
    """
    Extracts what CAN be reliably derived from the document text alone:
    ticker, company name, reporting period end, and fiscal year (derived
    from reporting_period_end's year, per spec -- NOT from the filename).
    filing_date and cik are intentionally left None here -- they should
    come from sec_registry.py (see that module's docstring for why).
    """
    identity = DocumentIdentity()
    identity.ticker = extract_ticker_from_text(text)
    identity.company_name = extract_company_name_from_text(text)
    identity.reporting_period_end = extract_reporting_period_end(text)

    if identity.reporting_period_end:
        identity.fiscal_year = fiscal_year_from_reporting_period_end(identity.reporting_period_end)
    else:
        logger.info(
            "fiscal_year could not be derived from reporting_period_end "
            "(which was itself not found) -- will remain null unless the "
            "SEC registry fallback supplies it."
        )

    return identity


if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("text_file", help="Path to a .txt/.md file of parsed filing text")
    args = arg_parser.parse_args()

    content = Path(args.text_file).read_text()
    identity = extract_identity_from_text(content)
    sections = extract_sections(content)

    print(json.dumps({
        "identity": vars(identity),
        "num_sections": len(sections),
        "sections": [s.to_dict() for s in sections],
    }, indent=2))
