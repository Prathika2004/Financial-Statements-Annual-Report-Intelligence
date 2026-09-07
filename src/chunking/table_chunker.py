"""
Detects and extracts financial tables from a section's text as standalone
chunks, so a dense number-table is never split mid-table or diluted by
surrounding prose in the same chunk.

Save as: sec-rag-project/src/chunking/table_chunker.py

Detection strategy: looks for contiguous blocks of GFM-style Markdown table
rows ("| cell | cell |"), which is what Docling produces for genuinely
gridded/bordered tables. This only fires on Docling-parsed documents --
pdfplumber's flat-text fallback has no reconstructable table structure, so
sections parsed that way simply have zero tables detected and fall through
entirely to section_chunker's prose splitting. This is a known, logged
limitation (see build_table_chunks docstring), not a silent failure.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Tuple

from config.settings import CHUNK_SIZE_TOKENS

logger = logging.getLogger(__name__)

MD_TABLE_ROW_PATTERN = re.compile(r"^\s*\|.*\|\s*$")
CAPTION_SEARCH_WINDOW_CHARS = 300
MIN_TABLE_ROWS = 2  # header + at least one data/separator row
# Tables use the same approximate budget as prose chunks. A small allowance
# keeps the repeated caption/header with each part.
MAX_TABLE_CHUNK_CHARS = CHUNK_SIZE_TOKENS * 4


@dataclass
class TableSpan:
    start: int
    end: int
    table_text: str
    caption: str = ""


def detect_markdown_tables(text: str) -> List[TableSpan]:
    """
    Scans line by line for contiguous runs of Markdown table rows and
    returns each run's character span + raw table text. Requires at least
    MIN_TABLE_ROWS consecutive matching lines so a single stray line
    containing a couple of pipe characters doesn't get misdetected as a table.
    """
    lines = text.split("\n")
    line_offsets = []
    offset = 0
    for line in lines:
        line_offsets.append(offset)
        offset += len(line) + 1

    spans = []
    i, n = 0, len(lines)
    while i < n:
        if MD_TABLE_ROW_PATTERN.match(lines[i]):
            start_line = i
            j = i
            while j < n and MD_TABLE_ROW_PATTERN.match(lines[j]):
                j += 1
            if j - start_line >= MIN_TABLE_ROWS:
                start_off = line_offsets[start_line]
                end_off = line_offsets[j - 1] + len(lines[j - 1])
                table_text = "\n".join(lines[start_line:j])
                spans.append(TableSpan(start=start_off, end=end_off, table_text=table_text))
            i = j
        else:
            i += 1
    return spans


def _find_caption(text: str, table_start: int) -> str:
    """
    Looks backward from a table's start position for the nearest non-empty
    line of prose within CAPTION_SEARCH_WINDOW_CHARS -- this is usually the
    sentence introducing the table ("The following table presents...").
    Falls back to an empty string (caller supplies a generic caption instead)
    if nothing suitable is found nearby.
    """
    window_start = max(0, table_start - CAPTION_SEARCH_WINDOW_CHARS)
    preceding_text = text[window_start:table_start]
    lines = [l.strip() for l in preceding_text.split("\n") if l.strip()]
    if not lines:
        return ""
    candidate = lines[-1]
    # avoid picking up a heading line itself as the caption
    if candidate.isupper() or len(candidate) < 10:
        return ""
    return candidate


def _split_table_by_rows(table_text: str, max_chars: int = MAX_TABLE_CHUNK_CHARS) -> List[str]:
    """Split an oversized Markdown table only between rows.

    The first two rows (header and separator) are repeated in every part so
    each embedded table fragment remains understandable on its own.
    """
    if len(table_text) <= max_chars:
        return [table_text]

    lines = table_text.splitlines()
    if len(lines) < 3:
        return [table_text]

    header = lines[:2]
    prefix = "\n".join(header)
    parts = []
    current_rows = []

    for row in lines[2:]:
        candidate_rows = current_rows + [row]
        candidate = prefix + "\n" + "\n".join(candidate_rows)
        if current_rows and len(candidate) > max_chars:
            parts.append(prefix + "\n" + "\n".join(current_rows))
            current_rows = [row]
        else:
            current_rows = candidate_rows

    if current_rows:
        parts.append(prefix + "\n" + "\n".join(current_rows))

    return parts or [table_text]


def build_table_chunks(section_text: str, section_item: str, section_title: str,
                        company_name: str, fiscal_year: int) -> Tuple[List[dict], List[Tuple[int, int]]]:
    """
    Finds every table in a section's text and returns (chunks, spans_to_remove).

    chunks: list of chunk dicts (chunk_type="table"), each with a caption
    prepended to the table markdown so the embedding captures semantic
    meaning, not just a wall of numbers.

    spans_to_remove: the (start, end) character ranges the caller should
    exclude before running prose-chunking on the remainder of the section,
    so table content doesn't get double-counted as both a table chunk and
    fragments of a text chunk.
    """
    table_spans = detect_markdown_tables(section_text)

    if not table_spans:
        return [], []

    chunks = []
    for idx, span in enumerate(table_spans):
        found_caption = _find_caption(section_text, span.start)
        caption = found_caption or f"Table from {section_item} ({section_title}), {company_name} FY{fiscal_year}"

        # Reserve space for the caption, including a possible part suffix.
        # A single exceptionally wide row can still exceed this budget because
        # rows are never cut in half.
        table_budget = max(200, MAX_TABLE_CHUNK_CHARS - len(caption) - 30)
        table_parts = _split_table_by_rows(span.table_text, max_chars=table_budget)
        for part_index, table_part in enumerate(table_parts, start=1):
            part_caption = (
                f"{caption} (part {part_index}/{len(table_parts)})"
                if len(table_parts) > 1
                else caption
            )
            chunk_text = f"{part_caption}\n\n{table_part}"
            chunks.append({
                "chunk_type": "table",
                "text": chunk_text,
                "caption": part_caption,
                "char_count": len(chunk_text),
                "section_item": section_item,
                "section_title": section_title,
                "table_index_in_section": idx,
                "table_part": part_index,
                "table_parts": len(table_parts),
            })

    logger.debug("Section %s: found %d table(s)", section_item, len(table_spans))
    spans_to_remove = [(s.start, s.end) for s in table_spans]
    return chunks, spans_to_remove


def remove_spans(text: str, spans_to_remove: List[Tuple[int, int]]) -> str:
    """Removes the given character spans (already-extracted tables) from
    text, so prose-chunking doesn't reprocess the same content."""
    if not spans_to_remove:
        return text
    spans_sorted = sorted(spans_to_remove, key=lambda s: s[0])
    result = []
    cursor = 0
    for start, end in spans_sorted:
        result.append(text[cursor:start])
        cursor = end
    result.append(text[cursor:])
    return "".join(result)
