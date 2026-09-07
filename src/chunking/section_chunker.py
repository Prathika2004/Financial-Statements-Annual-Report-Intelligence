"""
Splits each detected section (from Stage 3's metadata.json) into retrievable
chunks: tables extracted as standalone chunks first, then the remaining
prose recursively split to a target size with overlap.

Save as: sec-rag-project/src/chunking/section_chunker.py

Chunking never crosses a section boundary -- each section (Item 1, Item 1A,
or Intel's Heading 1/2/... fallback) is chunked independently, so a chunk
never mixes content from two unrelated topics. This is why Stage 3's clean
section detection mattered: chunking quality is directly downstream of it.
"""

import logging
import re
import sys
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.settings import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS
from src.chunking.table_chunker import build_table_chunks

logger = logging.getLogger(__name__)

# Rough heuristic: ~4 characters per token for English text. Avoids adding a
# tokenizer dependency at this stage -- good enough for sizing chunks
# consistently; exact token counts matter more at the embedding-model stage,
# where the actual tokenizer will be used directly.
CHARS_PER_TOKEN_ESTIMATE = 4
CHUNK_SIZE_CHARS = CHUNK_SIZE_TOKENS * CHARS_PER_TOKEN_ESTIMATE
CHUNK_OVERLAP_CHARS = CHUNK_OVERLAP_TOKENS * CHARS_PER_TOKEN_ESTIMATE

SEPARATOR_HIERARCHY = ["\n\n", "\n", ". ", " "]


def recursive_split_text(text: str, chunk_size_chars: int = CHUNK_SIZE_CHARS,
                          overlap_chars: int = CHUNK_OVERLAP_CHARS) -> List[str]:
    """
    Splits text hierarchically -- tries the coarsest separator (paragraph
    breaks) first, only descending to finer separators (lines, sentences,
    then a hard character cut as a last resort) for pieces still too large.
    Rebuilds chunks up to chunk_size_chars, carrying overlap_chars of
    trailing context into the next chunk so retrieval doesn't lose meaning
    right at a chunk boundary.
    """
    if not text.strip():
        return []
    if len(text) <= chunk_size_chars:
        return [text.strip()]

    def split_on(sep, s):
        parts = s.split(sep)
        return [p + sep for p in parts[:-1]] + [parts[-1]]

    def recursive(pieces, sep_idx):
        if sep_idx >= len(SEPARATOR_HIERARCHY):
            out = []
            for p in pieces:
                for i in range(0, len(p), chunk_size_chars):
                    out.append(p[i:i + chunk_size_chars])
            return out
        result = []
        for p in pieces:
            if len(p) <= chunk_size_chars:
                result.append(p)
            else:
                result.extend(recursive(split_on(SEPARATOR_HIERARCHY[sep_idx], p), sep_idx + 1))
        return result

    fine_pieces = recursive([text], 0)

    chunks = []
    current = ""
    for piece in fine_pieces:
        if len(current) + len(piece) <= chunk_size_chars:
            current += piece
        else:
            if current.strip():
                chunks.append(current.strip())
            overlap_tail = current[-overlap_chars:] if overlap_chars else ""
            current = overlap_tail + piece
    if current.strip():
        chunks.append(current.strip())

    return chunks


def chunk_section(section: dict, company_name: str, fiscal_year: int) -> List[dict]:
    """
    Chunks a single section (as found in metadata.json's "sections" array,
    with its text sliced from cleaned.md using start_position/end_position
    by the caller). Returns a list of chunk dicts (mix of "table" and
    "text" chunk_type), in document order.
    """
    section_text = section["text"]
    section_item = section["item"]
    section_title = section["title"]

    table_chunks, table_spans = build_table_chunks(
        section_text, section_item, section_title, company_name, fiscal_year
    )

    def make_text_chunks(text: str) -> List[dict]:
        return [
            {
                "chunk_type": "text",
                "text": piece,
                "char_count": len(piece),
                "section_item": section_item,
                "section_title": section_title,
            }
            for piece in recursive_split_text(text)
        ]

    # Emit prose and tables in their original order. Removing every table
    # first and then appending all table chunks would make a table at the end
    # of a section appear before its opening prose in retrieval/debug output.
    all_chunks = []
    cursor = 0
    table_chunks_by_index = {}
    for table_chunk in table_chunks:
        table_chunks_by_index.setdefault(table_chunk["table_index_in_section"], []).append(table_chunk)

    for table_index, (start, end) in enumerate(table_spans):
        all_chunks.extend(make_text_chunks(section_text[cursor:start]))
        all_chunks.extend(table_chunks_by_index[table_index])
        cursor = end
    all_chunks.extend(make_text_chunks(section_text[cursor:]))

    if not all_chunks:
        logger.debug("Section %s produced zero chunks (empty after table removal?)", section_item)

    return all_chunks


def chunk_document(metadata: dict, cleaned_text: str) -> List[dict]:
    """
    Main entry point for Stage 4. Takes a document's metadata.json content
    and its cleaned.md text, slices out each section's text using the
    stored character offsets, chunks each section independently, and
    returns the full flat list of chunk dicts with document-level identity
    metadata attached to every chunk (needed for Qdrant payload filtering
    later: ticker, fiscal_year, company_name).
    """
    doc = metadata["document"]
    sections = metadata["sections"]

    if not sections:
        logger.warning(
            "%s: metadata.json has zero sections -- cannot chunk. "
            "This document needs its Stage 3 extraction reviewed first.",
            doc["filename"],
        )
        return []

    all_chunks = []
    chunk_counter = 0

    for section_meta in sections:
        start, end = section_meta["start_position"], section_meta["end_position"]
        parent_section_id = re.sub(
            r"[^a-z0-9]+",
            "_",
            f"{doc['filename']}__{section_meta['item']}",
            flags=re.IGNORECASE,
        ).strip("_").lower()
        section_with_text = {
            "item": section_meta["item"],
            "title": section_meta["title"],
            "text": cleaned_text[start:end],
        }

        section_chunks = chunk_section(section_with_text, doc.get("company_name"), doc.get("fiscal_year"))

        for chunk in section_chunks:
            chunk_counter += 1
            chunk["chunk_id"] = f"{doc['ticker'] or 'UNK'}_{doc['fiscal_year'] or 'UNK'}_{chunk_counter:04d}"
            # These fields let retrieval expand a small chunk back to its
            # complete source section without duplicating full section text in
            # every Qdrant payload.
            chunk["parent_section_id"] = parent_section_id
            chunk["parent_section_start"] = start
            chunk["parent_section_end"] = end
            chunk["ticker"] = doc.get("ticker")
            chunk["fiscal_year"] = doc.get("fiscal_year")
            chunk["company_name"] = doc.get("company_name")
            chunk["source_filename"] = doc.get("filename")
            chunk["chunk_id"] = (
                f"{doc['ticker'] or 'UNK'}_{doc['fiscal_year'] or 'UNK'}_"
                f"{Path(doc['filename']).stem}_{chunk_counter:04d}"
            )
            all_chunks.append(chunk)

    num_table_chunks = sum(1 for c in all_chunks if c["chunk_type"] == "table")
    num_text_chunks = len(all_chunks) - num_table_chunks
    logger.info(
        "%s: produced %d chunks (%d table, %d text) from %d sections",
        doc["filename"], len(all_chunks), num_table_chunks, num_text_chunks, len(sections),
    )

    return all_chunks
