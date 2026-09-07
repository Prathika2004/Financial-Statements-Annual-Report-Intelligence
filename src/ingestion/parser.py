"""
Parses a PDF into structured markdown text, preserving table structure.

Save as: sec-rag-project/src/ingestion/parser.py

Primary path: Docling (IBM) -- layout-aware, preserves tables as proper
Markdown tables. Downloads its layout model (~500MB-1GB) from Hugging Face
on first use -- needs internet once, then cached at ~/.cache/huggingface/.

Fallback path: pdfplumber -- plain text extraction, no table structure.
Used automatically if Docling isn't installed or throws an error on a
specific file, so ingestion never hard-fails on one bad PDF.

CHANGE from v1: parse_pdf() now also returns Docling's raw dict export
(result.document.export_to_dict()) when Docling succeeds, so the caller can
persist raw_docling.json separately from the cleaned/metadata output --
downstream chunking stages may want the structured dict (with real table
cell boundaries) rather than re-parsing markdown.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from docling.document_converter import DocumentConverter
    _DOCLING_AVAILABLE = True
except ImportError:
    _DOCLING_AVAILABLE = False

def parse_with_docling(pdf_path: Path) -> Optional[dict]:
    """
    Converts a PDF using Docling.

    Markdown extraction is required. The structured JSON export is optional:
    some documents cause Docling/Pydantic to detect a circular reference while
    serializing that export, but their Markdown is still valid.
    """
    if not _DOCLING_AVAILABLE:
        logger.warning(
            "Docling not installed -- falling back to pdfplumber for %s",
            pdf_path.name,
        )
        return None

    try:
        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        document = result.document

        # If this fails, Docling conversion itself failed: use pdfplumber.
        markdown = document.export_to_markdown()

        # If only this fails, retain Docling Markdown and omit raw_docling.json.
        try:
            doc_dict = document.export_to_dict()
        except Exception as exc:
            logger.warning(
                "Docling converted %s, but structured JSON export failed (%s). "
                "Continuing with Docling Markdown.",
                pdf_path.name,
                exc,
            )
            doc_dict = None

        return {
            "markdown": markdown,
            "doc_dict": doc_dict,
        }

    except Exception as exc:
        logger.warning(
            "Docling failed on %s (%s) -- falling back to pdfplumber",
            pdf_path.name,
            exc,
        )
        return None


def parse_with_pdfplumber(pdf_path: Path) -> str:
    """Plain-text fallback extraction. No table structure preserved."""
    import pdfplumber

    full_text = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            full_text.append(page_text)
    return "\n".join(full_text)


def needs_ocr(pdf_path: Path, char_threshold: int = 20) -> bool:
    """Cheap check for whether a PDF is scanned (image-only, no text layer).
    Not wired into the pipeline yet -- this dataset is confirmed all-native-text."""
    import pdfplumber

    with pdfplumber.open(str(pdf_path)) as pdf:
        if len(pdf.pages) == 0:
            return True
        first_page_text = pdf.pages[0].extract_text() or ""
        return len(first_page_text.strip()) < char_threshold


def parse_pdf(pdf_path: Path) -> dict:
    """
    Main entry point. Returns a dict:
      {
        "text": str,               # markdown (Docling) or plain text (pdfplumber)
        "doc_dict": dict | None,   # Docling's structured export, None if fallback was used
        "parser_used": "docling" | "pdfplumber" | "none",
        "needs_ocr": bool,
      }
    """
    pdf_path = Path(pdf_path)

    if needs_ocr(pdf_path):
        logger.info(
            "%s appears to be scanned (no text layer) -- OCR branch not yet "
            "implemented, skipping this file for now.", pdf_path.name
        )
        return {"text": "", "doc_dict": None, "parser_used": "none", "needs_ocr": True}

    docling_result = parse_with_docling(pdf_path)

    if docling_result is not None:
        return {
            "text": docling_result["markdown"],
            "doc_dict": docling_result["doc_dict"],
            "parser_used": "docling",
            "needs_ocr": False,
        }

    text = parse_with_pdfplumber(pdf_path)
    return {"text": text, "doc_dict": None, "parser_used": "pdfplumber", "needs_ocr": False}


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("pdf_file", help="Path to a 10-K PDF")
    args = arg_parser.parse_args()

    result = parse_pdf(Path(args.pdf_file))
    print(f"Parser used: {result['parser_used']}")
    print(f"Text length: {len(result['text']):,} chars")
    print(f"Has doc_dict: {result['doc_dict'] is not None}")
    print(result["text"][:500])