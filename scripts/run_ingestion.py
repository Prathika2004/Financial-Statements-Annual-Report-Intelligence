"""
End-to-end runner for Stage 1: Parse -> Clean -> Extract Metadata.

Save as: sec-rag-project/scripts/run_ingestion.py

For every PDF found (recursively) under data/raw_pdfs/, produces:

  data/processed/<company>__<filename>/
      raw_docling.md      # Docling's raw markdown (or pdfplumber text if fallback was used)
      raw_docling.json     # Docling's structured dict export (null/absent if fallback was used)
      cleaned.md            # after cleaner.py's header/page-number stripping
      metadata.json         # document identity + sections + statistics (see schema below)

metadata.json schema:
  {
    "document": {filename, company_name, ticker, cik, filing_type,
                 filing_date, reporting_period_end, fiscal_year, source,
                 parser_used, status},
    "sections": [{item, title, start_position, end_position, text_char_count}, ...],
    "statistics": {num_sections, full_cleaned_text_char_count}
  }

Usage:
    python scripts/run_ingestion.py                # process everything
    python scripts/run_ingestion.py --limit 5       # first 5 only (testing)
    python scripts/run_ingestion.py --file amzn_10K_2022.pdf   # single file
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import RAW_PDF_DIR, PROCESSED_DIR
from src.ingestion.parser import parse_pdf
from src.ingestion.metadata_extractor import (
    extract_identity_from_text, extract_sections, fiscal_year_from_reporting_period_end,
)
from src.ingestion.sec_registry import load_registry, resolve_ticker_from_folder_name, lookup_filing
from src.cleaning.cleaner import clean_text

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_ingestion")


def process_one_pdf(pdf_path: Path, registry: dict) -> tuple:
    """Returns (metadata_dict, raw_markdown, raw_doc_dict_or_None, cleaned_text_or_None)."""
    folder_hint = pdf_path.parent.name
    parsed = parse_pdf(pdf_path)

    base_document = {
        "filename": pdf_path.name,
        "company_name": None,
        "ticker": None,
        "cik": None,
        "filing_type": "10-K",
        "filing_date": None,
        "reporting_period_end": None,
        "fiscal_year": None,
        "source": "SEC",
        "parser_used": parsed["parser_used"],
        "status": None,
    }

    if parsed["needs_ocr"] or not parsed["text"]:
        base_document["status"] = "skipped_needs_ocr" if parsed["needs_ocr"] else "skipped_empty"
        logger.warning("%s: %s", pdf_path.name, base_document["status"])
        metadata = {
            "document": base_document,
            "sections": [],
            "statistics": {"num_sections": 0, "full_cleaned_text_char_count": 0},
        }
        return metadata, parsed["text"], parsed["doc_dict"], None

    cleaned = clean_text(parsed["text"])

    # --- identity fields extracted from the document text itself ---
    identity = extract_identity_from_text(cleaned)

    # --- authoritative fields (cik, filing_date) + cross-check from EDGAR registry ---
    resolved_ticker = identity.ticker or resolve_ticker_from_folder_name(folder_hint, registry)
    registry_info = lookup_filing(
        resolved_ticker, identity.fiscal_year, registry, identity.reporting_period_end
    ) if resolved_ticker else {
        "cik": None, "company_name": None, "filing_date": None, "reporting_period_end": None,
    }

    if identity.ticker and resolved_ticker and identity.ticker != resolved_ticker:
        logger.warning(
            "%s: ticker mismatch -- in-document extraction found '%s' but folder-based "
            "resolution found '%s'. Using in-document value; verify manually.",
            pdf_path.name, identity.ticker, resolved_ticker,
        )

    final_ticker = identity.ticker or resolved_ticker
    final_company_name = identity.company_name or registry_info["company_name"]
    final_reporting_period_end = identity.reporting_period_end or registry_info["reporting_period_end"]
    final_fiscal_year = identity.fiscal_year or (
        fiscal_year_from_reporting_period_end(final_reporting_period_end)
        if final_reporting_period_end else None
    )

    if identity.reporting_period_end and registry_info["reporting_period_end"] and \
            identity.reporting_period_end != registry_info["reporting_period_end"]:
        logger.warning(
            "%s: reporting_period_end mismatch -- document says %s, EDGAR registry says %s. "
            "Using in-document value.",
            pdf_path.name, identity.reporting_period_end, registry_info["reporting_period_end"],
        )

    base_document.update({
        "company_name": final_company_name,
        "ticker": final_ticker,
        "cik": registry_info["cik"],
        "filing_date": registry_info["filing_date"],
        "reporting_period_end": final_reporting_period_end,
        "fiscal_year": final_fiscal_year,
        "status": "ok",
    })

    for field_name in ("company_name", "ticker", "cik", "filing_date", "reporting_period_end", "fiscal_year"):
        if base_document[field_name] is None:
            logger.info("%s: field '%s' could not be resolved -- left as null.", pdf_path.name, field_name)

    # --- section detection ---
    sections = extract_sections(cleaned)
    if not sections:
        logger.warning("%s: 0 sections detected -- see warnings above for why.", pdf_path.name)

    metadata = {
        "document": base_document,
        "sections": [s.to_dict() for s in sections],
        "statistics": {
            "num_sections": len(sections),
            "full_cleaned_text_char_count": len(cleaned),
        },
    }

    return metadata, parsed["text"], parsed["doc_dict"], cleaned


def save_outputs(pdf_path: Path, metadata: dict, raw_markdown: str, raw_doc_dict, cleaned_text: str):
    folder_hint = pdf_path.parent.name
    out_dir = PROCESSED_DIR / f"{folder_hint}__{pdf_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # NOTE: encoding="utf-8" is required explicitly on every write here.
    # Path.write_text() / open() without an encoding default to the OS locale
    # encoding -- on Windows that's cp1252, which cannot represent characters
    # SEC filings actually contain (e.g. the "\u2612" checkbox mark on the
    # cover page: "(Mark One) \u2612 ANNUAL REPORT..."), causing a crash.
    # Linux/Mac default to UTF-8, which is why this didn't surface in testing
    # there -- it's a genuine Windows-specific bug, not a one-off.
    (out_dir / "raw_docling.md").write_text(raw_markdown or "", encoding="utf-8")

    if raw_doc_dict is not None:
        with open(out_dir / "raw_docling.json", "w", encoding="utf-8") as f:
            json.dump(raw_doc_dict, f, indent=2, default=str, ensure_ascii=False)

    if cleaned_text is not None:
        (out_dir / "cleaned.md").write_text(cleaned_text, encoding="utf-8")

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N PDFs")
    parser.add_argument("--file", type=str, default=None, help="Process a single named PDF only")
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    registry = load_registry()

    if args.file:
        matches = list(RAW_PDF_DIR.rglob(args.file))
        if not matches:
            logger.error("Could not find %s anywhere under %s", args.file, RAW_PDF_DIR)
            return
        pdf_files = matches
    else:
        pdf_files = sorted(RAW_PDF_DIR.rglob("*.pdf"))
        if args.limit:
            pdf_files = pdf_files[: args.limit]

    if not pdf_files:
        logger.error("No PDFs found under %s (searched recursively).", RAW_PDF_DIR)
        return

    logger.info("Found %d PDF(s) under %s", len(pdf_files), RAW_PDF_DIR)

    ok_count, skipped_count, zero_section_count = 0, 0, 0

    for pdf_path in tqdm(pdf_files, desc="Ingesting filings"):
        if not pdf_path.exists():
            logger.error("Missing file: %s", pdf_path)
            continue

        metadata, raw_markdown, raw_doc_dict, cleaned_text = process_one_pdf(pdf_path, registry)
        out_dir = save_outputs(pdf_path, metadata, raw_markdown, raw_doc_dict, cleaned_text)

        status = metadata["document"]["status"]
        num_sections = metadata["statistics"]["num_sections"]

        if status == "ok" and num_sections > 0:
            ok_count += 1
        elif status == "ok" and num_sections == 0:
            zero_section_count += 1
        else:
            skipped_count += 1

        logger.info(
            "%s -> %s | parser=%s status=%s sections=%d",
            pdf_path.name, out_dir.name, metadata["document"]["parser_used"], status, num_sections,
        )

    print(f"\nDone. {ok_count} filings fully OK, {zero_section_count} OK-but-zero-sections "
          f"(needs manual review), {skipped_count} skipped.")
    print(f"Output saved to {PROCESSED_DIR}/")


if __name__ == "__main__":
    main()