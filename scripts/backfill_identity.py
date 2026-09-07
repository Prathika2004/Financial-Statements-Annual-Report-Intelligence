"""Regenerate document identity fields from existing cleaned documents.

Use this after improving identity extraction (company_name, ticker, cik,
filing_date, reporting_period_end, fiscal_year). It avoids re-running slow
PDF conversion when the raw and cleaned outputs are already valid -- mirrors
backfill_sections.py but for the "document" block instead of "sections".
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import PROCESSED_DIR
from src.ingestion.metadata_extractor import extract_identity_from_text, fiscal_year_from_reporting_period_end
from src.ingestion.sec_registry import load_registry, resolve_ticker_from_folder_name, lookup_filing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prefix",
        default="",
        help="Processed-folder prefix to update, for example 'nvdia__'. Omit to update all.",
    )
    args = parser.parse_args()

    registry = load_registry()

    folders = sorted(path for path in PROCESSED_DIR.glob(f"{args.prefix}*") if path.is_dir())
    if not folders:
        raise SystemExit(f"No processed folders found with prefix: {args.prefix}")

    for folder in folders:
        cleaned_path = folder / "cleaned.md"
        metadata_path = folder / "metadata.json"
        if not cleaned_path.exists() or not metadata_path.exists():
            print(f"Skipping {folder.name}: cleaned.md or metadata.json is missing")
            continue

        cleaned_text = cleaned_path.read_text(encoding="utf-8")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        doc = metadata["document"]

        if doc.get("status") != "ok":
            print(f"Skipping {folder.name}: status={doc.get('status')}")
            continue

        identity = extract_identity_from_text(cleaned_text)

        # Processed folders are named "{raw_pdf_parent_folder}__{pdf_stem}"
        # (see run_ingestion.py's folder_hint = pdf_path.parent.name). Using
        # the full processed-folder name here instead of just the prefix
        # feeds a completely different, longer string into the fuzzy matcher
        # and can match the wrong company.
        folder_hint = folder.name.split("__", 1)[0]
        resolved_ticker = identity.ticker or resolve_ticker_from_folder_name(folder_hint, registry)
        registry_info = lookup_filing(
            resolved_ticker, identity.fiscal_year, registry, identity.reporting_period_end
        ) if resolved_ticker else {
            "cik": None, "company_name": None, "filing_date": None, "reporting_period_end": None,
        }

        final_ticker = identity.ticker or resolved_ticker
        final_company_name = identity.company_name or registry_info["company_name"]
        final_reporting_period_end = identity.reporting_period_end or registry_info["reporting_period_end"]
        final_fiscal_year = identity.fiscal_year or (
            fiscal_year_from_reporting_period_end(final_reporting_period_end)
            if final_reporting_period_end else None
        )

        before = dict(doc)
        doc.update({
            "company_name": final_company_name,
            "ticker": final_ticker,
            "cik": registry_info["cik"],
            "filing_date": registry_info["filing_date"],
            "reporting_period_end": final_reporting_period_end,
            "fiscal_year": final_fiscal_year,
        })

        changed = {k: (before[k], doc[k]) for k in doc if before.get(k) != doc[k]}
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if changed:
            print(f"{folder.name}: updated {changed}")
        else:
            print(f"{folder.name}: unchanged")


if __name__ == "__main__":
    main()
