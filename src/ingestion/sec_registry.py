"""
Builds and caches a lookup table of company identity + filing-history data
from SEC EDGAR's own structured APIs -- the authoritative source for cik,
filing_date, and company_name.

Save as: sec-rag-project/src/ingestion/sec_registry.py

--- Why this exists (design rationale) ---
cik and filing_date are NOT reliably present as parseable text inside a
10-K's body: CIK is an EDGAR system identifier assigned at the account
level, not something companies print in the document, and "filing_date" is
metadata EDGAR records at submission time, not something every filer states
verbatim inside the PDF itself. Rather than write fragile regex hoping to
find these by luck, we pull them from EDGAR's own JSON APIs once and cache
them locally. reporting_period_end IS reliably in the document text (the
"fiscal year ended" cover-page declaration) AND independently available
here (as `reportDate`) -- we use both and cross-check, preferring the
in-document value but logging if they disagree.

--- Two API calls, both free, no key needed ---
1. https://www.sec.gov/files/company_tickers.json
   -> maps ticker to {cik, company title}, covers every US public company.
2. https://data.sec.gov/submissions/CIK{cik}.json
   -> per-company filing history: for each filing, `form`, `filingDate`,
      `reportDate` (== reporting_period_end), `accessionNumber`.

SEC requires a descriptive User-Agent with a real contact email on every
request, or it will 403 you -- set CONTACT_EMAIL below.
"""

import difflib
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

CONTACT_EMAIL = "your_email@example.com"  # <-- set this to a real address
HEADERS = {"User-Agent": f"AI Engineer RAG Project ({CONTACT_EMAIL})"}

REGISTRY_CACHE_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "company_registry.json"

# Manual overrides for folder names that fuzzy-matching might get wrong or
# that are ambiguous (e.g. "google" -> Alphabet's actual registrant name is
# "Alphabet Inc."). Edit this table, not the matching code, when a new
# company's folder name doesn't resolve correctly -- this keeps the matching
# logic itself generic rather than hardcoded per company.
FOLDER_NAME_TICKER_OVERRIDES = {
    "google": "GOOGL",
    "alphabet": "GOOGL",
}


def _fetch_json(url: str) -> dict:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_ticker_directory() -> dict:
    """Returns {TICKER: {"cik": "0001318605", "title": "Tesla, Inc."}, ...}"""
    logger.info("Fetching SEC's official ticker directory...")
    raw = _fetch_json("https://www.sec.gov/files/company_tickers.json")
    return {
        entry["ticker"].upper(): {
            "cik": str(entry["cik_str"]).zfill(10),
            "title": entry["title"],
        }
        for entry in raw.values()
    }


def fetch_10k_filing_history(cik: str) -> list:
    """
    Returns a list of dicts, one per original 10-K (10-K/A amendments
    excluded), each with: filing_date, reporting_period_end, accession_number.
    """
    logger.info("Fetching filing history for CIK %s...", cik)
    data = _fetch_json(f"https://data.sec.gov/submissions/CIK{cik}.json")
    recent = data["filings"]["recent"]

    filings = []
    for form, filing_date, report_date, accession in zip(
        recent["form"], recent["filingDate"], recent["reportDate"], recent["accessionNumber"]
    ):
        if form != "10-K":
            continue
        filings.append({
            "filing_date": filing_date,
            "reporting_period_end": report_date,
            "accession_number": accession,
        })
    return filings


def build_registry(tickers: list, save_path: Path = REGISTRY_CACHE_PATH) -> dict:
    """
    Builds the full registry for a list of tickers and saves it to disk.
    Run this once (or whenever you add new companies) rather than hitting
    EDGAR on every ingestion run.
    """
    ticker_directory = fetch_ticker_directory()
    registry = {}

    for ticker in tickers:
        ticker = ticker.upper()
        entry = ticker_directory.get(ticker)
        if not entry:
            logger.warning("Ticker '%s' not found in SEC's directory -- skipping.", ticker)
            continue

        cik = entry["cik"]
        try:
            filings = fetch_10k_filing_history(cik)
        except Exception as e:
            logger.error("Failed to fetch filing history for %s (CIK %s): %s", ticker, cik, e)
            filings = []

        registry[ticker] = {
            "company_name": entry["title"],
            "cik": cik,
            "filings": filings,
        }
        time.sleep(0.3)  # be polite to EDGAR

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(registry, f, indent=2)
    logger.info("Registry saved to %s (%d companies)", save_path, len(registry))
    return registry


def load_registry(path: Path = REGISTRY_CACHE_PATH) -> dict:
    if not path.exists():
        logger.warning(
            "No cached registry found at %s -- identity fields (cik, filing_date) "
            "will be null for all documents until you run build_registry(). "
            "See this module's __main__ block.",
            path,
        )
        return {}
    with open(path) as f:
        return json.load(f)


def resolve_ticker_from_folder_name(folder_name: str, registry: dict) -> Optional[str]:
    """
    Maps a raw folder name (e.g. "amazon", "BERKSHIRE HAT...") to a ticker
    already present in the registry, via: (1) manual override table,
    (2) exact ticker match, (3) fuzzy match against company names.
    Logs which method succeeded (or that all methods failed) for traceability.
    """
    normalized = re.sub(r"[^a-z0-9]", "", folder_name.lower())

    if normalized in FOLDER_NAME_TICKER_OVERRIDES:
        ticker = FOLDER_NAME_TICKER_OVERRIDES[normalized]
        logger.debug("Folder '%s' resolved via manual override -> %s", folder_name, ticker)
        return ticker

    if folder_name.upper() in registry:
        logger.debug("Folder '%s' matched directly as a ticker.", folder_name)
        return folder_name.upper()

    # fuzzy match against company names in the registry
    candidates = {
        ticker: re.sub(r"[^a-z0-9]", "", info["company_name"].lower())
        for ticker, info in registry.items()
    }
    matches = difflib.get_close_matches(normalized, candidates.values(), n=1, cutoff=0.4)
    if matches:
        matched_ticker = [t for t, name in candidates.items() if name == matches[0]][0]
        logger.info(
            "Folder '%s' fuzzy-matched to ticker %s (company: %s). "
            "Verify this is correct -- if not, add an override in FOLDER_NAME_TICKER_OVERRIDES.",
            folder_name, matched_ticker, registry[matched_ticker]["company_name"],
        )
        return matched_ticker

    logger.warning(
        "Could not resolve folder name '%s' to any ticker in the registry "
        "(no override, no exact match, no fuzzy match above threshold). "
        "cik/filing_date will be null for this document -- consider adding "
        "an entry to FOLDER_NAME_TICKER_OVERRIDES.",
        folder_name,
    )
    return None


def lookup_filing(ticker: str, fiscal_year: Optional[int], registry: dict,
                   reporting_period_end: Optional[str] = None) -> dict:
    """
    Given a ticker and the fiscal_year/reporting_period_end already extracted
    from the document text, finds the matching filing entry in the registry
    and returns {cik, company_name, filing_date, reporting_period_end} -- any
    field that can't be resolved is None, with a log explaining why.
    """
    result = {"cik": None, "company_name": None, "filing_date": None, "reporting_period_end": None}

    entry = registry.get(ticker)
    if not entry:
        logger.info("Ticker '%s' not present in registry -- cik/filing_date will be null.", ticker)
        return result

    result["cik"] = entry["cik"]
    result["company_name"] = entry["company_name"]

    # An exact date match is unambiguous and preferred: matching by fiscal_year
    # alone breaks for a 52/53-week fiscal calendar whose period-end date can
    # land in the same calendar year as fiscal_year's derived label OR the
    # next one (e.g. J&J's period ended 2023-01-01, labeled fiscal_year 2022,
    # would otherwise wrongly match a "2022"-prefixed registry entry for a
    # *different* filing). reporting_period_end is already reliably extracted
    # in-document, so use it directly when available.
    if reporting_period_end:
        for filing in entry["filings"]:
            if filing["reporting_period_end"] == reporting_period_end:
                result["filing_date"] = filing["filing_date"]
                result["reporting_period_end"] = filing["reporting_period_end"]
                return result
        logger.info(
            "%s: no registry filing exactly matches reporting_period_end=%s -- "
            "falling back to fiscal_year-based matching.",
            ticker, reporting_period_end,
        )

    if fiscal_year is None:
        logger.info(
            "No fiscal_year available to match against %s's filing history -- "
            "filing_date/reporting_period_end from registry will be null "
            "(in-document extraction may still have found reporting_period_end).",
            ticker,
        )
        return result

    for filing in entry["filings"]:
        if filing["reporting_period_end"].startswith(str(fiscal_year)):
            result["filing_date"] = filing["filing_date"]
            result["reporting_period_end"] = filing["reporting_period_end"]
            return result

    logger.info(
        "No filing found in %s's registry history matching fiscal_year=%s "
        "-- registry may not have been built with enough years of history.",
        ticker, fiscal_year,
    )
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Build/refresh the registry cache for your 20 companies. Run this once
    # (takes ~10-20 seconds), then metadata extraction reads the cached file
    # -- no network calls during actual ingestion.
    TICKERS = [
        "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "JNJ", "WMT",
        "XOM", "BRK-B", "PG", "KO", "NFLX", "INTC", "V", "MA", "PFE", "BA",
    ]
    build_registry(TICKERS)