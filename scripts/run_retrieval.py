"""
Manual CLI for Stage 7 hybrid retrieval -- ask a question, see what comes back.

Save as: sec-rag-project/scripts/run_retrieval.py

Usage:
    python scripts/run_retrieval.py "What are the main risk factors?"
    python scripts/run_retrieval.py "What was revenue growth?" --ticker TSLA --fiscal-year 2023
    python scripts/run_retrieval.py "cybersecurity risk" --chunk-type table --top-k 5
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.retrieval.retriever import build_retrieval_context, retrieve

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--fiscal-year", type=int, default=None)
    parser.add_argument("--chunk-type", default=None, choices=["text", "table"])
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    filters = {}
    if args.ticker:
        filters["ticker"] = args.ticker
    if args.fiscal_year:
        filters["fiscal_year"] = args.fiscal_year
    if args.chunk_type:
        filters["chunk_type"] = args.chunk_type

    context = build_retrieval_context()
    results = retrieve([args.query], context, filters=filters or None, top_k=args.top_k)

    print(f"\n{len(results)} result(s) for: {args.query!r}" + (f" (filters={filters})" if filters else ""))
    for i, hit in enumerate(results, start=1):
        print(f"\n[{i}] rrf_score={hit['rrf_score']:.4f} found_by={hit['found_by']} "
              f"| {hit['ticker']} FY{hit['fiscal_year']} {hit['section_item']} ({hit['chunk_type']})")
        print(f"    {hit['text'][:200]!r}")


if __name__ == "__main__":
    main()
