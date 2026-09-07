"""
Manual CLI for Stage 8: hybrid retrieval + cross-encoder reranking.

Save as: sec-rag-project/scripts/run_reranking.py

Usage:
    python scripts/run_reranking.py "What are the main risk factors?"
    python scripts/run_reranking.py "What was revenue growth?" --ticker TSLA --fiscal-year 2023
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.retrieval.retriever import build_retrieval_context, retrieve_and_rerank
from src.reranking.flashrank_reranker import load_reranker

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--fiscal-year", type=int, default=None)
    parser.add_argument("--chunk-type", default=None, choices=["text", "table"])
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    filters = {}
    if args.ticker:
        filters["ticker"] = args.ticker
    if args.fiscal_year:
        filters["fiscal_year"] = args.fiscal_year
    if args.chunk_type:
        filters["chunk_type"] = args.chunk_type

    context = build_retrieval_context()
    ranker = load_reranker()

    results = retrieve_and_rerank([args.query], context, ranker, filters=filters or None, final_k=args.top_k)

    print(f"\n{len(results)} result(s) for: {args.query!r}" + (f" (filters={filters})" if filters else ""))
    for i, hit in enumerate(results, start=1):
        print(f"\n[{i}] rerank_score={hit['score']:.4f} (pre-rerank rrf_score={hit['rrf_score']:.4f}, "
              f"found_by={hit['found_by']}) | {hit['ticker']} FY{hit['fiscal_year']} "
              f"{hit['section_item']} ({hit['chunk_type']})")
        print(f"    {hit['text'][:200]!r}")


if __name__ == "__main__":
    main()
