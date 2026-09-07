"""
Manual CLI for Stage 9: full RAG pipeline -- ask a question, get a grounded answer.

Save as: sec-rag-project/scripts/run_generation.py

Usage:
    python scripts/run_generation.py "What are the main risk factors related to cybersecurity?"
    python scripts/run_generation.py "What was revenue growth?" --ticker TSLA --fiscal-year 2023
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import OLLAMA_URL
from src.retrieval.retriever import build_retrieval_context
from src.reranking.flashrank_reranker import load_reranker
from src.generation.rag_pipeline import answer_question
from src.generation.llm_router import is_ollama_running

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--fiscal-year", type=int, default=None)
    args = parser.parse_args()

    if not is_ollama_running():
        raise SystemExit(f"Ollama isn't reachable at {OLLAMA_URL} -- is it running?")

    filters = {}
    if args.ticker:
        filters["ticker"] = args.ticker
    if args.fiscal_year:
        filters["fiscal_year"] = args.fiscal_year

    context = build_retrieval_context()
    ranker = load_reranker()

    result = answer_question(args.question, context, ranker, filters=filters or None)

    print(f"\nQuestion: {args.question}\n")
    print(f"Answer:\n{result['answer']}\n")

    print(f"Sources ({len(result['sources'])}):")
    for i, s in enumerate(result["sources"], start=1):
        print(f"  [{i}] {s['ticker']} FY{s['fiscal_year']} {s['section_item']} (rerank_score={s['score']:.4f})")

    unverified = result["numeric_check"]["unverified"]
    if unverified:
        print(f"\nWARNING: numbers in the answer not found in any source (possible fabrication): {unverified}")


if __name__ == "__main__":
    main()
