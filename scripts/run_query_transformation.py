"""
Manual CLI for verifying query transformation against a real LLM.

Save as: sec-rag-project/scripts/run_query_transformation.py

Every function in src/query_transformation/ was built and unit-tested
against FakeLLM (canned responses) before an LLM provider existed. This
script is the real-model check: it wires llm_router.generate() in as the
llm_generate callable and shows what each technique actually produces from
Qwen2.5, since a real model's output format (numbering style, verbosity,
instruction-following) can differ from what the parsing logic was tested
against.

Usage:
    python scripts/run_query_transformation.py "Explain AI" --technique rewrite
    python scripts/run_query_transformation.py "What was revenue growth?" --technique multi-query
    python scripts/run_query_transformation.py "How did revenue and headcount both change in 2023?" --technique decompose
    python scripts/run_query_transformation.py "What was Q3 2023 revenue?" --technique step-back
    python scripts/run_query_transformation.py "What was revenue growth?" --technique hyde
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import OLLAMA_URL
from src.generation.llm_router import generate, is_ollama_running
from src.query_transformation.query_rewriting import rewrite_query, step_back_query
from src.query_transformation.multi_query import generate_multi_queries
from src.query_transformation.decomposition import decompose_query
from src.query_transformation.hyde import generate_hypothetical_document, hyde_search

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--technique", required=True,
                         choices=["rewrite", "step-back", "multi-query", "decompose", "hyde"])
    args = parser.parse_args()

    if not is_ollama_running():
        raise SystemExit(f"Ollama isn't reachable at {OLLAMA_URL} -- is it running?")

    print(f"Original question: {args.question!r}\n")

    if args.technique == "rewrite":
        print("Rewritten:", repr(rewrite_query(args.question, generate)))

    elif args.technique == "step-back":
        print("Step-back:", repr(step_back_query(args.question, generate)))

    elif args.technique == "multi-query":
        variants = generate_multi_queries(args.question, generate, num_variants=3)
        print(f"{len(variants)} quer{'y' if len(variants) == 1 else 'ies'} for retrieve():")
        for v in variants:
            print(f"  - {v!r}")

    elif args.technique == "decompose":
        sub_questions = decompose_query(args.question, generate)
        print(f"{len(sub_questions)} sub-question(s):")
        for q in sub_questions:
            print(f"  - {q!r}")

    elif args.technique == "hyde":
        hypothetical = generate_hypothetical_document(args.question, generate)
        print(f"Hypothetical document:\n{hypothetical}\n")

        from src.embeddings.embedder import load_embedding_model
        from src.vectorstore.qdrant_client import get_client

        model = load_embedding_model()
        client = get_client()
        results = hyde_search(args.question, generate, model, client, top_k=3)

        print(f"Top {len(results)} HyDE dense-search result(s):")
        for r in results:
            print(f"  score={r['score']:.4f} | {r['ticker']} FY{r['fiscal_year']} {r['section_item']} "
                  f"| {r['text'][:100]!r}")


if __name__ == "__main__":
    main()
