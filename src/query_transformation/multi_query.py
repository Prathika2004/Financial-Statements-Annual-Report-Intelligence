"""
Multi-query expansion: generates several alternative phrasings of one
question, to widen dense/BM25 recall before hybrid fusion.

Save as: sec-rag-project/src/query_transformation/multi_query.py

generate_multi_queries()'s return value is a List[str] that plugs directly
into retriever.retrieve()/retrieve_and_rerank()'s `queries` parameter --
this is the exact extensibility point that parameter was built for (see
retriever.py's module docstring): each variant gets its own dense+BM25
search, and hybrid_fusion.reciprocal_rank_fusion() merges all of them.
"""

import logging
import re
import sys
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.query_transformation.llm_interface import LLMGenerate

logger = logging.getLogger(__name__)

MULTI_QUERY_PROMPT_TEMPLATE = (
    "Generate {num_variants} different ways to phrase the following "
    "question, each capturing the same information need from a different "
    "angle or wording. Return exactly {num_variants} lines, one question "
    "per line, with no numbering or extra text.\n\nQuestion: {question}"
)


def _parse_lines(raw_output: str) -> List[str]:
    lines = [line.strip() for line in raw_output.strip().split("\n")]
    # Strip a leading "1. " / "- " / "1) " numbering style if the LLM added
    # one despite being asked not to -- cheap insurance against a common
    # instruction-following failure mode, not a hard requirement.
    cleaned = [re.sub(r"^\d+[\.\)]\s*|^[-*]\s*", "", line) for line in lines]
    return [line for line in cleaned if line]


def generate_multi_queries(question: str, llm_generate: LLMGenerate, num_variants: int = 3) -> List[str]:
    """
    Returns [question] + up to num_variants alternative phrasings. The
    original question is always included and always first, so a caller
    that doesn't care about multi-query still gets sane, safe behavior.
    Falls back to just [question] if the LLM's output can't be parsed into
    any variants.
    """
    prompt = MULTI_QUERY_PROMPT_TEMPLATE.format(question=question, num_variants=num_variants)
    variants = _parse_lines(llm_generate(prompt))

    if not variants:
        logger.info("Multi-query generation produced no usable variants -- falling back to the original question only.")
        return [question]

    # De-duplicate while preserving order; original question always first.
    seen = {question.strip().lower()}
    result = [question]
    for variant in variants:
        key = variant.lower()
        if key not in seen:
            seen.add(key)
            result.append(variant)
    return result
