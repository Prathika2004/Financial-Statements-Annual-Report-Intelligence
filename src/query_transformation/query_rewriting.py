"""
Single-query transforms: rewriting/expansion and step-back generalization.

Save as: sec-rag-project/src/query_transformation/query_rewriting.py

Each function takes one user question and an llm_generate callable (see
llm_interface.py) and returns ONE resulting query string -- use it on its
own, or combine it with the original via multi_query.py.
"""

import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.query_transformation.llm_interface import LLMGenerate

logger = logging.getLogger(__name__)

REWRITE_PROMPT_TEMPLATE = (
    "Rewrite the following question to be clearer and more complete, "
    "expanding any abbreviations, while preserving its original meaning. "
    "Return only the rewritten question, nothing else.\n\nQuestion: {question}"
)

STEP_BACK_PROMPT_TEMPLATE = (
    "Given a specific question, write a more general question that steps back "
    "from its specific details (numbers, dates, quarters, fiscal years) to ask "
    "about the broader trend or topic instead. This will be used to retrieve "
    "broader background context alongside the specific answer.\n\n"
    "Example:\n"
    "Specific question: What was Apple's Q3 2023 revenue?\n"
    "General question: How has Apple's revenue trended over time?\n\n"
    "Now write the general question for this one. Return only the "
    "generalized question, nothing else.\n\n"
    "Specific question: {question}"
)


def rewrite_query(question: str, llm_generate: LLMGenerate) -> str:
    """Clarifies/expands an ambiguous or abbreviated question, e.g.
    "Explain AI" -> "Explain Artificial Intelligence". Falls back to the
    original question if the LLM returns nothing usable."""
    result = llm_generate(REWRITE_PROMPT_TEMPLATE.format(question=question)).strip()
    return result or question


def step_back_query(question: str, llm_generate: LLMGenerate) -> str:
    """Generalizes a specific question to capture broader context (e.g. a
    question about one fiscal year's numbers steps back to "how has this
    metric trended over time"). Falls back to the original question if the
    LLM returns nothing usable."""
    result = llm_generate(STEP_BACK_PROMPT_TEMPLATE.format(question=question)).strip()
    return result or question
