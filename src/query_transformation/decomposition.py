"""
Query decomposition: breaks a compound question into independently
retrievable sub-questions.

Save as: sec-rag-project/src/query_transformation/decomposition.py

Like multi_query.py, this returns a List[str] that plugs directly into
retriever.retrieve()'s `queries` parameter -- but the generation strategy
differs: multi-query rephrases the SAME information need several ways,
while decomposition splits one question into genuinely DIFFERENT
sub-questions (e.g. "How did revenue and headcount both change in 2023?"
-> ["How did revenue change in 2023?", "How did headcount change in
2023?"]). A question with nothing to decompose comes back as a
single-item list containing the original question, not forced into fake
sub-questions.
"""

import logging
import re
import sys
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.query_transformation.llm_interface import LLMGenerate

logger = logging.getLogger(__name__)

DECOMPOSITION_PROMPT_TEMPLATE = (
    "If the following question asks about more than one distinct thing, "
    "break it into separate, independently answerable sub-questions -- one "
    "per line, no numbering. If it is already a single, simple question, "
    "return it unchanged on one line.\n\nQuestion: {question}"
)


def decompose_query(question: str, llm_generate: LLMGenerate) -> List[str]:
    """Returns one or more sub-questions. Always returns at least
    [question] -- never an empty list -- if the LLM output can't be parsed."""
    raw_output = llm_generate(DECOMPOSITION_PROMPT_TEMPLATE.format(question=question))

    lines = [line.strip() for line in raw_output.strip().split("\n")]
    cleaned = [re.sub(r"^\d+[\.\)]\s*|^[-*]\s*", "", line) for line in lines]
    sub_questions = [line for line in cleaned if line]

    if not sub_questions:
        logger.info("Decomposition produced no usable sub-questions -- falling back to the original question.")
        return [question]
    return sub_questions
