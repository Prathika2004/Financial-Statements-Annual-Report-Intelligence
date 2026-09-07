"""
Output guardrails: citation enforcement on the generated answer.

Save as: sec-rag-project/src/guardrails/output_guardrails.py

--- Why this exists alongside numeric_verifier.py, not instead of it ---
numeric_verifier.py checks whether NUMBERS in the answer are grounded in
the sources. This checks a different, structural thing: does the answer
cite its sources at all (per SYSTEM_PROMPT rule 2), and are the citations
it uses valid -- referencing a source that was actually retrieved, not
"[Source 8]" when only 5 sources exist. A model fabricating a citation
number is exactly as much a grounding failure as fabricating a number;
both are "does this claim trace back to real retrieved text" checks, just
for different failure modes.
"""

import re
import sys
from pathlib import Path
from typing import Dict

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.generation.prompt_templates import is_refusal_answer

CITATION_PATTERN = re.compile(r"\[Source (\d+)")


def check_citations(answer: str, num_sources: int) -> Dict:
    """
    Extracts every "[Source N]" reference in the answer and checks:
      - at least one citation exists, UNLESS the answer is a refusal (a
        correct "I don't know" cites nothing, and must not be flagged)
      - every cited N is in range [1, num_sources] -- a reference to a
        source number that was never given is a fabricated citation.
    """
    cited_numbers = [int(n) for n in CITATION_PATTERN.findall(answer)]
    out_of_range = sorted({n for n in cited_numbers if n < 1 or n > num_sources})

    missing_citation = not cited_numbers and num_sources > 0 and not is_refusal_answer(answer)

    return {
        "cited_sources": sorted(set(cited_numbers)),
        "out_of_range_citations": out_of_range,
        "missing_citation": missing_citation,
    }
