"""
Numeric grounding check: flags dollar amounts, percentages, and other
figures in a generated answer that don't appear anywhere in the source
chunks it was generated from.

Save as: sec-rag-project/src/generation/numeric_verifier.py

--- What this catches, and what it deliberately doesn't ---
This is a presence check, not a correctness check: it confirms a number
extracted from the answer also appears, in the same normalized form,
somewhere among the numbers extracted from the source chunks -- not that
the model used it in a way that's actually correct for the question asked.
It will not catch a real source number misattributed to the wrong
company/year, and it won't recognize "$1.2 billion" and "$1,200,000,000" as
equivalent -- true unit normalization is out of scope here. What it DOES
reliably catch is the most damaging failure mode for a financial RAG
system: the model inventing a figure with zero grounding anywhere in the
retrieved context. A flagged number is a signal for a human (or a stricter
caller) to double-check -- not proof the answer is wrong. Some flags will
be false positives (e.g. a total the model correctly computed by adding two
sourced figures together).

--- Why numbers are compared as a set of extracted tokens, not substring
search over a blob of concatenated source text ---
Concatenating all source text into one string and stripping whitespace to
check "does this number appear as a substring" risks accidental collisions
across sentence/word boundaries once whitespace is gone (e.g. "...grew 12.
The next year..." and "...5%..." could spuriously combine into something a
naive substring check treats as containing "12.5"). Extracting distinct
number tokens from both the answer and the sources independently, then
checking exact membership in that token set, avoids this entirely.
"""

import re
import sys
from pathlib import Path
from typing import Dict, List

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

NUMBER_PATTERN = re.compile(
    r"\$?\d[\d,]*(?:\.\d+)?\s?(?:billion|million|thousand|%)?",
    re.IGNORECASE,
)

# Bare small integers are almost never a meaningful financial figure worth
# verifying (item numbers, list counts, single-digit counts) and would
# otherwise flood the report with noise.
MIN_DIGITS_TO_CHECK = 3


def _normalize(number_text: str) -> str:
    return re.sub(r"[,\s]", "", number_text).lower()


def extract_numbers(text: str) -> List[str]:
    """Returns every number-like substring in text with at least
    MIN_DIGITS_TO_CHECK digits, normalized (commas/whitespace stripped,
    lowercased) so "1,234" and "1234" -- or "1.2 billion" and "1.2billion"
    -- compare equal."""
    result = []
    for candidate in NUMBER_PATTERN.findall(text):
        if sum(c.isdigit() for c in candidate) >= MIN_DIGITS_TO_CHECK:
            result.append(_normalize(candidate))
    return result


def verify_numbers(answer: str, source_chunks: List[Dict]) -> Dict[str, List[str]]:
    """
    Extracts numeric claims from `answer` and checks each one for exact
    membership in the set of numbers extracted from `source_chunks`' text.

    Returns {"checked": [...], "unverified": [...]} -- "checked" is every
    distinct number found in the answer, "unverified" is the subset that
    doesn't appear anywhere in the sources.
    """
    source_numbers = set()
    for chunk in source_chunks:
        source_numbers.update(extract_numbers(chunk["text"]))

    answer_numbers = extract_numbers(answer)
    unverified = [n for n in answer_numbers if n not in source_numbers]

    return {
        "checked": sorted(set(answer_numbers)),
        "unverified": sorted(set(unverified)),
    }
