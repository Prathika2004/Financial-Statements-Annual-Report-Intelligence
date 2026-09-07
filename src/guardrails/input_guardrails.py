"""
Input guardrails: prompt-injection heuristics and basic PII detection on
the user's question, run before it ever reaches retrieval or generation.

Save as: sec-rag-project/src/guardrails/input_guardrails.py

--- Why heuristic pattern matching, not a trained classifier or an LLM call ---
Frameworks like NeMo Guardrails or Lakera use ML classifiers or an LLM call
for this -- either adds a real cost here: another LLM round-trip before the
already-slow generation call even starts, or a new heavy dependency. For a
single-user local tool where the actual threat model is "did something
weird get pasted in," not "an adversary is red-teaming this API," a fast,
transparent pattern check catches the obvious cases without either cost.

--- Be honest about what this does and doesn't do ---
Pattern matching catches known phrasings ("ignore previous instructions")
but not paraphrases or obfuscated variants -- a determined attacker gets
past this easily. This is a first, cheap check, not a security boundary.
This project's actual security posture (see config/settings.py's API_KEY
discussion) is "runs on localhost, for one user" -- these checks exist for
defense-in-depth and because Module 21 was explicitly asked for, not
because a hostile attacker is the realistic threat here.
"""

import re
import sys
from pathlib import Path
from typing import Dict, List

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

INJECTION_PATTERNS = [
    r"ignore (all |the )?(previous|prior|above) instructions",
    r"disregard (all |the )?(previous|prior|above) (instructions|rules)",
    r"reveal (your |the )?(system prompt|instructions)",
    r"what (is|are) your (system prompt|instructions)",
    r"forget (everything|all) (you|that)",
    r"new instructions?:",
    r"override (your |the )?(rules|instructions|guidelines)",
    r"jailbreak",
    r"do anything now",
    r"\bdan\b.{0,20}\bmode\b",
]
_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

# Flags PII in the QUESTION itself -- this financial-filings dataset never
# needs personal data to answer anything, so its presence is always
# incidental, not something the pipeline should echo back unnecessarily.
PII_PATTERNS = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b"),
}


def check_prompt_injection(text: str) -> List[str]:
    """Returns the matched injection-pattern strings (empty if none)."""
    return [p.pattern for p in _INJECTION_RE if p.search(text)]


def check_pii(text: str) -> List[str]:
    """Returns which PII categories were detected (empty if none)."""
    return [name for name, pattern in PII_PATTERNS.items() if pattern.search(text)]


def screen_input(question: str) -> Dict:
    """
    Runs all input guardrails against a user's question.

    "blocked" is True only for a likely prompt-injection attempt -- PII in
    the question is flagged but does NOT block the request: a user asking
    "what's Tesla's revenue, my email is x@y.com" isn't an attack, just
    noise worth logging, not refusing.
    """
    injection_matches = check_prompt_injection(question)
    pii_matches = check_pii(question)

    reasons = []
    if injection_matches:
        reasons.append(f"possible prompt injection ({len(injection_matches)} pattern match(es))")

    return {
        "blocked": bool(injection_matches),
        "reasons": reasons,
        "pii_detected": pii_matches,
    }
