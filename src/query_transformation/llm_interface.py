"""
The swappable LLM-calling interface every query-transformation technique
depends on.

Save as: sec-rag-project/src/query_transformation/llm_interface.py

--- Why this exists as its own tiny module ---
Every technique in this package (query rewriting/expansion/step-back,
multi-query, decomposition, HyDE) needs to ask an LLM to produce text from
a prompt -- but no LLM provider has been chosen yet (that's Stage 9's job,
src/generation/llm_router.py, currently unbuilt). Rather than hardcode a
specific provider here and rewrite this whole package later, every function
in this package takes an `llm_generate: LLMGenerate` parameter -- a plain
"prompt in, text out" callable -- instead of importing a provider
directly. When Stage 9 exists, llm_router.py should expose a function
matching this exact signature (e.g. `generate(prompt: str) -> str`,
wrapping whichever provider/model gets chosen), and every call site in this
package changes from a test double to the real thing with zero changes to
the transformation logic itself.

FakeLLM below is a canned-response stand-in used by this package's own
tests -- it is not a "cheap default provider". It exists purely so
multi_query.py / decomposition.py / hyde.py's parsing and control-flow
logic can be verified without a real model loaded.
"""

from typing import Callable, Dict, List, Optional

# A real implementation (Stage 9) takes a prompt string and returns the
# model's raw text response.
LLMGenerate = Callable[[str], str]


class FakeLLM:
    """
    A deterministic stand-in for a real llm_generate function, used only in
    this package's tests. Returns a canned response for an exact prompt
    match, or a caller-supplied default otherwise -- so a test can assert
    exactly what prompt each technique sent to the LLM, without needing a
    real model loaded.
    """

    def __init__(self, responses: Optional[Dict[str, str]] = None, default: str = ""):
        self.responses = responses or {}
        self.default = default
        self.calls: List[str] = []  # every prompt this was called with, in order

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.responses.get(prompt, self.default)
