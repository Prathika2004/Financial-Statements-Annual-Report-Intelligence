"""
LLM calling layer: talks to Ollama, serving Qwen2.5 7B locally.

Save as: sec-rag-project/src/generation/llm_router.py

--- Why this fulfills query_transformation's LLMGenerate contract ---
src/query_transformation/llm_interface.py defines LLMGenerate =
Callable[[str], str] specifically so query transformation could be built
before an LLM provider was chosen (see that module's docstring). generate()
below is that real implementation: a plain "prompt in, text out" function.
Anywhere in query_transformation that took `llm_generate` as a parameter
can now be called with `generate` (or `functools.partial(generate,
model=...)` for a non-default model) with no other code changes.

--- Why raw HTTP instead of the `ollama` PyPI package ---
`requests` is already a project dependency (used by sec_registry.py for
EDGAR's API) and Ollama's REST API is small and stable -- adding another
client library for two endpoints isn't worth it.

--- Why /api/chat, not /api/generate ---
/api/chat takes a system message and a user message as distinct roles,
matching Module 17's prompt structure (System Prompt -> Retrieved Context
-> User Question -> Instructions) directly, rather than requiring the
caller to concatenate everything into one raw prompt string and hope the
model's chat template still applies its instruction-tuning correctly.
prompt_templates.py builds exactly this {system, user} pair.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Iterator, List, Dict, Optional

import requests

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.settings import OLLAMA_URL, GENERATION_MODEL_NAME, GENERATION_TEMPERATURE

logger = logging.getLogger(__name__)

# A 7B model on CPU processing a full prompt (system instructions + several
# reranked source chunks, easily 1500-3000+ tokens) plus generating a
# multi-paragraph answer measurably exceeds a "short interactive request"
# budget -- 120s was tried first and timed out on a real query. This is a
# read timeout on an already-established connection, not a page-load wait.
REQUEST_TIMEOUT_SECONDS = 300


def is_ollama_running(url: str = OLLAMA_URL) -> bool:
    try:
        requests.get(url, timeout=3)
        return True
    except requests.exceptions.ConnectionError:
        return False


def chat(messages: List[Dict[str, str]], model: str = GENERATION_MODEL_NAME,
         temperature: float = GENERATION_TEMPERATURE, url: str = OLLAMA_URL) -> str:
    """
    Sends a list of {"role": "system"|"user"|"assistant", "content": ...}
    messages to Ollama's /api/chat and returns the model's reply text.
    """
    response = requests.post(
        f"{url}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def chat_stream(messages: List[Dict[str, str]], model: str = GENERATION_MODEL_NAME,
                 temperature: float = GENERATION_TEMPERATURE, url: str = OLLAMA_URL) -> Iterator[str]:
    """
    Same request as chat(), but yields each incremental content piece as it
    arrives from Ollama instead of waiting for and returning the full reply.
    Ollama's streaming response is newline-delimited JSON: every line is a
    {"message": {"content": "..."}, "done": bool, ...} object; "done": true
    marks the final line (its own content is always empty, plus timing
    stats this function doesn't need). Confirmed empirically against a real
    running Ollama instance -- this shape isn't just assumed from docs.
    """
    response = requests.post(
        f"{url}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature},
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
        stream=True,
    )
    response.raise_for_status()

    for line in response.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        content = chunk.get("message", {}).get("content", "")
        if content:
            yield content


def generate(prompt: str, system: Optional[str] = None, model: str = GENERATION_MODEL_NAME,
             temperature: float = GENERATION_TEMPERATURE, url: str = OLLAMA_URL) -> str:
    """
    LLMGenerate-compatible entry point (see module docstring): a single
    prompt string in, a single response string out. `system`, if given, is
    sent as a separate system-role message rather than concatenated into
    the prompt.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return chat(messages, model=model, temperature=temperature, url=url)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not is_ollama_running():
        raise SystemExit(f"Ollama isn't reachable at {OLLAMA_URL} -- is it running?")

    reply = generate("Reply with exactly the word: OK")
    print(f"Model replied: {reply!r}")
