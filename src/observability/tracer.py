"""
Structured request tracing: one JSON line per /ask(/stream) request,
capturing timing and outcome in one place.

Save as: sec-rag-project/src/observability/tracer.py

--- Why JSONL to a local file, not LangSmith/Phoenix/OpenTelemetry ---
Those are built for a team debugging a deployed, multi-user service with a
hosted dashboard to look at. This is a single-user local tool -- a plain
JSON-lines file that's grep/jq-able covers the actual need (was this
request slow, what did it retrieve, did it hit cache, did a guardrail
fire) without a new service dependency or an account to sign up for.
Nothing here precludes wiring in a real tracing backend later if this ever
becomes a multi-user deployment.
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.settings import TRACE_LOG_PATH


def log_trace(record: Dict, path: Path = TRACE_LOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    full_record = {"timestamp": time.time(), **record}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(full_record, ensure_ascii=False) + "\n")
