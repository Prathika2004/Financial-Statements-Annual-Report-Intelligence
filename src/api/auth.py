"""
Optional API key check (Module 22).

Save as: sec-rag-project/src/api/auth.py

Disabled by default -- see config/settings.py's API_KEY discussion for why
full auth/RBAC infrastructure doesn't fit a localhost-only, single-user
tool. Set the SEC_RAG_API_KEY environment variable to require an
"Authorization: Bearer <key>" header on the expensive endpoints (/ask,
/ask/stream) -- the moment this server is reachable from anywhere but
localhost, this is the one gate that should exist.
"""

import sys
from pathlib import Path

from fastapi import Header, HTTPException

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.settings import API_KEY


async def require_api_key(authorization: str = Header(default=None)):
    if API_KEY is None:
        return  # auth disabled -- the default
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
