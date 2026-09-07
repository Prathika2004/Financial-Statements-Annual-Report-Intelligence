"""
Pydantic request/response models for the API layer.

Save as: sec-rag-project/src/api/schemas.py
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

Technique = Literal["none", "rewrite", "step-back", "multi-query", "decompose", "hyde"]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to ask about the SEC filings.")
    ticker: Optional[str] = Field(None, description="Restrict retrieval to this ticker, e.g. 'TSLA'.")
    fiscal_year: Optional[int] = Field(None, description="Restrict retrieval to this fiscal year, e.g. 2023.")
    chunk_type: Optional[str] = Field(None, pattern="^(text|table)$",
                                       description="Restrict retrieval to 'text' or 'table' chunks.")
    technique: Technique = Field(
        "none",
        description=(
            "Optional query-transformation technique to apply before retrieval. "
            "Each one adds an extra LLM call before the answer-generation call, "
            "roughly doubling response time -- see src/query_transformation/ for what each does."
        ),
    )


class SourceOut(BaseModel):
    """
    Only the fields an API consumer actually needs to display/cite a
    source. retrieve_and_rerank() returns chunk dicts with several more
    internal fields (rrf_score, found_by, parent_section_id, etc.) --
    Pydantic silently drops fields not declared here when constructing
    from that dict, so this doubles as the API's public contract.
    """
    chunk_id: str
    ticker: Optional[str] = None
    fiscal_year: Optional[int] = None
    company_name: Optional[str] = None
    section_item: Optional[str] = None
    chunk_type: Optional[str] = None
    score: float
    text: str


class NumericCheckOut(BaseModel):
    checked: List[str]
    unverified: List[str]


class CitationCheckOut(BaseModel):
    cited_sources: List[int]
    out_of_range_citations: List[int]
    missing_citation: bool


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceOut]
    numeric_check: NumericCheckOut
    citation_check: Optional[CitationCheckOut] = None
    cached: bool = False
    blocked: bool = False
    low_confidence: bool = False
