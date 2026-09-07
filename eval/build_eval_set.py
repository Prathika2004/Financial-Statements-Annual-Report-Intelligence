"""
Builds the evaluation question set for RAGAS scoring.

Save as: sec-rag-project/eval/build_eval_set.py

--- Why these questions are hand-curated from verified source text, not
generated ---
RAGAS's reference-based metrics (Context Precision, Context Recall) compare
retrieval/generation against a "ground truth" answer -- a wrong reference
would silently corrupt every score computed against it. Every reference
answer below was pulled directly from a real chunk's text (grep'd out of
the actual processed filings, not written from memory) before being
phrased as a question. See each entry's "source" comment for where to
re-verify it if the underlying data ever changes.

--- Why one question is deliberately unanswerable ---
A RAG system that always produces a confident answer regardless of whether
the retrieved context supports it is the main hallucination risk (see
prompt_templates.py's system prompt). An eval set that only contains
answerable questions can't measure whether the "say I don't know" behavior
actually works -- Netflix is not one of the 10 companies in this dataset,
so this question has no correct answer to retrieve, on purpose.

Usage:
    python eval/build_eval_set.py
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import EVAL_SET_DIR

EVAL_QUESTIONS = [
    {
        "id": "tesla_net_income_fy2023",
        "question": "What was Tesla's net income attributable to common stockholders in fiscal year 2023?",
        "reference": "Tesla's net income attributable to common stockholders in fiscal year 2023 was $15.00 billion, "
                      "a favorable change of $2.44 billion compared to the prior year.",
        "filters": {"ticker": "TSLA", "fiscal_year": 2023},
        # source: data/processed/tesla__tesla_10K_2024/chunks.json, Item 7
    },
    {
        "id": "tesla_vehicle_deliveries_fy2023",
        "question": "How many consumer vehicles did Tesla produce and deliver in 2023?",
        "reference": "In 2023, Tesla produced 1,845,985 consumer vehicles and delivered 1,808,581 consumer vehicles.",
        "filters": {"ticker": "TSLA", "fiscal_year": 2023},
        # source: data/processed/tesla__tesla_10K_2024/chunks.json, Item 7
    },
    {
        "id": "tesla_automotive_sales_growth_fy2023",
        "question": "How much did Tesla's automotive sales revenue increase in 2023 compared to 2022?",
        "reference": "Tesla's automotive sales revenue increased $11.30 billion, or 17%, in the year ended "
                      "December 31, 2023 compared to the year ended December 31, 2022, primarily due to an "
                      "increase in combined Model 3/Model Y deliveries.",
        "filters": {"ticker": "TSLA", "fiscal_year": 2023},
        # source: data/processed/tesla__tesla_10K_2024/chunks.json, Item 7
    },
    {
        "id": "jnj_worldwide_sales_fy2023",
        "question": "How much did Johnson & Johnson's worldwide sales grow in 2023?",
        "reference": "Johnson & Johnson's worldwide sales increased 6.5% to $85.2 billion in 2023, compared to a "
                      "1.6% increase in 2022.",
        "filters": {"ticker": "JNJ", "fiscal_year": 2023},
        # source: data/processed/johnson__jnj_10K_2024/chunks.json, Item 7
    },
    {
        "id": "cybersecurity_risk_factors_general",
        "question": "What are the main risk factors related to cybersecurity that companies describe in their 10-K filings?",
        "reference": "Companies describe cybersecurity risks including unauthorized access, data theft, computer "
                      "viruses, ransomware, and other security incidents that could disrupt operations, lead to "
                      "data breaches, harm reputation, and result in legal liability or regulatory penalties. "
                      "Risks also stem from the complexity and geographic breadth of the systems that need to be "
                      "defended, including against state-sponsored threat actors.",
        "filters": None,
        # source: verified during Stage 8 testing against KO, TSLA, INTC Item 1C sections
    },
    {
        "id": "visa_cybersecurity_approach",
        "question": "What is Visa's approach to cybersecurity?",
        "reference": "Visa states that as a global company providing payment services to consumers and companies "
                      "around the world, trust is indispensable, and describes a cybersecurity program addressing "
                      "the risks inherent to operating global payment systems.",
        "filters": {"ticker": "V"},
        # source: data/processed/visa__v_10K_2024/chunks.json, Item 1C
    },
    {
        "id": "unanswerable_out_of_dataset_company",
        "question": "What was Netflix's revenue in fiscal year 2023?",
        "reference": "This question cannot be answered from the provided sources -- Netflix's 10-K filings are "
                      "not part of this dataset (the dataset covers Amazon, Apple, Berkshire Hathaway, Coca-Cola, "
                      "Intel, Johnson & Johnson, Microsoft, Nvidia, Tesla, and Visa only).",
        "filters": None,
        "expect_refusal": True,
        # deliberately unanswerable -- tests whether the system says so rather than hallucinating
    },
    {
        "id": "unanswerable_wrong_fiscal_year",
        "question": "What was Tesla's revenue in fiscal year 2030?",
        "reference": "This question cannot be answered from the provided sources -- fiscal year 2030 has not "
                      "occurred yet and no such filing exists in this dataset.",
        "filters": {"ticker": "TSLA", "fiscal_year": 2030},
        "expect_refusal": True,
        # deliberately unanswerable (and deliberately filtered to a fiscal_year with zero matching chunks)
    },
]


def build_eval_set(output_dir: Path = EVAL_SET_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "eval_questions.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(EVAL_QUESTIONS, f, indent=2, ensure_ascii=False)
    return output_path


if __name__ == "__main__":
    path = build_eval_set()
    print(f"Wrote {len(EVAL_QUESTIONS)} eval question(s) to {path}")
