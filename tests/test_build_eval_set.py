"""
Tests that the eval set is structurally well-formed. Does not check the
reference answers' factual accuracy against source documents -- that was
verified manually against real chunks when each question was written (see
build_eval_set.py's comments); this just guards against the file being
malformed or missing a required field.

Run with: pytest tests/test_build_eval_set.py -v
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from eval.build_eval_set import EVAL_QUESTIONS


def test_every_question_has_required_fields():
    for q in EVAL_QUESTIONS:
        assert q["id"]
        assert q["question"]
        assert q["reference"]
        assert "filters" in q  # explicit None is fine, missing key is not


def test_question_ids_are_unique():
    ids = [q["id"] for q in EVAL_QUESTIONS]
    assert len(ids) == len(set(ids))


def test_at_least_one_refusal_case_exists():
    """The eval set must be able to measure refusal behavior, not just
    correct-answer cases -- see build_eval_set.py's module docstring."""
    assert any(q.get("expect_refusal") for q in EVAL_QUESTIONS)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
