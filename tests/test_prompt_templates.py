"""
Tests for Stage 9 prompt assembly.

Run with: pytest tests/test_prompt_templates.py -v
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.generation.prompt_templates import (
    format_source_label, build_context_block, build_user_message, assemble_messages, SYSTEM_PROMPT,
    is_refusal_answer,
)


class TestRefusalDetection:

    def test_recognizes_common_refusal_phrasings(self):
        assert is_refusal_answer("This question cannot be answered from the provided sources.")
        assert is_refusal_answer("The provided filing excerpts do not contain this information.")
        assert is_refusal_answer("I don't know based on the given context.")

    def test_does_not_flag_a_normal_confident_answer(self):
        assert not is_refusal_answer("Tesla's net income in fiscal year 2023 was $15.00 billion.")

    def test_case_insensitive(self):
        assert is_refusal_answer("CANNOT BE ANSWERED given the sources provided.")


def make_chunk(text, ticker="TSLA", fiscal_year=2023, section_item="Item 1A",
                chunk_type="text", company_name="Tesla, Inc."):
    return {
        "text": text, "ticker": ticker, "fiscal_year": fiscal_year,
        "section_item": section_item, "chunk_type": chunk_type, "company_name": company_name,
    }


def test_format_source_label_includes_identity_and_section():
    label = format_source_label(make_chunk("text"), 1)
    assert label == "[Source 1: Tesla, Inc. FY2023, Item 1A]"


def test_format_source_label_flags_table_chunks():
    label = format_source_label(make_chunk("text", chunk_type="table"), 1)
    assert "(table)" in label


def test_build_context_block_numbers_sources_in_order():
    chunks = [make_chunk("first chunk text"), make_chunk("second chunk text")]
    block = build_context_block(chunks)
    assert "[Source 1:" in block
    assert "[Source 2:" in block
    assert block.index("[Source 1:") < block.index("[Source 2:")
    assert "first chunk text" in block
    assert "second chunk text" in block


def test_build_context_block_handles_no_sources():
    assert "No sources were retrieved" in build_user_message("a question", [])


def test_build_user_message_includes_question_and_context():
    chunks = [make_chunk("relevant text")]
    message = build_user_message("What was revenue?", chunks)
    assert "What was revenue?" in message
    assert "relevant text" in message


def test_assemble_messages_returns_system_then_user_role():
    chunks = [make_chunk("relevant text")]
    messages = assemble_messages("What was revenue?", chunks)
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[1]["role"] == "user"
    assert "What was revenue?" in messages[1]["content"]


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
