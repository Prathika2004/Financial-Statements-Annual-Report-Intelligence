from src.chunking.section_chunker import chunk_document, chunk_section
from src.chunking.table_chunker import _split_table_by_rows


def test_table_stays_between_surrounding_prose():
    section = {
        "item": "Item 8",
        "title": "Financial Statements",
        "text": (
            "Opening discussion before the table.\n\n"
            "| Year | Revenue |\n| --- | --- |\n| 2024 | 100 |\n\n"
            "Closing discussion after the table."
        ),
    }

    chunks = chunk_section(section, "Example Corp", 2024)

    assert [chunk["chunk_type"] for chunk in chunks] == ["text", "table", "text"]
    assert "Opening discussion" in chunks[0]["text"]
    assert "| 2024 | 100 |" in chunks[1]["text"]
    assert "Closing discussion" in chunks[2]["text"]


def test_chunks_include_document_scoped_id_and_parent_reference():
    metadata = {
        "document": {
            "filename": "example_10K_2024.pdf",
            "ticker": "EXM",
            "fiscal_year": 2024,
            "company_name": "Example Corp",
        },
        "sections": [
            {
                "item": "Item 1",
                "title": "Business",
                "start_position": 0,
                "end_position": 31,
            }
        ],
    }

    chunks = chunk_document(metadata, "Item 1. Business\nExample content.")

    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == "EXM_2024_example_10K_2024_0001"
    assert chunks[0]["parent_section_id"] == "example_10k_2024_pdf_item_1"
    assert chunks[0]["parent_section_start"] == 0
    assert chunks[0]["parent_section_end"] == 31


def test_oversized_table_is_split_at_row_boundaries_with_repeated_header():
    table = "| Year | Revenue |\n| --- | --- |\n" + "\n".join(
        f"| {year} | {'x' * 40} |" for year in range(2000, 2030)
    )

    parts = _split_table_by_rows(table, max_chars=300)

    assert len(parts) > 1
    assert all(part.startswith("| Year | Revenue |\n| --- | --- |") for part in parts)
    assert all(len(part) <= 300 for part in parts)
