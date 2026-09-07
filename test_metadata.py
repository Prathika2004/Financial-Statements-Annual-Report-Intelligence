"""Test script for metadata extraction"""
import sys
import os

os.chdir(r"C:\Users\Prathika\Documents\SEC\sec-rag-project")
sys.path.insert(0, ".")

from src.ingestion.metadata_extractor import extract_all_metadata, extract_ticker, extract_sections, extract_fiscal_year, extract_company_name, extract_cik, extract_filing_date, extract_reporting_period
from src.ingestion.parser import parse_pdf
from src.cleaning.cleaner import clean_text

# Test 1: Parse Amazon 2022 PDF and extract metadata
print("=" * 60)
print("TEST 1: Amazon 2022 PDF")
print("=" * 60)

result = parse_pdf("data/raw_pdfs/sec_pdfs/amazon/amzn_10K_2022.pdf")
print(f"Parser used: {result['parser_used']}")
print(f"Text length: {len(result['text']):,} chars")

cleaned = clean_text(result['text'])
print(f"Cleaned length: {len(cleaned):,} chars")

meta = extract_all_metadata(cleaned, filename="amzn_10K_2022.pdf", folder_hint="amazon")

print()
print("=== METADATA RESULTS ===")
print(f"Filename: {meta.filename}")
print(f"Company name: {meta.company_name}")
print(f"Ticker: {meta.ticker}")
print(f"CIK: {meta.cik}")
print(f"Filing type: {meta.filing_type}")
print(f"Filing date: {meta.filing_date}")
print(f"Reporting period end: {meta.reporting_period_end}")
print(f"Fiscal year: {meta.fiscal_year}")
print(f"Sections found: {len(meta.sections)}")

for s in meta.sections[:5]:
    print(f"  Item {s.item}: {s.title} ({s.start_position}-{s.end_position}, {s.text_char_count} chars)")

print(f"Status: {meta.status}")
print(f"Parser used: {meta.parser_used}")

# Test 2: Check ticker extraction separately
print()
print("=" * 60)
print("TEST 2: Ticker extraction test")
print("=" * 60)

ticker = extract_ticker(cleaned)
print(f"Ticker extracted: {ticker}")

# Test 3: Section detection
print()
print("=" * 60)
print("TEST 3: Section detection test")
print("=" * 60)

sections = extract_sections(cleaned)
print(f"Sections found: {len(sections)}")
for s in sections[:5]:
    print(f"  Item {s.item}: {s.title} ({s.start_position}-{s.end_position}, {s.text_char_count} chars)")

# Test 4: Fiscal year
print()
print("=" * 60)
print("TEST 4: Fiscal year extraction")
print("=" * 60)

fy = extract_fiscal_year(cleaned)
print(f"Fiscal year: {fy}")

# Test 5: Company name
print()
print("=" * 60)
print("TEST 5: Company name extraction")
print("=" * 60)

cn = extract_company_name(cleaned)
print(f"Company name: {cn}")

# Test 6: CIK
print()
print("=" * 60)
print("TEST 6: CIK extraction")
print("=" * 60)

cik = extract_cik(cleaned)
print(f"CIK: {cik}")

# Test 7: Filing date
print()
print("=" * 60)
print("TEST 7: Filing date extraction")
print("=" * 60)

fd = extract_filing_date(cleaned)
print(f"Filing date: {fd}")

# Test 8: Reporting period
print()
print("=" * 60)
print("TEST 8: Reporting period extraction")
print("=" * 60)

rp = extract_reporting_period(cleaned)
print(f"Reporting period end: {rp}")

print()
print("All tests completed!")