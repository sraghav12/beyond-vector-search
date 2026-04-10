#!/usr/bin/env python3
"""
fix_query_doc_ids.py
====================
Aligns evidence_docs in queries.json with the actual doc_ids in the corpus.

The fiscal year heuristic during fetching may assign different years than
what the queries expect. This script reads the corpus, builds a mapping
of available doc_ids per ticker, and updates queries.json to reference
the correct IDs.

Usage:
    python scripts/fix_query_doc_ids.py
    python scripts/fix_query_doc_ids.py --dry-run   # preview changes without writing
"""

import json
import re
import sys
import argparse
from pathlib import Path
from collections import defaultdict


def load_corpus_doc_ids(corpus_dir: str) -> dict[str, list[int]]:
    """
    Load all doc_ids from the largest corpus file.
    Returns: dict mapping ticker -> sorted list of fiscal years (descending)
    """
    corpus_path = Path(corpus_dir)
    corpus_files = list(corpus_path.glob("corpus_*.jsonl"))

    def extract_scale(p: Path) -> int:
        match = re.search(r"corpus_(\d+)", p.stem)
        return int(match.group(1)) if match else 0

    corpus_files.sort(key=extract_scale, reverse=True)

    if not corpus_files:
        print(f"ERROR: No corpus files found in {corpus_dir}")
        sys.exit(1)

    largest = corpus_files[0]
    print(f"Reading doc_ids from: {largest}")

    ticker_years = defaultdict(list)
    with open(largest) as f:
        for line in f:
            doc = json.loads(line)
            doc_id = doc["doc_id"]
            # Parse: TICKER_10K_YEAR
            match = re.match(r"(.+)_10K_(\d{4})", doc_id)
            if match:
                ticker = match.group(1)
                year = int(match.group(2))
                ticker_years[ticker].append(year)

    # Sort years descending for each ticker
    for ticker in ticker_years:
        ticker_years[ticker] = sorted(ticker_years[ticker], reverse=True)

    return dict(ticker_years)


def fix_doc_id(doc_id: str, ticker_years: dict[str, list[int]], year_shift: int) -> str:
    """
    Fix a single doc_id by shifting the year.
    If the shifted year exists in the corpus, use it. Otherwise keep original.
    """
    if doc_id in ("ALL", "ALL_PAIRS"):
        return doc_id

    match = re.match(r"(.+)_10K_(\d{4})", doc_id)
    if not match:
        return doc_id

    ticker = match.group(1)
    old_year = int(match.group(2))
    new_year = old_year + year_shift

    available = ticker_years.get(ticker, [])
    if new_year in available:
        return f"{ticker}_10K_{new_year}"
    elif old_year in available:
        return doc_id  # Original is fine
    else:
        # Neither works — use the most recent available year
        if available:
            return f"{ticker}_10K_{available[0]}"
        return doc_id


def main():
    parser = argparse.ArgumentParser(description="Fix query doc IDs to match corpus")
    parser.add_argument("--queries", default="data/queries/queries.json")
    parser.add_argument("--corpus-dir", default="data/processed")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()

    # Load corpus doc IDs
    ticker_years = load_corpus_doc_ids(args.corpus_dir)

    print(f"\nCorpus contains {sum(len(v) for v in ticker_years.values())} documents "
          f"across {len(ticker_years)} tickers")
    print("\nMost recent fiscal year per ticker:")
    for ticker in sorted(ticker_years):
        years = ticker_years[ticker]
        print(f"  {ticker:6s}: {years}")

    # Determine the year shift
    # Check what the most common "most recent year" is
    most_recent_years = [years[0] for years in ticker_years.values() if years]
    most_common_recent = max(set(most_recent_years), key=most_recent_years.count)
    print(f"\nMost common 'most recent' year in corpus: {most_common_recent}")

    # The queries were written assuming most recent = 2024
    # If corpus has 2025, shift = +1
    year_shift = most_common_recent - 2024
    print(f"Year shift needed: {year_shift:+d}")

    if year_shift == 0:
        print("No shift needed — queries already aligned!")
        return

    # Load queries
    with open(args.queries) as f:
        queries = json.load(f)

    # Fix each query
    changes = []
    for q in queries:
        old_docs = q["evidence_docs"]
        new_docs = [fix_doc_id(d, ticker_years, year_shift) for d in old_docs]

        if old_docs != new_docs:
            changes.append({
                "id": q["id"],
                "old": old_docs,
                "new": new_docs,
            })
            q["evidence_docs"] = new_docs

    print(f"\n{'='*60}")
    print(f"Changes: {len(changes)} queries updated out of {len(queries)}")
    print(f"{'='*60}")

    for c in changes[:20]:  # Show first 20
        print(f"  {c['id']}: {c['old']} → {c['new']}")
    if len(changes) > 20:
        print(f"  ... and {len(changes) - 20} more")

    # Write updated queries
    if not args.dry_run and changes:
        with open(args.queries, "w") as f:
            json.dump(queries, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Updated {args.queries}")
    elif args.dry_run:
        print(f"\n(Dry run — no files written)")
    else:
        print(f"\nNo changes needed.")


if __name__ == "__main__":
    main()