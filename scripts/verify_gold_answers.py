#!/usr/bin/env python3
"""
verify_gold_answers.py
======================
Helper script to verify and refine gold-standard answers against the actual corpus.

For each query, this script:
  1. Loads the relevant evidence documents from the corpus
  2. Searches for key financial terms and numbers
  3. Prints relevant snippets so you can verify/update the gold answer
  4. Optionally uses an LLM to auto-extract answers (requires API key)

Usage:
    # Show relevant snippets for all queries
    python scripts/verify_gold_answers.py

    # Verify a specific query
    python scripts/verify_gold_answers.py --query q001

    # Verify only tier 1 queries
    python scripts/verify_gold_answers.py --tier 1

    # Auto-extract answers using LLM (requires OPENAI_API_KEY)
    python scripts/verify_gold_answers.py --auto-extract --model gpt-4o-mini

    # Export verified answers
    python scripts/verify_gold_answers.py --export data/ground_truth/gold_answers.json
"""

import json
import re
import sys
import argparse
import logging
from pathlib import Path
from collections import defaultdict
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Financial keyword patterns for extraction
# ---------------------------------------------------------------------------

FINANCIAL_PATTERNS = {
    "revenue": [
        r"(?:total\s+)?(?:net\s+)?(?:revenue|sales)\s*[\$]?\s*([\d,\.]+)\s*(?:million|billion)?",
        r"(?:net\s+)?(?:revenue|sales)\s+(?:was|were|of)\s+\$\s*([\d,\.]+)\s*(?:million|billion)?",
    ],
    "net_income": [
        r"net\s+income\s*[\$]?\s*([\d,\.]+)\s*(?:million|billion)?",
        r"net\s+income\s+(?:was|of)\s+\$\s*([\d,\.]+)\s*(?:million|billion)?",
    ],
    "gross_margin": [
        r"gross\s+(?:margin|profit)\s*[\$]?\s*([\d,\.]+)\s*(?:million|billion|%|percent)?",
    ],
    "r_and_d": [
        r"research\s+and\s+development\s*[\$]?\s*([\d,\.]+)\s*(?:million|billion)?",
        r"R&D\s+(?:expense|spending)\s*[\$]?\s*([\d,\.]+)\s*(?:million|billion)?",
    ],
    "total_assets": [
        r"total\s+assets\s*[\$]?\s*([\d,\.]+)\s*(?:million|billion|trillion)?",
    ],
    "employees": [
        r"(?:approximately\s+)?([\d,]+)\s+(?:employees|associates|team members|full-time)",
    ],
    "operating_income": [
        r"operating\s+income\s*[\$]?\s*([\d,\.]+)\s*(?:million|billion)?",
    ],
    "dividends": [
        r"(?:dividends?\s+(?:paid|declared))\s*[\$]?\s*([\d,\.]+)\s*(?:million|billion)?",
        r"(?:per\s+share\s+)?dividend\s*[\$]?\s*([\d,\.]+)",
    ],
    "capex": [
        r"capital\s+expenditures?\s*[\$]?\s*([\d,\.]+)\s*(?:million|billion)?",
    ],
}


# ---------------------------------------------------------------------------
# Corpus Loader
# ---------------------------------------------------------------------------

def load_corpus_docs(corpus_dir: str, doc_ids: list[str]) -> dict[str, str]:
    """
    Load specific documents from the largest available corpus file.

    Args:
        corpus_dir: Path to data/processed/
        doc_ids: List of doc_id strings to load

    Returns:
        Dict mapping doc_id -> document text
    """
    corpus_path = Path(corpus_dir)

    # Find the largest corpus file (sort by the number in the filename)
    corpus_files = list(corpus_path.glob("corpus_*.jsonl"))
    if not corpus_files:
        logger.error(f"No corpus files found in {corpus_dir}")
        sys.exit(1)

    def extract_scale(p: Path) -> int:
        """Extract the numeric scale from a filename like corpus_150.jsonl."""
        match = re.search(r"corpus_(\d+)", p.stem)
        return int(match.group(1)) if match else 0

    corpus_files.sort(key=extract_scale, reverse=True)
    largest = corpus_files[0]
    logger.info(f"Loading from: {largest}")

    docs = {}
    with open(largest) as f:
        for line in f:
            record = json.loads(line)
            if doc_ids == ["ALL"] or record["doc_id"] in doc_ids:
                docs[record["doc_id"]] = record["text"]

    return docs


def load_all_docs(corpus_dir: str) -> dict[str, str]:
    """Load all documents from the largest corpus file."""
    return load_corpus_docs(corpus_dir, ["ALL"])


# ---------------------------------------------------------------------------
# Snippet Extraction
# ---------------------------------------------------------------------------

def find_relevant_snippets(
    text: str,
    keywords: list[str],
    context_chars: int = 300,
    max_snippets: int = 5,
) -> list[str]:
    """
    Find text snippets around keyword matches.

    Args:
        text: Full document text
        keywords: List of keywords/phrases to search for
        context_chars: Characters of context on each side
        max_snippets: Maximum snippets to return

    Returns:
        List of text snippets with context
    """
    snippets = []
    text_lower = text.lower()

    for keyword in keywords:
        keyword_lower = keyword.lower()
        start = 0
        while True:
            idx = text_lower.find(keyword_lower, start)
            if idx == -1:
                break

            snippet_start = max(0, idx - context_chars)
            snippet_end = min(len(text), idx + len(keyword) + context_chars)
            snippet = text[snippet_start:snippet_end].strip()

            # Add ellipses if truncated
            if snippet_start > 0:
                snippet = "..." + snippet
            if snippet_end < len(text):
                snippet = snippet + "..."

            snippets.append(snippet)
            start = idx + len(keyword)

            if len(snippets) >= max_snippets:
                break

        if len(snippets) >= max_snippets:
            break

    return snippets


def extract_financial_numbers(text: str, metric: str) -> list[str]:
    """
    Extract specific financial numbers from text using regex patterns.

    Args:
        text: Document text
        metric: One of the FINANCIAL_PATTERNS keys

    Returns:
        List of matched number strings
    """
    patterns = FINANCIAL_PATTERNS.get(metric, [])
    matches = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            matches.append(match.group(0))
    return matches[:10]  # Limit to 10 matches


# ---------------------------------------------------------------------------
# Query-specific keyword inference
# ---------------------------------------------------------------------------

def infer_keywords(query: dict) -> list[str]:
    """
    Infer search keywords from a query's text.

    Args:
        query: Query dict with 'text' field

    Returns:
        List of keywords to search for in documents
    """
    q_text = query["text"].lower()
    keywords = []

    # Financial metrics
    metric_keywords = {
        "revenue": ["revenue", "net sales", "total sales", "net revenue", "total net revenue"],
        "net income": ["net income", "net earnings"],
        "r&d": ["research and development", "R&D"],
        "gross margin": ["gross margin", "gross profit"],
        "operating": ["operating income", "operating profit", "operating margin"],
        "employees": ["employees", "associates", "headcount", "team members"],
        "total assets": ["total assets"],
        "deposits": ["total deposits", "deposits"],
        "dividends": ["dividend", "dividends paid"],
        "capital": ["capital expenditures", "capex", "capital and exploratory"],
        "backlog": ["backlog", "order backlog", "unfilled orders"],
        "segments": ["segment", "reportable segment", "operating segment"],
        "aum": ["assets under management", "AUM"],
        "reserves": ["proved reserves", "oil and gas reserves"],
        "debt": ["long-term debt", "total debt", "debt maturities"],
        "goodwill": ["goodwill", "intangible assets"],
        "risk": ["risk factor", "risks"],
        "climate": ["climate", "energy transition", "greenhouse", "emissions"],
        "ai": ["artificial intelligence", "machine learning", "AI"],
        "supply chain": ["supply chain", "manufacturing", "supplier"],
        "patent": ["patent", "intellectual property", "exclusivity"],
        "e-commerce": ["e-commerce", "digital", "omnichannel", "online"],
        "tax": ["effective tax rate", "income tax", "tax provision"],
        "currency": ["foreign currency", "exchange rate", "currency risk"],
        "buyback": ["repurchase", "buyback", "share repurchase"],
        "competition": ["competition", "competitive", "competitors"],
    }

    for metric, kws in metric_keywords.items():
        if metric in q_text or any(kw in q_text for kw in kws):
            keywords.extend(kws)

    # Always include some generic keywords from the query itself
    # Extract capitalized company-related words
    for word in query["text"].split():
        clean = word.strip("?.,()\"'")
        if len(clean) > 3 and clean[0].isupper():
            keywords.append(clean)

    return list(set(keywords))[:15]  # Dedupe and limit


# ---------------------------------------------------------------------------
# Main Verification Logic
# ---------------------------------------------------------------------------

def verify_query(
    query: dict,
    corpus_dir: str,
    verbose: bool = True,
) -> dict:
    """
    Verify a single query by finding relevant snippets in the corpus.

    Returns:
        Dict with verification results
    """
    query_id = query["id"]
    evidence = query.get("evidence_docs", [])

    # Handle ALL docs queries
    if evidence == ["ALL"] or evidence == ["ALL_PAIRS"]:
        docs = load_all_docs(corpus_dir)
    else:
        docs = load_corpus_docs(corpus_dir, evidence)

    if not docs:
        return {
            "query_id": query_id,
            "status": "no_docs",
            "message": f"No matching documents found for {evidence}",
        }

    # Infer keywords from the query
    keywords = infer_keywords(query)

    # Find snippets in each document
    all_snippets = {}
    for doc_id, text in docs.items():
        snippets = find_relevant_snippets(text, keywords, context_chars=250, max_snippets=3)
        if snippets:
            all_snippets[doc_id] = snippets

    if verbose:
        print(f"\n{'='*70}")
        print(f"Query {query_id} (Tier {query['tier']}: {query['tier_name']})")
        print(f"{'='*70}")
        print(f"Q: {query['text']}")
        print(f"Gold Answer: {query['gold_answer']}")
        print(f"Evidence Docs: {evidence}")
        print(f"Keywords searched: {keywords[:8]}...")
        print()

        if all_snippets:
            for doc_id, snippets in all_snippets.items():
                print(f"  --- {doc_id} ---")
                for i, snippet in enumerate(snippets, 1):
                    # Truncate very long snippets
                    if len(snippet) > 500:
                        snippet = snippet[:500] + "..."
                    print(f"  [{i}] {snippet}")
                    print()
        else:
            print("  ⚠ No relevant snippets found. Keywords may need adjustment.")

    return {
        "query_id": query_id,
        "status": "found" if all_snippets else "not_found",
        "docs_searched": list(docs.keys()),
        "docs_with_matches": list(all_snippets.keys()),
        "snippet_count": sum(len(s) for s in all_snippets.values()),
    }


# ---------------------------------------------------------------------------
# Export Gold Answers
# ---------------------------------------------------------------------------

def export_gold_answers(queries: list[dict], output_path: str):
    """Export a clean gold_answers.json for the evaluation framework."""
    gold = {}
    for q in queries:
        gold[q["id"]] = {
            "answer": q["gold_answer"],
            "tier": q["tier"],
            "tier_name": q["tier_name"],
            "difficulty": q["difficulty"],
            "answer_type": q.get("answer_type", "descriptive"),
            "evidence_docs": q["evidence_docs"],
            "verified": False,  # Set to True after manual verification
        }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(gold, f, indent=2)

    logger.info(f"Exported {len(gold)} gold answers to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify gold-standard answers against the corpus",
    )
    parser.add_argument(
        "--queries", type=str, default="data/queries/queries.json",
        help="Path to queries JSON file",
    )
    parser.add_argument(
        "--corpus-dir", type=str, default="data/processed",
        help="Path to processed corpus directory",
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Verify a specific query ID (e.g., q001)",
    )
    parser.add_argument(
        "--tier", type=int, default=None,
        help="Verify only queries in a specific tier (1-4)",
    )
    parser.add_argument(
        "--export", type=str, default=None,
        help="Export gold answers to specified path",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Show summary statistics only, no snippets",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load queries
    queries_path = Path(args.queries)
    if not queries_path.exists():
        logger.error(f"Queries file not found: {queries_path}")
        sys.exit(1)

    with open(queries_path) as f:
        queries = json.load(f)

    logger.info(f"Loaded {len(queries)} queries")

    # Print query distribution
    tier_counts = defaultdict(int)
    difficulty_counts = defaultdict(int)
    for q in queries:
        tier_counts[f"Tier {q['tier']}: {q['tier_name']}"] += 1
        difficulty_counts[q["difficulty"]] += 1

    print("\nQuery Distribution:")
    for tier, count in sorted(tier_counts.items()):
        print(f"  {tier}: {count} queries")
    print("\nDifficulty Distribution:")
    for diff, count in sorted(difficulty_counts.items()):
        print(f"  {diff}: {count} queries")

    # Export if requested
    if args.export:
        export_gold_answers(queries, args.export)
        return

    # Filter queries
    if args.query:
        queries = [q for q in queries if q["id"] == args.query]
        if not queries:
            logger.error(f"Query {args.query} not found")
            sys.exit(1)
    elif args.tier:
        queries = [q for q in queries if q["tier"] == args.tier]

    if args.summary:
        print(f"\nTotal queries to verify: {len(queries)}")
        return

    # Verify each query
    results = []
    for query in queries:
        result = verify_query(query, args.corpus_dir, verbose=True)
        results.append(result)

    # Summary
    found = sum(1 for r in results if r["status"] == "found")
    not_found = sum(1 for r in results if r["status"] == "not_found")
    no_docs = sum(1 for r in results if r["status"] == "no_docs")

    print(f"\n{'='*70}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Queries verified:    {len(results)}")
    print(f"  Snippets found:      {found}")
    print(f"  No snippets:         {not_found}")
    print(f"  Missing docs:        {no_docs}")


if __name__ == "__main__":
    main()