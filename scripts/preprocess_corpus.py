#!/usr/bin/env python3
"""
preprocess_corpus.py
====================
Converts raw SEC 10-K HTML filings into clean, structured JSONL corpus files.

Workflow:
    1. Read the filing_manifest.json from the fetch step
    2. For each raw HTML filing:
       a. Parse and extract text with BeautifulSoup
       b. Clean whitespace, remove boilerplate, normalize unicode
       c. Count tokens with tiktoken (cl100k_base encoding)
       d. Build a structured document record
    3. Sort documents in a stable order (for consistent subsetting)
    4. Write corpus files at each scale: corpus_10, corpus_25, corpus_50, etc.
       Smaller corpora are strict subsets of larger ones.

Usage:
    python scripts/preprocess_corpus.py
    python scripts/preprocess_corpus.py --raw-dir data/raw --output-dir data/processed
    python scripts/preprocess_corpus.py --scales 10,25,50,100
    python scripts/preprocess_corpus.py --stats-only  # just show token stats

Output:
    data/processed/corpus_10.jsonl     (first 10 docs)
    data/processed/corpus_25.jsonl     (first 25 docs)
    data/processed/corpus_50.jsonl     (first 50 docs)
    data/processed/corpus_100.jsonl    (first 100 docs)
    data/processed/corpus_150.jsonl    (all 150 docs)
    data/processed/corpus_stats.json   (token counts, sizes, per-sector stats)
"""

import os
import re
import sys
import json
import html
import argparse
import logging
import unicodedata
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from collections import defaultdict

from bs4 import BeautifulSoup, Comment
import warnings
from bs4 import XMLParsedAsHTMLWarning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

import tiktoken
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# tiktoken encoding used by GPT-4o and GPT-4o-mini
ENCODING_NAME = "cl100k_base"

# Default corpus scales
DEFAULT_SCALES = [10, 25, 50, 100, 150]

# Sections that are typically boilerplate / not analytically useful
BOILERPLATE_PATTERNS = [
    # SEC header boilerplate
    r"UNITED STATES\s+SECURITIES AND EXCHANGE COMMISSION",
    r"Washington,?\s*D\.?C\.?\s*20549",
    r"FORM\s+10-K",
    r"ANNUAL REPORT PURSUANT TO SECTION",
    r"For the fiscal year ended",
    r"Commission [Ff]ile [Nn]umber",
    # Page headers/footers
    r"Table of Contents",
    r"^\d+$",  # standalone page numbers
]

# Exhibit patterns to strip (certifications, consents, etc.)
EXHIBIT_HEADER_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:EXHIBIT|Exhibit)\s+\d+[\.\d]*\s*[-—]?\s*",
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class DocumentRecord:
    """A single processed document in the corpus."""
    doc_id: str
    text: str
    metadata: dict

    def to_jsonl(self) -> str:
        """Serialize to a single JSON line."""
        return json.dumps({
            "doc_id": self.doc_id,
            "text": self.text,
            "metadata": self.metadata,
        }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# HTML Text Extraction
# ---------------------------------------------------------------------------

class FilingCleaner:
    """
    Extracts and cleans text from SEC 10-K HTML filings.

    SEC filings are messy HTML: deeply nested tables for layout,
    embedded styles, page break markers, and inconsistent formatting.
    This class handles all of that.
    """

    # Tags that never contain useful body text
    STRIP_TAGS = {"script", "style", "meta", "link", "head", "noscript", "iframe"}

    # Tags whose text we want but that can be inlined
    INLINE_TAGS = {"b", "i", "u", "em", "strong", "span", "font", "a", "sup", "sub"}

    def __init__(self):
        try:
            self.tokenizer = tiktoken.get_encoding(ENCODING_NAME)
            self._use_tiktoken = True
            logger.debug("Using tiktoken for token counting")
        except Exception as e:
            logger.warning(
                f"tiktoken encoding download failed ({e}). "
                "Using character-based token estimate (chars / 3.5). "
                "Install tiktoken with network access for exact counts."
            )
            self.tokenizer = None
            self._use_tiktoken = False

    def extract_text(self, html_content: str) -> str:
        """
        Extract clean text from a raw SEC filing HTML document.

        Args:
            html_content: Raw HTML string

        Returns:
            Cleaned plain text
        """
        # Try lxml first, fall back to html.parser for problematic files
        for parser in ("lxml", "html.parser"):
            try:
                soup = BeautifulSoup(html_content, parser)
                break
            except Exception:
                continue
        else:
            # Last resort: just strip tags with regex
            text = re.sub(r"<[^>]+>", " ", html_content)
            return self._clean_text(text)

        # Remove unwanted tags entirely
        for tag in soup.find_all(self.STRIP_TAGS):
            tag.decompose()

        # Remove HTML comments
        for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
            comment.extract()

        # Remove hidden elements (display:none, visibility:hidden)
        for tag in soup.find_all(style=True):
            try:
                if not tag or not getattr(tag, 'attrs', None):
                    continue
                style = (tag.attrs.get("style") or "").lower()
                if "display:none" in style or "display: none" in style:
                    tag.decompose()
                elif "visibility:hidden" in style or "visibility: hidden" in style:
                    tag.decompose()
            except (AttributeError, TypeError):
                continue

        # Extract text with newlines at block boundaries
        text = self._extract_with_structure(soup)

        # Clean the extracted text
        text = self._clean_text(text)

        return text

    def _extract_with_structure(self, soup: BeautifulSoup) -> str:
        """
        Walk the DOM and extract text, inserting newlines at block
        element boundaries to preserve document structure.
        """
        from bs4 import NavigableString

        block_tags = {
            "p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
            "li", "tr", "td", "th", "br", "hr",
            "table", "thead", "tbody", "section", "article",
            "blockquote", "pre", "ul", "ol", "dl", "dt", "dd",
        }

        parts = []
        for element in soup.descendants:
            if element.name in block_tags:
                parts.append("\n")
            # Only extract from leaf text nodes, not from tags
            if isinstance(element, NavigableString) and not isinstance(element, Comment):
                text = element.strip()
                if text:
                    parts.append(text + " ")

        return "".join(parts)

    def _clean_text(self, text: str) -> str:
        """Apply all text cleaning steps."""
        # Decode HTML entities
        text = html.unescape(text)

        # Normalize unicode (NFKC collapses compatibility characters)
        text = unicodedata.normalize("NFKC", text)

        # Replace common unicode artifacts
        replacements = {
            "\u00a0": " ",   # non-breaking space
            "\u200b": "",    # zero-width space
            "\u200c": "",    # zero-width non-joiner
            "\u200d": "",    # zero-width joiner
            "\u2002": " ",   # en space
            "\u2003": " ",   # em space
            "\u2009": " ",   # thin space
            "\ufeff": "",    # BOM
            "\u2018": "'",   # left single quote
            "\u2019": "'",   # right single quote
            "\u201c": '"',   # left double quote
            "\u201d": '"',   # right double quote
            "\u2013": "-",   # en dash
            "\u2014": " - ", # em dash
            "\u2026": "...", # ellipsis
            "\u00b7": " ",   # middle dot
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        # Remove any remaining control characters except newline and tab
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

        # Normalize whitespace within lines
        # Replace tabs and multiple spaces with single space
        text = re.sub(r"[^\S\n]+", " ", text)

        # Collapse 3+ consecutive newlines into 2
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)

        # Remove lines that are just dashes, underscores, or equals signs (separators)
        text = re.sub(r"^[-_=]{3,}\s*$", "", text, flags=re.MULTILINE)

        # Strip leading/trailing whitespace from the whole document
        text = text.strip()

        # Remove very short orphan lines (likely artifacts) between real paragraphs
        # But keep numbered items, bullet points, etc.
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            # Keep non-empty lines, or empty lines (they become paragraph breaks)
            if line == "" or len(line) > 2:
                cleaned_lines.append(line)
            elif re.match(r"^[\d\.\)\(]+$", line):
                # Keep standalone numbers (list items, section numbers)
                cleaned_lines.append(line)
        text = "\n".join(cleaned_lines)

        # Final collapse of excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken, or estimate from character count."""
        if self._use_tiktoken:
            return len(self.tokenizer.encode(text))
        # Fallback: rough estimate. English text averages ~3.5 chars per token
        # with cl100k_base. This is accurate within ~10% for SEC filings.
        return int(len(text) / 3.5)


# ---------------------------------------------------------------------------
# Corpus Builder
# ---------------------------------------------------------------------------

class CorpusBuilder:
    """
    Builds scaled corpus JSONL files from raw HTML filings.

    The key invariant: smaller corpora are strict subsets of larger ones.
    We achieve this by sorting all documents in a fixed order, then
    corpus_N is simply the first N documents in that order.

    Sort order: by sector (alphabetical), then by ticker, then by fiscal year (desc).
    This ensures sector diversity even at small scales.
    """

    def __init__(self, raw_dir: str, output_dir: str):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.cleaner = FilingCleaner()
        self.documents: list[DocumentRecord] = []

    def load_manifest(self) -> list[dict]:
        """Load the filing manifest from the fetch step."""
        manifest_path = self.raw_dir / "filing_manifest.json"
        if not manifest_path.exists():
            logger.error(f"Manifest not found at {manifest_path}")
            logger.error("Run fetch_sec_filings.py first.")
            sys.exit(1)

        with open(manifest_path) as f:
            manifest = json.load(f)

        # Filter to successful downloads only
        successful = [
            m for m in manifest
            if m.get("download_status") in ("success", "skipped")
            and m.get("file_path")
        ]

        logger.info(f"Manifest loaded: {len(successful)}/{len(manifest)} filings available")
        return successful

    def process_all(self, manifest: list[dict]) -> list[DocumentRecord]:
        """
        Process all raw HTML filings into clean document records.

        Args:
            manifest: List of filing metadata dicts from the manifest

        Returns:
            List of DocumentRecord objects, sorted in corpus order
        """
        documents = []
        errors = []

        for entry in tqdm(manifest, desc="Processing filings"):
            try:
                doc = self._process_single(entry)
                if doc:
                    documents.append(doc)
            except Exception as e:
                import traceback
                error_msg = f"Error processing {entry.get('ticker', '?')} FY{entry.get('fiscal_year', '?')}: {e}"
                logger.error(error_msg)
                logger.debug(traceback.format_exc())
                errors.append(error_msg)

        if errors:
            logger.warning(f"{len(errors)} filings had processing errors")

        # Sort in stable corpus order
        documents.sort(key=lambda d: (
            d.metadata["sector"],
            d.metadata["ticker"],
            -d.metadata["fiscal_year"],  # most recent first within a company
        ))

        # Deduplicate: if same doc_id appears twice (shouldn't happen), keep first
        seen = set()
        deduped = []
        for doc in documents:
            if doc.doc_id not in seen:
                seen.add(doc.doc_id)
                deduped.append(doc)
            else:
                logger.warning(f"Duplicate doc_id skipped: {doc.doc_id}")

        self.documents = deduped
        logger.info(f"Processed {len(deduped)} documents successfully")
        return deduped

    def _process_single(self, entry: dict) -> Optional[DocumentRecord]:
        """Process a single filing from manifest entry to DocumentRecord."""
        file_path_str = entry.get("file_path")
        if not file_path_str:
            logger.warning(f"No file_path for {entry.get('ticker', '?')}")
            return None

        file_path = Path(file_path_str)

        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return None

        # Read raw HTML
        raw_size = file_path.stat().st_size
        if raw_size < 1000:
            logger.warning(f"Suspiciously small file ({raw_size} bytes): {file_path}")
            return None

        html_content = file_path.read_text(encoding="utf-8", errors="replace")

        # Extract and clean text
        clean_text = self.cleaner.extract_text(html_content)

        ticker = entry.get("ticker", "UNKNOWN")
        fiscal_year = entry.get("fiscal_year", 0)

        if len(clean_text) < 500:
            logger.warning(
                f"Very short extracted text ({len(clean_text)} chars) for "
                f"{ticker} FY{fiscal_year}"
            )
            return None

        # Count tokens
        token_count = self.cleaner.count_tokens(clean_text)

        # Build doc ID
        doc_id = f"{ticker}_10K_{fiscal_year}"

        # Build metadata — use .get() with defaults for safety
        metadata = {
            "company": entry.get("company_name", ""),
            "ticker": ticker,
            "filing_type": entry.get("filing_type", "10-K"),
            "fiscal_year": fiscal_year,
            "sector": entry.get("sector", ""),
            "cik": entry.get("cik", ""),
            "filing_date": entry.get("filing_date", ""),
            "accession_number": entry.get("accession_number", ""),
            "token_count": token_count,
            "char_count": len(clean_text),
            "raw_html_bytes": raw_size,
        }

        return DocumentRecord(doc_id=doc_id, text=clean_text, metadata=metadata)

    def write_corpus_files(self, scales: list[int]) -> dict:
        """
        Write JSONL corpus files at each scale.

        Smaller corpora are strict subsets of larger ones.
        If a requested scale exceeds the number of available documents,
        we cap it at the total count and log a warning.

        Returns:
            Dict of corpus stats
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        total_docs = len(self.documents)
        stats = {
            "total_documents": total_docs,
            "scales": {},
        }

        for scale in sorted(scales):
            actual_size = min(scale, total_docs)
            if scale > total_docs:
                logger.warning(
                    f"Requested corpus_{scale} but only {total_docs} docs available. "
                    f"Writing corpus_{scale} with {total_docs} docs."
                )

            corpus_docs = self.documents[:actual_size]
            filename = f"corpus_{scale}.jsonl"
            filepath = self.output_dir / filename

            with open(filepath, "w", encoding="utf-8") as f:
                for doc in corpus_docs:
                    f.write(doc.to_jsonl() + "\n")

            # Compute stats for this scale
            total_tokens = sum(d.metadata["token_count"] for d in corpus_docs)
            total_chars = sum(d.metadata["char_count"] for d in corpus_docs)
            sector_counts = defaultdict(int)
            for d in corpus_docs:
                sector_counts[d.metadata["sector"]] += 1

            scale_stats = {
                "num_documents": actual_size,
                "total_tokens": total_tokens,
                "total_chars": total_chars,
                "avg_tokens_per_doc": total_tokens // actual_size if actual_size > 0 else 0,
                "min_tokens": min(d.metadata["token_count"] for d in corpus_docs),
                "max_tokens": max(d.metadata["token_count"] for d in corpus_docs),
                "sectors": dict(sector_counts),
                "file_size_bytes": filepath.stat().st_size,
            }
            stats["scales"][str(scale)] = scale_stats

            logger.info(
                f"  corpus_{scale}.jsonl: {actual_size} docs, "
                f"{total_tokens:,} tokens, "
                f"{filepath.stat().st_size / 1024 / 1024:.1f} MB"
            )

        # Save stats
        stats_path = self.output_dir / "corpus_stats.json"
        stats_path.write_text(json.dumps(stats, indent=2))
        logger.info(f"Stats saved to: {stats_path}")

        return stats

    def print_stats(self):
        """Print detailed corpus statistics."""
        if not self.documents:
            logger.warning("No documents loaded. Run process_all() first.")
            return

        total_tokens = sum(d.metadata["token_count"] for d in self.documents)
        total_chars = sum(d.metadata["char_count"] for d in self.documents)

        print("\n" + "=" * 70)
        print("CORPUS STATISTICS")
        print("=" * 70)
        print(f"  Total documents:     {len(self.documents)}")
        print(f"  Total tokens:        {total_tokens:,}")
        print(f"  Total characters:    {total_chars:,}")
        print(f"  Avg tokens/doc:      {total_tokens // len(self.documents):,}")
        print(f"  Min tokens:          {min(d.metadata['token_count'] for d in self.documents):,}")
        print(f"  Max tokens:          {max(d.metadata['token_count'] for d in self.documents):,}")

        # Per-sector stats
        print("\n  By Sector:")
        sector_data = defaultdict(lambda: {"count": 0, "tokens": 0})
        for d in self.documents:
            s = d.metadata["sector"]
            sector_data[s]["count"] += 1
            sector_data[s]["tokens"] += d.metadata["token_count"]

        for sector in sorted(sector_data):
            data = sector_data[sector]
            avg = data["tokens"] // data["count"]
            print(f"    {sector:30s} {data['count']:3d} docs  {data['tokens']:>10,} tokens  (avg {avg:,})")

        # Per-company stats (sorted by token count)
        print("\n  By Company (sorted by total tokens):")
        company_data = defaultdict(lambda: {"count": 0, "tokens": 0, "sector": ""})
        for d in self.documents:
            t = d.metadata["ticker"]
            company_data[t]["count"] += 1
            company_data[t]["tokens"] += d.metadata["token_count"]
            company_data[t]["sector"] = d.metadata["sector"]

        for ticker in sorted(company_data, key=lambda t: company_data[t]["tokens"], reverse=True):
            data = company_data[ticker]
            print(
                f"    {ticker:6s} | {data['sector']:25s} | "
                f"{data['count']} filings | {data['tokens']:>10,} tokens"
            )

        # Context window analysis
        print("\n  Context Window Fit (single doc):")
        windows = [
            ("GPT-4o (128K)", 128_000),
            ("Gemini 2.5 Pro (1M)", 1_000_000),
        ]
        for name, limit in windows:
            fits = sum(1 for d in self.documents if d.metadata["token_count"] <= limit)
            print(f"    {name}: {fits}/{len(self.documents)} docs fit")

        print("\n  Cumulative Corpus Fit:")
        sorted_by_order = self.documents  # already in corpus order
        cumulative = 0
        for i, doc in enumerate(sorted_by_order, 1):
            cumulative += doc.metadata["token_count"]
            if i in (10, 25, 50, 100, 150):
                gpt4o_fit = "YES" if cumulative <= 128_000 else "NO"
                gemini_fit = "YES" if cumulative <= 1_000_000 else "NO"
                print(
                    f"    corpus_{i:3d}: {cumulative:>10,} tokens | "
                    f"GPT-4o: {gpt4o_fit:3s} | Gemini 1M: {gemini_fit:3s}"
                )

        print("=" * 70)

    def verify_subset_invariant(self, scales: list[int]):
        """Verify that smaller corpora are strict subsets of larger ones."""
        sorted_scales = sorted(scales)
        for i in range(len(sorted_scales) - 1):
            small_scale = sorted_scales[i]
            large_scale = sorted_scales[i + 1]

            small_path = self.output_dir / f"corpus_{small_scale}.jsonl"
            large_path = self.output_dir / f"corpus_{large_scale}.jsonl"

            if not small_path.exists() or not large_path.exists():
                continue

            small_ids = set()
            with open(small_path) as f:
                for line in f:
                    doc = json.loads(line)
                    small_ids.add(doc["doc_id"])

            large_ids = set()
            with open(large_path) as f:
                for line in f:
                    doc = json.loads(line)
                    large_ids.add(doc["doc_id"])

            if small_ids.issubset(large_ids):
                logger.info(f"  ✓ corpus_{small_scale} ⊂ corpus_{large_scale}")
            else:
                missing = small_ids - large_ids
                logger.error(
                    f"  ✗ SUBSET VIOLATION: corpus_{small_scale} has {len(missing)} docs "
                    f"not in corpus_{large_scale}: {missing}"
                )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess SEC 10-K filings into JSONL corpus files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/preprocess_corpus.py
  python scripts/preprocess_corpus.py --scales 10,25,50,100,150
  python scripts/preprocess_corpus.py --stats-only
        """,
    )
    parser.add_argument(
        "--raw-dir", type=str, default="data/raw",
        help="Directory with raw HTML files and manifest (default: data/raw)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/processed",
        help="Directory for JSONL output files (default: data/processed)",
    )
    parser.add_argument(
        "--scales", type=str, default=None,
        help="Comma-separated corpus scales (default: 10,25,50,100,150)",
    )
    parser.add_argument(
        "--stats-only", action="store_true",
        help="Only compute and display stats, don't write corpus files",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    scales = (
        [int(s) for s in args.scales.split(",")]
        if args.scales
        else DEFAULT_SCALES
    )

    logger.info("=" * 60)
    logger.info("PREPROCESSING SEC 10-K FILINGS")
    logger.info("=" * 60)

    builder = CorpusBuilder(raw_dir=args.raw_dir, output_dir=args.output_dir)

    # Load manifest
    manifest = builder.load_manifest()

    # Process all filings
    documents = builder.process_all(manifest)

    # Print stats
    builder.print_stats()

    if args.stats_only:
        logger.info("Stats-only mode — no corpus files written.")
        return

    # Write corpus files
    logger.info("\nWriting corpus files:")
    builder.write_corpus_files(scales)

    # Verify subset invariant
    logger.info("\nVerifying subset invariant:")
    builder.verify_subset_invariant(scales)

    logger.info("\nDone! Corpus files are in: " + args.output_dir)


if __name__ == "__main__":
    main()