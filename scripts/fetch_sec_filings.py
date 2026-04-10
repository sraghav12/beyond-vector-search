#!/usr/bin/env python3
"""
fetch_sec_filings.py
====================
Downloads SEC 10-K filings from EDGAR for the benchmark corpus.

Workflow:
    1. Resolve each ticker → CIK via the EDGAR company_tickers.json endpoint
    2. Fetch each company's filing history from the submissions API
    3. Identify 10-K filings for the requested fiscal years
    4. Download the primary HTML document for each filing
    5. Save raw HTML to data/raw/{TICKER}_10K_{YEAR}.html

Usage:
    python scripts/fetch_sec_filings.py --years 3 --output-dir data/raw
    python scripts/fetch_sec_filings.py --years 5 --tickers AAPL,NVDA,JPM
    python scripts/fetch_sec_filings.py --verify-only  # just check CIK resolution

EDGAR rate limit: 10 requests/second. We stay well under this.
EDGAR requires a User-Agent header with your name and email.
Set SEC_EDGAR_USER_AGENT in your .env file or environment:
    SEC_EDGAR_USER_AGENT="YourName your@email.com"
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from tqdm import tqdm
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# EDGAR API base URLs
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# Rate limiting: EDGAR allows 10 req/s, we use 150ms between requests for safety
REQUEST_DELAY = 0.15

# ---------------------------------------------------------------------------
# Target Companies
# ---------------------------------------------------------------------------

COMPANIES = [
    # Healthcare
    {"ticker": "JNJ",   "name": "Johnson & Johnson",     "cik": "200406",  "sector": "Healthcare"},
    {"ticker": "UNH",   "name": "UnitedHealth Group",    "cik": "731766",  "sector": "Healthcare"},
    {"ticker": "LLY",   "name": "Eli Lilly",             "cik": "59478",   "sector": "Healthcare"},
    {"ticker": "PFE",   "name": "Pfizer",                "cik": "78003",   "sector": "Healthcare"},
    {"ticker": "ABBV",  "name": "AbbVie",                "cik": "1551152", "sector": "Healthcare"},
    # Financials
    {"ticker": "JPM",   "name": "JPMorgan Chase",        "cik": "19617",   "sector": "Financials"},
    {"ticker": "BAC",   "name": "Bank of America",       "cik": "70858",   "sector": "Financials"},
    {"ticker": "GS",    "name": "Goldman Sachs",         "cik": "886982",  "sector": "Financials"},
    {"ticker": "WFC",   "name": "Wells Fargo",           "cik": "72971",   "sector": "Financials"},
    {"ticker": "BLK",   "name": "BlackRock",             "cik": "1364742", "sector": "Financials"},
    # Energy
    {"ticker": "XOM",   "name": "Exxon Mobil",           "cik": "34088",   "sector": "Energy"},
    {"ticker": "CVX",   "name": "Chevron",               "cik": "93410",   "sector": "Energy"},
    {"ticker": "COP",   "name": "ConocoPhillips",        "cik": "1163165", "sector": "Energy"},
    # Consumer Staples
    {"ticker": "WMT",   "name": "Walmart",               "cik": "104169",  "sector": "Consumer Staples"},
    {"ticker": "PG",    "name": "Procter & Gamble",       "cik": "80424",   "sector": "Consumer Staples"},
    {"ticker": "KO",    "name": "Coca-Cola",              "cik": "21344",   "sector": "Consumer Staples"},
    # Industrials
    {"ticker": "CAT",   "name": "Caterpillar",           "cik": "18230",   "sector": "Industrials"},
    {"ticker": "BA",    "name": "Boeing",                "cik": "12927",   "sector": "Industrials"},
    {"ticker": "DE",    "name": "Deere & Company",       "cik": "315189",  "sector": "Industrials"},
    {"ticker": "GE",    "name": "GE Aerospace",          "cik": "40545",   "sector": "Industrials"},
    # Consumer Discretionary
    {"ticker": "HD",    "name": "Home Depot",            "cik": "354950",  "sector": "Consumer Discretionary"},
    {"ticker": "MCD",   "name": "McDonald's",            "cik": "63908",   "sector": "Consumer Discretionary"},
    {"ticker": "COST",  "name": "Costco",                "cik": "909832",  "sector": "Consumer Discretionary"},
    # Materials
    {"ticker": "LIN",   "name": "Linde",                 "cik": "1707925", "sector": "Materials"},
    {"ticker": "FCX",   "name": "Freeport-McMoRan",      "cik": "831259",  "sector": "Materials"},
    # Technology
    {"ticker": "AAPL",  "name": "Apple",                 "cik": "320193",  "sector": "Technology"},
    {"ticker": "NVDA",  "name": "NVIDIA",                "cik": "1045810", "sector": "Technology"},
    {"ticker": "GOOGL", "name": "Alphabet",              "cik": "1652044", "sector": "Technology"},
    {"ticker": "META",  "name": "Meta Platforms",        "cik": "1326801", "sector": "Technology"},
    {"ticker": "AMD",   "name": "Advanced Micro Devices", "cik": "2488",   "sector": "Technology"},
]


@dataclass
class FilingMetadata:
    """Metadata for a single SEC filing."""
    ticker: str
    company_name: str
    cik: str
    sector: str
    filing_type: str
    fiscal_year: int
    filing_date: str
    accession_number: str
    primary_document: str
    file_path: Optional[str] = None
    download_status: str = "pending"  # pending, success, failed, skipped


# ---------------------------------------------------------------------------
# EDGAR API Client
# ---------------------------------------------------------------------------

class EdgarClient:
    """
    Client for the SEC EDGAR API.

    Handles rate limiting, retries, and the multi-step lookup:
        ticker → CIK → filing history → document download
    """

    def __init__(self, user_agent: str):
        if not user_agent or "@" not in user_agent:
            raise ValueError(
                "SEC EDGAR requires a User-Agent with your name and email.\n"
                "Set SEC_EDGAR_USER_AGENT='YourName your@email.com' in .env"
            )
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
        })
        self._last_request_time = 0.0

    def _rate_limit(self):
        """Enforce minimum delay between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((requests.exceptions.RequestException,)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _get(self, url: str) -> requests.Response:
        """Make a rate-limited GET request with retries."""
        self._rate_limit()
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        return response

    def verify_cik(self, cik: str) -> dict:
        """
        Fetch company submission data from EDGAR to verify CIK is valid.
        Returns the full submissions JSON.
        """
        padded_cik = cik.zfill(10)
        url = SUBMISSIONS_URL.format(cik=padded_cik)
        logger.debug(f"Fetching submissions: {url}")
        response = self._get(url)
        return response.json()

    def find_10k_filings(
        self, cik: str, ticker: str, company_name: str, sector: str, num_years: int = 3
    ) -> list[FilingMetadata]:
        """
        Find the most recent 10-K filings for a company.

        Args:
            cik: SEC Central Index Key
            ticker: Stock ticker symbol
            company_name: Company name
            sector: GICS sector
            num_years: How many annual filings to retrieve

        Returns:
            List of FilingMetadata objects for each 10-K found
        """
        submissions = self.verify_cik(cik)
        recent = submissions.get("filings", {}).get("recent", {})

        if not recent:
            logger.warning(f"No recent filings found for {ticker} (CIK: {cik})")
            return []

        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])

        # Find 10-K filings (exclude 10-K/A amendments)
        filings = []
        for i, form in enumerate(forms):
            if form == "10-K" and i < len(dates) and i < len(accessions) and i < len(primary_docs):
                # Extract fiscal year from the filing date
                # The filing date is when it was submitted; fiscal year is usually
                # the year before for companies with calendar-year fiscal years,
                # but we'll use the filing date year and let preprocessing refine it
                filing_date = dates[i]
                fiscal_year = self._infer_fiscal_year(filing_date)

                filing = FilingMetadata(
                    ticker=ticker,
                    company_name=company_name,
                    cik=cik,
                    sector=sector,
                    filing_type="10-K",
                    fiscal_year=fiscal_year,
                    filing_date=filing_date,
                    accession_number=accessions[i],
                    primary_document=primary_docs[i],
                )
                filings.append(filing)

                if len(filings) >= num_years:
                    break

        # If fewer than num_years found in recent, check older filings
        # EDGAR paginates old filings into separate files
        if len(filings) < num_years:
            older_files = submissions.get("filings", {}).get("files", [])
            for older_file_info in older_files:
                if len(filings) >= num_years:
                    break
                older_url = f"https://data.sec.gov/submissions/{older_file_info['name']}"
                try:
                    older_data = self._get(older_url).json()
                    older_forms = older_data.get("form", [])
                    older_dates = older_data.get("filingDate", [])
                    older_accessions = older_data.get("accessionNumber", [])
                    older_docs = older_data.get("primaryDocument", [])

                    for i, form in enumerate(older_forms):
                        if form == "10-K" and len(filings) < num_years:
                            filing_date = older_dates[i]
                            fiscal_year = self._infer_fiscal_year(filing_date)
                            filing = FilingMetadata(
                                ticker=ticker,
                                company_name=company_name,
                                cik=cik,
                                sector=sector,
                                filing_type="10-K",
                                fiscal_year=fiscal_year,
                                filing_date=filing_date,
                                accession_number=older_accessions[i],
                                primary_document=older_docs[i],
                            )
                            filings.append(filing)
                except Exception as e:
                    logger.warning(f"Error fetching older filings for {ticker}: {e}")

        if not filings:
            logger.warning(f"No 10-K filings found for {ticker} (CIK: {cik})")
        else:
            years = [f.fiscal_year for f in filings]
            logger.info(f"Found {len(filings)} 10-K filings for {ticker}: FY{years}")

        return filings

    def download_filing(self, filing: FilingMetadata, output_dir: Path) -> FilingMetadata:
        """
        Download the primary document for a filing.

        Args:
            filing: FilingMetadata with accession number and document name
            output_dir: Directory to save the raw HTML file

        Returns:
            Updated FilingMetadata with file_path and download_status
        """
        # Build the output filename
        filename = f"{filing.ticker}_10K_{filing.fiscal_year}.html"
        filepath = output_dir / filename

        # Skip if already downloaded
        if filepath.exists() and filepath.stat().st_size > 1000:
            logger.info(f"  Already exists: {filename} ({filepath.stat().st_size:,} bytes)")
            filing.file_path = str(filepath)
            filing.download_status = "skipped"
            return filing

        # Build the EDGAR archive URL
        # Accession number format: "0000320193-24-000123"
        # URL path needs it without dashes: "000032019324000123"
        accession_no_dashes = filing.accession_number.replace("-", "")

        url = ARCHIVES_URL.format(
            cik=filing.cik,
            accession=accession_no_dashes,
            document=filing.primary_document,
        )

        try:
            logger.info(f"  Downloading: {filename}")
            logger.debug(f"  URL: {url}")
            response = self._get(url)

            # Save the raw content
            output_dir.mkdir(parents=True, exist_ok=True)
            filepath.write_bytes(response.content)

            filing.file_path = str(filepath)
            filing.download_status = "success"
            logger.info(f"  Saved: {filename} ({len(response.content):,} bytes)")

        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                # Sometimes the primary document name doesn't match.
                # Try the filing index page to find the actual document.
                logger.warning(f"  404 for {filename}, trying index lookup...")
                filing = self._download_via_index(filing, output_dir)
            else:
                logger.error(f"  Failed to download {filename}: {e}")
                filing.download_status = "failed"

        except Exception as e:
            logger.error(f"  Failed to download {filename}: {e}")
            filing.download_status = "failed"

        return filing

    def _download_via_index(self, filing: FilingMetadata, output_dir: Path) -> FilingMetadata:
        """
        Fallback: fetch the filing index page and find the 10-K document.
        Some filings have the primary doc listed differently than expected.
        """
        accession_no_dashes = filing.accession_number.replace("-", "")
        index_url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{filing.cik}/{accession_no_dashes}/index.json"
        )

        try:
            response = self._get(index_url)
            index_data = response.json()

            # Look for the 10-K document in the filing index
            target_doc = None
            for item in index_data.get("directory", {}).get("item", []):
                name = item.get("name", "").lower()
                # Look for the main 10-K HTML document (not exhibits, not XML)
                if name.endswith(".htm") or name.endswith(".html"):
                    # Prefer files that look like the main filing
                    if any(kw in name for kw in ["10-k", "10k", "annual", "filing"]):
                        target_doc = item["name"]
                        break

            # If no obvious match, take the largest HTML file (usually the 10-K)
            if not target_doc:
                html_files = [
                    item for item in index_data.get("directory", {}).get("item", [])
                    if item.get("name", "").lower().endswith((".htm", ".html"))
                    and "ex" not in item.get("name", "").lower()[:3]  # skip exhibits
                ]
                if html_files:
                    # Sort by size descending, take the largest
                    html_files.sort(key=lambda x: int(x.get("size", "0")), reverse=True)
                    target_doc = html_files[0]["name"]

            if target_doc:
                doc_url = ARCHIVES_URL.format(
                    cik=filing.cik,
                    accession=accession_no_dashes,
                    document=target_doc,
                )
                response = self._get(doc_url)
                filename = f"{filing.ticker}_10K_{filing.fiscal_year}.html"
                filepath = output_dir / filename
                filepath.write_bytes(response.content)
                filing.file_path = str(filepath)
                filing.download_status = "success"
                filing.primary_document = target_doc
                logger.info(f"  Saved via index: {filename} ({len(response.content):,} bytes)")
            else:
                logger.error(f"  No suitable document found in filing index for {filing.ticker}")
                filing.download_status = "failed"

        except Exception as e:
            logger.error(f"  Index fallback failed for {filing.ticker}: {e}")
            filing.download_status = "failed"

        return filing

    @staticmethod
    def _infer_fiscal_year(filing_date: str) -> int:
        """
        Infer the fiscal year from the filing date.

        Most companies file their 10-K within 60-90 days of fiscal year end.
        - Calendar-year companies (Dec 31 FY end) file in Feb/Mar → FY = filing_year - 1
        - Some companies have non-calendar FY (e.g., Apple ends Sept, Walmart ends Jan)

        We use a simple heuristic: if filed in Jan-Mar, fiscal year = filing_year - 1.
        Otherwise fiscal year = filing_year.
        This isn't perfect but covers ~80% of cases. The preprocessing step
        can refine this by parsing the actual fiscal year from the filing text.
        """
        year = int(filing_date[:4])
        month = int(filing_date[5:7])

        # If filed in Q1, the fiscal year is almost certainly the prior calendar year
        if month <= 3:
            return year - 1
        return year


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_fetch(
    output_dir: str = "data/raw",
    num_years: int = 3,
    tickers: Optional[list[str]] = None,
    verify_only: bool = False,
    save_manifest: bool = True,
):
    """
    Main entry point: fetch 10-K filings for all target companies.

    Args:
        output_dir: Where to save raw HTML files
        num_years: How many years of filings per company (1-10)
        tickers: Optional subset of tickers to fetch (default: all 30)
        verify_only: If True, only verify CIK resolution without downloading
        save_manifest: If True, save a JSON manifest of all filings metadata
    """
    user_agent = os.getenv("SEC_EDGAR_USER_AGENT", "")
    if not user_agent:
        logger.error(
            "SEC_EDGAR_USER_AGENT not set.\n"
            "Add to .env: SEC_EDGAR_USER_AGENT=\"YourName your@email.com\"\n"
            "Or export: export SEC_EDGAR_USER_AGENT=\"YourName your@email.com\""
        )
        sys.exit(1)

    client = EdgarClient(user_agent)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Filter companies if specific tickers requested
    companies = COMPANIES
    if tickers:
        ticker_set = {t.upper() for t in tickers}
        companies = [c for c in COMPANIES if c["ticker"] in ticker_set]
        if not companies:
            logger.error(f"No matching companies for tickers: {tickers}")
            sys.exit(1)

    logger.info(f"Processing {len(companies)} companies, {num_years} years each")
    logger.info(f"Output directory: {output_path.resolve()}")

    # ---- Phase 1: Discover filings ----
    all_filings: list[FilingMetadata] = []
    errors = []

    logger.info("=" * 60)
    logger.info("PHASE 1: Discovering 10-K filings on EDGAR")
    logger.info("=" * 60)

    for company in tqdm(companies, desc="Discovering filings"):
        try:
            filings = client.find_10k_filings(
                cik=company["cik"],
                ticker=company["ticker"],
                company_name=company["name"],
                sector=company["sector"],
                num_years=num_years,
            )
            all_filings.extend(filings)
        except Exception as e:
            error_msg = f"Error discovering filings for {company['ticker']}: {e}"
            logger.error(error_msg)
            errors.append(error_msg)

    # Summary
    logger.info(f"\nDiscovered {len(all_filings)} filings across {len(companies)} companies")
    if errors:
        logger.warning(f"{len(errors)} companies had errors during discovery")

    if verify_only:
        logger.info("\n--- Verification Results ---")
        for company in companies:
            company_filings = [f for f in all_filings if f.ticker == company["ticker"]]
            status = f"✓ {len(company_filings)} filings" if company_filings else "✗ NO FILINGS"
            logger.info(f"  {company['ticker']:6s} | CIK {company['cik']:>10s} | {status}")
        return all_filings

    # ---- Phase 2: Download filings ----
    logger.info("=" * 60)
    logger.info("PHASE 2: Downloading filing documents")
    logger.info("=" * 60)

    for filing in tqdm(all_filings, desc="Downloading"):
        client.download_filing(filing, output_path)

    # ---- Summary ----
    success = sum(1 for f in all_filings if f.download_status in ("success", "skipped"))
    failed = sum(1 for f in all_filings if f.download_status == "failed")
    skipped = sum(1 for f in all_filings if f.download_status == "skipped")

    logger.info("=" * 60)
    logger.info("DOWNLOAD COMPLETE")
    logger.info(f"  Total filings:   {len(all_filings)}")
    logger.info(f"  Successful:      {success} ({skipped} already existed)")
    logger.info(f"  Failed:          {failed}")
    logger.info("=" * 60)

    # ---- Save manifest ----
    if save_manifest:
        manifest_path = output_path / "filing_manifest.json"
        manifest = [asdict(f) for f in all_filings]
        manifest_path.write_text(json.dumps(manifest, indent=2))
        logger.info(f"Manifest saved to: {manifest_path}")

    # Report failures
    if failed > 0:
        logger.warning("\nFailed downloads:")
        for f in all_filings:
            if f.download_status == "failed":
                logger.warning(f"  {f.ticker} FY{f.fiscal_year} ({f.accession_number})")

    return all_filings


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch SEC 10-K filings from EDGAR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/fetch_sec_filings.py --years 3
  python scripts/fetch_sec_filings.py --years 5 --tickers AAPL,NVDA,JPM
  python scripts/fetch_sec_filings.py --verify-only
        """,
    )
    parser.add_argument(
        "--years", type=int, default=3,
        help="Number of fiscal years to fetch per company (default: 3)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/raw",
        help="Directory for raw HTML files (default: data/raw)",
    )
    parser.add_argument(
        "--tickers", type=str, default=None,
        help="Comma-separated list of tickers to fetch (default: all 30)",
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Only verify CIK resolution, don't download anything",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    tickers = args.tickers.split(",") if args.tickers else None

    run_fetch(
        output_dir=args.output_dir,
        num_years=args.years,
        tickers=tickers,
        verify_only=args.verify_only,
    )