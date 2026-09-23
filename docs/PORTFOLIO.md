# Portfolio and interview walkthrough

## Resume bullet

Built a reproducible financial-QA benchmark comparing four retrieval strategies across 150 SEC filings and 43 source-supported questions; measured quality, latency and cost, corrected fiscal-year/evidence defects, and found vector retrieval cost about 17× less per query than structure-aware retrieval, which scored higher overall but without a conclusive paired difference.

Shorter version:

Built a reproducible LLM evaluation framework comparing three retrieval pipelines and a full-context baseline across 150 SEC filings and 43 source-supported questions; added 23 tests, measured judge scores, latency and cost, and corrected fiscal-year and evidence-coverage errors.

The full-context baseline exceeded the context limit on all 43 questions; it is an infeasibility result, not a fourth working retrieval pipeline. The roughly 17× cost comparison covers generation only, excluding indexing and judging. Do not describe the source annotations as independently human-verified. These bullets describe v1; do not reuse the old 75-question accuracy claims.

## Five-minute walkthrough

1. **Problem (30 seconds):** A financial answer must identify the correct company, fiscal year, metric and units. Finding related text is insufficient.
2. **System (60 seconds):** Show vector chunk retrieval, local section selection, recursive Python retrieval and the context-stuffing guard. All use the same generation model and frozen corpus.
3. **Engineering (60 seconds):** Show one source-backed reference, the manifest/checksum tests, cache fingerprints, and a result trace. Explain why a legacy NVIDIA `_2025` ID actually covers FY2026.
4. **Results (60 seconds):** Open the comparison chart. Structure-aware scores 0.605 versus vector 0.495, but costs roughly 17× more. Vector wins direct extraction, 0.835 versus 0.690. Explain the wide paired interval rather than claiming a decisive winner.
5. **Failure analysis (60 seconds):** Use q029's fiscal-period confusion and q041's incomplete company coverage. Explain why RLM's 8 abstentions stay in the denominator and why context infeasibility is separate from wrong answers.
6. **Limits (30 seconds):** Agent annotations, same-provider judge, a small curated subset, one run and one corpus size. Name human calibration and better temporal/multi-document retrieval as follow-up research.

## Demo without API spending

```bash
make prepare
make test
make report
```

Open `results/release_v1/comparison.png`, then inspect q001/q029/q041 in the raw JSONL files. This is a research/engineering portfolio artifact; a hosted chatbot is not necessary to explain its contribution.

## Why this fits AI/LLM application engineering

The work demonstrates retrieval implementation, model orchestration, data provenance, evaluation design, debugging, cost accounting, reproducibility and honest tradeoffs. It does not demonstrate training a foundation model or establish broad classical-ML/statistical expertise by itself.
