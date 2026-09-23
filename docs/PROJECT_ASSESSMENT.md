# Project resumption assessment

> Historical working document. Current release status and results are in [FINDINGS.md](FINDINGS.md) and [the v1 manifest](../data/release/v1/manifest.json).

Checked locally on September 13, 2026. This audit supersedes stale README status claims; it does not certify the benchmark results.

## Decision

Finish this project as a bounded retrieval evaluation study. It is technically feasible: implementations, datasets, saved experiments, and a working local test environment already exist. The remaining critical work is evidence quality, evaluation, and reproducibility. No new pipeline is needed to make it valuable.

Portfolio fit is strongest for applied AI/LLM engineering. For ML engineering, emphasize reliable pipelines, caching, failure handling, testing, and reproducibility. For data science, emphasize controlled comparisons, uncertainty, annotation quality, and error analysis. This project alone does not demonstrate model training or a broad classical-ML skill set.

## Independently checked state

- `.venv/bin/python -m pytest -q`: **16 passed in 14.66s**. Tests include fake pipelines and mocked providers; this is not a fresh live-API validation of all four systems.
- Corpus files contain 50, 100, and 150 unique document IDs; the largest spans 30 tickers. The README's five replicated filings description is obsolete.
- 75 questions: 20 single-hop, 20 multi-hop, 15 aggregation, 20 pairwise.
- Gold verification flags: **0/75 true**. This does not mean every answer is wrong.
- **61/75** gold answer strings contain the word `verify` or `likely` (case insensitive), indicating unresolved references or hedging.
- **66/75** questions have a different evidence-document set in `queries.json` versus `gold_answers.json`. A different filing can repeat comparative figures, so this is a review queue, not proof that all 66 are unanswerable.
- All explicitly named query evidence IDs exist at each current scale. However, gold-cited IDs absent at scales 50/100/150 number **20/12/1**, respectively. `BLK_10K_2024` is absent even at 150. Document-ID coverage does not establish passage-level answerability.
- Direct corpus check: the cover of `NVDA_10K_2025` says “For the fiscal year ended January 25, 2026”. Fiscal labels cannot be trusted without reconciliation.
- `docs/GOLD_VERIFICATION_DRAFT.md` contains useful proposed corrections, but its opening says 49 questions covered while a later section says all 75. Treat it as unvalidated working notes. Some proposals make claims about missing evidence that require independent review.
- Saved scored CSVs have unique query IDs within each file. Scales 50/100 have 68 rows per pipeline; scale 150 has 75. A scale comparison must use a common query subset.
- `pageindex` is a local structure-aware implementation. Describe it as such; these results do not establish the performance of an external PageIndex service.
- Important deliverables remain empty, including FINDINGS.md, pyproject.toml, Makefile, and several scaffolding/configuration files. Empty Python package initializers can be legitimate and need not be filled.
- Results and processed data are Git-ignored; a public reviewer cannot reproduce the local evidence from the README alone. Existing uncommitted implementation changes predate this audit and were preserved.

## Existing results — provisional only

Computed directly from per-pipeline scored CSVs, rather than copying earlier notes:

| Pipeline | Scale | Rows | Mean judge score |
|---|---:|---:|---:|
| vector_rag | 50 | 68 | 0.4824 |
| vector_rag | 100 | 68 | 0.4765 |
| vector_rag | 150 | 75 | 0.4493 |
| pageindex | 50 | 68 | 0.4029 |
| pageindex | 100 | 68 | 0.4809 |
| pageindex | 150 | 75 | 0.4533 |
| rlm | 150 | 75 | 0.2333 |

The RLM CSV contains 73 `ok` and two `error` rows; `ok` need not mean a substantive answer. Naive baseline CSVs contain zeros and `ok` statuses, while methodology describes context infeasibility: normalize interpretation before publication.

These are judge scores, not percent accuracy. Wrong golds can favor one strategy over another, so even the ranking is provisional. A difference of 0.004 between vector and structure-aware retrieval is not evidence of a meaningful winner. Old claims that comparisons remain fair despite incorrect golds are not justified.

## Completion gates, in order

1. **Freeze a defensible question set.** Reconcile the two answer/evidence files. Use explicit fiscal periods; define “most recent” relative to the frozen corpus. Preserve source URL/accession, document ID, supporting quote, and calculation where relevant. Resolve recast financial figures consistently. Independently verify the draft; do not bulk-flip flags. For missing financial statements (the draft flags WFC), obtain the referenced source or explicitly exclude affected questions. Prefer a smaller verified benchmark over 75 unreliable labels.
2. **Re-establish fair corpus comparisons.** After evidence corrections, verify passage availability at all included scales. Rebuild if needed. Score corpus-wide questions only where their full evidence universe is defined. Use the same eligible questions for scaling comparisons.
3. **Reuse experiments selectively.** Gold-only corrections can generally re-score unchanged predictions. Changed questions or corpus content invalidate affected predictions and caches. Preserve old outputs with a clear dataset/configuration version. Start with the scale-150 comparison; retain scaling only if validity gates pass.
4. **Make scoring defensible.** Validate numerical units and fiscal periods, define descriptive-answer rubrics, and obtain human labels for a stratified sample of roughly 25 outputs. Report judge agreement and disagreements. Add paired uncertainty estimates with their small-sample/company-dependence limitations. Report abstentions, errors, context infeasibility, costs, and latency separately.
5. **Ship reproducible evidence.** Complete FINDINGS and rewrite the README; remove unsupported superiority claims. Publish compact results, configuration and dataset manifests, dependency/environment information, source provenance, and exact commands. Validate a clean setup and a small smoke run. Add a brief walkthrough with successful and failed examples. A hosted UI is optional.

## Resources and estimate

- User confirmed **AI / LLM application engineer** as the target role and authorized **up to $25** for corrected benchmark API work. No deadline supplied.
- No additional access is required to continue local evidence review.
- The API cap is $25; execution still requires usable provider quota. Current keys/quota were not probed; do not infer their status from August notes or paste keys into chat.
- Planning estimate: **12–20 focused hours over about 1–2 weeks** for a reduced, defensible release; full repair of all ambiguous questions or missing filings may take longer. This is an estimate, not a completion guarantee.
- A **$10–25 reserve** is a planning allowance for selective reruns/rejudging, not a current-price quote. Historical artifacts suggest modest costs, but model availability, current prices, actual usage, and quota must be checked before execution. Local audit spend was $0.
- Human review of a small judge-calibration set adds credibility; the agent can prepare the passages and review sheet.

## Resume positioning

Safe present-tense claim: “Built a Python benchmark comparing vector retrieval, structure-aware retrieval, recursive LLM retrieval, and a context-stuffing baseline over 150 SEC filings and 75 financial QA questions, with cost/latency tracking and automated evaluation.”

After verification, replace the dataset size with the final retained count and add one measured finding. A result where vector retrieval matches a more complex system at lower cost is valuable if supported; there is no need to force the advanced method to win. Do not quote current judge scores as validated accuracy or claim statistical equivalence without evidence.

Next concrete action: reconcile source-backed gold answers and query evidence, starting with explicit-year numeric questions and the known NVIDIA/BlackRock defects.
