# Methodology — release v1

## Fixed experiment

- 150 distinct SEC 10-K filings, 30 companies, five stored vintages each.
- 43 questions: 20 single-hop, 17 multi-hop, three aggregation, three pairwise.
- `gpt-4o-mini` for all generation; `gpt-4o` for all actual judge calls.
- One fresh run per question/pipeline on the corrected release corpus; 172 records total.
- Release question/corpus snapshot frozen before these runs. Earlier exploratory results had already been seen, so the selection is not a pristine held-out test.

## Corpus and references

`data/release/v1/corpus_150.jsonl.gz` is the exact frozen processed corpus. The manifest records all filing accessions, individual text hashes, and aggregate file hashes. Ten January-year-end NVIDIA/Walmart fiscal labels were corrected from cover-page dates in the release copy. Filing text and immutable legacy document IDs were preserved.

“Most recent” means latest filing present for that company in the frozen corpus. BlackRock's most recent stored filing is FY2023. Fiscal periods vary across companies. Questions whose accounting definitions could change the correct answer were clarified before running.

Included reference answers have source excerpts, offsets, document hashes, accessions and calculation notes. The first 20 additionally passed a raw-HTML/XBRL inspection by the same agent. `verified: false` continues to mean no explicit independent human certification; `verification_status: agent_verified` identifies source checks actually completed.

The 32 excluded questions remain listed with individual reasons. Examples include missing WFC financial statements, treating book equity as market capitalization, missing or inapplicable metrics, and descriptive comparisons without reviewed rubrics. V1 does not claim to complete the original 75-label research dataset.

## Pipeline configuration

| Parameter | Value |
|---|---|
| Vector embedding model | text-embedding-3-small |
| Chunk size / overlap | 1024 / 128 tokens |
| Retrieved chunks | 5 |
| Structure-aware selected sections | up to 8 |
| RLM depth | 2 |
| RLM maximum root iterations | 5 |
| RLM cumulative token budget | 300000 |
| RLM internal soft timeout | 180 seconds |
| Runner timeout argument | 240 seconds |
| Inter-query delay | 1 second, outside query latency |

The CLI name `--rlm-max-subcalls` maps to the library's `max_iterations`; it is not a guarantee of exactly five sub-model calls. Internal budgets can be checked between calls and are not an account-level dollar cap. The runner uses thread-based timeouts and cannot forcibly terminate an in-flight API request. These are implementation limits, not guarantees otherwise implied by earlier comments.

The full-corpus baseline detects an over-window corpus before calling the provider. It is structurally infeasible in this configuration. Its `status=ok` means the guard executed correctly, not that an answer was produced.

The structure-aware pipeline is local code, not the external PageIndex service. The RLM prompt emphasizes numerical extraction and early final answers; multi-document comparison performance therefore measures this implementation and budget, not the best achievable recursive-LM system.

## Scoring and uncertainty

The headline metric is mean GPT-4o judge score on a 0–1 rubric; it is **not exact-match accuracy**. The exact judge system/user templates are frozen in `judge_config.json`. All attempts remain in the denominator; errors and explicit abstentions are not silently dropped. Context-infeasible naive rows are displayed separately, with no accuracy claim.

Actual calls use one fixed judge. The naive guard retains the generation model in a legacy CSV metadata field because no judge was called; the report recognizes this case explicitly. It rejects duplicate/missing rows and an unexpected model on substantive judged answers.

Paired differences align question IDs and use 10,000 bootstrap resamples of questions, seed 20260913. Questions share companies and evidence, so independence is imperfect. Intervals are exploratory summaries, not population-level significance claims. With only three aggregation/pairwise questions, their means are especially unstable.

Deterministic metrics saved by the original runner are supplementary only. Its any-matching-number heuristic can confuse years or incidental figures with the requested answer; the release does not promote it as validated numeric accuracy.

No independent human-vs-judge agreement statistic is claimed. Human calibration, repeated runs, broader descriptive rubrics, and a corrected scaling study are future research, not completed results.

## Costs and latency

Query latency is measured around the pipeline call, excluding initial index construction and the one-second pacing delay. Generation costs use recorded token usage and configured list prices. The one-time index estimate is included separately. No cold-start production latency or concurrency claim is made.

A dated-model-name price lookup defect caused initial judge costs to be recorded as zero. It is fixed and covered by a regression test. Historical judge usage was not saved, so `spend_estimate.json` reconstructs costs from stored prompts and response text at uncached rates. The $1.73 total is an estimate, not an invoice reconciliation.

## Reproducibility and provenance

Run `make prepare`, `make test`, and `make report` without keys. Direct dependency versions and the successful clean environment are recorded. `make benchmark` uses the release files explicitly and writes to `results/reproduction` by default, leaving published results intact.

Answer/index cache keys include corpus fingerprints. A clean checkout without `.cache` is required for a fresh latency experiment. Output resume alone does not clear answer caches. API model aliases can evolve, and model generation can vary; reproduction checks the method and artifacts, not guaranteed identical new outputs.
