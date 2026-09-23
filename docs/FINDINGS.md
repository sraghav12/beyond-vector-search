# Findings — release v1

Completed September 14, 2026. These findings supersede the April/August exploratory scores and earlier README tables.

## Overall results

43 included questions across 150 distinct filings; 172 pipeline records, 129 non-naive answers judged or assigned a failure/abstention score. No records are marked as execution errors, but this does not imply every pipeline produced a useful answer.

| Strategy | Mean judge score | Mean seconds/query | Estimated generation cost/query |
|---|---:|---:|---:|
| Vector RAG | 0.4953 | 1.447 | $0.000722 |
| Structure-aware | 0.6047 | 8.853 | $0.012113 |
| Recursive LM | 0.3442 | 14.026 | $0.015594 |
| Full corpus | Context infeasible on 43/43 | Not comparable | $0 |

Structure-aware retrieval costs **16.77×** as much per query as vector RAG and takes **6.12×** as long in this run. Its +0.1093 score difference has a paired bootstrap 95% interval of [−0.0674, +0.2837]. Do not call this superiority or equivalence: the data are too limited to establish either.

The RLM-minus-vector difference is −0.1512 (query-bootstrap interval [−0.2953, −0.0070]). This applies to this particular capped, extraction-oriented implementation. It is not a general claim about recursive language models. The interval also ignores dependence among questions sharing companies.

## By question type

| Tier | n | Vector | Structure-aware | RLM |
|---|---:|---:|---:|---:|
| Single-hop | 20 | 0.835 | 0.690 | 0.650 |
| Multi-hop | 17 | 0.224 | 0.571 | 0.094 |
| Aggregation | 3 | 0.067 | 0.333 | 0.067 |
| Pairwise | 3 | 0.200 | 0.500 | 0.000 |

Vector retrieval works well for direct extraction in this sample. Broader section context helps some multi-document comparisons. Exhaustive ranking and aggregation remain difficult. The aggregation and pairwise samples are too small for stable tier-level conclusions.

## Concrete failure examples

- **q001, Apple FY2024 revenue:** all three working pipelines returned $391,035 million or its equivalent.
- **q029, NVIDIA versus AMD growth:** vector retrieval answered using old periods; structure-aware retrieval also confused periods; RLM returned a single dollar amount instead of a growth comparison. Correct labels alone do not guarantee correct temporal retrieval.
- **q041, five-company revenue ranking:** vector and structure-aware outputs lacked several requested companies; RLM returned `NO_FINAL_ANSWER`. Broad comparison requires evidence coverage across every requested company.
- **RLM:** eight explicit `NO_FINAL_ANSWER` results remained in the denominator. Natural-language refusals and incorrect answers are also possible outside that count.

## What the audit changed

The original bank had 61 gold answers containing “verify” or “likely,” 66 inconsistent query/gold evidence lists, and incorrectly labeled fiscal years for January-year-end companies. V1 includes 43 source-backed questions and excludes 32 with individual reasons. All included evidence exists at scale 150. Ten metadata labels were corrected without changing filing text or document IDs.

The previous scaling axis was confounded: larger corpora introduced missing evidence. V1 deliberately makes no claim about degradation with corpus size. The retained subset was selected on evidence/rubric readiness after earlier experiments, not randomly or as a preregistered test set.

## Cost accounting

- Generation from recorded usage: $1.22242.
- One-time vector indexing estimate: $0.32083.
- Judge estimate reconstructed from stored prompts and responses: $0.18424.
- Tiny generation probe: $0.000003.
- **Estimated total: $1.73**, within the authorized $25 allowance.

The historical judge recorded zero cost because the provider returned a dated model ID not present in the price table. Snapshot-name normalization is now fixed and regression-tested. The original judge token counts were not persisted, so its historical cost is an estimate, not a provider-invoice reconciliation. Original metric CSVs preserve those zero fields; use `spend_estimate.json` for the explained estimate. No second paid benchmark was run merely to repair this accounting metadata.

Rates were checked against the official [GPT-4o mini](https://developers.openai.com/api/docs/models/gpt-4o-mini) and [GPT-4o](https://developers.openai.com/api/docs/models/gpt-4o) pages. Cached-token discounts are not used in the reconstructed judge estimate.

## Limits of the evidence

- Answers were source-checked by an AI agent. The first 20 had a second raw-HTML/XBRL check by the same agent. There is no independent human judge-agreement measurement.
- GPT-4o judges GPT-4o-mini generations: same-provider/model-family bias is possible.
- One generation model, one run, one corpus size; no repeated-seed or tuning study.
- The local structure-aware implementation and RLM prompt/budget are specific implementations, not comprehensive evaluations of those approach families.
- Questions overlap in companies and sources. Bootstrap intervals resample questions and do not account for those dependencies.
- Industry-specific accounting definitions and different fiscal year ends remain part of explicitly stated question scopes.

These limits make v1 an interview-ready exploratory engineering study, not a publication-grade claim of a universally best retrieval strategy.
