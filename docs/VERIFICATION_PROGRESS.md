# Source verification progress

> Historical working document. Current release status and results are in [FINDINGS.md](FINDINGS.md) and [the v1 manifest](../data/release/v1/manifest.json).

**Latest status:** 37 supported answers, three unresolved reviewed questions, and 35 not yet reviewed. See [the multi-hop review](MULTI_HOP_REVIEW.md) and [current manifest](../data/ground_truth/provenance/current_manifest.json). The single-hop batch below is the first completed milestone.

## September 13, 2026: single-hop batch complete

All 20 single-hop questions (q001–q020) now have answers checked directly against local SEC filing text. The question and gold files agree on answer text and supporting document IDs for this batch. The remaining 55 questions are pending.

This is **agent verification, not human sign-off**. `verified` remains false; `verification_status: agent_verified` records the work performed. Do not describe the full benchmark as verified.

### Review artifacts

- [Source excerpts and change history](../data/ground_truth/provenance/2026-09-13_single_hop.json): exact source text, character offsets, document hashes, accession numbers, source index URLs, previous answers/evidence, and interpretation notes.
- [Dataset revision manifest](../data/ground_truth/provenance/2026-09-13_manifest.json): file hashes, completed/pending question IDs, required reruns, and evidence gaps.

### Material corrections

| Question | Source-backed answer / decision |
|---|---|
| q002 NVIDIA R&D | $18,497 million, fiscal 2026 ended January 25, 2026; the corpus document's legacy ID ends in 2025. |
| q004 ExxonMobil revenue | Explicitly ask for total revenues and other income: $349,585 million for 2024. Sales and other operating revenue is a different line item. |
| q005 Pfizer revenue | Explicitly request originally reported FY2023 revenue: $58,496 million from PFE_10K_2023, avoiding later recast comparatives. |
| q009 Caterpillar margin | Reported operating margin of 20.2% in 2024, distinct from adjusted 20.7%. |
| q010 Goldman provision | $1,348 million for 2024; replaces an instruction to verify. |
| q012 Chevron Capex | $16,448 million for 2024; question explicitly excludes equity affiliate Capex. |
| q013 BlackRock AUM | About $10.0 trillion at December 31, 2023, the latest BlackRock filing available in this corpus. |
| q014 Home Depot comparable sales | −1.8% in fiscal 2024, from HD_10K_2024. |
| q015 Coca-Cola revenue | $47,061 million for 2024. |
| q016 Alphabet segments | Ask for the three segment-results categories; Other Bets itself combines multiple operating segments. |
| q017 AMD gross margin | Reported 49% in fiscal 2024. |

Other questions received precise answers, synchronized evidence, and/or source records. “Most recent” is interpreted relative to the frozen corpus for the company, not today's SEC inventory. Legacy document IDs have not been renamed.

### Checks and experiment consequences

Every stored excerpt was checked against its source character offsets and SHA-256 hash. All 20 reviewed question/gold pairs agree, retain human-review flags as false, and contain no “verify” or “likely” answer placeholders. The Services-revenue percentage calculation was checked separately.

Questions **q004, q005, q012, q016** have changed text and need fresh predictions. Old judge scores are stale under the corrected gold revision. Gold-only changes can reuse predictions only if question and corpus context are unchanged.

The updated evidence reveals PFE_10K_2023 missing at scales 50 and 100, and HD_10K_2024 missing at scale 50. All reviewed evidence is present at scale 150. Rebuild smaller corpora after the full evidence review; rerunning before that would waste budget. No corpus or previous result file was overwritten in this batch.

Next: review multi-hop questions q021–q040, then resolve aggregation/comparison rubrics and missing-source cases. Paid API spend so far: $0 of the authorized $25.
