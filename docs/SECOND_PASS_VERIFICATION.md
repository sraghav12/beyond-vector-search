# Second-pass answer verification

All **20/20 answers** are supported for their stated question scope. No answer changes were required.

This check used the original locally saved SEC HTML, rather than only the processed corpus excerpts. Numeric checks match XBRL concepts, reporting periods, USD units, scaling and applicable dimensions. Narrative answers were checked against raw filing text. This is a second pass by the same agent, not an independent human review or a fresh SEC download.

| ID | Confirmed answer | Check |
|---|---|---|
| q001 | $391,035 million ($391.035 billion). | XBRL facts |
| q002 | $18,497 million ($18.497 billion) in fiscal 2026, ended January 25, 2026. | XBRL facts |
| q003 | $177,556 million ($177.556 billion), reported total net revenue for FY2024. | XBRL facts |
| q004 | $349,585 million ($349.585 billion). | XBRL facts |
| q005 | $58,496 million ($58.496 billion). | XBRL facts |
| q006 | Approximately 2.1 million associates worldwide as of January 31, 2026. | Original filing text |
| q007 | $66,517 million ($66.517 billion). | XBRL facts |
| q008 | Advertising placements on its family of apps are its primary source of revenue. | Original filing text |
| q009 | 20.2% reported operating profit margin for FY2024. | Original filing text |
| q010 | $1,348 million ($1.348 billion). | XBRL facts |
| q011 | Innovative Medicine and MedTech. | Original filing text |
| q012 | $16,448 million ($16.448 billion). | XBRL facts |
| q013 | Approximately $10.0 trillion in assets under management as of December 31, 2023. | Original filing text |
| q014 | Comparable sales decreased 1.8% (growth rate −1.8%) in fiscal 2024. | Original filing text |
| q015 | $47,061 million ($47.061 billion). | XBRL facts |
| q016 | Google Services, Google Cloud, and Other Bets. Other Bets combines multiple operating segments that are not individually material. | Original filing text |
| q017 | 49% reported gross margin for FY2024. | XBRL facts + table text + recalculation |
| q018 | $84,039 million ($84.039 billion), for the fiscal year ended June 30, 2024. | XBRL facts |
| q019 | Approximately 24.6% ($96,169 million / $391,035 million × 100). | XBRL facts + recalculation |
| q020 | $344,758 million ($344.758 billion). | XBRL facts |

## Qualifications that must stay with the benchmark

- “Most recent” means the latest filing for that company in this frozen corpus. NVIDIA and Walmart legacy document IDs end in 2025 although their filings cover fiscal 2026; BlackRock is limited to FY2023.
- q004 explicitly includes other income; q005 explicitly uses originally reported revenue; q012 excludes affiliate Capex; q016 asks for segment-result categories rather than a literal count of underlying operating segments. These four changed questions still require fresh predictions.
- AMD’s 49% is the filing’s rounded whole-percent margin; the source amounts yield about 49.350%. Apple Services yields about 24.593%, rounded to 24.6%.
- Gold `verified` flags remain false. The user expressed general agreement, but this report records agent re-verification rather than inventing individual human decisions.

[Machine-readable raw-filing evidence](../data/ground_truth/provenance/2026-09-13_second_pass.json) · [Human review checklist](HUMAN_REVIEW.md)
