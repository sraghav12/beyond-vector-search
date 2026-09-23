# Multi-hop answer review

> Historical working document. Current release status and results are in [FINDINGS.md](FINDINGS.md) and [the v1 manifest](../data/release/v1/manifest.json).

Reviewed q021–q040: **17 source-backed corrections; 3 unresolved**. Total supported answers across both batches: **37/75**. Human review remains distinct from agent verification.

| ID | Corrected answer |
|---|---|
| q021 | Apple: $416,161 million in FY2025, versus NVIDIA: $215,938 million in fiscal 2026. Apple had higher revenue. |
| q022 | AMD spent approximately 23.4% of revenue on R&D ($8,091M / $34,639M) in FY2025, versus NVIDIA approximately 8.6% ($18,497M / $215,938M) in fiscal 2026. AMD was higher. |
| q023 | JPMorgan Chase: $58,471 million, versus Bank of America: $26,973 million for FY2024, using the comparative figures in their FY2025 filings. JPMorgan Chase was higher. |
| q024 | ExxonMobil invested more: $24,306 million in cash additions to property, plant and equipment, versus Chevron capital expenditures of $16,448 million in FY2024. |
| q025 | Eli Lilly had the higher margin: approximately 83.0% ((65,179−11,052)/65,179), versus Pfizer approximately 74.3% ((62,579−16,067)/62,579), for FY2025. |
| q026 | Apple revenue increased from $383,285 million in FY2023 to $391,035 million in FY2024, up $7,750 million (approximately 2.0%). |
| q027 | Walmart net sales were $706,413 million in fiscal 2026 and Costco net sales were $269,912 million in FY2025. The ratio was approximately 2.62× (706,413/269,912). |
| q028 | JPMorgan Chase had higher total assets at December 31, 2024: $4,002,814 million, versus Bank of America $3,261,299 million. |
| q029 | NVIDIA revenue grew approximately 65.5% ($215,938M versus $130,497M) in fiscal 2026; AMD grew approximately 34.3% ($34,639M versus $25,785M) in FY2025. NVIDIA grew faster. |
| q031 | Procter & Gamble paid approximately $4.08 per common share in its fiscal year ended June 30, 2025, versus Coca-Cola $2.04 per share in the year ended December 31, 2025. P&G was approximately twice as high. |
| q032 | JPMorgan Chase had higher FY2024 return on common equity: 18%, versus Goldman Sachs 12.7%. |
| q033 | Reality Labs operating loss increased from $16,120 million in FY2023 to $17,729 million in FY2024, a $1,609 million increase in loss (approximately 10.0%). |
| q034 | At December 31, 2025, ExxonMobil reported $34,241 million of long-term debt, versus ConocoPhillips $22,424 million. ExxonMobil was higher by $11,817 million. |
| q035 | Linde had the higher FY2025 reported operating margin: approximately 26.3% ($8,923M/$33,986M), versus Freeport-McMoRan approximately 25.2% ($6,518M/$25,915M). |
| q036 | Apple had more cash and cash equivalents: $35,934 million at September 27, 2025, versus Alphabet $30,708 million at December 31, 2025. Apple was higher by $5,226 million. |
| q039 | Johnson & Johnson spent approximately 15.6% of revenue on R&D ($14,665M/$94,193M), versus AbbVie approximately 14.9% ($9,096M/$61,160M), in FY2025. Johnson & Johnson was slightly higher. |
| q040 | GE Aerospace consolidated revenue increased from $35,348 million in 2023 to $38,702 million in 2024, up $3,354 million (approximately 9.5%). Equipment revenue benefited from pricing and customer/product mix; services grew from spare-parts volume, pricing and internal shop-visit workscope. |

## Unresolved items

- **q030:** Deere presents segment operating profit and consolidated pretax income, rather than the same explicit consolidated operating-income line as Caterpillar. Define a comparable metric before grading.
- **q037:** WFC Item 8 incorporates financial statements from the Annual Report to Shareholders, not present in the processed wrapper filing. Obtain source financials before grading.
- **q038:** McDonald's International Developmental Licensed Markets & Corporate segment includes corporate activity. Do not equate that full segment with strictly non-US geographic revenue without a supported convention.

## Reproducibility

[Exact evidence, previous values and calculations](../data/ground_truth/provenance/2026-09-13_multi_hop.json) · [Current dataset manifest](../data/ground_truth/provenance/current_manifest.json)

Changed questions in this batch: q023, q024, q025, q039, q040. These require fresh predictions. The current manifest lists evidence gaps at each scale. Old result files and corpora were preserved. The existing runner does not automatically exclude unresolved questions merely because a verification status was added; select a validated subset explicitly before running.

Checks: every saved excerpt matches its document hash and text offsets; all 37 supported question/gold/evidence pairs agree. Arithmetic for the nine derived comparisons was recomputed from the cited inputs. No paid API calls.
