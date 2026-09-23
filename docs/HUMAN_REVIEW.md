# Human review: first 20 questions

**Second-pass check completed:** all 20 answers were confirmed against the original locally saved SEC HTML, using XBRL facts and raw narrative text. See [the per-answer results](SECOND_PASS_VERIFICATION.md). This is agent verification; the checkboxes below remain available for explicit human decisions.

Start with q001, q002, and q004. These cover a straightforward extraction, a fiscal-year correction, and a clarified accounting definition. Review the remaining entries afterward.

For each item, check that the excerpt supports the answer, the year and units match, and the question asks for that exact metric. For tables, read the year headers before choosing a column. Quotes below preserve source text; they were extracted by the agent and are not independent human verification. Open the SEC filing index link when you need wider context or want to spot-check the extraction against the original filing.

Use exactly one decision per question: **Accept**, **Needs correction**, or **Unsure**. Write a short reason for the latter two. An unchecked item is pending. You can edit this file or send decisions in chat, such as `q001 accept; q002 unsure — check fiscal year`. The agent will apply explicit decisions to the verification records; creating this checklist does not mark anything human-reviewed.

Do not use the older GOLD_VERIFICATION_DRAFT.md as the current answer list; it contains superseded proposals.

---

## q001

**Question:** What was Apple's total net sales (revenue) for fiscal year 2024?

**Proposed answer:** $391,035 million ($391.035 billion).

**Interpretation:** FY2024 ended September 28, 2024; comparative income-statement column.

**Source:** AAPL_10K_2025 — [SEC filing index](https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/0000320193-25-000079-index.html); filed 2025-10-31. Exact local-text offsets: 126378–126854.

```text
ng notes.

Apple Inc. | 2025 Form 10-K | 28

Apple Inc.
CONSOLIDATED STATEMENTS OF OPERATIONS
(In millions, except number of shares, which are reflected in thousands, and per-share amounts)

Years ended

September 27,
2025

September 28,
2024

September 30,
2023

Net sales:

Products
307,003

294,866

298,085

Services
109,158

96,169

85,200

Total net sales
416,161

391,035

383,285

Cost of sales:

Products
194,116

185,233

189,282

Services
26,844

25,119

24,855

To
```

- [ ] Accept
- [ ] Needs correction
- [ ] Unsure

**Reviewer note:**

---

## q002

**Question:** How much did NVIDIA spend on research and development in its most recent fiscal year reported in its 10-K?

**Proposed answer:** $18,497 million ($18.497 billion) in fiscal 2026, ended January 25, 2026.

**Interpretation:** Latest NVIDIA filing in frozen corpus is FY2026 despite legacy doc_id ending 2025.

**Source:** NVDA_10K_2025 — [SEC filing index](https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/0001045810-26-000021-index.html); filed 2026-02-25. Exact local-text offsets: 207385–207751.

```text
gross margin was an unfavorable impact of 2.6% and 2.3% in fiscal years 2026 and 2025, respectively.

Operating Expenses

Year Ended

Jan 25, 2026

Jan 26, 2025

Change

Change

($ in millions)

Research and development
18,497

12,914

5,583

43

Sales, general and administrative
4,579

3,491

1,088

31

Total operating expenses
23,076

16,405

6,671

41
The incre
```

- [ ] Accept
- [ ] Needs correction
- [ ] Unsure

**Reviewer note:**

---

## q003

**Question:** What was JPMorgan Chase's total net revenue for fiscal year 2024?

**Proposed answer:** $177,556 million ($177.556 billion), reported total net revenue for FY2024.

**Interpretation:** FY2024 reported basis; do not substitute managed-basis revenue.

**Source:** JPM_10K_2025 — [SEC filing index](https://www.sec.gov/Archives/edgar/data/19617/000162828026008131/0001628280-26-008131-index.html); filed 2026-02-13. Exact local-text offsets: 195922–196289.

```text

Financial
THREE-YEAR SUMMARY OF CONSOLIDATED FINANCIAL HIGHLIGHTS (unaudited)

As of or for the year ended December 31,
(in millions, except per share, ratio, employee data and where otherwise noted)

2025

2024

2023

Selected income statement data

Total net revenue

182,447

177,556

(g)
158,104

Total noninterest expense

95,640

91,797

(g)
87,172

Pre-provis
```

- [ ] Accept
- [ ] Needs correction
- [ ] Unsure

**Reviewer note:**

---

## q004

**Question:** What was ExxonMobil's total revenues and other income for fiscal year 2024?

**Proposed answer:** $349,585 million ($349.585 billion).

**Interpretation:** Explicitly use total revenues and other income, including affiliate and other income; sales and other operating revenue alone is $339,247 million. Question changed; rerun required.

**Previous question:** What was ExxonMobil's total revenue for fiscal year 2024?

The question wording changed. Please review whether the revised scope is clear as well as whether the answer is correct.

**Source:** XOM_10K_2025 — [SEC filing index](https://www.sec.gov/Archives/edgar/data/34088/000003408826000045/0000034088-26-000045-index.html); filed 2026-02-18. Exact local-text offsets: 256736–257223.

```text
able of Contents
The information in the Notes to Consolidated Financial Statements is an integral part of these statements.
CONSOLIDATED STATEMENT OF INCOME

(millions of dollars)
Note
Reference
Number
2025
2024
2023

Revenues and other income

Sales and other operating revenue

3
323,905

339,247

334,697

Income from equity affiliates

8
5,064

6,194

6,385

Other income

3,269

4,144

3,500

Total revenues and other income

332,238

349,585

344,582

Costs and other deductions

C
```

- [ ] Accept
- [ ] Needs correction
- [ ] Unsure

**Reviewer note:**

---

## q005

**Question:** What was Pfizer's total revenue for fiscal year 2023, as originally reported in its fiscal 2023 10-K?

**Proposed answer:** $58,496 million ($58.496 billion).

**Interpretation:** Use the original FY2023 statement, not a recast comparative in a later filing. Question changed; rerun required.

**Previous question:** What was Pfizer's total revenue for fiscal year 2023?

The question wording changed. Please review whether the revised scope is clear as well as whether the answer is correct.

**Source:** PFE_10K_2023 — [SEC filing index](https://www.sec.gov/Archives/edgar/data/78003/000007800324000039/0000078003-24-000039-index.html); filed 2024-02-22. Exact local-text offsets: 343675–344071.

```text
 New York

February 22, 2024

Pfizer Inc.

2023 Form 10-K

51

Consolidated Statements of Income
Pfizer Inc. and Subsidiary Companies

Year Ended December 31,

(MILLIONS, EXCEPT PER SHARE DATA)

2023

2022

2021

Revenues:

Product revenues (a)

50,914

91,793

73,636

Alliance revenues (a)

7,582

8,537

7,652

Total revenues

58,496

100,330

81,288

Costs and expenses:

Cost of sales (b), (
```

- [ ] Accept
- [ ] Needs correction
- [ ] Unsure

**Reviewer note:**

---

## q006

**Question:** How many employees did Walmart report in its most recent 10-K filing?

**Proposed answer:** Approximately 2.1 million associates worldwide as of January 31, 2026.

**Interpretation:** Latest Walmart filing in frozen corpus covers fiscal 2026 despite legacy doc_id ending 2025.

**Source:** WMT_10K_2025 — [SEC filing index](https://www.sec.gov/Archives/edgar/data/104169/000010416926000055/0000104169-26-000055-index.html); filed 2026-03-13. Exact local-text offsets: 19405–19666.

```text
mobile applications and service offerings, as well as our supply chain, combined with approximately 2.1 million associates as of January 31, 2026, to better serve our customers. Our strategies increasingly include the use of AI-powered tools to support customer
```

- [ ] Accept
- [ ] Needs correction
- [ ] Unsure

**Reviewer note:**

---

## q007

**Question:** What was Boeing's total revenue for fiscal year 2024?

**Proposed answer:** $66,517 million ($66.517 billion).

**Interpretation:** FY2024 comparative column; dollars in millions.

**Source:** BA_10K_2025 — [SEC filing index](https://www.sec.gov/Archives/edgar/data/12927/000162828026004357/0001628280-26-004357-index.html); filed 2026-01-30. Exact local-text offsets: 103124–103470.

```text
 our government services business remains stable.

Consolidated Results of Operations
The following table summarizes key indicators of consolidated results of operations:

(Dollars in millions, except per share data)

Years ended December 31,
2025

2024

2023

Revenues
$89,463

$66,517

$77,794

GAAP

Earnings/(loss) from operations
$4,281

($1
```

- [ ] Accept
- [ ] Needs correction
- [ ] Unsure

**Reviewer note:**

---

## q008

**Question:** What is Meta Platforms' primary source of revenue according to its most recent 10-K?

**Proposed answer:** Advertising placements on its family of apps are its primary source of revenue.

**Interpretation:** Latest Meta filing in frozen corpus; business description directly answers the question.

**Source:** META_10K_2025 — [SEC filing index](https://www.sec.gov/Archives/edgar/data/1326801/000162828026003942/0001628280-26-003942-index.html); filed 2026-01-29. Exact local-text offsets: 25364–25639.

```text
Currently, we generate substantially all of our revenue from selling advertising placements on our family of apps to marketers, which is reflected in FoA. Ads on our platform enable marketers to reach people across a range of marketing objectives, such as generating leads or
```

- [ ] Accept
- [ ] Needs correction
- [ ] Unsure

**Reviewer note:**

---

## q009

**Question:** What was Caterpillar's operating profit margin for fiscal year 2024?

**Proposed answer:** 20.2% reported operating profit margin for FY2024.

**Interpretation:** Reported operating margin; 20.7% is adjusted and is not the requested metric.

**Source:** CAT_10K_2025 — [SEC filing index](https://www.sec.gov/Archives/edgar/data/18230/000001823026000008/0000018230-26-000008-index.html); filed 2026-02-13. Exact local-text offsets: 119632–119923.

```text
Operating profit as a percent of sales and revenues was 16.5 percent in 2025, compared with 20.2 percent in 2024. Adjusted operating profit margin was 17.2 percent in 2025, compared with 20.7 percent in 2024.
• Profit per share for 2025 was $18.81, and excluding the items in the table below
```

- [ ] Accept
- [ ] Needs correction
- [ ] Unsure

**Reviewer note:**

---

## q010

**Question:** What was Goldman Sachs' provision for credit losses in fiscal year 2024?

**Proposed answer:** $1,348 million ($1.348 billion).

**Interpretation:** FY2024 provision, not FY2025 net benefit; table units millions.

**Source:** GS_10K_2025 — [SEC filing index](https://www.sec.gov/Archives/edgar/data/886982/000088698226000091/0000886982-26-000091-index.html); filed 2026-02-25. Exact local-text offsets: 353106–353396.

```text
sion for credit losses on loans and lending commitments.
The table below presents our provision for credit losses.

Year Ended December

$ in millions
2025
2024

2023

Provision for credit losses
(1,113)

1,348

1,028

Goldman Sachs 2025 Form 10-K

71

THE GOLDMAN SACHS GROUP, INC. AND SUB
```

- [ ] Accept
- [ ] Needs correction
- [ ] Unsure

**Reviewer note:**

---

## q011

**Question:** What are the main business segments reported by Johnson & Johnson in its most recent 10-K?

**Proposed answer:** Innovative Medicine and MedTech.

**Interpretation:** Latest Johnson & Johnson filing in the frozen corpus.

**Source:** JNJ_10K_2025 — [SEC filing index](https://www.sec.gov/Archives/edgar/data/200406/000020040626000016/0000200406-26-000016-index.html); filed 2026-02-11. Exact local-text offsets: 19282–19448.

```text
The Company is organized into two business segments: Innovative Medicine and MedTech. Additional information required by this item is incorporated herein by reference
```

- [ ] Accept
- [ ] Needs correction
- [ ] Unsure

**Reviewer note:**

---

## q012

**Question:** What were Chevron's capital expenditures (Capex), excluding equity affiliate capital expenditures, for fiscal year 2024?

**Proposed answer:** $16,448 million ($16.448 billion).

**Interpretation:** Company Capex excludes separately disclosed equity affiliate Capex. Replaced ambiguous legacy capital-and-exploratory wording; rerun required.

**Previous question:** What was Chevron's capital and exploratory expenditures for fiscal year 2024?

The question wording changed. Please review whether the revised scope is clear as well as whether the answer is correct.

**Source:** CVX_10K_2025 — [SEC filing index](https://www.sec.gov/Archives/edgar/data/93410/000009341026000078/0000093410-26-000078-index.html); filed 2026-02-24. Exact local-text offsets: 211070–211421.

```text
es, which represents the cash from operations available to creditors and investors after investing in the business.

Year ended December 31

Millions of dollars
2025

2024

2023

Net cash provided by operating activities
33,939

31,492

35,609

Less: Capital expenditures
17,347

16,448

15,829

Free Cash Flow
16,592

15,044

19,780

Adjusted Free Ca
```

- [ ] Accept
- [ ] Needs correction
- [ ] Unsure

**Reviewer note:**

---

## q013

**Question:** What was BlackRock's total assets under management (AUM) as reported in its most recent 10-K?

**Proposed answer:** Approximately $10.0 trillion in assets under management as of December 31, 2023.

**Interpretation:** Latest available BlackRock filing is FY2023; BLK_10K_2024 is absent.

**Source:** BLK_10K_2023 — [SEC filing index](https://www.sec.gov/Archives/edgar/data/1364742/000095017024019271/0000950170-24-019271-index.html); filed 2024-02-23. Exact local-text offsets: 5943–6084.

```text
$10.0 trillion of assets under management ("AUM") at December 31, 2023. With approximately 19,800 employees in more than 30 countries who ser
```

- [ ] Accept
- [ ] Needs correction
- [ ] Unsure

**Reviewer note:**

---

## q014

**Question:** What was Home Depot's comparable sales (comp sales) growth rate for fiscal year 2024?

**Proposed answer:** Comparable sales decreased 1.8% (growth rate −1.8%) in fiscal 2024.

**Interpretation:** Company-wide comparable sales, not U.S.-only or total net sales. Required passage is in FY2024 filing; smaller corpora need evidence revalidation.

**Source:** HD_10K_2024 — [SEC filing index](https://www.sec.gov/Archives/edgar/data/354950/000035495025000085/0000354950-25-000085-index.html); filed 2025-03-21. Exact local-text offsets: 152281–152533.

```text
Total comparable sales decreased 1.8% in fiscal 2024, reflecting a 1.0% decrease in comparable customer transactions and a 0.9% decrease in comparable average ticket compared to fiscal 2023. The decrease in comparable customer transactions primarily re
```

- [ ] Accept
- [ ] Needs correction
- [ ] Unsure

**Reviewer note:**

---

## q015

**Question:** What was Coca-Cola's net revenue for fiscal year 2024?

**Proposed answer:** $47,061 million ($47.061 billion).

**Interpretation:** Narrative explicitly identifies FY2024 net operating revenues.

**Source:** KO_10K_2025 — [SEC filing index](https://www.sec.gov/Archives/edgar/data/21344/000162828026010047/0001628280-26-010047-index.html); filed 2026-02-20. Exact local-text offsets: 250621–250796.

```text
Net operating revenues were $47,941 million in 2025, compared to $47,061 million in 2024, an increase of $880 million, or 2%.
The following table illustrates, on a percentage 
```

- [ ] Accept
- [ ] Needs correction
- [ ] Unsure

**Reviewer note:**

---

## q016

**Question:** Under which three categories does Alphabet report its segment results in its most recent 10-K?

**Proposed answer:** Google Services, Google Cloud, and Other Bets. Other Bets combines multiple operating segments that are not individually material.

**Interpretation:** Distinguish three segment-results categories from the underlying operating-segment count. Question changed; rerun required.

**Previous question:** How many operating segments does Alphabet (Google) report in its most recent 10-K?

The question wording changed. Please review whether the revised scope is clear as well as whether the answer is correct.

**Source:** GOOGL_10K_2025 — [SEC filing index](https://www.sec.gov/Archives/edgar/data/1652044/000165204426000018/0001652044-26-000018-index.html); filed 2026-02-05. Exact local-text offsets: 316774–316923.

```text
We report our segment results as Google Services, Google Cloud, and Other Bets:
• Google Services includes products and services such as ads, Android
```

**Source:** GOOGL_10K_2025 — [SEC filing index](https://www.sec.gov/Archives/edgar/data/1652044/000165204426000018/0001652044-26-000018-index.html); filed 2026-02-05. Exact local-text offsets: 317633–317743.

```text
Other Bets is a combination of multiple operating segments that are not individually material. Revenues from O
```

- [ ] Accept
- [ ] Needs correction
- [ ] Unsure

**Reviewer note:**

---

## q017

**Question:** What was AMD's gross margin percentage for fiscal year 2024?

**Proposed answer:** 49% reported gross margin for FY2024.

**Interpretation:** FY2024 comparative gross margin; GAAP gross profit divided by revenue rounds to reported 49%.

**Source:** AMD_10K_2025 — [SEC filing index](https://www.sec.gov/Archives/edgar/data/2488/000000248826000018/0000002488-26-000018-index.html); filed 2026-02-04. Exact local-text offsets: 231668–232229.

```text
e future. Substantially all of our sales transactions are denominated in U.S. dollars.

Comparison of Gross Margin, Expenses, Interest Expense, Other Income (expense) and Income Taxes
The following is a summary of certain Consolidated Statement of Operations data for 2025 and 2024:

December 27, 2025

December 28, 2024

(In millions, except for percentages)

Net revenue
34,639

25,785

Cost of sales
16,456

12,114

Amortization of acquisition-related intangibles
1,031

946

Gross profit
17,152

12,725

Gross margin
50

49

Research and development
8,091


```

- [ ] Accept
- [ ] Needs correction
- [ ] Unsure

**Reviewer note:**

---

## q018

**Question:** What was Procter & Gamble's net sales for fiscal year 2024?

**Proposed answer:** $84,039 million ($84.039 billion), for the fiscal year ended June 30, 2024.

**Interpretation:** Consolidated earnings statement, FY2024 column, units millions.

**Source:** PG_10K_2025 — [SEC filing index](https://www.sec.gov/Archives/edgar/data/80424/000008042425000076/0000080424-25-000076-index.html); filed 2025-08-04. Exact local-text offsets: 182006–182322.

```text
eriorate.

/s/ Deloitte & Touche LLP

Cincinnati, Ohio

August 4, 2025

36 The Procter & Gamble Company
Consolidated Statements of Earnings

Amounts in millions except per share amounts; fiscal years ended June 30
2025

2024

2023

NET SALES
84,284

84,039

82,006

Cost of products sold
41,164

40,848

42,760

Sell
```

- [ ] Accept
- [ ] Needs correction
- [ ] Unsure

**Reviewer note:**

---

## q019

**Question:** What percentage of Apple's total net sales came from the Services segment in fiscal year 2024?

**Proposed answer:** Approximately 24.6% ($96,169 million / $391,035 million × 100).

**Interpretation:** FY2024 Services net sales divided by FY2024 total net sales; 24.593458...%.

**Source:** AAPL_10K_2025 — [SEC filing index](https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/0000320193-25-000079-index.html); filed 2025-10-31. Exact local-text offsets: 126378–126854.

```text
ng notes.

Apple Inc. | 2025 Form 10-K | 28

Apple Inc.
CONSOLIDATED STATEMENTS OF OPERATIONS
(In millions, except number of shares, which are reflected in thousands, and per-share amounts)

Years ended

September 27,
2025

September 28,
2024

September 30,
2023

Net sales:

Products
307,003

294,866

298,085

Services
109,158

96,169

85,200

Total net sales
416,161

391,035

383,285

Cost of sales:

Products
194,116

185,233

189,282

Services
26,844

25,119

24,855

To
```

**Calculation:** 96,169 / 391,035 × 100 = 24.593451%, rounded to 24.6%.

- [ ] Accept
- [ ] Needs correction
- [ ] Unsure

**Reviewer note:**

---

## q020

**Question:** What was the total stockholders' equity for JPMorgan Chase as of December 31, 2024?

**Proposed answer:** $344,758 million ($344.758 billion).

**Interpretation:** Total stockholders equity at December 31, 2024, including preferred equity.

**Source:** JPM_10K_2025 — [SEC filing index](https://www.sec.gov/Archives/edgar/data/19617/000162828026008131/0001628280-26-008131-index.html); filed 2026-02-13. Exact local-text offsets: 197843–198210.

```text
osses

777,332

681,320

571,552

Loans

1,493,429

1,347,988

1,323,706

Total assets

4,424,900

4,002,814

3,875,393

Deposits

2,559,320

2,406,032

2,400,688

Long-term debt

435,206

401,418

391,825

Common stockholders' equity

342,393

324,708

300,474

Total stockholders' equity

362,438

344,758

327,878

Employees

318,512

317,233

309,926

Credit qual
```

- [ ] Accept
- [ ] Needs correction
- [ ] Unsure

**Reviewer note:**

---
