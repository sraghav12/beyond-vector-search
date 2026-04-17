# Beyond Vector Search

**A benchmark comparing RAG retrieval strategies on real-world financial QA over SEC 10-K filings.**

---

## Why I Built This

Most RAG systems default to vector search — chunk the document, embed it, retrieve the top-k chunks. It works fine for short documents. It breaks badly on long, structured, legally dense documents like SEC 10-K filings, where the answer to a question might be a single number buried in a 1.2-million-character filing that is split across dozens of tables and sections.

I'm a CMU student researching how well different retrieval strategies actually work when documents are long and the questions are precise. Vector search is the baseline everyone assumes is good enough. I wanted to put that assumption to the test by running it head-to-head against two alternatives that take a fundamentally different approach to the problem.

---

## What It Does

This project benchmarks **three retrieval pipelines** on the same set of financial questions, over the same corpus of real SEC 10-K filings:

| Pipeline | Strategy |
|----------|----------|
| `vector_rag` | Standard RAG — chunk + embed + retrieve top-k |
| `pageindex` | Section-aware retrieval — index by document structure, not arbitrary chunks |
| `rlm` | Recursive Language Models — the LLM writes and executes its own REPL code to navigate the corpus |

A fourth pipeline, `naive_llm`, stuffs as much of the corpus as fits into a single prompt — the dumbest possible baseline.

Answers are scored by an LLM judge (0.0 to 1.0) against gold-standard answers extracted from the actual filings.

---

## The Corpus

Five major public companies, real SEC 10-K filings:

| Company | Filing | Approx. Size |
|---------|--------|-------------|
| Apple | AAPL 10-K FY2024 | 205K chars |
| NVIDIA | NVDA 10-K FY2025 | 341K chars |
| JPMorgan Chase | JPM 10-K FY2024 | 1.2M chars |
| Pfizer | PFE 10-K FY2023 | 664K chars |
| ExxonMobil | XOM 10-K FY2024 | ~450K chars |

Each filing is replicated across 10, 25, 50, 100, and 150-document corpus scales to test how retrieval quality degrades as the haystack grows.

---

## Questions

Three tiers of question difficulty:

- **Tier 1 (single-hop):** Extract one number from one filing. "What was Apple's total revenue for FY2024?"
- **Tier 2 (multi-hop):** Combine information across sections or filings. "Which company had higher R&D growth?"
- **Tier 3 (pairwise):** Compare and rank across companies.

---

## What I Am Working On Right Now

### Current status

The benchmark framework is complete. All three pipelines run end-to-end. The judge scores answers automatically.

**RLM pipeline** is where most of the active work is. The RLM uses a REPL environment where the LLM writes Python code, executes it against the corpus, and iterates until it finds the answer. It is the most capable approach for long-document retrieval but also the most complex to get right.

Recent fixes completed:
- Fixed a `KeyError` bug in the prompt template where f-string placeholders (`{a!r}`, `{hit}`) inside embedded REPL example code were being misinterpreted by Python's `str.format`
- Added keyword-anchored slicing for large filings (JPMorgan 1.2M chars, Pfizer 664K chars) that exceed the sub-LLM context window
- Added native Claude/Anthropic backend support alongside OpenAI and Gemini
- Fixed `.env` loading so shell-level empty env vars don't block `.env` values from loading
- Added `--query-delay` flag for inter-query rate-limit pacing

**Current blocker:** API rate limits. The RLM sends large prompts (full corpus metadata + filing excerpts) on every iteration. Free-tier limits (Anthropic: 10K TPM, Gemini: daily cap exhausted) are too low for a full benchmark run. Actively seeking research API credits through CMU and academic programs.

### Early results (5 questions, scale=150)

| Pipeline | Model | single_hop score |
|----------|-------|-----------------|
| vector_rag | gpt-4o-mini | 0.361 |
| pageindex | gpt-4o-mini | 0.207 |
| rlm | gpt-4o-mini | 3/5 correct (partial run) |
| rlm | claude-haiku-4-5 | 1/5 complete (rate-limited on q2-q5) |

Apple revenue (q001): RLM with Claude correctly returned `$391,035 million`. The REPL loop selected the right filing, sliced it with keyword anchoring, and extracted the exact figure.

---

## Project Structure

```
beyond-vector-search/
├── data/
│   ├── processed/          # Corpus JSONL files at each scale (10, 25, 50, 100, 150)
│   ├── queries/            # Question bank with gold answers and difficulty tiers
│   └── ground_truth/       # Gold answer index
├── pipelines/
│   ├── naive_llm/          # Baseline: stuff everything into one prompt
│   ├── vector_rag/         # Standard chunk + embed + retrieve
│   ├── pageindex/          # Section-aware retrieval
│   └── rlm/                # Recursive LM with REPL execution
├── evaluation/
│   ├── judge.py            # LLM-as-judge scoring (0.0 to 1.0)
│   └── runner.py           # Pipeline orchestration and result logging
├── scripts/
│   └── run_benchmark.py    # CLI entrypoint for all benchmark runs
└── results/
    ├── raw/                # Per-query JSONL results
    └── metrics/            # Scored CSVs and summary tables
```

---

## Running It

```bash
# Install dependencies
pip install -r requirements.txt

# Run the RLM pipeline, 5 questions, scale 150
python scripts/run_benchmark.py \
  --pipeline rlm \
  --model claude-haiku-4-5-20251001 \
  --scale 150 \
  --limit 5 \
  --query-timeout 900 \
  --rlm-max-timeout 780 \
  --query-delay 30 \
  --verbose

# Run all pipelines at all scales (full sweep)
python scripts/run_benchmark.py --pipeline all --scale all --model gpt-4o-mini
```

API keys in `.env`:
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
```

---

## Motivation

Vector search is the default assumption in almost every production RAG system. The question this project is trying to answer is: **how much does that assumption cost you when documents are long, structured, and require precise numerical extraction?**

SEC 10-K filings are a good stress test because:
- They are long (100K to 1.2M characters per filing)
- The answers are exact numbers — easy to judge right or wrong
- They are publicly available and legally significant
- They have consistent structure that section-aware retrieval can exploit

If an RLM that writes its own retrieval code can significantly outperform vector search on this task, it is evidence that retrieval strategy matters more than the field currently acknowledges — and that the standard chunk-and-embed approach is a meaningful bottleneck, not just a design choice.

---

*CMU student project. Active work in progress.*
