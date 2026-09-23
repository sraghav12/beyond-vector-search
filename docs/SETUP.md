# Setup — release v1

Validated on macOS with Python 3.13.5 in a newly created virtual environment. Source code supports Python >=3.11, but other Python/OS combinations were not tested in this release.

## Offline reproduction

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-release.txt
make prepare
make test
make report
```

`make prepare` extracts the included 19 MB gzip snapshot to `data/processed/release_v1/corpus_150.jsonl` and verifies checksums. No fresh SEC download is needed. `make report` rebuilds `results/release_v1/report.json` from included raw JSONL and scored CSVs. Tests do not call model APIs.

The release is installable as a package after dependencies are installed:

```bash
pip install --no-deps -e .
```

`requirements-release.txt` pins direct runtime/test dependencies. `results/release_v1/environment.json` records the original environment; `environment_clean.json` records the separately installed validation environment. The older broad `requirements.txt` contains optional development dependencies and is not the release install path.

## Paid reproduction

Create a local `.env` with `OPENAI_API_KEY` (see `.env.example`). The release requires OpenAI access to GPT-4o mini, GPT-4o, and text-embedding-3-small. Existing xAI, Gemini and Anthropic keys are not needed for this experiment.

```bash
make benchmark OUTPUT=results/my_reproduction
python scripts/release_report.py --results results/my_reproduction
```

To see the exact selection without spending:

```bash
python scripts/run_benchmark.py --pipeline all --scale 150 --model gpt-4o-mini   --queries data/release/v1/queries.json --gold data/release/v1/gold_answers.json   --corpus-dir data/processed/release_v1 --judge-model gpt-4o --dry-run
```

Use a clean checkout with no pipeline caches for a fresh latency experiment. The runner resumes result files and each pipeline can reuse cached answers, so a different output directory alone does not guarantee fresh generations. The report requires one row per included question; do not append repeated experiments into the same output files.

The original generation/indexing/judging estimate was $1.73. This is not a hard spending cap, and provider prices and available models can change. RLM generated code executes locally; use trusted data and an isolated environment. Its internal token/time budgets do not forcibly stop an in-flight provider request.

## Layout

- `data/release/v1/`: frozen questions, references, exclusions, corpus archive, manifests.
- `data/ground_truth/provenance/`: source passages and audit history.
- `results/release_v1/raw/`: 172 fresh pipeline records.
- `results/release_v1/metrics/`: per-answer judge scores and reasoning.
- `results/release_v1/report.json`: checked summary and paired intervals.
- `results/release_v1/spend_estimate.json`: explicit historical cost estimate.
- `scripts/release_report.py`: offline report builder.

Earlier audit documents describe historical intermediate states. README, FINDINGS and the frozen v1 manifest are the current release references.
