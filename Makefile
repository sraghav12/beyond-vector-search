PYTHON ?= python
OUTPUT ?= results/reproduction

.PHONY: prepare test report benchmark

prepare:
	mkdir -p data/processed/release_v1
	gzip -dc data/release/v1/corpus_150.jsonl.gz > data/processed/release_v1/corpus_150.jsonl
	shasum -a 256 -c data/release/v1/SHA256SUMS

test:
	$(PYTHON) -m pytest -q

report:
	$(PYTHON) scripts/release_report.py

benchmark: prepare
	BVS_DISABLED_PROVIDERS=anthropic,gemini BVS_JUDGE_MODEL=gpt-4o $(PYTHON) scripts/run_benchmark.py --pipeline all --scale 150 --model gpt-4o-mini --queries data/release/v1/queries.json --gold data/release/v1/gold_answers.json --corpus-dir data/processed/release_v1 --output-dir $(OUTPUT) --judge-model gpt-4o --rlm-max-subcalls 5 --rlm-token-budget 300000 --rlm-max-timeout 180 --query-timeout 240 --query-delay 1
