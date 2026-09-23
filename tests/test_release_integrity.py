import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "data/release/v1"


def test_frozen_release_hashes_and_evidence_are_consistent():
    manifest = json.loads((RELEASE / "manifest.json").read_text())
    for path, digest in manifest["files_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    corpus = gzip.decompress((RELEASE / "corpus_150.jsonl.gz").read_bytes())
    assert hashlib.sha256(corpus).hexdigest() == manifest["corpus_uncompressed_sha256"]
    docs = {d["doc_id"]: d for d in map(json.loads, corpus.splitlines())}
    queries = json.loads((RELEASE / "queries.json").read_text())
    golds = json.loads((RELEASE / "gold_answers.json").read_text())
    excluded = {q["query_id"] for q in json.loads((RELEASE / "exclusions.json").read_text())}
    assert set(golds).isdisjoint(excluded)
    for q in queries:
        gold = golds[q["id"]]
        assert q["gold_answer"] == gold["answer"]
        assert q["evidence_docs"] == gold["evidence_docs"]
        assert set(q["evidence_docs"]) <= set(docs)
        provenance = json.loads((ROOT / gold["provenance_file"]).read_text())["records"][q["id"]]
        assert provenance["question"] == q["text"]
        assert provenance["answer"] == gold["answer"]
        for evidence in provenance["evidence"]:
            text = docs[evidence["doc_id"]]["text"]
            assert hashlib.sha256(text.encode()).hexdigest() == evidence["text_sha256"]
            assert text[evidence["start_char"]:evidence["end_char"]] == evidence["quote"]
