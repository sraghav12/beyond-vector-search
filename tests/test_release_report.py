import pytest

from scripts.release_report import paired_interval, summarize, validate_judge_row


def row(qid, score, **kwargs):
    return dict(query_id=qid, judge_score=score, status="ok", answer="answer",
                latency_ms=1000, cost_usd=0.01, **kwargs)


def test_context_infeasibility_is_not_zero_accuracy():
    records = [row("q1", 0)]
    records[0]["answer"] = "EXCEEDS_CONTEXT"
    result = summarize(records)
    assert result["mean_judge_score"] is None
    assert result["infeasible"] == 1


def test_failures_stay_in_denominator():
    records = [row("q1", 1), row("q2", 0)]
    records[1]["status"] = "error"
    result = summarize(records)
    assert result["mean_judge_score"] == 0.5
    assert result["errors"] == 1


def test_paired_interval_aligns_question_ids_and_rejects_missing_rows():
    left = [row("q1", 0.5), row("q2", 1)]
    right = [row("q2", 0.5), row("q1", 0)]
    assert paired_interval(left, right) == {"difference": 0.5, "ci95": [0.5, 0.5]}
    with pytest.raises(ValueError, match="same unique"):
        paired_interval(left, right[:1])


def test_duplicate_questions_cannot_inflate_report():
    with pytest.raises(ValueError, match="duplicate"):
        summarize([row("q1", 1), row("q1", 0)])


def test_local_infeasibility_guard_does_not_require_a_judge_call():
    validate_judge_row({"answer": "EXCEEDS_CONTEXT"},
                       {"judge_model": "gpt-4o-mini", "judge_score": "0",
                        "judge_reasoning": "degenerate answer: EXCEEDS_CONTEXT"})
    with pytest.raises(ValueError, match="unexpected judge"):
        validate_judge_row({"answer": "A substantive answer"},
                           {"judge_model": "gpt-4o-mini", "judge_score": "1"})
