from evaluation.metrics import compute_match_metrics, extract_numeric_values, normalize_text


def test_normalize_text_collapses_case_and_spacing():
    assert normalize_text(" Costco   Paid Members ") == "costco paid members"


def test_extract_numeric_values_respects_global_scale_marker():
    values = extract_numeric_values("81,000 (in thousands)")
    assert values == [81_000_000]


def test_numeric_metrics_distinguish_strict_vs_scale_invariant_matches():
    metrics = compute_match_metrics(
        "Costco reported 81,000 paid members in 2025.",
        "81 million",
        answer_type="numeric",
    )

    assert metrics["numeric_exact_match"] is False
    assert metrics["numeric_scale_invariant_match"] is True
    assert metrics["strict_match"] is False
    assert metrics["lenient_match"] is True


def test_numeric_metrics_treat_explicit_in_thousands_as_exact_match():
    metrics = compute_match_metrics(
        "81,000 (in thousands)",
        "81 million",
        answer_type="numeric",
    )

    assert metrics["numeric_exact_match"] is True
    assert metrics["strict_match"] is True
