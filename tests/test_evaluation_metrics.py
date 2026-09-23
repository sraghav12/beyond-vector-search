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


def test_containment_fires_when_gold_appears_inside_free_text():
    metrics = compute_match_metrics(
        "The company's headquarters are in Cupertino, California.",
        "Cupertino, California",
        answer_type="descriptive",
    )

    assert metrics["normalized_exact_match"] is False
    assert metrics["normalized_containment_match"] is True
    assert metrics["strict_match"] is False
    assert metrics["lenient_match"] is True


def test_numeric_scale_only_match_isolates_magnitude_errors():
    wrong_scale = compute_match_metrics(
        "Revenue was $391 million.", "$391 billion", answer_type="numeric",
    )
    assert wrong_scale["numeric_exact_match"] is False
    assert wrong_scale["numeric_scale_only_match"] is True

    right_scale = compute_match_metrics(
        "Revenue was $391.0 billion.", "$391 billion", answer_type="numeric",
    )
    assert right_scale["numeric_exact_match"] is True
    assert right_scale["numeric_scale_only_match"] is False


def test_free_text_sentence_with_correct_number_scores_numeric_not_exact():
    # Realistic pipeline output: a full sentence carrying the correct figure.
    metrics = compute_match_metrics(
        "Apple's total net sales (revenue) for fiscal year 2024 is $391,035 million.",
        "$391.0 billion ($391,035 million)",
        answer_type="numeric",
    )

    assert metrics["normalized_exact_match"] is False
    assert metrics["numeric_exact_match"] is True
    assert metrics["strict_match"] is True
