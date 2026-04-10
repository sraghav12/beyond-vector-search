import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import pandas as pd  # type: ignore[import-untyped]


_SCALE_MAP = {
    "k": Decimal("1000"),
    "thousand": Decimal("1000"),
    "thousands": Decimal("1000"),
    "m": Decimal("1000000"),
    "million": Decimal("1000000"),
    "millions": Decimal("1000000"),
    "b": Decimal("1000000000"),
    "billion": Decimal("1000000000"),
    "billions": Decimal("1000000000"),
    "t": Decimal("1000000000000"),
    "trillion": Decimal("1000000000000"),
    "trillions": Decimal("1000000000000"),
}

_NUMBER_RE = re.compile(
    r"""
    (?<![A-Za-z0-9])
    (?P<prefix>[$])?
    (?P<number>[-+]?\d[\d,]*(?:\.\d+)?)
    \s*
    (?P<suffix>
        thousand|thousands|
        million|millions|
        billion|billions|
        trillion|trillions|
        k|m|b|t|%
    )?
    """,
    re.IGNORECASE | re.VERBOSE,
)

_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9.%$-]+")
_GLOBAL_SCALE_RE = re.compile(
    r"\bin\s+(thousands?|millions?|billions?|trillions?)\b",
    re.IGNORECASE,
)

_REL_TOL = Decimal("0.001")
_ABS_TOL = Decimal("0.000001")


def normalize_text(text: str) -> str:
    text = text or ""
    text = text.lower().strip()
    text = _NON_ALNUM_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def normalized_exact_match(predicted: str, gold: str) -> bool:
    return normalize_text(predicted) == normalize_text(gold)


def _global_multiplier(text: str) -> Optional[Decimal]:
    match = _GLOBAL_SCALE_RE.search(text or "")
    if not match:
        return None
    return _SCALE_MAP.get(match.group(1).lower())


def extract_numeric_values(text: str) -> list[Decimal]:
    text = text or ""
    values: list[Decimal] = []
    global_multiplier = _global_multiplier(text)

    for match in _NUMBER_RE.finditer(text):
        raw_number = match.group("number")
        suffix = (match.group("suffix") or "").lower()

        try:
            value = Decimal(raw_number.replace(",", ""))
        except (InvalidOperation, AttributeError):
            continue

        if suffix == "%":
            multiplier = Decimal("1")
        elif suffix:
            multiplier = _SCALE_MAP.get(suffix, Decimal("1"))
        else:
            multiplier = global_multiplier or Decimal("1")

        values.append(value * multiplier)

    deduped: list[Decimal] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped


def _values_close(a: Decimal, b: Decimal) -> bool:
    diff = abs(a - b)
    largest = max(abs(a), abs(b), Decimal("1"))
    return diff <= max(_ABS_TOL, _REL_TOL * largest)


def numeric_exact_match(predicted: str, gold: str) -> Optional[bool]:
    predicted_values = extract_numeric_values(predicted)
    gold_values = extract_numeric_values(gold)

    if not predicted_values or not gold_values:
        return None

    for predicted_value in predicted_values:
        for gold_value in gold_values:
            if _values_close(predicted_value, gold_value):
                return True
    return False


def numeric_scale_invariant_match(predicted: str, gold: str) -> Optional[bool]:
    strict_match = numeric_exact_match(predicted, gold)
    if strict_match is not None and strict_match:
        return True

    predicted_values = extract_numeric_values(predicted)
    gold_values = extract_numeric_values(gold)
    if not predicted_values or not gold_values:
        return None

    allowed_scales = (
        Decimal("1000"),
        Decimal("1000000"),
        Decimal("1000000000"),
        Decimal("1000000000000"),
    )

    for predicted_value in predicted_values:
        for gold_value in gold_values:
            if predicted_value == 0 or gold_value == 0:
                continue
            larger = max(abs(predicted_value), abs(gold_value))
            smaller = min(abs(predicted_value), abs(gold_value))
            ratio = larger / smaller
            for scale in allowed_scales:
                if _values_close(ratio, scale):
                    return True
    return False


def compute_match_metrics(
    predicted: str,
    gold: str,
    answer_type: Optional[str] = None,
) -> dict:
    answer_type = (answer_type or "").lower()
    normalized_match = normalized_exact_match(predicted, gold)

    numeric_match = None
    numeric_scale_match = None
    if answer_type == "numeric":
        numeric_match = numeric_exact_match(predicted, gold)
        numeric_scale_match = numeric_scale_invariant_match(predicted, gold)

    strict_match = normalized_match
    lenient_match = normalized_match
    if answer_type == "numeric":
        strict_match = bool(numeric_match) or normalized_match
        lenient_match = strict_match or bool(numeric_scale_match)

    return {
        "normalized_exact_match": normalized_match,
        "numeric_exact_match": numeric_match,
        "numeric_scale_invariant_match": numeric_scale_match,
        "strict_match": strict_match,
        "lenient_match": lenient_match,
    }


# ── Token-level F1 ────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    return normalize_text(text).split()


def f1_score(predicted: str, gold: str) -> float:
    predicted_tokens = tokenize(predicted)
    gold_tokens = tokenize(gold)
    if not predicted_tokens or not gold_tokens:
        return 0.0
    predicted_counts: dict[str, int] = {}
    for t in predicted_tokens:
        predicted_counts[t] = predicted_counts.get(t, 0) + 1
    gold_counts: dict[str, int] = {}
    for t in gold_tokens:
        gold_counts[t] = gold_counts.get(t, 0) + 1
    matched = sum(
        min(predicted_counts.get(t, 0), gold_counts[t]) for t in gold_counts
    )
    if matched == 0:
        return 0.0
    precision = matched / len(predicted_tokens)
    recall = matched / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ── LLM Judge ────────────────────────────────────────────────────────────────

_JUDGE_PROMPT = """\
You are an expert evaluator. Given a question, a gold-standard reference answer,
and a predicted answer, rate the predicted answer's correctness from 0.0 to 1.0.

- 1.0 = perfectly correct, captures all key information
- 0.5 = partially correct, some key info present but incomplete or minor errors
- 0.0 = completely wrong or irrelevant

Question: {question}
Gold Answer: {gold}
Predicted Answer: {predicted}

Respond with ONLY a JSON object: {{"score": <float>, "reasoning": "<brief explanation>"}}\
"""

_JUDGE_OPPOSITE = {
    "gpt-4o-mini": "gemini-2.5-flash",
    "gpt-4o": "gemini-2.5-flash",
    "gemini-2.5-flash": "gpt-4o-mini",
    "gemini-2.0-flash": "gpt-4o-mini",
    "gemini-1.5-flash": "gpt-4o-mini",
}

_judge_cache: Optional[dict] = None  # type: ignore[type-arg]


def _get_judge_cache() -> dict:  # type: ignore[type-arg]
    global _judge_cache
    if _judge_cache is None:
        try:
            import diskcache  # type: ignore[import-untyped]
            _judge_cache = diskcache.Cache(  # type: ignore[assignment]
                Path(__file__).parent.parent / ".cache" / "judge"
            )
        except ImportError:
            _judge_cache = {}
    return _judge_cache  # type: ignore[return-value]


def llm_judge(
    question: str,
    predicted: str,
    gold: str,
    model: str = "gpt-4o-mini",
) -> tuple[float, str]:
    """Score predicted answer 0.0-1.0 using the opposite model to avoid self-eval bias."""
    import litellm  # type: ignore[import-untyped]

    judge_model = _JUDGE_OPPOSITE.get(model, "gpt-4o-mini")
    cache_key = f"judge|{judge_model}|{question}|{gold}|{predicted}"
    cache = _get_judge_cache()

    if cache_key in cache:
        cached = cache[cache_key]
        return cached["score"], cached["reasoning"]

    prompt = _JUDGE_PROMPT.format(
        question=question, gold=gold, predicted=predicted
    )
    try:
        response = litellm.completion(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        parsed = json.loads(raw)
        score = float(parsed.get("score", 0.0))
        reasoning = str(parsed.get("reasoning", ""))
    except Exception as exc:
        score = 0.0
        reasoning = f"judge error: {exc}"

    cache[cache_key] = {"score": score, "reasoning": reasoning}
    return score, reasoning


# ── Unified metric bundle ─────────────────────────────────────────────────────

def compute_all_metrics(
    question: str,
    predicted: str,
    gold: str,
    answer_type: Optional[str] = None,
    judge_model: Optional[str] = None,
) -> dict:
    match_metrics = compute_match_metrics(predicted, gold, answer_type=answer_type)
    f1 = f1_score(predicted, gold)
    judge_score: Optional[float] = None
    judge_reasoning: Optional[str] = None
    if judge_model:
        judge_score, judge_reasoning = llm_judge(
            question, predicted, gold, model=judge_model
        )
    return {
        **match_metrics,
        "f1": f1,
        "llm_judge": judge_score,
        "judge_reasoning": judge_reasoning,
    }


# ── Batch scoring ─────────────────────────────────────────────────────────────

def score_results_file(
    results_path: str,
    gold_answers_path: str,
    judge_model: Optional[str] = None,
) -> "pd.DataFrame":
    with open(gold_answers_path, encoding="utf-8") as fh:
        gold_answers: dict = json.load(fh)

    rows = []
    with open(results_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            query_id = record.get("query_id", "")
            gold_entry = gold_answers.get(query_id, {})
            gold = gold_entry.get("answer", record.get("gold_answer", ""))
            predicted = record.get("answer", "")
            question = record.get("query_text", "")
            answer_type = gold_entry.get("answer_type", record.get("answer_type"))

            if record.get("status") != "ok":
                m: dict = {
                    "normalized_exact_match": False,
                    "numeric_exact_match": None,
                    "numeric_scale_invariant_match": None,
                    "strict_match": False,
                    "lenient_match": False,
                    "f1": 0.0,
                    "llm_judge": None,
                    "judge_reasoning": None,
                }
            else:
                m = compute_all_metrics(
                    question, predicted, gold,
                    answer_type=answer_type,
                    judge_model=judge_model,
                )

            rows.append(
                {
                    "query_id": query_id,
                    "tier": record.get("tier"),
                    "tier_name": record.get("tier_name"),
                    "difficulty": record.get("difficulty"),
                    "pipeline": record.get("pipeline_name"),
                    "model": record.get("model"),
                    "scale": record.get("corpus_scale"),
                    "status": record.get("status"),
                    "latency_ms": record.get("latency_ms"),
                    "cost_usd": record.get("cost_usd"),
                    "answer_preview": str(predicted)[:120],
                    "gold_preview": str(gold)[:120],
                    **m,
                }
            )

    import pandas as pd  # lazy import so non-scoring paths stay lightweight
    df = pd.DataFrame(rows)
    out_path = (
        Path(results_path).parent.parent
        / "metrics"
        / (Path(results_path).stem + "_scored.csv")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


# ── Aggregate stats ───────────────────────────────────────────────────────────

def aggregate_metrics(scored_df: "pd.DataFrame") -> dict:
    import pandas as pd

    numeric_cols = ["strict_match", "lenient_match", "f1", "llm_judge"]
    existing = [c for c in numeric_cols if c in scored_df.columns]

    def _safe_mean(series):
        s = pd.to_numeric(series, errors="coerce").dropna()
        return float(s.mean()) if len(s) else None

    overall = {col: _safe_mean(scored_df[col]) for col in existing}

    by_tier: dict = {}
    if "tier_name" in scored_df.columns:
        for tier, group in scored_df.groupby("tier_name", dropna=False):
            by_tier[str(tier)] = {col: _safe_mean(group[col]) for col in existing}

    by_difficulty: dict = {}
    if "difficulty" in scored_df.columns:
        for diff, group in scored_df.groupby("difficulty", dropna=False):
            by_difficulty[str(diff)] = {col: _safe_mean(group[col]) for col in existing}

    latency_stats: dict = {}
    if "latency_ms" in scored_df.columns:
        lat: pd.Series = pd.to_numeric(scored_df["latency_ms"], errors="coerce").dropna()  # type: ignore[assignment]
        if len(lat):
            latency_stats = {
                "p50_ms": float(lat.quantile(0.50)),
                "p95_ms": float(lat.quantile(0.95)),
                "p99_ms": float(lat.quantile(0.99)),
            }

    cost_stats: dict = {}
    if "cost_usd" in scored_df.columns:
        cost: pd.Series = pd.to_numeric(scored_df["cost_usd"], errors="coerce").dropna()  # type: ignore[assignment]
        if len(cost):
            cost_stats = {
                "avg_cost_per_query": float(cost.mean()),
                "total_cost": float(cost.sum()),
            }

    return {
        "overall": overall,
        "by_tier": by_tier,
        "by_difficulty": by_difficulty,
        "latency": latency_stats,
        "cost": cost_stats,
    }