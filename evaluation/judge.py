"""LLM-as-judge scorer for benchmark results.

Usage:
    from evaluation.judge import LLMJudge

    judge = LLMJudge(model="gpt-4o-mini")
    result = judge.score("What was Apple's revenue?", "$391B", "$391.0 billion", "numeric")
    print(result.score)  # 1.0

    df = judge.score_results_file(
        "results/raw/naive_llm_gpt-4o-mini_10.jsonl",
        "data/ground_truth/gold_answers.json",
        output_path="results/metrics/naive_llm_gpt-4o-mini_10_scored.csv",
    )
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import diskcache  # type: ignore[import-untyped]
import litellm  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]

# ── Cross-model judge selection ───────────────────────────────────────────────

# Preferred cross-provider judges — Claude is best: different company, different arch
_CROSS_JUDGE: dict[str, str] = {
    "gpt-4o-mini": "anthropic/claude-haiku-4-5-20251001",
    "gpt-4o": "anthropic/claude-haiku-4-5-20251001",
    "gemini-2.5-flash": "anthropic/claude-haiku-4-5-20251001",
    "gemini-2.0-flash": "anthropic/claude-haiku-4-5-20251001",
    "gemini-1.5-flash": "anthropic/claude-haiku-4-5-20251001",
    "claude-haiku": "gpt-4o-mini",
    "claude-sonnet": "gpt-4o-mini",
    "claude-opus": "gpt-4o-mini",
}

# Same-provider fallbacks (different tier) when cross-provider key is unavailable
_SAME_PROVIDER_FALLBACK: dict[str, str] = {
    "gpt-4o-mini": "gpt-4o",
    "gpt-4o": "gpt-4o-mini",
    "gemini-2.5-flash": "gemini-2.0-flash",
    "gemini-2.0-flash": "gemini-1.5-flash",
    "gemini-1.5-flash": "gemini-2.0-flash",
    "claude-haiku-4-5-20251001": "gpt-4o-mini",
}

_DEGENERATE_PREFIXES = ("error:", "exceeds_context", "exceeds context")


def _has_key(provider: str) -> bool:
    """Check whether an API key for the given provider is present in the environment."""
    import os
    if provider == "openai":
        return bool(os.environ.get("OPENAI_API_KEY"))
    if provider == "anthropic":
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    if provider == "gemini":
        return bool(
            os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GOOGLE_GENERATIVEAI_API_KEY")
        )
    return False


def _provider(model: str) -> str:
    """Infer provider from a litellm model string."""
    m = model.lower()
    if m.startswith("anthropic/") or "claude" in m:
        return "anthropic"
    if m.startswith("gemini/") or "gemini" in m:
        return "gemini"
    return "openai"


def get_recommended_judge(evaluated_model: str) -> str:
    """Return the best available judge, preferring Claude > Gemini > GPT (cross-provider)."""
    key = evaluated_model.lower().split("/")[-1]

    # find the ideal cross-provider judge
    ideal: Optional[str] = None
    for pattern, judge in _CROSS_JUDGE.items():
        if pattern in key:
            ideal = judge
            break
    if ideal is None:
        # default: prefer Claude, fall through to others
        ideal = "anthropic/claude-haiku-4-5-20251001"

    # use ideal if its provider key is available
    if _has_key(_provider(ideal)):
        return ideal

    # try other cross-provider options in priority order
    for candidate in [
        "anthropic/claude-haiku-4-5-20251001",
        "gemini/gemini-2.0-flash",
        "gpt-4o",
        "gpt-4o-mini",
    ]:
        if _has_key(_provider(candidate)) and _provider(candidate) != _provider(evaluated_model):
            import warnings
            warnings.warn(
                f"Preferred judge '{ideal}' unavailable. Using '{candidate}'.",
                stacklevel=2,
            )
            return candidate

    # last resort: same-provider different tier (note self-eval bias)
    fallback = _SAME_PROVIDER_FALLBACK.get(key)
    if fallback and _has_key(_provider(fallback)):
        import warnings
        warnings.warn(
            f"No cross-provider judge available. Falling back to '{fallback}' "
            "(same-provider — self-evaluation bias may affect scores).",
            stacklevel=2,
        )
        return fallback

    return ideal  # will fail at call time with a clear auth error


# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert financial analyst evaluating AI-generated answers against \
gold-standard reference answers. Score the predicted answer on a scale from \
0.0 to 1.0 based on factual correctness and completeness.

Scoring guide:
  1.0 — Fully correct. All key facts match the gold answer.
  0.8 — Mostly correct. Minor omission or slightly imprecise phrasing, but \
the core fact is right.
  0.5 — Partially correct. Gets some key information but misses or \
contradicts other key parts.
  0.2 — Mostly wrong. A relevant attempt but the main fact is incorrect.
  0.0 — Completely wrong, irrelevant, refused, or empty.

Answer-type-specific guidance:
  numeric   — Accept equivalent representations ($391B ≈ $391.0 billion ≈ \
391,035 million). Minor rounding within 1% is fine (score ≥ 0.8). \
Wrong order of magnitude = 0.0.
  comparative — Check direction (higher/lower) AND magnitude. Correct \
direction but wrong magnitude = 0.5.
  descriptive — Score on coverage of key concepts, not exact wording.
  ranking   — Partial credit for partially correct orderings.

Respond with ONLY valid JSON (no markdown fences):
{"score": <float 0.0-1.0>, "reasoning": "<one sentence>"}\
"""

_USER_TEMPLATE = """\
Answer type: {answer_type}

Question: {question}

Gold answer: {gold}

Predicted answer: {predicted}\
"""


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class JudgeResult:
    score: float
    reasoning: str
    judge_model: str
    cost_usd: float = 0.0
    cached: bool = False
    error: Optional[str] = None


# ── Judge class ───────────────────────────────────────────────────────────────

class LLMJudge:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        cache_dir: Optional[str] = None,
    ) -> None:
        self.model = model
        _cache_path = Path(cache_dir) if cache_dir else (
            Path(__file__).parent.parent / ".cache" / "judge"
        )
        self._cache: diskcache.Cache = diskcache.Cache(_cache_path)
        self._total_cost: float = 0.0
        self._calls: int = 0
        self._cache_hits: int = 0

    # ── public API ────────────────────────────────────────────────────────────

    def score(
        self,
        question: str,
        predicted: str,
        gold: str,
        answer_type: Optional[str] = None,
    ) -> JudgeResult:
        """Score one predicted answer. Returns JudgeResult with score 0.0-1.0."""
        # Degenerate cases — don't waste API calls
        if not predicted or not predicted.strip():
            return JudgeResult(
                score=0.0, reasoning="empty answer",
                judge_model=self.model, cached=False,
            )
        pred_lower = predicted.strip().lower()
        for prefix in _DEGENERATE_PREFIXES:
            if pred_lower.startswith(prefix):
                return JudgeResult(
                    score=0.0, reasoning=f"degenerate answer: {predicted[:80]}",
                    judge_model=self.model, cached=False,
                )

        judge_model = get_recommended_judge(self.model)
        cache_key = _cache_key(judge_model, question, predicted, gold, answer_type or "")

        if cache_key in self._cache:
            self._cache_hits += 1
            cached_val: dict = self._cache[cache_key]
            return JudgeResult(
                score=cached_val["score"],
                reasoning=cached_val["reasoning"],
                judge_model=judge_model,
                cost_usd=0.0,
                cached=True,
            )

        result = self._call_judge(judge_model, question, predicted, gold, answer_type)

        # If the primary judge hit a quota/auth/404 error, try the same-provider fallback
        if result.error is not None and _is_recoverable(result.error):
            fallback = _SAME_PROVIDER_FALLBACK.get(judge_model.split("/")[-1])
            if fallback is None and _has_key("openai"):
                fallback = "gpt-4o"
            if fallback and fallback != judge_model:
                import warnings
                warnings.warn(
                    f"Judge '{judge_model}' failed ({result.error[:80]}...). "
                    f"Retrying with '{fallback}'.",
                    stacklevel=3,
                )
                result = self._call_judge(fallback, question, predicted, gold, answer_type)
                judge_model = fallback

        if result.error is None:
            self._cache[cache_key] = {
                "score": result.score,
                "reasoning": result.reasoning,
            }
        self._total_cost += result.cost_usd
        self._calls += 1
        return result

    def score_batch(
        self,
        items: list[dict],
        show_progress: bool = False,
    ) -> list[JudgeResult]:
        """Score a list of dicts with keys: question, predicted, gold, answer_type (optional)."""
        try:
            from tqdm import tqdm  # type: ignore[import-untyped]
        except ImportError:
            tqdm = None  # type: ignore[assignment]

        iterable = items
        if show_progress and tqdm is not None:
            iterable = tqdm(items, desc=f"judge/{self.model}", unit="q")

        return [
            self.score(
                item["question"],
                item["predicted"],
                item["gold"],
                item.get("answer_type"),
            )
            for item in iterable
        ]

    def score_results_file(
        self,
        results_path: str,
        gold_answers_path: str,
        output_path: Optional[str] = None,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Read a runner JSONL, judge every row, return scored DataFrame."""
        with open(gold_answers_path, encoding="utf-8") as fh:
            gold_answers: dict = json.load(fh)

        rows = []
        with open(results_path, encoding="utf-8") as fh:
            lines = [l.strip() for l in fh if l.strip()]

        try:
            from tqdm import tqdm  # type: ignore[import-untyped]
            iter_lines = tqdm(lines, desc="judging", unit="q") if show_progress else lines
        except ImportError:
            iter_lines = lines

        for line in iter_lines:
            record = json.loads(line)
            query_id = record.get("query_id", "")
            gold_entry = gold_answers.get(query_id, {})
            gold = gold_entry.get("answer", record.get("gold_answer", ""))
            predicted = record.get("answer", "")
            question = record.get("query_text", "")
            answer_type = gold_entry.get("answer_type", record.get("answer_type"))

            if record.get("status") != "ok":
                judge_score = 0.0
                judge_reasoning = f"pipeline error: {record.get('error_message', '')}"
                judge_cost = 0.0
                judge_cached = False
            else:
                jr = self.score(question, predicted, gold, answer_type)
                judge_score = jr.score
                judge_reasoning = jr.reasoning
                judge_cost = jr.cost_usd
                judge_cached = jr.cached

            rows.append(
                {
                    "query_id": query_id,
                    "tier": record.get("tier"),
                    "tier_name": record.get("tier_name"),
                    "difficulty": record.get("difficulty"),
                    "answer_type": answer_type,
                    "pipeline": record.get("pipeline_name"),
                    "model": record.get("model"),
                    "scale": record.get("corpus_scale"),
                    "status": record.get("status"),
                    "latency_ms": record.get("latency_ms"),
                    "cost_usd": record.get("cost_usd"),
                    "judge_score": judge_score,
                    "judge_reasoning": judge_reasoning,
                    "judge_cost_usd": judge_cost,
                    "judge_cached": judge_cached,
                    "answer_preview": str(predicted)[:120],
                    "gold_preview": str(gold)[:120],
                }
            )

        df = pd.DataFrame(rows)

        _path = output_path or str(
            Path(results_path).parent.parent
            / "metrics"
            / (Path(results_path).stem + "_judged.csv")
        )
        Path(_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(_path, index=False)

        self._print_summary(df, _path)
        return df

    def stats(self) -> dict:
        return {
            "model": self.model,
            "calls": self._calls,
            "cache_hits": self._cache_hits,
            "total_cost_usd": round(self._total_cost, 6),
        }

    @property
    def total_cost_usd(self) -> float:
        return self._total_cost

    # ── internals ─────────────────────────────────────────────────────────────

    def _call_judge(
        self,
        judge_model: str,
        question: str,
        predicted: str,
        gold: str,
        answer_type: Optional[str],
    ) -> JudgeResult:
        user_msg = _USER_TEMPLATE.format(
            answer_type=answer_type or "general",
            question=question,
            gold=gold,
            predicted=predicted,
        )
        try:
            response = litellm.completion(
                model=judge_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0,
            )
            raw = response.choices[0].message.content.strip()
            score, reasoning = _parse_judge_response(raw)
            cost = _extract_cost(response)
        except Exception as exc:
            return JudgeResult(
                score=0.0,
                reasoning=f"judge call failed: {exc}",
                judge_model=judge_model,
                cost_usd=0.0,
                error=str(exc),
            )

        return JudgeResult(
            score=score,
            reasoning=reasoning,
            judge_model=judge_model,
            cost_usd=cost,
        )

    def _print_summary(self, df: pd.DataFrame, output_path: str) -> None:
        n = len(df)
        mean_score = df["judge_score"].mean()
        judge_cost = df["judge_cost_usd"].sum()
        cached_pct = 100 * df["judge_cached"].sum() / n if n else 0

        print(f"\nJudge summary  ({Path(output_path).name})")
        print(f"  queries      : {n}")
        print(f"  mean score   : {mean_score:.3f}")
        print(f"  cached       : {cached_pct:.0f}%")
        print(f"  judge cost   : ${judge_cost:.4f}")

        if "tier_name" in df.columns:
            print("\n  by tier:")
            for tier, g in df.groupby("tier_name", dropna=False):
                print(f"    {str(tier):20s}  {g['judge_score'].mean():.3f}")

        if "difficulty" in df.columns:
            print("\n  by difficulty:")
            for diff, g in df.groupby("difficulty", dropna=False):
                print(f"    {str(diff):12s}  {g['judge_score'].mean():.3f}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_recoverable(error_str: str) -> bool:
    """True for quota / auth / not-found errors where a different model might work."""
    s = error_str.lower()
    return any(kw in s for kw in (
        "quota", "rate limit", "ratelimit", "not found", "404",
        "invalid api key", "authentication", "permission", "billing",
    ))


def _cache_key(
    judge_model: str,
    question: str,
    predicted: str,
    gold: str,
    answer_type: str,
) -> str:
    blob = "\n".join([judge_model, question, predicted, gold, answer_type])
    return hashlib.sha256(blob.encode()).hexdigest()


def _parse_judge_response(raw: str) -> tuple[float, str]:
    # strip markdown fences
    text = re.sub(r"^```[a-z]*\n?", "", raw.strip())
    text = re.sub(r"\n?```$", "", text).strip()

    # attempt JSON parse
    try:
        parsed = json.loads(text)
        score = float(parsed["score"])
        reasoning = str(parsed.get("reasoning", ""))
        return max(0.0, min(1.0, score)), reasoning
    except (json.JSONDecodeError, KeyError, ValueError):
        pass

    # regex fallback
    m = re.search(r'"?score"?\s*[:=]\s*([0-9.]+)', text, re.IGNORECASE)
    if m:
        try:
            score = max(0.0, min(1.0, float(m.group(1))))
            r = re.search(r'"?reasoning"?\s*[:=]\s*"([^"]+)"', text, re.IGNORECASE)
            reasoning = r.group(1) if r else "parsed via regex fallback"
            return score, reasoning
        except ValueError:
            pass

    return 0.0, f"parse failed: {raw[:120]}"


def _extract_cost(response) -> float:
    try:
        usage = response.usage
        model = response.model or ""
        from pipelines.base import compute_cost
        return compute_cost(model, usage.prompt_tokens, usage.completion_tokens)
    except Exception:
        return 0.0
