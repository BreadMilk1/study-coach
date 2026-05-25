"""LLM-as-judge utility for Study Coach (P2.1-②).

Implements `judge_response`: a 4-criteria rubric evaluator
(relevance / accuracy / citation_quality / **pedagogical_fit**) backed by an
LLM call. The `rubric` parameter is injected into the prompt so the same
function can serve Tutor today and Quiz/Plan in P2.1-④/⑤ via different rubric
templates (A→C evolution path).

Score is normalised to 0..1; verdict is "pass" if score ≥ 0.6 else "weak";
weak_dims contains dimension names that scored ≤ 3. Parsing is tolerant of
markdown-fenced JSON; on parse failure we return a neutral 3/5 score so the
Judge Guard does not nuke responses purely because the judge model produced
malformed JSON.
"""

import json
import re
from pathlib import Path
from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage

_DIMENSIONS: tuple[str, ...] = (
    # Grounding side
    "relevance",
    "accuracy",
    "citation_quality",
    # Pedagogical side — split from prior single pedagogical_fit so weak_dims
    # surface a specific failure mode (e.g. heavy jargon hidden by good examples).
    "accessibility",
    "example_quality",
    "learner_level_fit",
)

# P2.1-④ Quiz judge dimensions: focuses on question/answer quality rather than
# pedagogical accessibility (accessibility doesn't apply to MCQ stems).
QUIZ_DIMENSIONS: tuple[str, ...] = (
    "question_quality",
    "option_plausibility",
    "answer_correctness",
    "explanation_clarity",
    "difficulty_calibration",
)

# P2.1-⑤ Plan judge dimensions: focuses on milestone quality (specificity /
# granularity / time feasibility / topic coverage / actionability).
PLAN_DIMENSIONS: tuple[str, ...] = (
    "milestone_specificity",
    "milestone_granularity",
    "time_feasibility",
    "topic_coverage",
    "actionability",
)

_WEAK_DIM_THRESHOLD = 3  # ≤ this surfaces in weak_dims
_PASS_SCORE_THRESHOLD = 0.6


class JudgeResult(TypedDict):
    score: float                       # 0..1
    verdict: Literal["pass", "weak"]
    weak_dims: list[str]
    reasoning: str


def load_tutor_rubric() -> str:
    """Return the Tutor judge prompt template (with {question}/{context}/{answer} placeholders)."""
    path = Path(__file__).parent / "prompts" / "judge_tutor.txt"
    return path.read_text(encoding="utf-8")


def load_quiz_rubric() -> str:
    """Return the Quiz judge prompt template (P2.1-④)."""
    path = Path(__file__).parent / "prompts" / "judge_quiz.txt"
    return path.read_text(encoding="utf-8")


def load_plan_rubric() -> str:
    """Return the Plan judge prompt template (P2.1-⑤)."""
    path = Path(__file__).parent / "prompts" / "judge_plan.txt"
    return path.read_text(encoding="utf-8")


def _build_prompt(*, question: str, answer: str, context: str, rubric: str) -> str:
    return rubric.format(question=question, answer=answer, context=context)


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_JSON_RE = re.compile(r"(\{[^{}]*\})", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    m = _JSON_FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    m = _BARE_JSON_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    return None


def _normalise(parsed: dict | None, dimensions: tuple[str, ...]) -> JudgeResult:
    if parsed is None:
        # Neutral fallback: don't kill a response because the judge babbled.
        return JudgeResult(
            score=0.6,
            verdict="pass",
            weak_dims=[],
            reasoning="Judge output parsing failed; defaulted to neutral fallback.",
        )

    dim_scores: dict[str, float] = {}
    for dim in dimensions:
        raw = parsed.get(dim, 3)
        try:
            v = float(raw)
        except (TypeError, ValueError):
            v = 3.0
        v = max(1.0, min(5.0, v))
        dim_scores[dim] = v

    avg = sum(dim_scores.values()) / len(dimensions)
    score = avg / 5.0
    verdict: Literal["pass", "weak"] = (
        "pass" if score >= _PASS_SCORE_THRESHOLD else "weak"
    )
    weak_dims = [d for d, v in dim_scores.items() if v <= _WEAK_DIM_THRESHOLD]
    reasoning = str(parsed.get("reasoning", "")).strip()
    return JudgeResult(
        score=round(score, 4),
        verdict=verdict,
        weak_dims=weak_dims,
        reasoning=reasoning,
    )


async def judge_response(
    *,
    question: str,
    answer: str,
    context: str,
    rubric: str,
    judge_llm,
    dimensions: tuple[str, ...] | None = None,
) -> JudgeResult:
    """Run LLM-as-judge with the given rubric template.

    `dimensions` overrides which JSON keys are extracted and averaged. Defaults
    to the 6-dim Tutor rubric set; pass `QUIZ_DIMENSIONS` for Quiz, etc.
    """
    dims = dimensions or _DIMENSIONS
    prompt = _build_prompt(
        question=question, answer=answer, context=context, rubric=rubric
    )
    response = await judge_llm.ainvoke([HumanMessage(content=prompt)])
    raw = getattr(response, "content", "") or ""
    return _normalise(_extract_json(raw), dims)
