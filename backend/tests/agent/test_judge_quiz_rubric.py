"""Cut ④d — Quiz-specific judge rubric tests.

Extends judge_response() to accept a `dimensions` override (default = tutor 6).
Adds load_quiz_rubric() + QUIZ_DIMENSIONS for the Quiz path.
"""
import pytest
from langchain_core.messages import AIMessage

from app.agent.judge import QUIZ_DIMENSIONS, judge_response, load_quiz_rubric


class StubJudgeLLM:
    def __init__(self, content: str):
        self.content = content

    async def ainvoke(self, messages, **_kwargs):
        return AIMessage(content=self.content)


def test_load_quiz_rubric_template_has_dims_and_placeholders():
    text = load_quiz_rubric()
    for dim in QUIZ_DIMENSIONS:
        assert dim in text, f"rubric missing dimension {dim}"
    for placeholder in ("{question}", "{answer}", "{context}"):
        assert placeholder in text, f"rubric missing placeholder {placeholder}"


def test_quiz_dimensions_are_distinct_from_tutor_dimensions():
    from app.agent.judge import _DIMENSIONS as TUTOR_DIMENSIONS

    # Quiz judge cares about question/answer correctness, not learner-level
    # accessibility — they share concept, not literal names.
    assert set(QUIZ_DIMENSIONS) != set(TUTOR_DIMENSIONS)
    # All quiz dims should be present in the rubric file (validated above).


@pytest.mark.asyncio
async def test_judge_response_with_quiz_rubric_passes_when_all_strong():
    payload = (
        '{"question_quality":5,"option_plausibility":5,"answer_correctness":5,'
        '"explanation_clarity":5,"difficulty_calibration":5,'
        '"reasoning":"Well-formed MCQ, plausible distractors, correct explanation."}'
    )
    result = await judge_response(
        question="What is HyDE?",
        answer="generated quiz prompt text",
        context="HyDE rewrites queries.",
        rubric=load_quiz_rubric(),
        judge_llm=StubJudgeLLM(payload),
        dimensions=QUIZ_DIMENSIONS,
    )

    assert result["verdict"] == "pass"
    assert result["score"] == pytest.approx(1.0)
    assert result["weak_dims"] == []


@pytest.mark.asyncio
async def test_judge_response_with_quiz_rubric_flags_weak_answer_correctness():
    payload = (
        '{"question_quality":5,"option_plausibility":5,"answer_correctness":2,'
        '"explanation_clarity":5,"difficulty_calibration":5,'
        '"reasoning":"Marked answer is wrong."}'
    )
    result = await judge_response(
        question="?",
        answer="quiz prompt",
        context="ctx",
        rubric=load_quiz_rubric(),
        judge_llm=StubJudgeLLM(payload),
        dimensions=QUIZ_DIMENSIONS,
    )

    assert "answer_correctness" in result["weak_dims"]
    # Tutor dimension names must NOT leak into quiz-judged weak_dims
    assert "accessibility" not in result["weak_dims"]


@pytest.mark.asyncio
async def test_judge_response_default_dimensions_unchanged_for_tutor_path():
    """Calling judge_response WITHOUT `dimensions` keeps the tutor 6-dim contract."""
    payload = (
        '{"relevance":5,"accuracy":5,"citation_quality":5,'
        '"accessibility":5,"example_quality":5,"learner_level_fit":5,'
        '"reasoning":"solid"}'
    )
    result = await judge_response(
        question="?",
        answer="a",
        context="c",
        rubric="ignored — stub LLM doesn't read the prompt",
        judge_llm=StubJudgeLLM(payload),
    )
    assert result["verdict"] == "pass"
    assert result["score"] == pytest.approx(1.0)
