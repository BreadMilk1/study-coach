"""Cut ⑤d — Plan judge rubric: dimensions + loader + scoring."""
from langchain_core.messages import AIMessage

from app.agent.judge import (
    PLAN_DIMENSIONS,
    QUIZ_DIMENSIONS,
    _DIMENSIONS as TUTOR_DIMENSIONS,
    judge_response,
    load_plan_rubric,
)


class StubJudgeLLM:
    def __init__(self, response_text: str):
        self.response_text = response_text

    async def ainvoke(self, messages, **_kwargs):
        return AIMessage(content=self.response_text)


def test_plan_dimensions_are_five_and_disjoint_from_tutor_and_quiz():
    assert len(PLAN_DIMENSIONS) == 5
    plan_set = set(PLAN_DIMENSIONS)
    assert plan_set.isdisjoint(TUTOR_DIMENSIONS)
    assert plan_set.isdisjoint(QUIZ_DIMENSIONS)


def test_load_plan_rubric_returns_template_with_placeholders():
    text = load_plan_rubric()
    assert "{question}" in text
    assert "{context}" in text
    assert "{answer}" in text
    for dim in PLAN_DIMENSIONS:
        assert dim in text


async def test_judge_response_with_plan_dims_strong_plan_passes():
    llm = StubJudgeLLM(
        '{"milestone_specificity": 5, "milestone_granularity": 5, '
        '"time_feasibility": 5, "topic_coverage": 4, "actionability": 5, '
        '"reasoning": "well scoped"}'
    )
    result = await judge_response(
        question="Make a plan",
        answer="...",
        context="",
        rubric=load_plan_rubric(),
        judge_llm=llm,
        dimensions=PLAN_DIMENSIONS,
    )
    assert result["verdict"] == "pass"
    assert result["weak_dims"] == []


async def test_judge_response_with_plan_dims_weak_plan_flags_weak_dims():
    llm = StubJudgeLLM(
        '{"milestone_specificity": 1, "milestone_granularity": 2, '
        '"time_feasibility": 1, "topic_coverage": 2, "actionability": 1, '
        '"reasoning": "vague"}'
    )
    result = await judge_response(
        question="Make a plan",
        answer="...",
        context="",
        rubric=load_plan_rubric(),
        judge_llm=llm,
        dimensions=PLAN_DIMENSIONS,
    )
    assert result["verdict"] == "weak"
    assert "milestone_specificity" in result["weak_dims"]
    assert "actionability" in result["weak_dims"]
