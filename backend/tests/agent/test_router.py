"""Router intent-classification tests (P2.1-① RED-1).

Pure function `route_intent(message) -> Literal["tutor","quiz","plan"]`.
Keyword-based first cut; LLM structured-output upgrade deferred to later phases.
"""

from app.agent.router import route_intent


def test_router_returns_quiz_for_quiz_keywords():
    assert route_intent("quiz me on hyde") == "quiz"
    assert route_intent("测我一下") == "quiz"
    assert route_intent("请出题考我") == "quiz"
    assert route_intent("practice please") == "quiz"
    assert route_intent("test me on RAG") == "quiz"


def test_router_returns_plan_for_plan_keywords():
    assert route_intent("帮我做学习计划") == "plan"
    assert route_intent("set a study plan") == "plan"
    assert route_intent("我的目标是 next week 考试") == "plan"
    assert route_intent("schedule my week") == "plan"
    assert route_intent("复习计划怎么安排") == "plan"


def test_router_defaults_to_tutor_when_no_keyword_match():
    assert route_intent("what is HyDE?") == "tutor"
    assert route_intent("") == "tutor"
    assert route_intent("解释一下 ReAct") == "tutor"
    assert route_intent("Explain reranking") == "tutor"


def test_router_is_case_insensitive():
    assert route_intent("QUIZ me") == "quiz"
    assert route_intent("Plan something for me") == "plan"
    assert route_intent("TEST ME ON BM25") == "quiz"


def test_router_prioritizes_quiz_over_plan_when_both_match():
    assert route_intent("plan a quiz for me") == "quiz"
    assert route_intent("帮我安排学习计划并测我一下") == "quiz"
