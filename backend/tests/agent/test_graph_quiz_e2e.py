"""Cut ④e — Graph e2e tests for the Quiz path.

Two ainvoke turns prove the full quiz session lifecycle through the real
LangGraph topology:
  START → memory_hydrator → router → quiz → judge → memory_writer → END

Turn 1 (GENERATE): user "quiz me on HyDE" → quiz_master auto-creates goal+topic,
generates Q, persists, sets active_quiz_question_id, returns formatted prompt.
Turn 2 (GRADE): re-enter with active_quiz_question_id + user answer → quiz_master
grades, updates mastery, clears active_quiz_question_id.
"""
from datetime import datetime

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.agent.graph import build_graph
from app.agent.memory_updater import build_memory_writer
from app.agent.quiz_master import build_quiz_master
from app.db.models import Base
from app.db.repositories import (
    GoalRepository,
    MasteryRepository,
    MistakeRepository,
    QuestionRepository,
    TopicRepository,
    UserRepository,
)


_QUIZ_JSON = """[
  {
    "prompt": "What does HyDE rewrite?",
    "options": ["A) Queries", "B) Documents", "C) Embeddings", "D) Answers"],
    "answer": "A",
    "explanation": "HyDE rewrites the user query into a hypothetical answer."
  }
]"""

_PASS_QUIZ_JUDGE = (
    '{"question_quality":5,"option_plausibility":5,"answer_correctness":5,'
    '"explanation_clarity":5,"difficulty_calibration":5,"reasoning":"solid"}'
)


class StubGeneratorLLM:
    async def ainvoke(self, messages, **_kwargs):
        return AIMessage(content=_QUIZ_JSON)


class StubJudgeLLM:
    def __init__(self, payload: str = _PASS_QUIZ_JUDGE):
        self.payload = payload
        self.last_prompt: str | None = None
        self.ainvoke_count = 0

    async def ainvoke(self, messages, **_kwargs):
        self.ainvoke_count += 1
        self.last_prompt = messages[-1].content if messages else ""
        return AIMessage(content=self.payload)


class StubTutorLLM:
    """Unused in quiz path; satisfies build_graph(llm=...) signature."""

    async def astream(self, messages, **_kwargs):
        from langchain_core.messages import AIMessageChunk
        yield AIMessageChunk(content="tutor never runs in these tests")


class StubRetriever:
    def search(self, query, top_k=5):
        return []


@pytest.fixture
def session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,
    )
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def _build_quiz_master(session):
    return build_quiz_master(
        llm=StubGeneratorLLM(),
        topic_repo=TopicRepository(session),
        question_repo=QuestionRepository(session),
        mistake_repo=MistakeRepository(session),
        mastery_repo=MasteryRepository(session),
        goal_repo=GoalRepository(session),
        now_fn=lambda: datetime(2026, 5, 21, 12, 0),
    )


def _build_memory_writer(session):
    return build_memory_writer(
        mastery_repo=MasteryRepository(session),
        mistake_repo=MistakeRepository(session),
    )


async def test_quiz_generate_turn_through_graph_persists_question_and_sets_active(session):
    users = UserRepository(session)
    user = users.get_or_create("fp-e2e-gen")
    graph = build_graph(retriever=StubRetriever(), llm=StubTutorLLM())

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="quiz me on HyDE")],
            "user_id": user.id,
        },
        config={
            "configurable": {
                "quiz_master": _build_quiz_master(session),
                "memory_writer": _build_memory_writer(session),
                "judge_llm": StubJudgeLLM(),
            }
        },
    )

    # active_quiz_question_id set so next turn can grade
    assert result.get("active_quiz_question_id") is not None
    # Question text surfaced
    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert ai_msgs and "What does HyDE rewrite?" in ai_msgs[-1].content
    # Persisted question is queryable
    questions = QuestionRepository(session)
    persisted = questions.get_by_id(result["active_quiz_question_id"])
    assert persisted is not None
    assert persisted.answer == "A"
    # Judge passed (no degrade)
    assert result.get("degraded", False) is False


async def test_quiz_grade_turn_correct_answer_bumps_mastery_through_graph(session):
    users = UserRepository(session)
    goals = GoalRepository(session)
    topics = TopicRepository(session)
    questions = QuestionRepository(session)
    user = users.get_or_create("fp-e2e-grade")
    goal = goals.create(user_id=user.id, title="X")
    topic = topics.create(goal_id=goal.id, name="HyDE")
    question = questions.create(
        topic_id=topic.id,
        prompt="?",
        options_json=["A) x", "B) y", "C) z", "D) w"],
        answer="A",
        explanation="Because A.",
    )

    graph = build_graph(retriever=StubRetriever(), llm=StubTutorLLM())

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="A")],
            "user_id": user.id,
            "active_quiz_question_id": question.id,
        },
        config={
            "configurable": {
                "quiz_master": _build_quiz_master(session),
                "memory_writer": _build_memory_writer(session),
                "judge_llm": StubJudgeLLM(),
            }
        },
    )

    # Mastery bumped via QuizMaster's update_mastery call
    mastery = MasteryRepository(session)
    assert mastery.get_for_user(user.id) == {"HyDE": pytest.approx(0.1)}
    # No mistake recorded for correct answer
    mistakes = MistakeRepository(session)
    assert mistakes.get_due_for_user(user.id, now=datetime(2026, 6, 1)) == []
    # active_quiz_question_id cleared
    assert result.get("active_quiz_question_id") is None
    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert ai_msgs and "Correct" in ai_msgs[-1].content


async def test_quiz_judge_uses_quiz_rubric_not_tutor_rubric(session):
    """Quiz path must inject judge_quiz rubric so dimensions match QUIZ_DIMENSIONS."""
    users = UserRepository(session)
    user = users.get_or_create("fp-rubric-check")
    judge = StubJudgeLLM()
    graph = build_graph(retriever=StubRetriever(), llm=StubTutorLLM())

    await graph.ainvoke(
        {
            "messages": [HumanMessage(content="quiz me on HyDE")],
            "user_id": user.id,
        },
        config={
            "configurable": {
                "quiz_master": _build_quiz_master(session),
                "memory_writer": _build_memory_writer(session),
                "judge_llm": judge,
            }
        },
    )

    assert judge.last_prompt is not None
    # Quiz rubric mentions dimensions specific to quiz (not tutor's "accessibility").
    assert "question_quality" in judge.last_prompt
    assert "option_plausibility" in judge.last_prompt
    # Sanity: tutor-only dimension should NOT appear in the quiz judge prompt.
    assert "learner_level_fit" not in judge.last_prompt


async def test_quiz_grade_path_skips_judge_llm_entirely(session):
    """Cut ④g: grade output is deterministic; judging it against the quiz rubric
    (designed for question quality) produces meaningless weak verdicts. So the
    judge node must short-circuit on quiz_action='grade'.
    """
    users = UserRepository(session)
    goals = GoalRepository(session)
    topics = TopicRepository(session)
    questions = QuestionRepository(session)
    user = users.get_or_create("fp-grade-skip")
    goal = goals.create(user_id=user.id, title="X")
    topic = topics.create(goal_id=goal.id, name="HyDE")
    question = questions.create(
        topic_id=topic.id,
        prompt="?",
        options_json=["A) x", "B) y", "C) z", "D) w"],
        answer="A",
        explanation="Because A.",
    )

    judge = StubJudgeLLM()
    graph = build_graph(retriever=StubRetriever(), llm=StubTutorLLM())

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="A")],
            "user_id": user.id,
            "active_quiz_question_id": question.id,
        },
        config={
            "configurable": {
                "quiz_master": _build_quiz_master(session),
                "memory_writer": _build_memory_writer(session),
                "judge_llm": judge,
            }
        },
    )

    # Judge LLM must NOT be called on a grade turn.
    assert judge.ainvoke_count == 0
    # Graph still completes cleanly; mastery still bumped (memory_writer fired).
    assert result.get("active_quiz_question_id") is None
    assert result.get("degraded", False) is False
    mastery = MasteryRepository(session)
    assert mastery.get_for_user(user.id) == {"HyDE": pytest.approx(0.1)}


async def test_quiz_generate_path_still_runs_judge_llm(session):
    """Cut ④g sanity-check: skip is GRADE-only. GENERATE still gets judged
    (this is the valuable PDCA gate on LLM-generated questions)."""
    users = UserRepository(session)
    user = users.get_or_create("fp-gen-judge")
    judge = StubJudgeLLM()
    graph = build_graph(retriever=StubRetriever(), llm=StubTutorLLM())

    await graph.ainvoke(
        {
            "messages": [HumanMessage(content="quiz me on HyDE")],
            "user_id": user.id,
        },
        config={
            "configurable": {
                "quiz_master": _build_quiz_master(session),
                "memory_writer": _build_memory_writer(session),
                "judge_llm": judge,
            }
        },
    )

    assert judge.ainvoke_count == 1  # GENERATE turn still judged
