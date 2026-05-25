"""Cut ④b — Quiz tools tests.

Tests grow per tool. Order: update_mastery (DB-only, simplest) → record_mistake
(DB + SM-2) → grade_quiz_answer (DB lookup + compare) → generate_quiz (LLM).
"""
from datetime import datetime, timedelta

import pytest
from langchain_core.messages import AIMessage
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.agent.tools.quiz import (
    generate_quiz,
    grade_quiz_answer,
    record_mistake,
    update_mastery,
)
from app.agent.tools.schemas import GradeOut, MasteryOut, MistakeOut, QuizOut
from app.db.models import Base
from app.db.repositories import (
    GoalRepository,
    MasteryRepository,
    MistakeRepository,
    QuestionRepository,
    TopicRepository,
    UserRepository,
)


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def _seed_user_with_topic(session):
    users = UserRepository(session)
    goals = GoalRepository(session)
    topics = TopicRepository(session)
    user = users.get_or_create("fp-tools")
    goal = goals.create(user_id=user.id, title="Quiz prep")
    topic = topics.create(goal_id=goal.id, name="HyDE")
    return user, topic


def test_update_mastery_applies_delta_and_returns_new_score(session):
    user, topic = _seed_user_with_topic(session)
    mastery = MasteryRepository(session)
    mastery.upsert(user_id=user.id, topic_id=topic.id, score=0.5)

    out = update_mastery(
        user_id=user.id,
        topic_id=topic.id,
        delta=0.2,
        mastery_repo=mastery,
    )

    assert isinstance(out, MasteryOut)
    assert out.new_score == pytest.approx(0.7)
    assert mastery.get_for_user(user.id) == {"HyDE": pytest.approx(0.7)}


def test_update_mastery_creates_row_when_missing(session):
    user, topic = _seed_user_with_topic(session)
    mastery = MasteryRepository(session)

    out = update_mastery(
        user_id=user.id,
        topic_id=topic.id,
        delta=0.3,
        mastery_repo=mastery,
    )

    assert out.new_score == pytest.approx(0.3)


def _seed_question(session, topic_id: str):
    questions = QuestionRepository(session)
    return questions.create(
        topic_id=topic_id,
        prompt="What does HyDE stand for?",
        options_json=[
            "A) Hypothetical Document Embedding",
            "B) Hybrid Document Encoder",
            "C) High-Density Embedding",
            "D) None of the above",
        ],
        answer="A",
        explanation="HyDE = Hypothetical Document Embedding.",
    )


def test_record_mistake_persists_with_first_wrong_srs_schedule(session):
    user, topic = _seed_user_with_topic(session)
    question = _seed_question(session, topic.id)
    mistakes = MistakeRepository(session)

    fixed_now = datetime(2026, 5, 21, 12, 0)
    out = record_mistake(
        user_id=user.id,
        question_id=question.id,
        user_answer="B",
        mistake_repo=mistakes,
        now=fixed_now,
    )

    assert isinstance(out, MistakeOut)
    # First-time-wrong SM-2 schedule: interval=1, ease=1.7
    assert out.srs_due_at == fixed_now + timedelta(days=1)
    # The mistake_id is queryable and shows up in due list once today rolls over.
    due_tomorrow = mistakes.get_due_for_user(user.id, now=fixed_now + timedelta(days=2))
    assert out.mistake_id in due_tomorrow


def test_grade_quiz_answer_marks_match_as_correct(session):
    _, topic = _seed_user_with_topic(session)
    question = _seed_question(session, topic.id)
    questions = QuestionRepository(session)

    out = grade_quiz_answer(
        question_id=question.id,
        user_answer="A",
        question_repo=questions,
    )

    assert isinstance(out, GradeOut)
    assert out.correct is True
    assert out.correct_answer == "A"
    assert "Hypothetical Document Embedding" in out.explanation


def test_grade_quiz_answer_is_forgiving_of_case_and_whitespace(session):
    _, topic = _seed_user_with_topic(session)
    question = _seed_question(session, topic.id)
    questions = QuestionRepository(session)

    out = grade_quiz_answer(
        question_id=question.id,
        user_answer="  a  ",
        question_repo=questions,
    )

    assert out.correct is True


def test_grade_quiz_answer_marks_mismatch_as_incorrect(session):
    _, topic = _seed_user_with_topic(session)
    question = _seed_question(session, topic.id)
    questions = QuestionRepository(session)

    out = grade_quiz_answer(
        question_id=question.id,
        user_answer="B",
        question_repo=questions,
    )

    assert out.correct is False
    assert out.correct_answer == "A"


class StubQuizLLM:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.last_prompt: str | None = None

    async def ainvoke(self, messages, **_kwargs):
        self.last_prompt = messages[-1].content if messages else ""
        return AIMessage(content=self.response_text)


_ONE_QUESTION_JSON = """[
  {
    "prompt": "What does HyDE rewrite?",
    "options": ["A) Queries", "B) Documents", "C) Embeddings", "D) Answers"],
    "answer": "A",
    "explanation": "HyDE rewrites the user query into a hypothetical answer."
  }
]"""


async def test_generate_quiz_parses_llm_json_and_persists_question(session):
    _, topic = _seed_user_with_topic(session)
    questions = QuestionRepository(session)
    llm = StubQuizLLM(_ONE_QUESTION_JSON)

    out = await generate_quiz(
        topic_id=topic.id,
        topic_name="HyDE",
        difficulty="medium",
        n=1,
        llm=llm,
        question_repo=questions,
    )

    assert isinstance(out, QuizOut)
    assert len(out.questions) == 1
    q = out.questions[0]
    assert q.answer == "A"
    assert q.prompt == "What does HyDE rewrite?"
    assert "HyDE" in llm.last_prompt
    assert "medium" in llm.last_prompt.lower()

    # Persisted: id round-trips through DB
    persisted = questions.get_by_id(q.id)
    assert persisted is not None
    assert persisted.prompt == q.prompt
    assert persisted.topic_id == topic.id


_FENCED_JSON = """```json
[
  {
    "prompt": "Q?",
    "options": ["A) x", "B) y", "C) z", "D) w"],
    "answer": "C",
    "explanation": "Because C."
  }
]
```"""


async def test_generate_quiz_tolerates_markdown_json_fence(session):
    _, topic = _seed_user_with_topic(session)
    questions = QuestionRepository(session)
    llm = StubQuizLLM(_FENCED_JSON)

    out = await generate_quiz(
        topic_id=topic.id,
        topic_name="HyDE",
        n=1,
        llm=llm,
        question_repo=questions,
    )

    assert len(out.questions) == 1
    assert out.questions[0].answer == "C"


async def test_generate_quiz_embeds_context_chunks_in_prompt(session):
    """Cut ④h: when source chunks are passed, the LLM prompt must include their
    content so the model can ground the generated question."""
    _, topic = _seed_user_with_topic(session)
    questions = QuestionRepository(session)
    llm = StubQuizLLM(_ONE_QUESTION_JSON)

    chunks = [
        {"chunk_id": "topic7:p3:0", "content": "HyDE stands for Hypothetical Document Embedding."},
        {"chunk_id": "topic7:p4:0", "content": "HyDE rewrites a user query by generating a hypothetical answer first."},
    ]

    await generate_quiz(
        topic_id=topic.id,
        topic_name="HyDE",
        n=1,
        llm=llm,
        question_repo=questions,
        context_chunks=chunks,
    )

    assert llm.last_prompt is not None
    assert "Hypothetical Document Embedding" in llm.last_prompt
    assert "hypothetical answer first" in llm.last_prompt
    # Prompt must instruct LLM to ground in the chunks (not invent).
    assert "ground" in llm.last_prompt.lower() or "source" in llm.last_prompt.lower()


async def test_generate_quiz_without_context_chunks_falls_back_to_ungrounded(session):
    """No chunks → prompt skips source section entirely (current behavior)."""
    _, topic = _seed_user_with_topic(session)
    questions = QuestionRepository(session)
    llm = StubQuizLLM(_ONE_QUESTION_JSON)

    await generate_quiz(
        topic_id=topic.id,
        topic_name="HyDE",
        n=1,
        llm=llm,
        question_repo=questions,
    )

    assert llm.last_prompt is not None
    # No source-chunk markers in the prompt
    assert "Hypothetical Document Embedding" not in llm.last_prompt
