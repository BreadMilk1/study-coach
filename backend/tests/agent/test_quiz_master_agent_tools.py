"""Cut P2.3-①b — unit tests for the 2 LLM-facing tool wrappers in quiz_master_agent.

Each test exercises:
  1. Closure injection — LLM-visible args do not include user_id / repos, yet
     the tool can use them via the factory closure.
  2. JSON return shape — every tool returns a JSON-serializable string.
  3. For persist_quiz_question: round-trip persistence on valid input, and
     {"error": ...} JSON on Pydantic validation failure (LLM self-correct path).

Business logic depth is NOT re-tested here — Pydantic validation is exercised
by Pydantic itself. The contract under test is the WRAPPER, not the validator.
"""
import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.agent.quiz_master_agent import _make_quiz_tools
from app.db.models import Base
from app.db.repositories import (
    GoalRepository,
    QuestionRepository,
    TopicRepository,
    UserRepository,
)


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


class StubRetriever:
    def __init__(self, chunks=None):
        self.chunks = chunks or []
        self.calls: list[tuple[str, int]] = []

    def search(self, query, top_k=5):
        self.calls.append((query, top_k))
        return self.chunks[:top_k]


def test_retriever_search_uses_closure_retriever_and_returns_json():
    retriever = StubRetriever(chunks=[
        {"chunk_id": "c1", "content": "HyDE = Hypothetical Document Embedding.", "page": 1},
        {"chunk_id": "c2", "content": "Embed the hypothetical doc, search corpus.", "page": 2},
    ])
    tools = _make_quiz_tools(
        user_id="u1", retriever=retriever,
        topic_repo=None, question_repo=None, goal_repo=None,
    )
    tool = next(t for t in tools if t.name == "retriever_search")

    assert "query" in tool.args
    assert "top_k" in tool.args
    assert "retriever" not in tool.args
    assert "user_id" not in tool.args

    out = tool.invoke({"query": "HyDE", "top_k": 1})
    parsed = json.loads(out)
    assert retriever.calls == [("HyDE", 1)]
    assert isinstance(parsed, list)
    assert parsed[0]["chunk_id"] == "c1"


def test_persist_quiz_question_creates_goal_topic_question_on_valid_input(session):
    user = UserRepository(session).get_or_create("fp-quiz-1")
    goal_repo = GoalRepository(session)
    topic_repo = TopicRepository(session)
    question_repo = QuestionRepository(session)

    tools = _make_quiz_tools(
        user_id=user.id, retriever=None,
        topic_repo=topic_repo, question_repo=question_repo, goal_repo=goal_repo,
    )
    tool = next(t for t in tools if t.name == "persist_quiz_question")

    assert "topic" in tool.args
    assert "prompt" in tool.args
    assert "options" in tool.args
    assert "answer" in tool.args
    assert "explanation" in tool.args
    assert "user_id" not in tool.args

    out = tool.invoke({
        "topic": "HyDE",
        "prompt": "What does HyDE stand for?",
        "options": ["A) Hypothesis-Driven Experimentation",
                    "B) Hypothetical Document Embedding",
                    "C) High-Yield Data Encoding",
                    "D) Hybrid Document Engine"],
        "answer": "B",
        "explanation": "HyDE = Hypothetical Document Embedding (Gao et al. 2022).",
    })
    parsed = json.loads(out)
    assert parsed["persisted"] is True
    assert "question_id" in parsed
    assert "topic_id" in parsed

    q = question_repo.get_by_id(parsed["question_id"])
    assert q is not None
    assert q.answer == "B"
    assert q.options_json == ["A) Hypothesis-Driven Experimentation",
                              "B) Hypothetical Document Embedding",
                              "C) High-Yield Data Encoding",
                              "D) Hybrid Document Engine"]
    assert goal_repo.list_active_for_user(user.id), "goal auto-created"


def test_persist_quiz_question_returns_error_json_on_wrong_options_count(session):
    user = UserRepository(session).get_or_create("fp-quiz-2")
    goal_repo = GoalRepository(session)
    topic_repo = TopicRepository(session)
    question_repo = QuestionRepository(session)

    tools = _make_quiz_tools(
        user_id=user.id, retriever=None,
        topic_repo=topic_repo, question_repo=question_repo, goal_repo=goal_repo,
    )
    tool = next(t for t in tools if t.name == "persist_quiz_question")

    out = tool.invoke({
        "topic": "BM25",
        "prompt": "What is BM25?",
        "options": ["A) An embedding model", "B) A ranking function"],
        "answer": "B",
        "explanation": "BM25 is a ranking function used by search engines.",
    })
    parsed = json.loads(out)
    assert "error" in parsed
    assert "options" in parsed["error"]


def test_persist_quiz_question_returns_error_json_on_invalid_option_prefix(session):
    user = UserRepository(session).get_or_create("fp-quiz-3")
    goal_repo = GoalRepository(session)
    topic_repo = TopicRepository(session)
    question_repo = QuestionRepository(session)

    tools = _make_quiz_tools(
        user_id=user.id, retriever=None,
        topic_repo=topic_repo, question_repo=question_repo, goal_repo=goal_repo,
    )
    tool = next(t for t in tools if t.name == "persist_quiz_question")

    out = tool.invoke({
        "topic": "reranking",
        "prompt": "What is a reranker?",
        "options": ["a) bad prefix", "B) ok", "C) ok", "D) ok"],
        "answer": "B",
        "explanation": "A reranker reorders search results by relevance.",
    })
    parsed = json.loads(out)
    assert "error" in parsed
    assert "option[0]" in parsed["error"] or "must start with" in parsed["error"]
