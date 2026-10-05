"""Cut P2.3-①c — unit tests for the quiz_master_agent loop body.

Test surface:
  1. natural_stop — model emits final summary with no tool calls
  2. budget_exhausted — model keeps calling tools past max_iter
  3. llm_call_failed — LLM ainvoke raises (e.g. Ollama 400 for no-tools model)
  4. tool_error_self_correction — invalid schema → ToolMessage → model retries
  5. valid_persist_round_trip — full happy path: persist → summary → active_quiz_question_id set
  6. quiz_action_always_generate — agent never sees GRADE turns by contract
"""
import json
from datetime import datetime

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.agent.agent_trace import AgentTrace
from app.agent.quiz_master_agent import (
    build_quiz_master_agent,
    _infer_quiz_action,
)
from app.db.models import Base, Question
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


class ScriptedLLM:
    """LLM that emits a scripted sequence of responses."""
    def __init__(self, responses: list[AIMessage]):
        self.responses = list(responses)
        self.bound_tools = None
        self.calls = 0

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    async def ainvoke(self, messages, **_kwargs):
        if self.calls >= len(self.responses):
            raise IndexError("ScriptedLLM ran out of responses")
        resp = self.responses[self.calls]
        self.calls += 1
        return resp


class FailingLLM:
    bound_tools = None
    def bind_tools(self, tools):
        self.bound_tools = tools
        return self
    async def ainvoke(self, messages, **_kwargs):
        raise ConnectionRefusedError("ollama is down")


def _ai(content="", tool_calls=None, input_tokens=10, output_tokens=5):
    msg = AIMessage(content=content, tool_calls=tool_calls or [])
    msg.usage_metadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    return msg


async def test_natural_stop_renders_persisted_question(session):
    user = UserRepository(session).get_or_create("fp-loop-1")
    goal_repo = GoalRepository(session)
    topic_repo = TopicRepository(session)
    question_repo = QuestionRepository(session)

    valid_persist_args = {
        "topic": "HyDE",
        "prompt": "What does HyDE stand for?",
        "options": ["A) Hypothesis-Driven Experimentation",
                    "B) Hypothetical Document Embedding",
                    "C) High-Yield Data Encoding",
                    "D) Hybrid Document Engine"],
        "answer": "B",
        "explanation": "HyDE = Hypothetical Document Embedding.",
    }
    llm = ScriptedLLM([
        _ai(tool_calls=[{"name": "retriever_search",
                         "args": {"query": "HyDE"}, "id": "tc-1"}]),
        _ai(tool_calls=[{"name": "persist_quiz_question",
                         "args": valid_persist_args, "id": "tc-2"}]),
        _ai(content="UNTRUSTED MODEL FINAL TEXT"),
    ])

    agent = build_quiz_master_agent(
        llm=llm,
        topic_repo=topic_repo, question_repo=question_repo, goal_repo=goal_repo,
        retriever=None,
        now_fn=lambda: datetime(2026, 5, 24),
    )
    result = await agent({
        "messages": [HumanMessage(content="quiz me on HyDE")],
        "user_id": user.id,
    })

    assert result["quiz_action"] == "generate"
    assert result["active_quiz_question_id"] is not None
    assert "agent_trace" in result
    assert result["agent_trace"]["exit_reason"] == "natural_stop"
    assert result["agent_trace"]["total_iterations"] == 3
    assert result["agent_trace"]["total_tool_calls"] == 2
    content = result["messages"][0].content
    assert "What does HyDE stand for?" in content
    assert "B) Hypothetical Document Embedding" in content
    assert "Reply with A, B, C, or D." in content
    assert "UNTRUSTED MODEL FINAL TEXT" not in content
    assert isinstance(result["messages"][0], AIMessage)


async def test_budget_exhausted_degrades_gracefully(session):
    user = UserRepository(session).get_or_create("fp-loop-2")
    goal_repo = GoalRepository(session)
    topic_repo = TopicRepository(session)
    question_repo = QuestionRepository(session)

    looping_responses = [
        _ai(tool_calls=[{"name": "retriever_search",
                         "args": {"query": "x"}, "id": f"tc-{i}"}])
        for i in range(10)
    ]
    llm = ScriptedLLM(looping_responses)

    agent = build_quiz_master_agent(
        llm=llm,
        topic_repo=topic_repo, question_repo=question_repo, goal_repo=goal_repo,
        retriever=None,
        max_iter=3,
    )
    result = await agent({
        "messages": [HumanMessage(content="quiz me on chunking")],
        "user_id": user.id,
    })

    assert result["degraded"] is True
    assert result["agent_trace"]["exit_reason"] == "budget_exhausted"
    assert "budget" in result["messages"][0].content.lower() or "⚠️" in result["messages"][0].content
    assert result.get("active_quiz_question_id") is None


async def test_llm_call_failed_degrades_gracefully(session):
    user = UserRepository(session).get_or_create("fp-loop-3")
    goal_repo = GoalRepository(session)
    topic_repo = TopicRepository(session)
    question_repo = QuestionRepository(session)

    agent = build_quiz_master_agent(
        llm=FailingLLM(),
        topic_repo=topic_repo, question_repo=question_repo, goal_repo=goal_repo,
    )
    result = await agent({
        "messages": [HumanMessage(content="quiz me on HyDE")],
        "user_id": user.id,
    })

    assert result["degraded"] is True
    assert result["agent_trace"]["exit_reason"] == "llm_call_failed"
    assert "ConnectionRefused" in (result["agent_trace"]["llm_error"] or "")


async def test_tool_error_self_correction_via_toolmessage(session):
    user = UserRepository(session).get_or_create("fp-loop-4")
    goal_repo = GoalRepository(session)
    topic_repo = TopicRepository(session)
    question_repo = QuestionRepository(session)

    bad_args = {
        "topic": "BM25",
        "prompt": "What is BM25?",
        "options": ["A) bad", "B) bad", "C) bad"],
        "answer": "A",
        "explanation": "BM25 is...",
    }
    good_args = {
        "topic": "BM25",
        "prompt": "What is BM25?",
        "options": ["A) embedding", "B) ranking function",
                    "C) tokenizer", "D) reranker"],
        "answer": "B",
        "explanation": "BM25 is a probabilistic ranking function.",
    }
    llm = ScriptedLLM([
        _ai(tool_calls=[{"name": "persist_quiz_question",
                         "args": bad_args, "id": "tc-bad"}]),
        _ai(tool_calls=[{"name": "persist_quiz_question",
                         "args": good_args, "id": "tc-good"}]),
        _ai(content="Quiz on BM25 ready"),
    ])

    agent = build_quiz_master_agent(
        llm=llm,
        topic_repo=topic_repo, question_repo=question_repo, goal_repo=goal_repo,
    )
    result = await agent({
        "messages": [HumanMessage(content="quiz me on BM25")],
        "user_id": user.id,
    })

    assert result["quiz_action"] == "generate"
    assert result["active_quiz_question_id"] is not None
    assert result["agent_trace"]["exit_reason"] == "natural_stop"
    assert result["agent_trace"]["tool_errors"] == 1
    breakdown = result["agent_trace"]["tool_call_breakdown"]
    assert breakdown.get("persist_quiz_question") == 2


async def test_valid_persist_round_trip_writes_active_quiz_question_id(session):
    user = UserRepository(session).get_or_create("fp-loop-5")
    goal_repo = GoalRepository(session)
    topic_repo = TopicRepository(session)
    question_repo = QuestionRepository(session)

    persist_args = {
        "topic": "embeddings",
        "prompt": "What is an embedding?",
        "options": ["A) A vector representation",
                    "B) A type of database",
                    "C) A search algorithm",
                    "D) A loss function"],
        "answer": "A",
        "explanation": "An embedding is a dense vector representation of text.",
    }
    llm = ScriptedLLM([
        _ai(tool_calls=[{"name": "persist_quiz_question",
                         "args": persist_args, "id": "tc-1"}]),
        _ai(content="Quiz ready"),
    ])

    agent = build_quiz_master_agent(
        llm=llm,
        topic_repo=topic_repo, question_repo=question_repo, goal_repo=goal_repo,
    )
    result = await agent({
        "messages": [HumanMessage(content="quiz me on embeddings")],
        "user_id": user.id,
    })

    persisted_id = result["active_quiz_question_id"]
    assert persisted_id is not None
    fetched = question_repo.get_by_id(persisted_id)
    assert fetched is not None
    assert fetched.answer == "A"


async def test_final_quiz_text_without_successful_persist_degrades_instead(session):
    user = UserRepository(session).get_or_create("fp-loop-no-persist")
    goal_repo = GoalRepository(session)
    topic_repo = TopicRepository(session)
    question_repo = QuestionRepository(session)

    invalid_persist_args = {
        "topic": "RRF",
        "prompt": "What is RRF?",
        "options": ["A) Only one option"],
        "answer": "A",
        "explanation": "RRF combines rankings.",
    }
    llm = ScriptedLLM([
        _ai(tool_calls=[{"name": "persist_quiz_question",
                         "args": invalid_persist_args, "id": "tc-bad"}]),
        _ai(content=(
            "Here is your quiz question:\n\n"
            "What is RRF?\n\n"
            "A) Reciprocal Rank Fusion\n"
            "B) Random Ranking Filter\n"
            "C) Recursive Retrieval Format\n"
            "D) Ranked Result File\n\n"
            "Answer: A"
        )),
    ])

    agent = build_quiz_master_agent(
        llm=llm,
        topic_repo=topic_repo, question_repo=question_repo, goal_repo=goal_repo,
    )
    result = await agent({
        "messages": [HumanMessage(content="quiz me on RRF")],
        "user_id": user.id,
    })

    content = result["messages"][0].content
    assert result["degraded"] is True
    assert result.get("active_quiz_question_id") is None
    assert result["agent_trace"]["tool_errors"] == 1
    assert result["agent_trace"]["exit_reason"] == "quiz_persist_failed"
    assert "couldn't save a gradeable quiz question" in content.lower()
    assert "A) Reciprocal Rank Fusion" not in content


def test_infer_quiz_action_always_returns_generate():
    """Agent never sees GRADE turns by dispatcher contract."""
    import time
    trace = AgentTrace(t_start=time.monotonic())
    assert _infer_quiz_action(trace) == "generate"
    trace.record_tool_call("persist_quiz_question", {}, '{"question_id":"q"}', error=False)
    assert _infer_quiz_action(trace) == "generate"


# --- Batch A: LLM error detail boundary ------------------------------------

_MARKER = "SECRET_OPAQUE_MARKER_7f3a"
_SAFE_CONNECTION = "ConnectionRefusedError: Could not connect to the model service."


class MarkerCrashingLLM:
    """Fails the first LLM call with an opaque, detail-bearing exception."""

    def __init__(self):
        self.calls = 0

    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, messages, **_kwargs):
        self.calls += 1
        raise ConnectionRefusedError(
            f"[Errno 61] Connection refused to ollama at "
            f"http://127.0.0.1:11434/api/chat?api_key=sk-live-{_MARKER}"
        )


async def test_llm_failure_detail_is_projected_in_internal_and_public_trace(
    session, monkeypatch
):
    from app.agent import quiz_master_agent as quiz_agent_mod

    events: list[dict] = []
    monkeypatch.setattr(quiz_agent_mod, "get_stream_writer", lambda: events.append)

    user = UserRepository(session).get_or_create("fp-loop-marker")
    question_repo = QuestionRepository(session)
    llm = MarkerCrashingLLM()
    agent = build_quiz_master_agent(
        llm=llm,
        topic_repo=TopicRepository(session),
        question_repo=question_repo,
        goal_repo=GoalRepository(session),
    )

    result = await agent({
        "messages": [HumanMessage(content="quiz me on HyDE")],
        "user_id": user.id,
    })

    # One LLM call, clean degrade, no gradeable question persisted.
    assert llm.calls == 1
    assert result["degraded"] is True
    assert result.get("active_quiz_question_id") is None
    assert session.query(Question).count() == 0

    internal = result["agent_trace"]
    assert internal["exit_reason"] == "llm_call_failed"
    assert internal["llm_error"] == _SAFE_CONNECTION
    assert internal["total_iterations"] == 0

    public = [e for e in events if e.get("type") == "agent_run"]
    assert len(public) == 1
    assert public[0]["run"]["node"] == "quiz"
    assert public[0]["run"]["exit_reason"] == "llm_call_failed"
    assert public[0]["run"]["llm_error"] == _SAFE_CONNECTION

    assert _MARKER not in json.dumps(result, default=str)
    assert _MARKER not in json.dumps(events, default=str)


# ---------------------------------------------------------------------------
# Batch C — content blocks on tool-call-only turns.
# ---------------------------------------------------------------------------


class RecordingScriptedLLM(ScriptedLLM):
    """ScriptedLLM that also keeps the exact message list of every call."""

    def __init__(self, responses: list[AIMessage]):
        super().__init__(responses)
        self.message_history: list[list] = []

    async def ainvoke(self, messages, **_kwargs):
        self.message_history.append(list(messages))
        return await super().ainvoke(messages, **_kwargs)


def _block_ai(content, tool_calls=None, input_tokens=10, output_tokens=5):
    msg = AIMessage(content=content, tool_calls=tool_calls or [])
    msg.usage_metadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    return msg


async def test_tool_call_only_block_messages_keep_raw_arguments_and_identity(session):
    """Content blocks on tool-call turns must not disturb the real tool loop.

    The intermediate assistant envelopes keep their block content and raw
    tool-call arguments; success is identified by the persisted question row,
    and the visible GENERATE text never leaks the answer or explanation.
    """
    user = UserRepository(session).get_or_create("fp-loop-blocks")
    goal_repo = GoalRepository(session)
    topic_repo = TopicRepository(session)
    question_repo = QuestionRepository(session)

    search_args = {"query": "HyDE"}
    persist_args = {
        "topic": "HyDE",
        "prompt": "What does HyDE stand for?",
        "options": ["A) a", "B) b", "C) c", "D) d"],
        "answer": "B",
        "explanation": "HyDE = Hypothetical Document Embedding.",
    }
    first_blocks = [
        {"type": "text", "text": "I will search the notes first."},
        {"type": "thinking", "thinking": "plan the retrieval"},
    ]
    second_blocks = [{"type": "text", "text": "Now persist the question."}]
    final_blocks = [{"type": "text", "text": "UNTRUSTED MODEL FINAL TEXT"}]

    llm = RecordingScriptedLLM([
        _block_ai(first_blocks,
                  tool_calls=[{"name": "retriever_search",
                               "args": search_args, "id": "tc-1"}]),
        _block_ai(second_blocks,
                  tool_calls=[{"name": "persist_quiz_question",
                               "args": persist_args, "id": "tc-2"}]),
        _block_ai(final_blocks),
    ])

    agent = build_quiz_master_agent(
        llm=llm,
        topic_repo=topic_repo, question_repo=question_repo, goal_repo=goal_repo,
        retriever=None,
        now_fn=lambda: datetime(2026, 5, 24),
    )
    result = await agent({
        "messages": [HumanMessage(content="quiz me on HyDE")],
        "user_id": user.id,
    })

    assert result["quiz_action"] == "generate"
    assert result["agent_trace"]["exit_reason"] == "natural_stop"
    assert result["agent_trace"]["tool_call_breakdown"] == {
        "retriever_search": 1,
        "persist_quiz_question": 1,
    }

    # Result identity is the persisted row, not the model's final prose.
    question_id = result["active_quiz_question_id"]
    assert question_id is not None
    row = session.get(Question, question_id)
    assert row is not None
    assert row.prompt == persist_args["prompt"]
    assert row.answer == persist_args["answer"]
    assert row.explanation == persist_args["explanation"]

    content = result["messages"][0].content
    assert isinstance(content, str)
    assert persist_args["prompt"] in content
    assert "B) b" in content
    assert "Reply with A, B, C, or D." in content
    assert persist_args["explanation"] not in content
    assert "UNTRUSTED MODEL FINAL TEXT" not in content

    # Raw assistant envelopes (block content + tool_calls) are re-sent verbatim:
    # the loop must not flatten them through the text boundary helper.
    second_call = llm.message_history[1]
    first_ai = next(m for m in second_call if isinstance(m, AIMessage))
    assert first_ai.content == first_blocks
    assert first_ai.tool_calls[0]["args"] == search_args

    third_call = llm.message_history[2]
    tool_messages = [m for m in third_call if isinstance(m, ToolMessage)]
    assert any(question_id in m.content for m in tool_messages)
