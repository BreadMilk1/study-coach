"""Cut P2.2-①c — loop body + plan_action inference + error handling.

Stub LLM is scripted: each .ainvoke() returns the next preset AIMessage. Some
have tool_calls (forcing the loop to dispatch + iterate), the final one has
content but no tool_calls (natural stop).

Tests:
  1. natural_stop after retriever_search → update_study_plan → final summary
  2. budget_exhausted when LLM never stops calling tools
  3. llm_call_failed degrades cleanly without persisting
  4. tool error becomes a ToolMessage and loop continues (self-correction)
  5. plan_action inference: generate when get_existing_plan absent/null
  6. plan_action inference: check_in when get_existing_plan returned non-null
  7. plan_action inference fallback: generate when zero tools called
  8. _extract_topic regression — closes a P2.1-⑤i loose end
"""
import json
import socket
from datetime import datetime

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.agent.planner_agent import build_planner_agent
from app.db.models import Base
from app.db.repositories import (
    GoalRepository,
    MasteryRepository,
    MistakeRepository,
    PlanRepository,
    UserRepository,
)


class ScriptedLLM:
    """Returns the next AIMessage from a preset list per .ainvoke() call.

    The harness wraps this with .bind_tools(tools); since our loop only reads
    the returned AIMessage and its .tool_calls, we don't need a real
    BaseChatModel — just a duck-typed ainvoke + bind_tools no-op.
    """
    def __init__(self, responses: list[AIMessage]):
        self.responses = list(responses)
        self.idx = 0
        self.calls = 0
        self.seen: list[list] = []

    def bind_tools(self, _tools):
        return self  # pass-through; tools are dispatched outside the LLM

    async def ainvoke(self, messages, **_kwargs):
        self.calls += 1
        self.seen.append(list(messages))
        if self.idx >= len(self.responses):
            raise AssertionError("ScriptedLLM exhausted — loop called more times than expected")
        msg = self.responses[self.idx]
        self.idx += 1
        return msg


class CrashingLLM:
    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, messages, **_kwargs):
        raise ConnectionError("ollama unreachable")


class StubRetriever:
    def __init__(self, chunks=None):
        self.chunks = chunks or []

    def search(self, query, top_k=5):
        return self.chunks[:top_k]


def _msg(content="", tool_calls=None, input_tokens=10, output_tokens=5):
    msg = AIMessage(content=content, tool_calls=tool_calls or [])
    msg.usage_metadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    return msg


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


def _build(session, llm, retriever=None):
    return build_planner_agent(
        llm=llm,
        plan_repo=PlanRepository(session),
        goal_repo=GoalRepository(session),
        mastery_repo=MasteryRepository(session),
        mistake_repo=MistakeRepository(session),
        retriever=retriever or StubRetriever(),
        now_fn=lambda: datetime(2026, 5, 23, 12, 0),
        max_iter=10,
    )


async def test_loop_natural_stop_emits_plan_and_records_trace(session):
    user = UserRepository(session).get_or_create("fp-loop-1")
    goal_repo = GoalRepository(session)
    goal_repo.create(user_id=user.id, title="G")

    llm = ScriptedLLM([
        _msg(tool_calls=[{
            "name": "retriever_search",
            "args": {"query": "HyDE", "top_k": 3},
            "id": "c1",
        }]),
        _msg(tool_calls=[{
            "name": "update_study_plan",
            "args": {"milestones": [
                {"title": "Read HyDE", "due_at": "2026-05-25", "done": False, "topic": "HyDE"},
            ]},
            "id": "c2",
        }]),
        _msg(content="📋 Plan: Read HyDE by 2026-05-25.", tool_calls=[]),
    ])
    agent = _build(session, llm, retriever=StubRetriever(chunks=[
        {"chunk_id": "c1", "content": "HyDE definition", "page": 1},
    ]))

    update = await agent({
        "messages": [HumanMessage(content="make a plan on HyDE")],
        "user_id": user.id,
    })

    assert update["plan_action"] == "generate"
    assert update["active_plan_id"]
    assert "Read HyDE" in update["messages"][0].content
    trace = update["agent_trace"]
    assert trace["exit_reason"] == "natural_stop"
    assert trace["total_iterations"] == 3
    assert trace["total_tool_calls"] == 2
    assert trace["tool_call_breakdown"] == {
        "retriever_search": 1, "update_study_plan": 1,
    }


async def test_loop_budget_exhaustion_degrades_without_persisting(session):
    user = UserRepository(session).get_or_create("fp-loop-2")
    GoalRepository(session).create(user_id=user.id, title="G")

    # LLM keeps calling retriever_search forever; loop must bail at max_iter.
    forever_calls = [
        _msg(tool_calls=[{"name": "retriever_search", "args": {"query": "x"}, "id": f"c{i}"}])
        for i in range(12)
    ]
    llm = ScriptedLLM(forever_calls)
    agent = _build(session, llm)

    update = await agent({
        "messages": [HumanMessage(content="plan on X")],
        "user_id": user.id,
    })

    assert update["agent_trace"]["exit_reason"] == "budget_exhausted"
    # No plan should have been persisted on this path
    assert update.get("active_plan_id") is None
    # User-visible disclaimer text
    assert "reasoning budget" in update["messages"][0].content.lower() \
        or "budget" in update["messages"][0].content.lower()


async def test_loop_llm_error_degrades_with_disclaimer(session):
    user = UserRepository(session).get_or_create("fp-loop-3")
    GoalRepository(session).create(user_id=user.id, title="G")
    agent = _build(session, CrashingLLM())

    update = await agent({
        "messages": [HumanMessage(content="plan on something")],
        "user_id": user.id,
    })

    assert update["agent_trace"]["exit_reason"] == "llm_call_failed"
    assert "ConnectionError" in update["agent_trace"]["llm_error"]
    assert "could not reach" in update["messages"][0].content.lower() \
        or "model" in update["messages"][0].content.lower()


async def test_loop_tool_error_feeds_back_and_self_corrects(session):
    """First update_study_plan call has a bad milestone shape → ToolMessage
    error → LLM retries with correct shape → natural stop."""
    user = UserRepository(session).get_or_create("fp-loop-4")
    GoalRepository(session).create(user_id=user.id, title="G")

    llm = ScriptedLLM([
        _msg(tool_calls=[{
            "name": "update_study_plan",
            "args": {"milestones": [{"WRONG_KEY": "no title"}]},  # invalid → tool error
            "id": "c1",
        }]),
        _msg(tool_calls=[{
            "name": "update_study_plan",
            "args": {"milestones": [
                {"title": "M1", "due_at": None, "done": False, "topic": "X"},
            ]},
            "id": "c2",
        }]),
        _msg(content="Plan ready.", tool_calls=[]),
    ])
    agent = _build(session, llm)

    update = await agent({
        "messages": [HumanMessage(content="plan on X")],
        "user_id": user.id,
    })

    assert update["agent_trace"]["exit_reason"] == "natural_stop"
    assert update["agent_trace"]["tool_errors"] == 1
    # Second call succeeded → plan persisted
    assert update.get("active_plan_id")


async def test_plan_action_generate_when_no_get_existing_plan_called(session):
    user = UserRepository(session).get_or_create("fp-loop-5")
    GoalRepository(session).create(user_id=user.id, title="G")
    llm = ScriptedLLM([_msg(content="just text", tool_calls=[])])
    agent = _build(session, llm)

    update = await agent({
        "messages": [HumanMessage(content="x")],
        "user_id": user.id,
    })
    # No tools called → fallback inference is "generate"
    assert update["plan_action"] == "generate"


async def test_plan_action_check_in_when_get_existing_plan_returned_nonnull(session):
    user = UserRepository(session).get_or_create("fp-loop-6")
    goal_repo = GoalRepository(session)
    goal = goal_repo.create(user_id=user.id, title="G")
    plan_repo = PlanRepository(session)
    plan_repo.create(goal_id=goal.id, milestones_json=[
        {"title": "old M", "done": False, "topic": "HyDE"},
    ])

    llm = ScriptedLLM([
        _msg(tool_calls=[{"name": "get_existing_plan", "args": {}, "id": "c1"}]),
        _msg(content="progress: 0/1 done", tool_calls=[]),
    ])
    agent = _build(session, llm)

    update = await agent({
        "messages": [HumanMessage(content="how is the plan going")],
        "user_id": user.id,
    })

    assert update["plan_action"] == "check_in"


async def test_plan_action_generate_when_get_existing_plan_returned_null(session):
    """get_existing_plan called BUT returned the literal "null" sentinel → generate."""
    user = UserRepository(session).get_or_create("fp-loop-7")
    GoalRepository(session).create(user_id=user.id, title="G")

    llm = ScriptedLLM([
        _msg(tool_calls=[{"name": "get_existing_plan", "args": {}, "id": "c1"}]),
        # Loop now sees "null" — model decides no existing plan, drafts a new one
        _msg(tool_calls=[{
            "name": "update_study_plan",
            "args": {"milestones": [
                {"title": "M1", "due_at": None, "done": False, "topic": "x"},
            ]},
            "id": "c2",
        }]),
        _msg(content="Plan made.", tool_calls=[]),
    ])
    agent = _build(session, llm)

    update = await agent({
        "messages": [HumanMessage(content="plan")],
        "user_id": user.id,
    })

    assert update["plan_action"] == "generate"


async def test_extract_topic_strips_mindmap_suffix_without_corrupting_english():
    """Regression for the P2.1-⑤i character-set vs word-suffix strip bug.
    Topic 'Spam' must not become 'Sp' after suffix stripping."""
    from app.agent.planner_agent import _extract_topic_for_agent_prompt

    assert _extract_topic_for_agent_prompt("make a plan on Spam") == "Spam"
    assert _extract_topic_for_agent_prompt("帮我做学习计划 on HyDE 画脑图") == "HyDE"
    assert _extract_topic_for_agent_prompt("plan on BM25?") == "BM25"
    assert _extract_topic_for_agent_prompt("帮我做学习计划 on HyDE！") == "HyDE"


# --- Batch A: LLM error detail boundary ------------------------------------

_MARKER = "SECRET_OPAQUE_MARKER_7f3a"
_SAFE_CONNECTION = "ConnectionError: Could not connect to the model service."


class MarkerCrashingLLM:
    """Fails the first LLM call with an opaque, detail-bearing exception."""

    def __init__(self):
        self.calls = 0

    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, messages, **_kwargs):
        self.calls += 1
        raise ConnectionError(
            f"POST http://127.0.0.1:11434/api/chat failed: "
            f"Authorization: Bearer sk-live-{_MARKER}"
        )


async def test_llm_failure_detail_is_projected_in_internal_and_public_trace(
    session, monkeypatch
):
    from app.agent import planner_agent as planner_agent_mod

    events: list[dict] = []
    monkeypatch.setattr(planner_agent_mod, "get_stream_writer", lambda: events.append)

    user = UserRepository(session).get_or_create("fp-loop-marker")
    goal_repo = GoalRepository(session)
    goal_repo.create(user_id=user.id, title="G")
    llm = MarkerCrashingLLM()
    agent = _build(session, llm)

    update = await agent({
        "messages": [HumanMessage(content="plan on something")],
        "user_id": user.id,
    })

    # One LLM call, clean degrade, unchanged business fallback text.
    assert llm.calls == 1
    assert update["degraded"] is True
    # The loop never reached a confirmed persist → no active plan id.
    assert "active_plan_id" not in update

    internal = update["agent_trace"]
    assert internal["exit_reason"] == "llm_call_failed"
    assert internal["llm_error"] == _SAFE_CONNECTION
    assert internal["total_iterations"] == 0

    public = [e for e in events if e.get("type") == "agent_run"]
    assert len(public) == 1
    assert public[0]["run"]["exit_reason"] == "llm_call_failed"
    assert public[0]["run"]["llm_error"] == _SAFE_CONNECTION

    assert _MARKER not in json.dumps(update, default=str)
    assert _MARKER not in json.dumps(events, default=str)

    # The failing LLM never reached a persist tool → no plan side effect.
    active = goal_repo.list_active_for_user(user.id)
    assert PlanRepository(session).get_by_goal(active[0].id) is None


# ---------------------------------------------------------------------------
# Batch D1 — strict final-content consumption in the Planner agent loop.
#
# Only the model is faked; the real loop, tools, trace and repositories run.
# A final response without a usable text body degrades through the existing
# `llm_call_failed` path (warning assistant + failed agent_run), which is a
# completed Chat turn — NOT the Tutor's no-assistant / no-done boundary.
# ---------------------------------------------------------------------------

_D1_SENTINEL = "D1_SENTINEL_DO_NOT_LEAK"
_D1_MALFORMED = "LLM response content is not a supported text shape"
_D1_EMPTY = "LLM response text has no non-whitespace body"
_LLM_FAILED_TEXT = "⚠️ Could not reach the planner model. Please try again."
_GENERIC_LLM_ERROR = "LLMError: Model request failed."


@pytest.fixture
def no_network(monkeypatch):
    """Fail immediately on any real DNS / outbound socket attempt."""

    def deny(*_args, **_kwargs):
        raise AssertionError("D1 consumer test attempted real network access")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(socket, "getaddrinfo", deny)


def _plan_tool_call_msg(call_id: str = "c1"):
    return _msg(tool_calls=[{
        "name": "update_study_plan",
        "args": {"milestones": [
            {"title": "Read HyDE", "due_at": "2026-05-25", "done": False, "topic": "HyDE"},
        ]},
        "id": call_id,
    }])


def _d1_events(monkeypatch) -> list[dict]:
    from app.agent import planner_agent as planner_agent_mod

    events: list[dict] = []
    monkeypatch.setattr(planner_agent_mod, "get_stream_writer", lambda: events.append)
    return events


async def test_agent_loop_final_unknown_block_degrades_instead_of_showing_payload(
    session, monkeypatch, no_network
):
    events = _d1_events(monkeypatch)
    user = UserRepository(session).get_or_create("fp-d1-loop-unknown")
    goal_repo = GoalRepository(session)
    goal = goal_repo.create(user_id=user.id, title="G")

    llm = ScriptedLLM([
        _plan_tool_call_msg(),
        _msg(content=[{"type": "unknown_vendor_block", "text": _D1_SENTINEL}]),
    ])
    agent = _build(session, llm)

    update = await agent({
        "messages": [HumanMessage(content="make a plan on HyDE")],
        "user_id": user.id,
    })

    assert update["degraded"] is True
    assert update["messages"][0].content == _LLM_FAILED_TEXT
    trace = update["agent_trace"]
    assert trace["exit_reason"] == "llm_call_failed"
    assert trace["llm_error"] == _GENERIC_LLM_ERROR
    # The rejected iteration is still recorded, with its usage.
    assert trace["total_iterations"] == 2
    assert trace["output_tokens"] == 10
    assert trace["total_tool_calls"] == 1
    # No "this turn succeeded" plan id is claimed by the node update.
    assert "active_plan_id" not in update

    # The tool already committed the plan before the model failed to summarise.
    saved = PlanRepository(session).get_by_goal(goal.id)
    assert saved is not None
    assert [m["title"] for m in saved.milestones_json] == ["Read HyDE"]

    tokens = [e for e in events if e.get("type") == "token"]
    assert [e["text"] for e in tokens] == [_LLM_FAILED_TEXT]
    payload = json.dumps(update, default=str) + json.dumps(events, default=str)
    assert _D1_SENTINEL not in payload
    assert _D1_MALFORMED not in payload
    public = [e for e in events if e.get("type") == "agent_run"]
    assert len(public) == 1
    assert public[0]["run"]["exit_reason"] == "llm_call_failed"
    assert public[0]["run"]["llm_error"] == _GENERIC_LLM_ERROR


@pytest.mark.parametrize(
    "final_content",
    [
        "",
        "   ",
        "\n\n",
        [],
        [{"type": "thinking", "thinking": _D1_SENTINEL}],
    ],
    ids=["empty-str", "blank-str", "newlines", "empty-list", "reasoning-only"],
)
async def test_agent_loop_final_without_usable_body_degrades_with_llm_call_failed(
    session, monkeypatch, no_network, final_content
):
    events = _d1_events(monkeypatch)
    user = UserRepository(session).get_or_create("fp-d1-loop-empty")
    GoalRepository(session).create(user_id=user.id, title="G")

    llm = ScriptedLLM([_msg(content=final_content, tool_calls=[])])
    agent = _build(session, llm)

    update = await agent({
        "messages": [HumanMessage(content="make a plan on HyDE")],
        "user_id": user.id,
    })

    trace = update["agent_trace"]
    assert trace["exit_reason"] == "llm_call_failed", (
        "a final response without a usable body must not be reported as natural_stop"
    )
    assert trace["llm_error"] == _GENERIC_LLM_ERROR
    assert trace["total_iterations"] == 1
    assert update["messages"][0].content == _LLM_FAILED_TEXT
    tokens = [e for e in events if e.get("type") == "token"]
    assert [e["text"] for e in tokens] == [_LLM_FAILED_TEXT]
    payload = json.dumps(update, default=str) + json.dumps(events, default=str)
    assert _D1_SENTINEL not in payload
    assert _D1_EMPTY not in payload
    assert _D1_MALFORMED not in payload


async def test_agent_loop_only_narrows_the_helper_value_error(
    session, monkeypatch, no_network
):
    """An unrelated failure inside the terminal branch must not be converted
    into the llm_call_failed degrade."""
    from app.agent import planner_agent as planner_agent_mod

    def explode(_content):
        raise RuntimeError("unrelated terminal-branch failure")

    monkeypatch.setattr(planner_agent_mod, "require_text", explode)
    user = UserRepository(session).get_or_create("fp-d1-loop-narrow")
    GoalRepository(session).create(user_id=user.id, title="G")
    agent = _build(session, ScriptedLLM([_msg(content="fine", tool_calls=[])]))

    with pytest.raises(RuntimeError, match="unrelated terminal-branch failure"):
        await agent({
            "messages": [HumanMessage(content="make a plan on HyDE")],
            "user_id": user.id,
        })


async def test_agent_loop_final_text_blocks_keep_original_whitespace_and_order(
    session, no_network
):
    user = UserRepository(session).get_or_create("fp-d1-loop-whitespace")
    GoalRepository(session).create(user_id=user.id, title="G")
    llm = ScriptedLLM([_msg(
        content=[
            {"type": "thinking", "thinking": _D1_SENTINEL},
            {"type": "text", "text": "  first\n"},
            "second\n\n",
            {"type": "text", "text": "  third  "},
        ],
        tool_calls=[],
    )])
    agent = _build(session, llm)

    update = await agent({
        "messages": [HumanMessage(content="make a plan on HyDE")],
        "user_id": user.id,
    })

    assert update["agent_trace"]["exit_reason"] == "natural_stop"
    assert update["messages"][0].content == "  first\nsecond\n\n  third  "
    assert _D1_SENTINEL not in update["messages"][0].content


async def test_agent_loop_keeps_tool_call_responses_and_matching_tool_message_ids(
    session, no_network
):
    """Tool-call turns are complete messages: never extracted, never rebuilt,
    and the ToolMessage keeps the original tool_call_id."""
    from langchain_core.messages import ToolMessage

    user = UserRepository(session).get_or_create("fp-d1-loop-history")
    GoalRepository(session).create(user_id=user.id, title="G")
    tool_turn = _msg(
        content=[{"type": "unknown_vendor_block", "text": _D1_SENTINEL}],
        tool_calls=[{"name": "retriever_search", "args": {"query": "HyDE"}, "id": "call-1"}],
    )
    llm = ScriptedLLM([tool_turn, _msg(content="Final summary.", tool_calls=[])])
    agent = _build(session, llm, retriever=StubRetriever(chunks=[
        {"chunk_id": "c1", "content": "HyDE definition", "page": 1},
    ]))

    update = await agent({
        "messages": [HumanMessage(content="make a plan on HyDE")],
        "user_id": user.id,
    })

    assert update["agent_trace"]["exit_reason"] == "natural_stop"
    second_call = llm.seen[1]
    assert second_call[2] is tool_turn  # the original AIMessage, unmodified
    assert second_call[2].tool_calls[0]["id"] == "call-1"
    tool_message = second_call[3]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.tool_call_id == "call-1"
