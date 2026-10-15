"""D4b-2 Planner real-SDK consumer contract — agent loop + deterministic node.

Scope (D4b-2-PLAN.md): OpenAI ``gpt-4o-mini`` Chat Completions only, through the
product's own ``get_chat_model`` -> ``ChatOpenAI`` -> OpenAI SDK ->
``httpx.MockTransport`` chain. The consumer under test is always the real
``build_planner_agent`` / ``build_planner`` node with the real repositories;
only the provider socket, the retrieval corpus and the model's scripted answers
are synthetic.

Nothing here patches a factory, an adapter, the SDK parser, the node or a
repository. Every request is checked against an expectation built in this file
from the frozen prompt templates and this case's declared synthetic inputs --
never by comparing an observed value with itself.

Case map: P1-P5 agent loop, P6-P10 deterministic node. P8 is parametrized over
two soft-degradation triggers, so the file holds 11 test functions.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import sys
import sysconfig
from datetime import date, datetime
from pathlib import Path

import httpx
import pytest
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import select
from sqlalchemy.orm import Session
from pydantic import ValidationError

from app.agent.agent_trace import AgentTrace
from app.agent.planner import build_planner
from app.agent.planner_agent import (
    _BUDGET_EXHAUSTED_MSG,
    _LLM_FAILED_MSG,
    build_planner_agent,
)
from app.agent.tools.schemas import Milestone
from app.db.models import Base, Goal, Plan, PlanMilestone
from app.db.repositories import (
    GoalRepository,
    MasteryRepository,
    MistakeRepository,
    PlanRepository,
    TopicRepository,
    UserRepository,
)
from app.db.session import make_engine
from tests.llm.sdk_transport_support import (  # noqa: F401  (frozen, read-only)
    assert_module_from_backend,
    deny_sockets,
    offline_env,
    offline_openai,
    reclaim_own_task,
    send,
    wait_for,
)

OPENAI_MODEL = "gpt-4o-mini"
NOW = datetime(2026, 5, 23, 12, 0)
TODAY_ISO = "2026-05-23"
EXPECTED_AUTH_ERROR = "AuthenticationError: Model authentication failed."

# ---------------------------------------------------------------------------
# Synthetic inputs. These are this file's declared case inputs; the wire
# expectations below are derived from them and from the frozen prompt
# templates, never from an observed request body.
# ---------------------------------------------------------------------------

SYNTHETIC_CHUNK = {
    "chunk_id": "chunk-1",
    "content": "HyDE is a hypothetical document embeddings method.",
    "page": 1,
}
SYNTHETIC_CHUNKS = [SYNTHETIC_CHUNK]

PLAN_USER_MSG = "make a plan on HyDE"
MINDMAP_USER_MSG = "make a plan on HyDE mindmap"
CHECK_IN_USER_MSG = "how is my HyDE plan going?"
FINAL_SUMMARY = "  📋 Plan ready.\n\n1. Read HyDE paper\n2. Practise reranking  \n"

MILESTONE_ONE = {"title": "Read HyDE paper", "due_at": "2026-05-25", "done": False, "topic": "HyDE"}
MILESTONE_TWO = {"title": "Practise reranking", "due_at": "2026-05-30", "done": False, "topic": "Rerank"}
MILESTONE_TITLES = ["Read HyDE paper", "Practise reranking"]
MILESTONE_PAYLOAD = [MILESTONE_ONE, MILESTONE_TWO]

SEED_MILESTONES = [
    {"title": "Read HyDE paper", "due_at": "2026-05-20", "done": False, "topic": "HyDE"},
    {"title": "Summarise RAG", "due_at": "2026-05-22", "done": True, "topic": "RAG"},
    {"title": "Practise reranking", "due_at": "2026-05-30", "done": False, "topic": "Rerank"},
]
SEED_TOTAL = 3
SEED_DONE = 1
SEED_OVERDUE_TITLES = "Read HyDE paper"
SKIP_NOTE = "Auto-adjust skipped: model output didn't match plan schema."

MINDMAP_MERMAID = "mindmap\n  root((HyDE))\n    Read HyDE paper"
MINDMAP_OUTLINE = "- HyDE\n  - Read HyDE paper"

# ---------------------------------------------------------------------------
# Prompt oracles: test-owned literals plus synthetic inputs.
#
# These four literals are copied into this file verbatim. The expectation side
# never reads the product's prompt directory, never imports a production
# template constant and never calls a production renderer, so a change in
# production wording must break these expectations instead of being followed
# by them.
# ---------------------------------------------------------------------------

GENERATE_TEMPLATE = """You are a study-plan generator for an exam coach.

Subject: {topic}
{context_section}
Student weak topics (mastery < 0.4): {weak_topics}
Exam date (optional): {exam_date}
Today: {today}

Generate 3 to 7 milestones as a JSON array. Each milestone:
- title: short, concrete, actionable (≤ 12 words)
- due_at: ISO date "YYYY-MM-DD" between today and exam_date (or null if no exam_date)
- done: false
- topic: a sub-topic string (or null)

Distribute due_at dates evenly. Cover the weak topics if any. GROUND in the source chunks if provided.

Output ONLY a JSON array, no prose, no fences.
"""
CHECK_IN_TEMPLATE = """You are a study-plan adjuster for an exam coach.

Current milestones (JSON):
{current_milestones}

Progress summary:
- Done: {done_count} / {total_count}
- Overdue (titles): {overdue_titles}
- Weak topics (mastery < 0.4): {weak_topics}
- Recent mistakes: {recent_mistake_count}

User's check-in message: {user_msg}

Adjust the milestones conservatively. You MAY:
- Mark an item done only if the user explicitly said that specific item is complete.
- Reopen an item only if the user explicitly said that specific item is not complete.
- Postpone overdue unfinished items (push due_at later).

You MUST keep id/topic_id fields unchanged when present.
You MAY NOT add, delete, reorder, rewrite titles, change mastery, or change quiz state.

Output ONLY the updated JSON array, no prose, no fences. Schema same as input.
"""
PLANNER_AGENT_SYSTEM_PROMPT_GOLDEN = """You are a study coach planner agent.

The user wants either a new study plan or a check-in on an existing plan.

Your job:
1. Read the user's message to understand what topic they care about.
2. If you don't know what plan (if any) they already have, call `get_existing_plan` first.
3. If they want a NEW plan or use explicit re-plan keywords (帮我做 / make a plan / 重做):
   - Call `retriever_search` with the topic to ground in their source materials.
   - Call `update_study_plan` with 3-7 specific, dated milestones.
4. If they want a CHECK-IN (existing plan + 进度 / check-in / 调整 / etc):
   - Call `compute_progress` to see what's done/overdue.
   - Call `update_study_plan` with the adjusted milestone list.
   - Preserve milestone id/topic_id values from the existing plan.
   - Do not add, delete, or reorder milestones during check-in.
5. If they mention mindmap / 脑图 / mind map / 思维导图: call `generate_mindmap`.
6. When done, write a short markdown summary for the user with the milestones (and mindmap if generated). Do NOT call more tools after the summary.

Today is {today}. Be concise. Call tools to act, prose to summarize."""
PLANNER_MINDMAP_PROMPT_GOLDEN = """You are a study-plan mindmap generator.

Topic: {topic}

Milestones:
{milestone_lines}

Output two parts:

1. A valid mermaid `mindmap` block wrapped in ```mermaid fences, root labelled with the topic, branches for each milestone.
2. After the fence, a markdown bullet outline of the same structure.

Do NOT add any other commentary.
"""


def expected_generate_prompt(*, topic: str, chunks: list[dict], weak_topics: list[str],
                             exam_date: datetime | None, today: date) -> str:
    """The GENERATE user message this case must put on the wire.

    Rebuilt here from this file's own literal template plus this case's own
    inputs, including the chunk block. If the node changes how it renders
    chunks, weak topics, exam date or today, this expectation stops matching.
    """
    lines = "\n".join(f"[{i + 1}] {c.get('content', '')}" for i, c in enumerate(chunks))
    context_section = f"Source chunks:\n{lines}\n" if chunks else ""
    return GENERATE_TEMPLATE.format(
        topic=topic,
        context_section=context_section,
        weak_topics=", ".join(weak_topics) if weak_topics else "(none)",
        exam_date=exam_date.isoformat() if exam_date else "(none)",
        today=today.isoformat(),
    )


def expected_check_in_prompt(*, current_milestones: str, done_count: int,
                             total_count: int, overdue_titles: str,
                             weak_topics: list[str], recent_mistake_count: int,
                             user_msg: str) -> str:
    """The CHECK-IN user message this case must put on the wire.

    Rendered from this file's own literal, never from the product's prompt file.
    """
    return CHECK_IN_TEMPLATE.format(
        current_milestones=current_milestones,
        done_count=done_count,
        total_count=total_count,
        overdue_titles=overdue_titles,
        weak_topics=", ".join(weak_topics) if weak_topics else "(none)",
        recent_mistake_count=recent_mistake_count,
        user_msg=user_msg,
    )


def expected_agent_system_prompt() -> str:
    """The agent system prompt, with this case's date bound in.

    Formatted from this file's own literal, never from ``planner_agent``.
    """
    return PLANNER_AGENT_SYSTEM_PROMPT_GOLDEN.format(today=TODAY_ISO)


def assert_plain_messages(body: dict, expected_prompt: str) -> str:
    """This file's complete plain-prompt expectation: one user message, whole.

    The whole ``messages`` list is compared with exact equality -- not a prefix,
    a substring, a normalised fragment or a heading. The expected prompt is
    returned so a caller can go on asserting against the same value.
    """
    assert body["messages"] == [{"role": "user", "content": expected_prompt}]
    return expected_prompt


def expected_mindmap_prompt(*, topic: str, milestones: list[dict]) -> str:
    """The whole internal mindmap user message, from this file's own literal.

    Built from the literal plus this case's controlled inputs, so the whole
    prompt is compared -- not a heading or the first fragment.
    """
    lines = "\n".join(f"- {m['title']}" for m in milestones) or "- (no milestones)"
    return PLANNER_MINDMAP_PROMPT_GOLDEN.format(topic=topic, milestone_lines=lines)


# ---------------------------------------------------------------------------
# Scripted responses and the wire assertions.
# ---------------------------------------------------------------------------


def request_json(request: httpx.Request) -> dict:
    return json.loads(request.content)


def error_body(message: str, *, code: int) -> dict:
    return {"error": {"message": message, "type": "invalid_request_error", "code": code}}


def assistant_tool_calls(*calls, content=None, finish_reason: str = "tool_calls",
                         prompt_tokens: int = 11, completion_tokens: int = 7) -> dict:
    """One assistant turn that requests tool calls, in the given order."""
    return {
        "id": "chatcmpl-synthetic",
        "object": "chat.completion",
        "created": 1767225600,
        "model": OPENAI_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {"id": call_id, "type": "function",
                         "function": {"name": name, "arguments": arguments}}
                        for call_id, name, arguments in calls
                    ],
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
    }


def assistant_text(text: str) -> dict:
    """One assistant turn with no tool calls: a natural stop."""
    return {
        "id": "chatcmpl-synthetic",
        "object": "chat.completion",
        "created": 1767225600,
        "model": OPENAI_MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


class ScriptedResponder:
    """Dispatch real requests by whether the adapter bound tools or not.

    A bound turn (``tools`` in the body) is an outer agent-loop iteration; a
    tools-free turn is the internal mindmap call, which uses the same base
    model without tools. Keying on that difference is what stops the internal
    mindmap turn from being mistaken for the next outer iteration.

    Script entries are either a JSON body (served 200), a ``(body, status)``
    pair, or a callable taking the request.
    """

    def __init__(self, *, bound=(), plain=(), on_request=None):
        self.bound = list(bound)
        self.plain = list(plain or [])
        self.on_request = on_request
        self.bound_calls = 0
        self.plain_calls = 0
        self.seen: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = request_json(request)
        self.seen.append(body)
        if self.on_request is not None:
            self.on_request(request, body)
        if "tools" in body:
            script, counter = self.bound, "bound_calls"
        else:
            script, counter = self.plain, "plain_calls"
        index = getattr(self, counter)
        assert index < len(script), f"the agent sent more turns than scripted ({len(script)})"
        setattr(self, counter, index + 1)
        entry = script[index]
        if callable(entry):
            return entry(request)
        if isinstance(entry, tuple):
            payload, status = entry
        else:
            payload, status = entry, 200
        return httpx.Response(status, json=payload,
                              headers={"content-type": "application/json"})


def assert_chat_request(request: httpx.Request, *, bound: bool) -> dict:
    """Method / endpoint / model / non-stream, plus tool presence."""
    assert request.method == "POST"
    assert request.url.scheme == "https"
    assert request.url.host == "api.openai.com"
    assert request.url.path == "/v1/chat/completions"
    assert "api-key" not in request.headers and "x-goog-api-key" not in request.headers
    body = request_json(request)
    assert body["model"] == OPENAI_MODEL
    assert body.get("stream") is not True, "this slice is non-stream only"
    assert ("tools" in body) is bound
    return body


# ---------------------------------------------------------------------------
# Tool schema expectations, written out from the tool signatures in
# app/agent/planner_agent.py. The wire schema for ``milestones`` is an array of
# plain objects; the inner Milestone rules are a separate business layer and are
# deliberately not invented into the JSON schema.
# ---------------------------------------------------------------------------

MILESTONE_ITEMS = {"additionalProperties": True, "type": "object"}

EXPECTED_TOOL_SCHEMAS: list[dict] = [
    {
        "name": "retriever_search",
        "description_first_line": "Search the user's uploaded PDF corpus for chunks relevant to a topic.",
        "parameters": {
            "properties": {"query": {"type": "string"}, "top_k": {"default": 5, "type": "integer"}},
            "required": ["query"],
            "type": "object",
        },
    },
    {
        "name": "get_existing_plan",
        "description_first_line": "Return the user's currently active study plan, if any.",
        "parameters": {"properties": {}, "type": "object"},
    },
    {
        "name": "update_study_plan",
        "description_first_line": "Persist a list of milestones as the user's study plan (upsert).",
        "parameters": {
            "properties": {"milestones": {"items": MILESTONE_ITEMS, "type": "array"}},
            "required": ["milestones"],
            "type": "object",
        },
    },
    {
        "name": "generate_mindmap",
        "description_first_line": "Generate a mermaid mindmap + markdown outline for a study plan.",
        "parameters": {
            "properties": {"milestones": {"items": MILESTONE_ITEMS, "type": "array"},
                           "topic": {"type": "string"}},
            "required": ["topic", "milestones"],
            "type": "object",
        },
    },
    {
        "name": "compute_progress",
        "description_first_line": "Compute deterministic progress summary for the user's active plan.",
        "parameters": {"properties": {}, "type": "object"},
    },
]


def assert_tool_set(body: dict) -> None:
    """The five planner tools, their schemas and their order, on this request."""
    tools = body["tools"]
    assert [t["type"] for t in tools] == ["function"] * 5
    assert [t["function"]["name"] for t in tools] == [s["name"] for s in EXPECTED_TOOL_SCHEMAS]
    for wire, expected in zip(tools, EXPECTED_TOOL_SCHEMAS):
        function = wire["function"]
        assert function["parameters"] == expected["parameters"], function["name"]
        description = function["description"]
        assert isinstance(description, str) and description.strip(), function["name"]
        assert description.splitlines()[0].strip() == expected["description_first_line"]


def bound_bodies(harness) -> list[dict]:
    return [assert_chat_request(r, bound=True) for r in harness.requests]


def plain_bodies(harness) -> list[dict]:
    return [assert_chat_request(r, bound=False) for r in harness.requests]


def assert_agent_request(body: dict, *, user_msg: str, rounds: list[dict]) -> None:
    """Check one bound request end to end, without any positional guessing.

    The expectation is this case's *declared* layout: the system/user prefix,
    then every assistant/tool round that must already be in the history -- in
    order, with its call id, tool name, JSON-decoded arguments and the decoded
    ToolMessage result. Nothing is skipped and nothing is compared with an
    observed value of its own; dynamic ids in results come from real rows the
    case reads out of a second session.

    ``rounds`` entries are ``{"calls": [(call_id, tool_name, args), ...],
    "results": [decoded_result, ...]}``, where one assistant turn may request
    several tools and therefore yields several ToolMessages in call order.
    """
    messages = body["messages"]
    assert messages[0] == {"role": "system", "content": expected_agent_system_prompt()}
    assert messages[1] == {"role": "user", "content": user_msg}

    # Each round contributes one assistant turn plus one ToolMessage per call.
    expected_messages = 2 + sum(1 + len(r["calls"]) for r in rounds)
    assert len(messages) == expected_messages, "the history is exactly prefix + declared pairs"
    cursor = 2
    for round_index, spec in enumerate(rounds):
        assistant = messages[cursor]
        cursor += 1
        assert assistant["role"] == "assistant", f"round {round_index}"
        # A tool-only assistant turn is natively serialised with null content.
        assert assistant["content"] is None, f"round {round_index} assistant content"
        assert [c["id"] for c in assistant["tool_calls"]] == [
            call_id for call_id, _name, _args in spec["calls"]
        ], f"round {round_index} call ids"
        for wire_call, (call_id, name, args) in zip(assistant["tool_calls"], spec["calls"]):
            assert wire_call["type"] == "function", call_id
            assert wire_call["function"]["name"] == name, call_id
            assert json.loads(wire_call["function"]["arguments"]) == args, call_id
        for (call_id, _name, _args), result in zip(spec["calls"], spec["results"]):
            tool_message = messages[cursor]
            cursor += 1
            assert tool_message["role"] == "tool", call_id
            assert tool_message["tool_call_id"] == call_id
            assert json.loads(tool_message["content"]) == result, call_id


# ---------------------------------------------------------------------------
# Shared fixtures: a real file-backed SQLite under tmp_path, plus a way to
# prove commit visibility from a second session.
# ---------------------------------------------------------------------------


@pytest.fixture
def db(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'planner_contract.db'}")
    Base.metadata.create_all(engine)
    session = Session(engine)

    def fresh() -> Session:
        """A brand-new session over the same file DB: committed rows only."""
        return Session(engine)

    session.info["fresh"] = fresh
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def seed_user(session, fingerprint: str):
    return UserRepository(session).get_or_create(fingerprint)


def seed_goal_with_topics(session, user_id: str, title: str, topic_names: list[str]):
    goal = GoalRepository(session).create(user_id=user_id, title=title)
    topics = {name: TopicRepository(session).create(goal_id=goal.id, name=name)
              for name in topic_names}
    return goal, topics


def seed_plan(session, goal_id: str, milestones: list[dict]) -> Plan:
    """Write a plan through the production upsert, so ids are real rows."""
    return PlanRepository(session).update_milestones(goal_id=goal_id, milestones=milestones)


def count_rows(session, model) -> int:
    return len(list(session.execute(select(model)).scalars()))


def plan_titles(session, goal_id: str) -> list[str]:
    plan = PlanRepository(session).get_by_goal(goal_id)
    assert plan is not None
    return [m["title"] for m in plan.milestones_json]


class SyntheticRetriever:
    """A deterministic in-test corpus. Only retrieval is synthetic."""

    def __init__(self, chunks=None):
        self.chunks = list(chunks or [])
        self.calls: list[tuple[str, int]] = []

    def search(self, query, top_k=5):
        self.calls.append((query, top_k))
        return self.chunks[:top_k]


class ModelObserver(AsyncCallbackHandler):
    """Read-only observation of the real model round trips.

    ``on_chat_model_start`` records the exact message objects the adapter was
    asked to send; ``on_llm_end`` records the SDK-parsed response message.
    Neither callback changes the result, the messages or the flow.
    """

    def __init__(self):
        self.starts: list[list] = []
        self.ends: list[object] = []

    async def on_chat_model_start(self, serialized, messages, **kwargs):
        self.starts.append(messages)

    async def on_llm_end(self, response, **kwargs):
        self.ends.append(response.generations[0][0].message)


def observed_messages(observer: ModelObserver, index: int) -> list:
    """The exact message objects the adapter was handed on round ``index``.

    LangChain hands the callback one conversation batch per run; this case
    always has exactly one.
    """
    batches = observer.starts[index]
    assert len(batches) == 1, "this case sends exactly one conversation per round"
    return batches[0]


def build_agent(session, llm, *, retriever=None, max_iter: int = 10):
    return build_planner_agent(
        llm=llm,
        plan_repo=PlanRepository(session),
        goal_repo=GoalRepository(session),
        mastery_repo=MasteryRepository(session),
        mistake_repo=MistakeRepository(session),
        retriever=retriever if retriever is not None else SyntheticRetriever(),
        now_fn=lambda: NOW,
        max_iter=max_iter,
    )


def build_node(session, llm, *, retriever=None):
    return build_planner(
        llm=llm,
        plan_repo=PlanRepository(session),
        goal_repo=GoalRepository(session),
        mastery_repo=MasteryRepository(session),
        mistake_repo=MistakeRepository(session),
        retriever=retriever if retriever is not None else SyntheticRetriever(),
        now_fn=lambda: NOW,
    )


# ===========================================================================
# P1 -- agent loop: three bound turns plus one internal mindmap turn
# ===========================================================================


async def test_P1_agent_loop_full_success_with_mindmap(offline_openai, db):
    """P1: natural stop with a committed Plan, a real mindmap and verbatim prose."""
    user = seed_user(db, "fp-d4b2-p1")
    goal, _topics = seed_goal_with_topics(db, user.id, "Exam goal", ["HyDE", "Rerank"])
    retriever = SyntheticRetriever(SYNTHETIC_CHUNKS)
    observer = ModelObserver()

    # Sampled from a *fresh* session as each of the four requests arrives, so
    # the recorded list doubles as a DB time-point trace of the run.
    plan_rows_per_request: list[int] = []
    fresh = db.info["fresh"]

    def on_request(request, body):
        probe = fresh()
        try:
            plan_rows_per_request.append(count_rows(probe, Plan))
        finally:
            probe.close()

    responder = ScriptedResponder(
        bound=[
            assistant_tool_calls(("call_rs", "retriever_search",
                                  json.dumps({"query": "HyDE", "top_k": 3}))),
            assistant_tool_calls(
                ("call_up", "update_study_plan",
                 json.dumps({"milestones": MILESTONE_PAYLOAD})),
                ("call_mm", "generate_mindmap",
                 json.dumps({"topic": "HyDE", "milestones": MILESTONE_PAYLOAD})),
            ),
            assistant_text(FINAL_SUMMARY),
        ],
        plain=[assistant_text(f"```mermaid\n{MINDMAP_MERMAID}\n```\n\n{MINDMAP_OUTLINE}")],
        on_request=on_request,
    )
    offline_openai.respond(responder)

    agent = build_agent(db, offline_openai.model(callbacks=[observer]), retriever=retriever)
    update = await agent({"messages": [HumanMessage(content=PLAN_USER_MSG)], "user_id": user.id})

    # -- four real HTTP turns: three bound, one internal mindmap -----------
    assert len(offline_openai.requests) == 4
    assert (responder.bound_calls, responder.plain_calls) == (3, 1)
    shapes = [("tools" in request_json(r)) for r in offline_openai.requests]
    assert shapes == [True, True, False, True], (
        "the tools-free turn is the internal mindmap call, not an outer round"
    )
    bodies = [assert_chat_request(r, bound=bound) for r, bound in zip(offline_openai.requests, shapes)]

    for body in (bodies[0], bodies[1], bodies[3]):
        assert_tool_set(body)

    # The internal mindmap turn: one tools-free user message, compared whole
    # against this file's own literal filled with this case's inputs.
    mindmap_prompt = expected_mindmap_prompt(topic="HyDE", milestones=MILESTONE_PAYLOAD)
    assert_plain_messages(bodies[2], mindmap_prompt)
    # Commit visibility as each request landed: before round 1, before round 2
    # (the update tool has not run yet), then after the tool committed, both
    # for the internal mindmap turn and for round 3.
    assert plan_rows_per_request == [0, 0, 1, 1], (
        "a second session saw the committed Plan before the mindmap request"
    )

    # -- history on the wire, checked against this case's declared layout ---
    # A turn's own response is only visible to the *next* request, so round 2
    # carries round 1's pair and round 3 carries both rounds' pairs.
    probe = fresh()
    try:
        saved = PlanRepository(probe).get_by_goal(goal.id)
        assert saved is not None
        plan_result = {
            "plan_id": saved.id,
            "milestones_count": len(MILESTONE_PAYLOAD),
            "updated_at": saved.updated_at.isoformat(),
        }
    finally:
        probe.close()
    mindmap_result = {"mermaid_src": MINDMAP_MERMAID, "markdown_outline": MINDMAP_OUTLINE}

    RETRIEVER_ROUND = {
        "calls": [("call_rs", "retriever_search", {"query": "HyDE", "top_k": 3})],
        "results": [SYNTHETIC_CHUNKS],
    }
    PERSIST_ROUND = {
        "calls": [
            ("call_up", "update_study_plan", {"milestones": MILESTONE_PAYLOAD}),
            ("call_mm", "generate_mindmap",
             {"topic": "HyDE", "milestones": MILESTONE_PAYLOAD}),
        ],
        "results": [plan_result, mindmap_result],
    }

    assert_agent_request(bodies[0], user_msg=PLAN_USER_MSG, rounds=[])
    assert_agent_request(bodies[1], user_msg=PLAN_USER_MSG, rounds=[RETRIEVER_ROUND])
    assert_agent_request(bodies[3], user_msg=PLAN_USER_MSG,
                         rounds=[RETRIEVER_ROUND, PERSIST_ROUND])

    # -- the SDK's own AIMessage objects are what went into the next round --
    assert len(observer.starts) == 4 and len(observer.ends) == 4
    assert observed_messages(observer, 1)[2] is observer.ends[0], (
        "the SDK's own AIMessage object is what entered round 2"
    )
    assert observed_messages(observer, 2)[0].content == mindmap_prompt, (
        "the internal mindmap turn sent exactly the prompt the wire was checked against"
    )
    assert observed_messages(observer, 3)[4] is observer.ends[1], (
        "the SDK's own AIMessage object is what entered round 3"
    )
    assert observed_messages(observer, 3)[3].tool_call_id == "call_rs", (
        "the retriever result is still in history on the final round"
    )

    # -- natural stop, real ids, real committed rows -----------------------
    assert "degraded" not in update
    assert update["plan_action"] == "generate"
    assert update["messages"][0].content == FINAL_SUMMARY, (
        "the final summary keeps the model's original whitespace and body"
    )
    trace = update["agent_trace"]
    assert trace["exit_reason"] == "natural_stop"
    assert trace["total_iterations"] == 3
    assert trace["total_tool_calls"] == 3
    assert trace["tool_call_breakdown"] == {
        "retriever_search": 1, "update_study_plan": 1, "generate_mindmap": 1,
    }
    assert trace["tool_errors"] == 0
    assert retriever.calls == [("HyDE", 3)]
    assert update["last_context"] == "[1] " + SYNTHETIC_CHUNK["content"]

    probe = fresh()
    try:
        plan = probe.get(Plan, update["active_plan_id"])
        assert plan is not None and plan.goal_id == goal.id
        rows = sorted(probe.execute(select(PlanMilestone)).scalars(), key=lambda r: r.sort_order)
        assert [r.title for r in rows] == MILESTONE_TITLES
    finally:
        probe.close()


# ===========================================================================
# P2 -- business-invalid milestones, then self-correction
# ===========================================================================


async def test_P2_agent_tool_error_then_self_correction(offline_openai, db):
    """P2: the first update is business-invalid; only the retry creates a Plan."""
    user = seed_user(db, "fp-d4b2-p2")
    goal, _topics = seed_goal_with_topics(db, user.id, "Exam goal", ["HyDE"])

    bad_args = {"milestones": [{"due_at": "2026-05-25", "done": False, "topic": "HyDE"}]}
    bound = [
        assistant_tool_calls(("call_bad", "update_study_plan", json.dumps(bad_args))),
        assistant_tool_calls(("call_fix", "update_study_plan",
                              json.dumps({"milestones": [MILESTONE_ONE]}))),
        assistant_text("  Done.\n"),
    ]

    plan_rows_at_retry: list[int] = []
    fresh = db.info["fresh"]

    def on_request(request, body):
        if "tools" not in body:
            return
        if sum(1 for seen in responder.seen if "tools" in seen) == 2:
            probe = fresh()
            try:
                plan_rows_at_retry.append(count_rows(probe, Plan))
            finally:
                probe.close()

    responder = ScriptedResponder(bound=bound, on_request=on_request)
    offline_openai.respond(responder)

    agent = build_agent(db, offline_openai.model())
    update = await agent({"messages": [HumanMessage(content=PLAN_USER_MSG)], "user_id": user.id})

    assert len(offline_openai.requests) == 3
    bodies = bound_bodies(offline_openai)
    for body in bodies:
        assert_tool_set(body)

    probe = fresh()
    try:
        saved = PlanRepository(probe).get_by_goal(goal.id)
        assert saved is not None
        fix_result = {
            "plan_id": saved.id,
            "milestones_count": 1,
            "updated_at": saved.updated_at.isoformat(),
        }
    finally:
        probe.close()

    BAD_ROUND = {
        "calls": [("call_bad", "update_study_plan", bad_args)],
        "results": [{"error": "invalid milestone at title: Field required"}],
    }
    FIX_ROUND = {
        "calls": [("call_fix", "update_study_plan", {"milestones": [MILESTONE_ONE]})],
        "results": [fix_result],
    }
    assert_agent_request(bodies[0], user_msg=PLAN_USER_MSG, rounds=[])
    assert_agent_request(bodies[1], user_msg=PLAN_USER_MSG, rounds=[BAD_ROUND])
    assert_agent_request(bodies[2], user_msg=PLAN_USER_MSG, rounds=[BAD_ROUND, FIX_ROUND])

    # Top-level tool args expose only ``milestones``; there is no top-level
    # title to get wrong, so that error case is deliberately not written.
    assert set(bad_args) == {"milestones"}
    assert "title" not in EXPECTED_TOOL_SCHEMAS[2]["parameters"]["properties"]

    # No Plan existed when the retry was requested, and the seeded Goal stood.
    assert plan_rows_at_retry == [0]
    probe = fresh()
    try:
        goals = list(probe.execute(select(Goal)).scalars())
        assert [g.id for g in goals] == [goal.id]
    finally:
        probe.close()

    assert update["plan_action"] == "generate"
    assert update["agent_trace"]["exit_reason"] == "natural_stop"
    assert update["agent_trace"]["total_iterations"] == 3
    assert update["agent_trace"]["tool_errors"] == 1
    assert update["agent_trace"]["tool_call_breakdown"] == {"update_study_plan": 2}

    probe = fresh()
    try:
        assert plan_titles(probe, goal.id) == [MILESTONE_ONE["title"]]
    finally:
        probe.close()


# ===========================================================================
# P3 -- the Plan is already committed when the next model turn fails
# ===========================================================================


async def test_P3_agent_401_after_commit_degrades_but_keeps_plan(offline_openai, db):
    """P3: a 401 on the following turn degrades; the committed Plan survives."""
    user = seed_user(db, "fp-d4b2-p3")
    goal, _topics = seed_goal_with_topics(db, user.id, "Exam goal", ["HyDE"])
    provider_marker = "synthetic-provider-marker-do-not-leak"

    bound = [
        assistant_tool_calls(("call_up", "update_study_plan",
                              json.dumps({"milestones": [MILESTONE_ONE]}))),
        (error_body(provider_marker, code=401), 401),
    ]
    plan_rows_at_failure: list[int] = []
    fresh = db.info["fresh"]

    def on_request(request, body):
        if "tools" not in body:
            return
        if sum(1 for seen in responder.seen if "tools" in seen) == 2:
            probe = fresh()
            try:
                plan_rows_at_failure.append(count_rows(probe, Plan))
            finally:
                probe.close()

    responder = ScriptedResponder(bound=bound, on_request=on_request)
    offline_openai.respond(responder)

    agent = build_agent(db, offline_openai.model())
    update = await agent({"messages": [HumanMessage(content=PLAN_USER_MSG)], "user_id": user.id})

    assert len(offline_openai.requests) == 2, "401 is not retried by the SDK"
    bodies = bound_bodies(offline_openai)
    for body in bodies:
        assert_tool_set(body)
    assert plan_rows_at_failure == [1], "the Plan was committed before the 401"

    # The failing request still carried the whole prior turn, including the
    # saved plan id produced by the update tool.
    probe = fresh()
    try:
        saved = PlanRepository(probe).get_by_goal(goal.id)
        assert saved is not None
        update_result = {
            "plan_id": saved.id,
            "milestones_count": 1,
            "updated_at": saved.updated_at.isoformat(),
        }
    finally:
        probe.close()
    UPDATE_ROUND = {
        "calls": [("call_up", "update_study_plan", {"milestones": [MILESTONE_ONE]})],
        "results": [update_result],
    }
    assert_agent_request(bodies[0], user_msg=PLAN_USER_MSG, rounds=[])
    assert_agent_request(bodies[1], user_msg=PLAN_USER_MSG, rounds=[UPDATE_ROUND])

    assert update["degraded"] is True
    assert update["messages"][0].content == _LLM_FAILED_MSG
    trace = update["agent_trace"]
    assert trace["exit_reason"] == "llm_call_failed"
    assert trace["llm_error"] == EXPECTED_AUTH_ERROR
    assert trace["total_iterations"] == 1, "the failing turn never became an iteration"
    assert trace["total_tool_calls"] == 1
    assert trace["tool_call_breakdown"] == {"update_study_plan": 1}
    assert "active_plan_id" not in update, "no confirmed persist is claimed"
    assert update["plan_action"] == "generate"

    assert provider_marker not in update["messages"][0].content
    assert provider_marker not in trace["llm_error"]
    assert provider_marker not in json.dumps(update, default=str)

    probe = fresh()
    try:
        assert plan_titles(probe, goal.id) == [MILESTONE_ONE["title"]]
    finally:
        probe.close()


# ===========================================================================
# P4 -- budget exhaustion on the dedicated max_iter=2 branch
# ===========================================================================


async def test_P4_agent_budget_exhausted_without_plan(offline_openai, db):
    """P4: two logical rounds, two tool calls, nothing persisted."""
    user = seed_user(db, "fp-d4b2-p4")

    assert inspect.signature(build_planner_agent).parameters["max_iter"].default == 10, (
        "this case drives the max_iter parameter; the product default must stay 10"
    )

    responder = ScriptedResponder(bound=[
        assistant_tool_calls(("c1", "retriever_search", json.dumps({"query": "a"}))),
        assistant_tool_calls(("c2", "retriever_search", json.dumps({"query": "b"}))),
    ])
    offline_openai.respond(responder)

    agent = build_agent(db, offline_openai.model(), max_iter=2)
    update = await agent({"messages": [HumanMessage(content=PLAN_USER_MSG)], "user_id": user.id})

    assert len(offline_openai.requests) == 2, "one HTTP per logical round"
    bodies = bound_bodies(offline_openai)
    for body in bodies:
        assert_tool_set(body)

    # Round 2 carried round 1's complete pair; the synthetic retriever has no
    # chunks here, so the tool result is the literal empty JSON list.
    FIRST_ROUND = {
        "calls": [("c1", "retriever_search", {"query": "a"})],
        "results": [[]],
    }
    SECOND_ROUND = {
        "calls": [("c2", "retriever_search", {"query": "b"})],
        "results": [[]],
    }
    assert_agent_request(bodies[0], user_msg=PLAN_USER_MSG, rounds=[])
    assert_agent_request(bodies[1], user_msg=PLAN_USER_MSG, rounds=[FIRST_ROUND])
    assert SECOND_ROUND["calls"][0][0] not in bodies[1]["messages"][2]["tool_calls"][0]["id"], (
        "round 2's own call is not yet in its own request"
    )

    trace = update["agent_trace"]
    assert trace["exit_reason"] == "budget_exhausted"
    assert trace["total_iterations"] == 2
    assert trace["total_tool_calls"] == 2
    assert trace["tool_call_breakdown"] == {"retriever_search": 2}
    assert update["degraded"] is True
    assert update["messages"][0].content == _BUDGET_EXHAUSTED_MSG
    assert "active_plan_id" not in update

    probe = db.info["fresh"]()
    try:
        assert count_rows(probe, Plan) == 0
        assert count_rows(probe, Goal) == 0, "no goal is auto-created"
    finally:
        probe.close()


# ===========================================================================
# P5 -- first body gate, explicit cancellation before any tool runs
# ===========================================================================


async def test_P5_agent_cancelled_on_first_body(offline_openai, db):
    """P5: cancel the node task while the very first body is still open."""
    user = seed_user(db, "fp-d4b2-p5")

    gate = asyncio.Event()  # never set: only cancellation can end this body
    body = offline_openai.body([wait_for(gate), send(b"never-reached")])
    offline_openai.respond(
        lambda request: httpx.Response(
            200, headers={"content-type": "application/json"}, stream=body
        )
    )
    agent = build_agent(db, offline_openai.model())

    task = asyncio.create_task(
        agent({"messages": [HumanMessage(content=PLAN_USER_MSG)], "user_id": user.id})
    )
    try:
        await asyncio.wait_for(body.started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as excinfo:
            await asyncio.wait_for(asyncio.shield(task), timeout=5)
        assert type(excinfo.value) is asyncio.CancelledError
        assert task.cancelled()

        assert len(offline_openai.requests) == 1
        body_request = bound_bodies(offline_openai)[0]
        assert_tool_set(body_request)
        assert_agent_request(body_request, user_msg=PLAN_USER_MSG, rounds=[])
        assert body.bytes_sent == 0, "the body delivered nothing, so no tool had run"
        # Observed before the finally and before the fixture closes the SDK
        # clients. Recorded, never asserted -- it proves a local aclose only.
        print(f"D4b P5 fixture_pre_close_count={body.close_count}")
    finally:
        await reclaim_own_task(task)

    probe = db.info["fresh"]()
    try:
        assert count_rows(probe, Goal) == 0
        assert count_rows(probe, Plan) == 0
    finally:
        probe.close()


# ===========================================================================
# P6 -- deterministic GENERATE
# ===========================================================================


async def test_P6_deterministic_generate_commits_goal_then_plan(offline_openai, db):
    """P6: one tools-free turn; the Goal exists before it, the Plan after it."""
    user = seed_user(db, "fp-d4b2-p6")
    retriever = SyntheticRetriever(SYNTHETIC_CHUNKS)
    fresh = db.info["fresh"]
    goals_at_request: list[int] = []
    plans_at_request: list[int] = []

    def on_request(request, body):
        probe = fresh()
        try:
            goals_at_request.append(count_rows(probe, Goal))
            plans_at_request.append(count_rows(probe, Plan))
        finally:
            probe.close()

    model_text = "```json\n" + json.dumps(MILESTONE_PAYLOAD, ensure_ascii=False) + "\n```"
    responder = ScriptedResponder(plain=[assistant_text(model_text)], on_request=on_request)
    offline_openai.respond(responder)

    node = build_node(db, offline_openai.model(), retriever=retriever)
    update = await node({"messages": [HumanMessage(content=PLAN_USER_MSG)], "user_id": user.id})

    assert len(offline_openai.requests) == 1
    body = plain_bodies(offline_openai)[0]
    expected_prompt = expected_generate_prompt(
        topic="HyDE", chunks=SYNTHETIC_CHUNKS, weak_topics=[], exam_date=None, today=NOW.date(),
    )
    assert_plain_messages(body, expected_prompt)
    assert "Subject: HyDE" in expected_prompt
    assert f"Source chunks:\n[1] {SYNTHETIC_CHUNK['content']}\n" in expected_prompt
    assert "Student weak topics (mastery < 0.4): (none)" in expected_prompt
    assert "Exam date (optional): (none)" in expected_prompt
    assert f"Today: {TODAY_ISO}" in expected_prompt

    assert goals_at_request == [1], "the Goal is committed before the model call"
    assert plans_at_request == [0], "no Plan existed at that point"

    text = update["messages"][0].content
    assert text.startswith("📋 **Study Plan**")
    assert "1. [ ] Read HyDE paper — due 2026-05-25 *(HyDE)*" in text
    assert "2. [ ] Practise reranking — due 2026-05-30 *(Rerank)*" in text
    assert "```mermaid" not in text, "no mindmap keyword was used"
    assert update["plan_action"] == "generate"
    assert update["last_context"] == "[1] " + SYNTHETIC_CHUNK["content"]
    assert retriever.calls == [("HyDE", 5)]

    probe = fresh()
    try:
        goal = probe.execute(select(Goal)).scalar_one()
        assert goal.title == "Default Study Goal"
        assert goal.user_id == user.id
        plan = probe.get(Plan, update["active_plan_id"])
        assert plan is not None and plan.goal_id == goal.id
        rows = sorted(probe.execute(select(PlanMilestone)).scalars(), key=lambda r: r.sort_order)
        assert [r.title for r in rows] == MILESTONE_TITLES
        assert [r.due_at.date().isoformat() for r in rows] == ["2026-05-25", "2026-05-30"]
        assert all(r.done is False for r in rows)
    finally:
        probe.close()


# ===========================================================================
# P7 / P8 -- deterministic CHECK-IN
# ===========================================================================


def _seed_check_in_plan(db, fingerprint: str):
    user = seed_user(db, fingerprint)
    goal, topics = seed_goal_with_topics(db, user.id, "Exam goal", ["HyDE", "RAG", "Rerank"])
    plan = seed_plan(db, goal.id, SEED_MILESTONES)
    return user, goal, topics, plan


def _check_in_expectation(plan: Plan) -> str:
    """The CHECK-IN prompt for the seeded plan, built from the seed itself.

    ``current_milestones`` is the seeded row's stored JSON, i.e. this case's
    declared input; the progress figures are recomputed here from the seed
    rather than read back from the request.
    """
    return expected_check_in_prompt(
        current_milestones=json.dumps(plan.milestones_json, ensure_ascii=False),
        done_count=SEED_DONE, total_count=SEED_TOTAL,
        overdue_titles=SEED_OVERDUE_TITLES, weak_topics=[], recent_mistake_count=0,
        user_msg=CHECK_IN_USER_MSG,
    )


async def test_P7_deterministic_check_in_updates_same_plan(offline_openai, db):
    """P7: same Plan identity, ids and topic ids kept, only allowed fields move."""
    user, goal, topics, plan = _seed_check_in_plan(db, "fp-d4b2-p7")
    plan_id_before = plan.id
    expected_prompt = _check_in_expectation(plan)

    adjusted = []
    for row in plan.milestones_json:
        if row["title"] == "Read HyDE paper":
            adjusted.append({**row, "done": True, "due_at": "2026-05-21"})
        else:
            adjusted.append(dict(row))

    offline_openai.respond(ScriptedResponder(plain=[assistant_text(json.dumps(adjusted))]))

    node = build_node(db, offline_openai.model())
    update = await node({
        "messages": [HumanMessage(content=CHECK_IN_USER_MSG)],
        "user_id": user.id,
        "active_plan_id": plan.id,
    })

    assert len(offline_openai.requests) == 1
    body = plain_bodies(offline_openai)[0]
    assert_plain_messages(body, expected_prompt)
    assert f"- Done: {SEED_DONE} / {SEED_TOTAL}" in expected_prompt
    assert f"- Overdue (titles): {SEED_OVERDUE_TITLES}" in expected_prompt
    assert "- Weak topics (mastery < 0.4): (none)" in expected_prompt
    assert "- Recent mistakes: 0" in expected_prompt

    assert update["plan_action"] == "check_in"
    assert update["active_plan_id"] == plan_id_before, "same Plan identity"
    text = update["messages"][0].content
    assert text.startswith("📊 **Progress Check-in**")
    assert "1. [x] Read HyDE paper — due 2026-05-21 *(HyDE)*" in text
    assert "Auto-adjust skipped" not in text
    assert "- Done: 2 / 3" in text, "the displayed counts match the displayed rows"

    probe = db.info["fresh"]()
    try:
        assert count_rows(probe, Plan) == 1, "no second Plan was written"
        saved = probe.get(Plan, plan_id_before)
        assert saved is not None and saved.goal_id == goal.id
        assert [m["title"] for m in saved.milestones_json] == [m["title"] for m in SEED_MILESTONES]
        by_title = {m["title"]: m for m in saved.milestones_json}
        assert all(m["id"] for m in saved.milestones_json)
        assert by_title["Read HyDE paper"]["topic_id"] == topics["HyDE"].id
        assert by_title["Summarise RAG"]["topic_id"] == topics["RAG"].id
        assert by_title["Practise reranking"]["topic_id"] == topics["Rerank"].id
        assert by_title["Read HyDE paper"]["done"] is True
        assert by_title["Read HyDE paper"]["due_at"] == "2026-05-21"
        assert by_title["Practise reranking"]["done"] is False
    finally:
        probe.close()


@pytest.mark.parametrize("trigger", ["http_503_exhausted", "legal_body_without_milestones"])
async def test_P8_deterministic_check_in_soft_degradation(offline_openai, db, trigger):
    """P8: on either trigger the existing Plan is left field-for-field alone."""
    user, goal, _topics, plan = _seed_check_in_plan(db, f"fp-d4b2-p8-{trigger}")
    plan_id_before = plan.id
    before_milestones = json.dumps(plan.milestones_json, ensure_ascii=False, sort_keys=True)
    before_updated_at = plan.updated_at.isoformat()
    expected_prompt = _check_in_expectation(plan)

    if trigger == "http_503_exhausted":
        responder = ScriptedResponder(plain=[(error_body("synthetic", code=503), 503)] * 3)
        expected_requests = 3
    else:
        responder = ScriptedResponder(plain=[assistant_text("Looking good — on track.")])
        expected_requests = 1
    offline_openai.respond(responder)

    node = build_node(db, offline_openai.model())
    update = await node({
        "messages": [HumanMessage(content=CHECK_IN_USER_MSG)],
        "user_id": user.id,
        "active_plan_id": plan.id,
    })

    assert len(offline_openai.requests) == expected_requests
    bodies = plain_bodies(offline_openai)
    for body in bodies:
        assert body["messages"] == [{"role": "user", "content": expected_prompt}], (
            "every attempt carried the same CHECK-IN prompt"
        )
    # Every retry of this one logical invocation re-sent the identical parsed
    # JSON body, not merely matching sub-fields. Headers are not compared --
    # the SDK is free to add its own per-request headers.
    assert len(bodies) == expected_requests
    assert all(body == bodies[0] for body in bodies), (
        f"all {len(bodies)} attempts must carry the same request body"
    )

    assert update["plan_action"] == "check_in"
    assert update["active_plan_id"] == plan_id_before
    text = update["messages"][0].content
    assert text.startswith("📊 **Progress Check-in**")
    assert SKIP_NOTE in text
    assert "📋 **Study Plan**" not in text, "degradation is not a fresh generation"

    probe = db.info["fresh"]()
    try:
        assert count_rows(probe, Plan) == 1
        saved = probe.get(Plan, plan_id_before)
        assert saved is not None and saved.goal_id == goal.id
        after = json.dumps(saved.milestones_json, ensure_ascii=False, sort_keys=True)
        assert after == before_milestones
        assert saved.updated_at.isoformat() == before_updated_at, "the row was not rewritten"
    finally:
        probe.close()


# ===========================================================================
# P9 -- deterministic GENERATE whose internal mindmap turn fails
# ===========================================================================


async def test_P9_deterministic_mindmap_failure_keeps_plan(offline_openai, db):
    """P9: the mindmap turn fails, the Plan stands, the outline comes from rows."""
    user = seed_user(db, "fp-d4b2-p9")
    retriever = SyntheticRetriever(SYNTHETIC_CHUNKS)
    fresh = db.info["fresh"]
    plan_rows_at_mindmap: list[int] = []

    def on_request(request, body):
        probe = fresh()
        try:
            plan_rows_at_mindmap.append(count_rows(probe, Plan))
        finally:
            probe.close()

    responder = ScriptedResponder(
        plain=[
            assistant_text(json.dumps(MILESTONE_PAYLOAD, ensure_ascii=False)),
            (error_body("synthetic", code=401), 401),
        ],
        on_request=on_request,
    )
    offline_openai.respond(responder)

    node = build_node(db, offline_openai.model(), retriever=retriever)
    update = await node({"messages": [HumanMessage(content=MINDMAP_USER_MSG)], "user_id": user.id})

    assert len(offline_openai.requests) == 2, (
        "the mindmap keyword in the user message is what adds the second turn"
    )
    shapes = ["tools" in request_json(r) for r in offline_openai.requests]
    assert shapes == [False, False], "the deterministic node never binds tools"
    bodies = plain_bodies(offline_openai)
    assert bodies[0]["messages"] == [{
        "role": "user",
        "content": expected_generate_prompt(
            topic="HyDE", chunks=SYNTHETIC_CHUNKS, weak_topics=[], exam_date=None,
            today=NOW.date(),
        ),
    }]
    # The mindmap turn is compared whole against the frozen tools/plan.py
    # template, with the topic and milestones the generate step committed.
    assert bodies[1]["messages"] == [{
        "role": "user",
        "content": expected_mindmap_prompt(topic="HyDE", milestones=MILESTONE_PAYLOAD),
    }]
    assert plan_rows_at_mindmap == [0, 1], (
        "no Plan at the GENERATE turn, and a committed Plan at the mindmap turn"
    )

    text = update["messages"][0].content
    assert text.startswith("📋 **Study Plan**")
    assert "```mermaid" not in text, "a failed mindmap yields no mermaid block"
    assert "**Outline**" in text
    fallback = "- HyDE\n" + "\n".join(f"  - {t}" for t in MILESTONE_TITLES)
    assert fallback in text, "the outline is derived from the committed milestones"

    probe = fresh()
    try:
        goal = probe.execute(select(Goal)).scalar_one()
        assert plan_titles(probe, goal.id) == MILESTONE_TITLES, "the Plan was not rolled back"
    finally:
        probe.close()


# ===========================================================================
# P10 -- deterministic GENERATE cancelled while the body is open
# ===========================================================================


async def test_P10_deterministic_generate_cancelled(offline_openai, db):
    """P10: the Goal committed before the call is kept; no Plan is written."""
    user = seed_user(db, "fp-d4b2-p10")

    gate = asyncio.Event()
    body = offline_openai.body([wait_for(gate), send(b"never-reached")])
    offline_openai.respond(
        lambda request: httpx.Response(
            200, headers={"content-type": "application/json"}, stream=body
        )
    )
    node = build_node(db, offline_openai.model(), retriever=SyntheticRetriever(SYNTHETIC_CHUNKS))

    task = asyncio.create_task(
        node({"messages": [HumanMessage(content=PLAN_USER_MSG)], "user_id": user.id})
    )
    try:
        await asyncio.wait_for(body.started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as excinfo:
            await asyncio.wait_for(asyncio.shield(task), timeout=5)
        assert type(excinfo.value) is asyncio.CancelledError
        assert task.cancelled()

        assert len(offline_openai.requests) == 1
        body_request = plain_bodies(offline_openai)[0]
        # Not merely "no tools": the whole request is compared against the
        # frozen GENERATE template filled with this case's inputs.
        assert body_request["messages"] == [{
            "role": "user",
            "content": expected_generate_prompt(
                topic="HyDE", chunks=SYNTHETIC_CHUNKS, weak_topics=[],
                exam_date=None, today=NOW.date(),
            ),
        }]
        print(f"D4b P10 fixture_pre_close_count={body.close_count}")
    finally:
        await reclaim_own_task(task)

    probe = db.info["fresh"]()
    try:
        goals = list(probe.execute(select(Goal)).scalars())
        assert len(goals) == 1, "the Goal created before the model call is kept"
        assert goals[0].user_id == user.id
        assert count_rows(probe, Plan) == 0
    finally:
        probe.close()


# ===========================================================================
# Self-checks
# ===========================================================================


def test_plan_sources_are_exact_for_this_checkout():
    """Exact resolved identity for every module this slice exercises."""
    import app.agent.agent_trace as agent_trace_module
    import app.agent.planner as planner_module
    import app.agent.planner_agent as planner_agent_module
    import app.agent.tools.plan as plan_tools_module
    import app.agent.tools.schemas as schemas_module
    import langchain_openai.chat_models.base as openai_adapter
    import openai
    import openai._base_client as openai_base_client

    assert_module_from_backend(planner_agent_module, "app/agent/planner_agent.py")
    assert_module_from_backend(planner_module, "app/agent/planner.py")
    assert_module_from_backend(plan_tools_module, "app/agent/tools/plan.py")
    assert_module_from_backend(schemas_module, "app/agent/tools/schemas.py")
    assert_module_from_backend(agent_trace_module, "app/agent/agent_trace.py")
    assert_module_from_backend(sys.modules[__name__], "tests/agent/test_sdk_planner_contract.py")

    purelib = Path(sysconfig.get_path("purelib")).resolve()
    expected_purelib = Path(
        sys.prefix, "lib",
        f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages",
    ).resolve()
    assert purelib == expected_purelib, "the install root is this interpreter's site-packages"
    assert Path(openai_adapter.__file__).resolve() == (
        purelib / "langchain_openai" / "chat_models" / "base.py"
    )
    assert Path(openai.__file__).resolve() == purelib / "openai" / "__init__.py"
    assert Path(openai_base_client.__file__).resolve() == (
        purelib / "openai" / "_base_client.py"
    )


def test_wire_schema_is_not_the_milestone_business_schema():
    """The wire carries plain objects; title rules live in the schema model."""
    assert Milestone.model_validate(MILESTONE_ONE).title == "Read HyDE paper"
    update_schema = EXPECTED_TOOL_SCHEMAS[2]["parameters"]
    assert update_schema["properties"]["milestones"] == {"items": MILESTONE_ITEMS, "type": "array"}
    assert update_schema["required"] == ["milestones"]
    # Narrow failure: the rejection must be pydantic's, and it must be about
    # the ``title`` field specifically, not some unrelated error.
    with pytest.raises(ValidationError) as excinfo:
        Milestone.model_validate({"due_at": "2026-05-25", "done": False, "topic": "HyDE"})
    assert type(excinfo.value) is ValidationError
    locs = [error["loc"] for error in excinfo.value.errors()]
    assert ("title",) in locs, locs
    assert all(loc and loc[0] == "title" for loc in locs), locs
    assert AgentTrace(t_start=0.0).serialize()["exit_reason"] == "in_flight"


# ===========================================================================
# Prompt-oracle mechanism checks (P2).
#
# IMPORTANT: these four cases are NOT SDK, provider or full agent-loop
# evidence. They are pure test logic over synthetic parsed bodies: no node is
# run, no model is called, no production file is edited and no factory is
# patched. The real SDK business cases earlier in this file remain the healthy
# execution path and are not replaced or weakened by anything here.
#
# Each case proves exactly two things, in this order:
#   1. the test-owned literal expectation ACCEPTS the healthy text the product
#      really renders today. The healthy side is read from the frozen
#      production source named by the task's mutation matrix, right here and
#      only here -- never from the golden and never from an observed request.
#   2. the same complete comparator REJECTS an in-memory mutant that differs
#      only at that one instruction/context marker, with the roles, the model
#      and every other request field left untouched.
#
# Ten green meta cases are not proof that the literals are independent and are
# not offered as such: independence is a separate claim about the literals
# themselves, checked by inspection and by the real SDK cases above.
# ===========================================================================

_ORACLE_MUTANT = " SYNTHETIC-ORACLE-MUTANT"

_PLANNER_ORACLE_CASES = [
    ("P-GENERATE", "You are a study-plan generator for an exam coach."),
    ("P-CHECK-IN", "You are a study-plan adjuster for an exam coach."),
    ("P-AGENT", "You are a study coach planner agent."),
    ("P-MINDMAP", "You are a study-plan mindmap generator."),
]


@pytest.mark.parametrize("case_id,marker", _PLANNER_ORACLE_CASES)
def test_oracle_planner_prompt_accepts_healthy_and_rejects_mutant(case_id, marker):
    """Healthy accepted, in-memory mutant rejected, by this file's own literals."""
    prompts_dir = Path(__file__).resolve().parents[2] / "app" / "agent" / "prompts"

    if case_id == "P-GENERATE":
        healthy = (prompts_dir / "planner_generate.txt").read_text(encoding="utf-8").format(
            topic="HyDE",
            context_section=(
                "Source chunks:\n"
                + "\n".join(
                    f"[{i + 1}] {c.get('content', '')}" for i, c in enumerate(SYNTHETIC_CHUNKS)
                )
                + "\n"
            ),
            weak_topics="(none)",
            exam_date="(none)",
            today=TODAY_ISO,
        )
        expected = expected_generate_prompt(
            topic="HyDE", chunks=SYNTHETIC_CHUNKS, weak_topics=[],
            exam_date=None, today=NOW.date(),
        )
        kind = "plain"
    elif case_id == "P-CHECK-IN":
        seeded = json.dumps(SEED_MILESTONES, ensure_ascii=False)
        healthy = (prompts_dir / "planner_check_in.txt").read_text(encoding="utf-8").format(
            current_milestones=seeded,
            done_count=SEED_DONE,
            total_count=SEED_TOTAL,
            overdue_titles=SEED_OVERDUE_TITLES,
            weak_topics="(none)",
            recent_mistake_count=0,
            user_msg=CHECK_IN_USER_MSG,
        )
        expected = expected_check_in_prompt(
            current_milestones=seeded, done_count=SEED_DONE, total_count=SEED_TOTAL,
            overdue_titles=SEED_OVERDUE_TITLES, weak_topics=[], recent_mistake_count=0,
            user_msg=CHECK_IN_USER_MSG,
        )
        kind = "plain"
    elif case_id == "P-AGENT":
        # Imported here and nowhere else: the healthy candidate only.
        from app.agent.planner_agent import _AGENT_SYSTEM_PROMPT

        healthy = _AGENT_SYSTEM_PROMPT.format(today=TODAY_ISO)
        expected = expected_agent_system_prompt()
        kind = "agent"
    else:  # P-MINDMAP
        from app.agent.tools.plan import _MINDMAP_PROMPT

        healthy = _MINDMAP_PROMPT.format(
            topic="HyDE",
            milestone_lines="\n".join(f"- {m['title']}" for m in MILESTONE_PAYLOAD),
        )
        expected = expected_mindmap_prompt(topic="HyDE", milestones=MILESTONE_PAYLOAD)
        kind = "plain"

    def body_for(text):
        if kind == "agent":
            return {"messages": [
                {"role": "system", "content": text},
                {"role": "user", "content": PLAN_USER_MSG},
            ]}
        return {"messages": [{"role": "user", "content": text}]}

    def compare(text):
        if kind == "agent":
            assert_agent_request(body_for(text), user_msg=PLAN_USER_MSG, rounds=[])
        else:
            assert_plain_messages(body_for(text), expected)

    # 1. The healthy production text passes this file's own complete comparator.
    compare(healthy)

    # 2. Both shape checks happen before the expected failure, never inside it.
    assert healthy.count(marker) == 1, f"{case_id}: marker must be unique in healthy"
    mutant = healthy.replace(marker, marker + _ORACLE_MUTANT, 1)
    assert mutant != healthy, f"{case_id}: the mutant must differ from healthy"

    # 3. Only the comparator call is inside the expected rejection.
    with pytest.raises(AssertionError):
        compare(mutant)
