"""D4b-3 Quiz real-SDK consumer contract — deterministic node + agent loop.

Scope (D4b-3-QUIZ.md): OpenAI ``gpt-4o-mini`` Chat Completions only, through the
product's own ``get_chat_model`` -> ``ChatOpenAI`` -> OpenAI SDK ->
``httpx.MockTransport`` chain. The consumer under test is always the real
``build_quiz_master`` / ``build_quiz_master_agent`` node with the real
repositories; only the provider socket, the retrieval corpus and the model's
scripted answers are synthetic.

No Graph, no SSE, no Grade path, no real model and no real database: the DB is
a throw-away file-backed SQLite under ``tmp_path``.

Wire tool schemas come from the tool signatures and do **not** encode "exactly
four options" or "answer is A/B/C/D" — those are the separate
``QuizQuestionPersist`` business layer, enforced before any repository write.
This file keeps the two apart instead of inventing SDK-level strictness.

Case map: Q1-Q3 deterministic GENERATE, Q4-Q10 agent loop.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import sys
import sysconfig
from datetime import datetime
from pathlib import Path

import httpx
import pytest
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import HumanMessage
from sqlalchemy import select
from sqlalchemy.orm import Session
from pydantic import ValidationError

from app.agent.quiz_master import build_quiz_master
from app.agent.quiz_master_agent import (
    _BUDGET_EXHAUSTED_MSG,
    _LLM_FAILED_MSG,
    _QUIZ_PERSIST_FAILED_MSG,
    build_quiz_master_agent,
)
from app.agent.tools.schemas import QuizQuestionPersist
from app.db.models import Base, Goal, Question, Topic
from app.db.repositories import (
    GoalRepository,
    MasteryRepository,
    MistakeRepository,
    QuestionRepository,
    TopicRepository,
    UserRepository,
)
from app.db.session import make_engine
from app.llm.content import EMPTY_CONTENT_MESSAGE
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
EXPECTED_GENERIC_LLM_ERROR = "LLMError: Model request failed."
QUIZ_USER_MSG = "quiz me on HyDE"

SYNTHETIC_CHUNK = {
    "chunk_id": "chunk-1",
    "content": "HyDE rewrites a query into a hypothetical answer before embedding.",
    "page": 1,
}
SYNTHETIC_CHUNKS = [SYNTHETIC_CHUNK]

QUESTION_PROMPT = "What does HyDE generate before embedding?"
QUESTION_OPTIONS = ["A) A hypothetical answer", "B) A summary", "C) A keyword list", "D) A title"]
QUESTION_ANSWER = "A"
QUESTION_EXPLANATION = "HyDE expands the query into a pseudo-document before embedding."
QUESTION_TOPIC = "HyDE"

PERSIST_ARGS = {
    "topic": QUESTION_TOPIC,
    "prompt": QUESTION_PROMPT,
    "options": QUESTION_OPTIONS,
    "answer": QUESTION_ANSWER,
    "explanation": QUESTION_EXPLANATION,
}
THREE_OPTION_ARGS = {**PERSIST_ARGS, "options": QUESTION_OPTIONS[:3]}

QUIZ_MODEL_TEXT = json.dumps(
    [{
        "prompt": QUESTION_PROMPT,
        "options": QUESTION_OPTIONS,
        "answer": QUESTION_ANSWER,
        "explanation": QUESTION_EXPLANATION,
    }],
    ensure_ascii=False,
)


# ---------------------------------------------------------------------------
# Prompt oracles: test-owned literals plus synthetic inputs.
#
# These three literals are copied into this file verbatim. The expectation side
# never imports a production template constant and never calls a production
# renderer, so changed production wording must fail these expectations rather
# than be followed by them.
# ---------------------------------------------------------------------------

QUIZ_PROMPT_BASE_GOLDEN = """You are a quiz question generator for an exam coach.

Generate {n} multiple-choice question(s) on the topic: {topic_name}.
Difficulty: {difficulty}.
{context_section}
Output ONLY a JSON array. Each item must have:
- prompt: the question text
- options: exactly 4 strings, each prefixed with "A) " "B) " "C) " "D) "
- answer: a single letter A/B/C/D matching the correct option
- explanation: 1-2 sentences explaining why the answer is correct

Example shape:
[
  {{"prompt": "...", "options": ["A) ...", "B) ...", "C) ...", "D) ..."], "answer": "A", "explanation": "..."}}
]

CRITICAL FORMATTING: Each option MUST start with exactly "A) " / "B) " / "C) " / "D) "
(letter, closing paren, space). Never output bare option text without the prefix.
Do NOT number with "1) " "2) " or use bullet points.
"""
QUIZ_GROUNDED_CONTEXT_TEMPLATE_GOLDEN = """
GROUND your questions strictly in these source chunks. Do NOT invent facts not
supported by them; if a chunk doesn't cover an angle, don't ask about it.

{chunks}
"""
QUIZ_AGENT_SYSTEM_PROMPT_GOLDEN = """You are a study coach quiz generator.

The user wants a multiple-choice quiz question on a topic.

Your job:
1. Read the user's message to extract the topic.
2. Call `retriever_search` with the topic to ground in the user's PDF source material.
3. Write a single multiple-choice question:
   - One clear, unambiguous question prompt
   - Exactly 4 options, each prefixed "A) ", "B) ", "C) ", "D) "
   - Distractors should be plausible (not obviously wrong)
   - One correct answer letter
   - A 1-2 sentence explanation grounded in the retrieved source chunks
4. Call `persist_quiz_question` with the question, options, answer, and explanation.
5. After persistence succeeds, write a short markdown reply to the user showing
   the question. Do NOT call more tools.

Today is {today}. Difficulty: medium. Ground strictly in retrieved chunks; do
not invent facts."""


def expected_generate_prompt(*, topic_name: str, difficulty: str, n: int,
                             context_chunks: list[dict] | None) -> str:
    """The GENERATE user message this case must put on the wire.

    Rendered from this file's own literals, never from ``app.agent.tools.quiz``.
    """
    lines = "\n".join(f"[{i + 1}] {c.get('content', '')}" for i, c in enumerate(context_chunks or []))
    context_section = (
        QUIZ_GROUNDED_CONTEXT_TEMPLATE_GOLDEN.format(chunks=lines) if context_chunks else ""
    )
    return QUIZ_PROMPT_BASE_GOLDEN.format(
        n=n, topic_name=topic_name, difficulty=difficulty, context_section=context_section,
    )


def assert_plain_messages(body: dict, expected_prompt: str) -> str:
    """This file's complete plain-prompt expectation: one user message, whole.

    The whole ``messages`` list is compared with exact equality -- not a prefix,
    a substring, a normalised fragment or a heading. The expected prompt is
    returned so a caller can go on asserting against the same value.
    """
    assert body["messages"] == [{"role": "user", "content": expected_prompt}]
    return expected_prompt


def expected_agent_system_prompt() -> str:
    return QUIZ_AGENT_SYSTEM_PROMPT_GOLDEN.format(today=TODAY_ISO)


# ---------------------------------------------------------------------------
# Scripted responses and wire assertions.
# ---------------------------------------------------------------------------


def request_json(request: httpx.Request) -> dict:
    return json.loads(request.content)


def error_body(message: str, *, code: int) -> dict:
    return {"error": {"message": message, "type": "invalid_request_error", "code": code}}


def assistant_tool_calls(*calls, content=None, finish_reason: str = "tool_calls") -> dict:
    """One assistant turn requesting tool calls, in the given order."""
    return {
        "id": "chatcmpl-synthetic",
        "object": "chat.completion",
        "created": 1767225600,
        "model": OPENAI_MODEL,
        "choices": [{
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
        }],
        "usage": {"prompt_tokens": 13, "completion_tokens": 9, "total_tokens": 22},
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
        "usage": {"prompt_tokens": 13, "completion_tokens": 9, "total_tokens": 22},
    }


class ScriptedResponder:
    """Dispatch real requests by whether the adapter bound tools or not.

    A tools-free turn is the deterministic node's single ``generate_quiz``
    call; a bound turn is one agent-loop iteration.
    """

    def __init__(self, *, bound=(), plain=(), on_request=None):
        self.bound = list(bound)
        self.plain = list(plain)
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
        assert index < len(script), f"the consumer sent more turns than scripted ({len(script)})"
        setattr(self, counter, index + 1)
        entry = script[index]
        if callable(entry):
            return entry(request)
        payload, status = entry if isinstance(entry, tuple) else (entry, 200)
        return httpx.Response(status, json=payload,
                              headers={"content-type": "application/json"})


def assert_chat_request(request: httpx.Request, *, bound: bool) -> dict:
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
# Tool schema expectations, written from the tool signatures in
# app/agent/quiz_master_agent.py. The wire schema does NOT carry the
# four-option or A/B/C/D rules.
# ---------------------------------------------------------------------------

EXPECTED_TOOL_SCHEMAS: list[dict] = [
    {
        "name": "retriever_search",
        "description_first_line": "Search the user's PDF corpus for chunks relevant to a quiz topic.",
        "parameters": {
            "properties": {"query": {"type": "string"}, "top_k": {"default": 5, "type": "integer"}},
            "required": ["query"],
            "type": "object",
        },
    },
    {
        "name": "persist_quiz_question",
        "description_first_line": "Save the multiple-choice quiz question to the database.",
        "parameters": {
            "properties": {
                "answer": {"type": "string"},
                "explanation": {"type": "string"},
                "options": {"items": {"type": "string"}, "type": "array"},
                "prompt": {"type": "string"},
                "topic": {"type": "string"},
            },
            "required": ["topic", "prompt", "options", "answer", "explanation"],
            "type": "object",
        },
    },
]


def assert_tool_set(body: dict) -> None:
    tools = body["tools"]
    assert [t["type"] for t in tools] == ["function"] * 2
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
    ToolMessage result. Nothing is skipped, and dynamic ids in results come
    from real rows the case reads out of a second session.

    ``rounds`` entries are ``{"calls": [(call_id, tool_name, args), ...],
    "results": [decoded_result, ...]}``.
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
# Shared fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture
def db(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'quiz_contract.db'}")
    Base.metadata.create_all(engine)
    session = Session(engine)

    def fresh() -> Session:
        return Session(engine)

    session.info["fresh"] = fresh
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def seed_user(session, fingerprint: str):
    return UserRepository(session).get_or_create(fingerprint)


def count_rows(session, model) -> int:
    return len(list(session.execute(select(model)).scalars()))


class SyntheticRetriever:
    """A deterministic in-test corpus. Only retrieval is synthetic."""

    def __init__(self, chunks=None):
        self.chunks = list(chunks or [])
        self.calls: list[tuple[str, int]] = []

    def search(self, query, top_k=5):
        self.calls.append((query, top_k))
        return self.chunks[:top_k]


class ModelObserver(AsyncCallbackHandler):
    """Read-only observation of the real model round trips."""

    def __init__(self):
        self.starts: list = []
        self.ends: list[object] = []

    async def on_chat_model_start(self, serialized, messages, **kwargs):
        self.starts.append(messages)

    async def on_llm_end(self, response, **kwargs):
        self.ends.append(response.generations[0][0].message)


def observed_messages(observer: ModelObserver, index: int) -> list:
    """The exact message objects the adapter was handed on round ``index``."""
    batches = observer.starts[index]
    assert len(batches) == 1, "this case sends exactly one conversation per round"
    return batches[0]


def build_det(session, llm, *, retriever=None):
    return build_quiz_master(
        llm=llm,
        topic_repo=TopicRepository(session),
        question_repo=QuestionRepository(session),
        mistake_repo=MistakeRepository(session),
        mastery_repo=MasteryRepository(session),
        goal_repo=GoalRepository(session),
        retriever=retriever if retriever is not None else SyntheticRetriever(),
        now_fn=lambda: NOW,
    )


def build_loop(session, llm, *, retriever=None, max_iter: int = 6):
    return build_quiz_master_agent(
        llm=llm,
        topic_repo=TopicRepository(session),
        question_repo=QuestionRepository(session),
        goal_repo=GoalRepository(session),
        retriever=retriever if retriever is not None else SyntheticRetriever(),
        now_fn=lambda: NOW,
        max_iter=max_iter,
    )


# ===========================================================================
# Q1 -- deterministic GENERATE, success
# ===========================================================================


async def test_Q1_deterministic_generate_persists_question(offline_openai, db):
    """Q1: one tools-free turn; Goal/Topic/chunks are committed before it."""
    user = seed_user(db, "fp-d4b3-q1")
    retriever = SyntheticRetriever(SYNTHETIC_CHUNKS)
    fresh = db.info["fresh"]
    goals_at_request: list[int] = []
    topics_at_request: list[int] = []
    chunks_at_request: list[list] = []
    questions_at_request: list[int] = []

    def on_request(request, body):
        probe = fresh()
        try:
            goals_at_request.append(count_rows(probe, Goal))
            topics = list(probe.execute(select(Topic)).scalars())
            topics_at_request.append(len(topics))
            chunks_at_request.append(list(topics[0].source_chunks) if topics else [])
            questions_at_request.append(count_rows(probe, Question))
        finally:
            probe.close()

    responder = ScriptedResponder(plain=[assistant_text(QUIZ_MODEL_TEXT)], on_request=on_request)
    offline_openai.respond(responder)

    node = build_det(db, offline_openai.model(), retriever=retriever)
    update = await node({"messages": [HumanMessage(content=QUIZ_USER_MSG)], "user_id": user.id})

    assert len(offline_openai.requests) == 1
    body = plain_bodies(offline_openai)[0]
    expected_prompt = expected_generate_prompt(
        topic_name=QUESTION_TOPIC, difficulty="medium", n=1, context_chunks=SYNTHETIC_CHUNKS,
    )
    assert_plain_messages(body, expected_prompt)
    assert "Generate 1 multiple-choice question(s) on the topic: HyDE." in expected_prompt
    assert "Difficulty: medium." in expected_prompt
    assert f"[1] {SYNTHETIC_CHUNK['content']}" in expected_prompt

    assert goals_at_request == [1], "the Goal was committed before the model call"
    assert topics_at_request == [1], "the Topic was committed before the model call"
    assert chunks_at_request == [[SYNTHETIC_CHUNK["chunk_id"]]], "source_chunks committed"
    assert questions_at_request == [0], "no Question existed yet"

    text = update["messages"][0].content
    assert text.startswith(f"📝 Quiz on {QUESTION_TOPIC}:")
    assert QUESTION_PROMPT in text
    for option in QUESTION_OPTIONS:
        assert option in text
    assert text.endswith("Reply with A, B, C, or D.")
    assert update["quiz_action"] == "generate"
    assert update["last_context"] == QUESTION_EXPLANATION
    assert retriever.calls == [(QUESTION_TOPIC, 5)]

    probe = fresh()
    try:
        question = probe.get(Question, update["active_quiz_question_id"])
        assert question is not None
        assert question.prompt == QUESTION_PROMPT
        assert question.options_json == QUESTION_OPTIONS
        assert question.answer == QUESTION_ANSWER
        assert question.explanation == QUESTION_EXPLANATION
        topic = probe.get(Topic, question.topic_id)
        assert topic is not None and topic.name == QUESTION_TOPIC
        assert count_rows(probe, Question) == 1
    finally:
        probe.close()


# ===========================================================================
# Q2 -- deterministic GENERATE with a legal empty body
# ===========================================================================


async def test_Q2_deterministic_empty_body_fails_narrowly(offline_openai, db):
    """Q2: ``content=''`` fails at the strict text boundary, before any write."""
    user = seed_user(db, "fp-d4b3-q2")
    retriever = SyntheticRetriever(SYNTHETIC_CHUNKS)
    fresh = db.info["fresh"]
    committed: list[tuple[int, int, list, int]] = []
    committed_topic_id: list[str] = []
    committed_goal_id: list[str] = []

    def on_request(request, body):
        probe = fresh()
        try:
            topics = list(probe.execute(select(Topic)).scalars())
            goal = probe.execute(select(Goal)).scalar_one()
            committed_topic_id.append(topics[0].id)
            committed_goal_id.append(goal.id)
            committed.append((
                count_rows(probe, Goal), len(topics),
                list(topics[0].source_chunks) if topics else [],
                count_rows(probe, Question),
            ))
        finally:
            probe.close()

    responder = ScriptedResponder(plain=[assistant_text("")], on_request=on_request)
    offline_openai.respond(responder)

    node = build_det(db, offline_openai.model(), retriever=retriever)
    with pytest.raises(ValueError) as excinfo:
        await node({"messages": [HumanMessage(content=QUIZ_USER_MSG)], "user_id": user.id})

    assert len(offline_openai.requests) == 1
    body = plain_bodies(offline_openai)[0]
    assert_plain_messages(body, expected_generate_prompt(
        topic_name=QUESTION_TOPIC, difficulty="medium", n=1,
        context_chunks=SYNTHETIC_CHUNKS,
    ))

    assert type(excinfo.value) is ValueError
    assert str(excinfo.value) == EMPTY_CONTENT_MESSAGE
    assert committed == [(1, 1, [SYNTHETIC_CHUNK["chunk_id"]], 0)], (
        "Goal, Topic and source_chunks were committed before the call and stay"
    )

    probe = fresh()
    try:
        assert count_rows(probe, Goal) == 1
        assert count_rows(probe, Topic) == 1
        assert count_rows(probe, Question) == 0, "no fabricated question"
        # Not just "a row is still there": the same Topic row still carries
        # this case's full synthetic chunk id set, read as the real column.
        topic = probe.execute(select(Topic)).scalar_one()
        assert topic.id == committed_topic_id[0]
        assert list(topic.source_chunks) == [SYNTHETIC_CHUNK["chunk_id"]]
        assert SYNTHETIC_CHUNK["chunk_id"] in list(topic.source_chunks)
        goal = probe.execute(select(Goal)).scalar_one()
        assert goal.id == committed_goal_id[0]
        assert topic.goal_id == committed_goal_id[0]
        assert count_rows(probe, Question) == 0, "no fabricated question"
    finally:
        probe.close()


# ===========================================================================
# Q3 -- deterministic GENERATE when the SDK retries a 503 to exhaustion
# ===========================================================================


async def test_Q3_deterministic_503_propagates_raw_sdk_error(offline_openai, db):
    """Q3: the real SDK exception crosses the consumer's current boundary."""
    import openai

    user = seed_user(db, "fp-d4b3-q3")
    retriever = SyntheticRetriever(SYNTHETIC_CHUNKS)
    fresh = db.info["fresh"]
    expected_chunk_ids = [SYNTHETIC_CHUNK["chunk_id"]]
    # One snapshot per attempt, taken from a second session *before* that
    # request is served, so the pre-request identity is on record rather than
    # reconstructed afterwards.
    before_snapshots: list[dict] = []

    def on_request(request, body):
        probe = fresh()
        try:
            goal = probe.execute(select(Goal)).scalar_one()
            topic = probe.execute(select(Topic)).scalar_one()
            before_snapshots.append({
                "goal_id": goal.id,
                "goal_user_id": goal.user_id,
                "topic_id": topic.id,
                "topic_goal_id": topic.goal_id,
                "topic_name": topic.name,
                # Copy the real column: it stores chunk IDs, not whole chunks.
                "source_chunks": list(topic.source_chunks),
                "question_count": count_rows(probe, Question),
            })
        finally:
            probe.close()

    responder = ScriptedResponder(
        plain=[(error_body("synthetic", code=503), 503)] * 3, on_request=on_request,
    )
    offline_openai.respond(responder)

    node = build_det(db, offline_openai.model(), retriever=retriever)
    with pytest.raises(openai.InternalServerError) as excinfo:
        await node({"messages": [HumanMessage(content=QUIZ_USER_MSG)], "user_id": user.id})

    assert len(offline_openai.requests) == 3, "the SDK retried 503 three times"
    expected_prompt = expected_generate_prompt(
        topic_name=QUESTION_TOPIC, difficulty="medium", n=1, context_chunks=SYNTHETIC_CHUNKS,
    )
    bodies = plain_bodies(offline_openai)
    for body in bodies:
        # Every attempt carried the same GENERATE prompt, compared whole.
        assert_plain_messages(body, expected_prompt)
    # All three attempts of this one logical invocation re-sent the identical
    # parsed JSON body, not merely matching sub-fields. Headers are not
    # compared -- the SDK may add its own per-request headers.
    assert len(bodies) == 3
    assert all(body == bodies[0] for body in bodies), (
        "all three attempts must carry the same request body"
    )

    assert type(excinfo.value) is openai.InternalServerError

    # The pre-request snapshot was stable across all three attempts, and it
    # really belongs to this case's synthetic user.
    assert len(before_snapshots) == 3
    assert all(snap == before_snapshots[0] for snap in before_snapshots), (
        "Goal/Topic identity, linkage and chunk ids did not drift between attempts"
    )
    first = before_snapshots[0]
    assert first["goal_user_id"] == user.id, "the seeded Goal belongs to the synthetic user"
    assert first["topic_name"] == QUESTION_TOPIC
    assert first["topic_goal_id"] == first["goal_id"]
    assert first["source_chunks"] == expected_chunk_ids
    assert expected_chunk_ids == [SYNTHETIC_CHUNK["chunk_id"]]
    assert [s["question_count"] for s in before_snapshots] == [0, 0, 0]

    # After the failure, a new session must still show the *same* Goal and
    # Topic rows -- compared against the recorded pre-request ids, not just
    # against whatever is current now.
    probe = fresh()
    try:
        assert count_rows(probe, Goal) == 1
        assert count_rows(probe, Topic) == 1
        assert count_rows(probe, Question) == 0
        goal = probe.execute(select(Goal)).scalar_one()
        topic = probe.execute(select(Topic)).scalar_one()
        assert goal.id == first["goal_id"]
        assert goal.user_id == first["goal_user_id"]
        assert topic.id == first["topic_id"]
        assert topic.goal_id == first["topic_goal_id"]
        assert topic.name == first["topic_name"]
        # The failed GENERATE call did not clear the grounded chunk ids.
        assert list(topic.source_chunks) == first["source_chunks"] == expected_chunk_ids
    finally:
        probe.close()


# ===========================================================================
# Q4 -- loop happy path whose final body is empty
# ===========================================================================


async def test_Q4_loop_persists_then_final_uses_saved_question(offline_openai, db):
    """Q4: the final answer is built from the stored Question, not the model text."""
    user = seed_user(db, "fp-d4b3-q4")
    observer = ModelObserver()
    fresh = db.info["fresh"]
    questions_per_request: list[int] = []

    def on_request(request, body):
        probe = fresh()
        try:
            questions_per_request.append(count_rows(probe, Question))
        finally:
            probe.close()

    responder = ScriptedResponder(
        bound=[
            assistant_tool_calls(("c_rs", "retriever_search",
                                  json.dumps({"query": QUESTION_TOPIC, "top_k": 2}))),
            assistant_tool_calls(("c_ps", "persist_quiz_question", json.dumps(PERSIST_ARGS))),
            # A legal model body with no usable text: unlike the Planner loop,
            # the Quiz loop does not require_text on its final turn.
            assistant_text(""),
        ],
        on_request=on_request,
    )
    offline_openai.respond(responder)

    node = build_loop(db, offline_openai.model(callbacks=[observer]),
                      retriever=SyntheticRetriever(SYNTHETIC_CHUNKS))
    update = await node({"messages": [HumanMessage(content=QUIZ_USER_MSG)], "user_id": user.id})

    assert len(offline_openai.requests) == 3
    bodies = bound_bodies(offline_openai)
    for body in bodies:
        assert_tool_set(body)

    probe = fresh()
    try:
        saved = probe.execute(select(Question)).scalar_one()
        persist_result = {
            "question_id": saved.id,
            "topic_id": saved.topic_id,
            "persisted": True,
        }
    finally:
        probe.close()

    RETRIEVER_ROUND = {
        "calls": [("c_rs", "retriever_search", {"query": QUESTION_TOPIC, "top_k": 2})],
        "results": [SYNTHETIC_CHUNKS],
    }
    PERSIST_ROUND = {
        "calls": [("c_ps", "persist_quiz_question", PERSIST_ARGS)],
        "results": [persist_result],
    }
    assert_agent_request(bodies[0], user_msg=QUIZ_USER_MSG, rounds=[])
    assert_agent_request(bodies[1], user_msg=QUIZ_USER_MSG, rounds=[RETRIEVER_ROUND])
    assert_agent_request(bodies[2], user_msg=QUIZ_USER_MSG,
                         rounds=[RETRIEVER_ROUND, PERSIST_ROUND])

    tool_result = persist_result
    assert questions_per_request == [0, 0, 1], (
        "the Question was already committed when the final request was sent"
    )

    assert len(observer.starts) == 3 and len(observer.ends) == 3
    assert observed_messages(observer, 1)[2] is observer.ends[0]
    assert observed_messages(observer, 2)[4] is observer.ends[1]

    trace = update["agent_trace"]
    assert trace["exit_reason"] == "natural_stop"
    assert trace["total_iterations"] == 3
    assert trace["total_tool_calls"] == 2
    assert trace["tool_call_breakdown"] == {
        "retriever_search": 1, "persist_quiz_question": 1,
    }
    assert trace["tool_errors"] == 0
    assert update["quiz_action"] == "generate"
    assert update["active_quiz_question_id"] == tool_result["question_id"]

    expected_text = (
        "📝 Quiz:\n\n" + QUESTION_PROMPT + "\n\n" + "\n".join(QUESTION_OPTIONS)
        + "\n\nReply with A, B, C, or D."
    )
    assert update["messages"][0].content == expected_text
    assert update["messages"][0].content != "", "the empty model body is not the answer"

    probe = fresh()
    try:
        question = probe.get(Question, update["active_quiz_question_id"])
        assert question is not None
        assert question.prompt == QUESTION_PROMPT
        assert question.options_json == QUESTION_OPTIONS
        assert question.answer == QUESTION_ANSWER
    finally:
        probe.close()


# ===========================================================================
# Q5 -- loop self-corrects a business-invalid persist
# ===========================================================================


async def test_Q5_loop_three_options_then_self_correction(offline_openai, db):
    """Q5: the four-option rule lives in the schema model, not on the wire."""
    user = seed_user(db, "fp-d4b3-q5")
    fresh = db.info["fresh"]
    snapshot_at_retry: list[tuple[int, int, int]] = []

    def on_request(request, body):
        if "tools" not in body:
            return
        if sum(1 for seen in responder.seen if "tools" in seen) != 2:
            return
        probe = fresh()
        try:
            snapshot_at_retry.append((
                count_rows(probe, Goal), count_rows(probe, Topic), count_rows(probe, Question),
            ))
        finally:
            probe.close()

    responder = ScriptedResponder(
        bound=[
            assistant_tool_calls(("c_bad", "persist_quiz_question", json.dumps(THREE_OPTION_ARGS))),
            assistant_tool_calls(("c_fix", "persist_quiz_question", json.dumps(PERSIST_ARGS))),
            assistant_text("Here is your question."),
        ],
        on_request=on_request,
    )
    offline_openai.respond(responder)

    node = build_loop(db, offline_openai.model())
    update = await node({"messages": [HumanMessage(content=QUIZ_USER_MSG)], "user_id": user.id})

    assert len(offline_openai.requests) == 3
    bodies = bound_bodies(offline_openai)
    for body in bodies:
        assert_tool_set(body)

    # The wire schema does not carry minItems: the SDK layer never rejected it.
    options_schema = EXPECTED_TOOL_SCHEMAS[1]["parameters"]["properties"]["options"]
    assert options_schema == {"items": {"type": "string"}, "type": "array"}
    assert "minItems" not in options_schema

    probe = fresh()
    try:
        saved = probe.execute(select(Question)).scalar_one()
        fix_result = {"question_id": saved.id, "topic_id": saved.topic_id, "persisted": True}
    finally:
        probe.close()

    # The tool's error text is rebuilt from the schema model itself, so it is
    # anchored to the business validator rather than copied off the wire.
    with pytest.raises(ValidationError) as schema_error:
        QuizQuestionPersist(**THREE_OPTION_ARGS)
    assert schema_error.value.errors()[0]["loc"] == ("options",)
    error_message = f"invalid at options: {schema_error.value.errors()[0]['msg']}"

    # Both pairs are declared and checked: the rejected one and the corrected
    # one, with the real question/topic ids in the success result.
    BAD_ROUND = {
        "calls": [("c_bad", "persist_quiz_question", THREE_OPTION_ARGS)],
        "results": [{"error": error_message}],
    }
    FIX_ROUND = {
        "calls": [("c_fix", "persist_quiz_question", PERSIST_ARGS)],
        "results": [fix_result],
    }
    assert_agent_request(bodies[0], user_msg=QUIZ_USER_MSG, rounds=[])
    assert_agent_request(bodies[1], user_msg=QUIZ_USER_MSG, rounds=[BAD_ROUND])
    assert_agent_request(bodies[2], user_msg=QUIZ_USER_MSG, rounds=[BAD_ROUND, FIX_ROUND])

    # Note: the rejected arguments are compared as decoded JSON by
    # assert_agent_request above, never as a raw ``arguments`` string -- the
    # SDK is free to re-space that JSON on the wire.

    assert snapshot_at_retry == [(0, 0, 0)], (
        "only the User was seeded; no Goal, Topic or Question existed yet"
    )

    trace = update["agent_trace"]
    assert trace["exit_reason"] == "natural_stop"
    assert trace["total_iterations"] == 3
    assert trace["tool_errors"] == 1
    assert trace["tool_call_breakdown"] == {"persist_quiz_question": 2}

    probe = fresh()
    try:
        assert count_rows(probe, Question) == 1, "only the corrected call wrote"
        question = probe.execute(select(Question)).scalar_one()
        assert question.options_json == QUESTION_OPTIONS
    finally:
        probe.close()


# ===========================================================================
# Q6 -- loop stops naturally without persisting anything
# ===========================================================================


async def test_Q6_loop_natural_stop_without_persist(offline_openai, db):
    """Q6: current behaviour degrades instead of auto-persisting."""
    user = seed_user(db, "fp-d4b3-q6")

    responder = ScriptedResponder(bound=[assistant_text("Here is a question about HyDE: ...")])
    offline_openai.respond(responder)

    node = build_loop(db, offline_openai.model())
    update = await node({"messages": [HumanMessage(content=QUIZ_USER_MSG)], "user_id": user.id})

    assert len(offline_openai.requests) == 1
    assert_tool_set(bound_bodies(offline_openai)[0])
    assert_agent_request(bound_bodies(offline_openai)[0], user_msg=QUIZ_USER_MSG, rounds=[])

    assert update["degraded"] is True
    assert update["messages"][0].content == _QUIZ_PERSIST_FAILED_MSG
    assert update["active_quiz_question_id"] is None
    trace = update["agent_trace"]
    assert trace["exit_reason"] == "quiz_persist_failed"
    assert trace["total_iterations"] == 1
    assert trace["total_tool_calls"] == 0
    assert "Here is a question about HyDE" not in json.dumps(update, default=str)

    probe = db.info["fresh"]()
    try:
        assert count_rows(probe, Question) == 0
        assert count_rows(probe, Goal) == 0
        assert count_rows(probe, Topic) == 0
    finally:
        probe.close()


# ===========================================================================
# Q7 -- loop's very first model call fails
# ===========================================================================


async def test_Q7_loop_first_503_degrades_before_any_tool(offline_openai, db):
    """Q7: three HTTP attempts, one logical invocation, nothing written."""
    user = seed_user(db, "fp-d4b3-q7")
    marker = "synthetic-provider-marker-do-not-leak"

    responder = ScriptedResponder(bound=[(error_body(marker, code=503), 503)] * 3)
    offline_openai.respond(responder)

    node = build_loop(db, offline_openai.model())
    update = await node({"messages": [HumanMessage(content=QUIZ_USER_MSG)], "user_id": user.id})

    assert len(offline_openai.requests) == 3
    bodies = bound_bodies(offline_openai)
    for body in bodies:
        assert_tool_set(body)
        # All three attempts of the same logical invocation carry the identical
        # input; the SDK did not alter the request between retries.
        assert_agent_request(body, user_msg=QUIZ_USER_MSG, rounds=[])
    assert [request_json(r) for r in offline_openai.requests] == [bodies[0]] * 3

    assert update["degraded"] is True
    assert update["messages"][0].content == _LLM_FAILED_MSG
    trace = update["agent_trace"]
    assert trace["exit_reason"] == "llm_call_failed"
    assert trace["llm_error"] == EXPECTED_GENERIC_LLM_ERROR
    assert trace["total_iterations"] == 0, "the failing invoke never became an iteration"
    assert trace["total_tool_calls"] == 0
    assert marker not in update["messages"][0].content
    assert marker not in trace["llm_error"]
    assert marker not in json.dumps(update, default=str)

    probe = db.info["fresh"]()
    try:
        assert count_rows(probe, Goal) == 0
        assert count_rows(probe, Topic) == 0
        assert count_rows(probe, Question) == 0
    finally:
        probe.close()


# ===========================================================================
# Q8 -- loop's final turn fails after a successful persist
# ===========================================================================


async def test_Q8_loop_503_after_persist_keeps_question(offline_openai, db):
    """Q8: the committed Question survives; the degrade carries no pointer."""
    user = seed_user(db, "fp-d4b3-q8")
    marker = "synthetic-provider-marker-do-not-leak"
    fresh = db.info["fresh"]
    snapshot_at_second: list[tuple[int, int, int]] = []

    def on_request(request, body):
        if "tools" not in body:
            return
        if sum(1 for seen in responder.seen if "tools" in seen) != 2:
            return
        probe = fresh()
        try:
            snapshot_at_second.append((
                count_rows(probe, Goal), count_rows(probe, Topic), count_rows(probe, Question),
            ))
        finally:
            probe.close()

    responder = ScriptedResponder(
        bound=[
            assistant_tool_calls(("c_ps", "persist_quiz_question", json.dumps(PERSIST_ARGS))),
            (error_body(marker, code=503), 503),
            (error_body(marker, code=503), 503),
            (error_body(marker, code=503), 503),
        ],
        on_request=on_request,
    )
    offline_openai.respond(responder)

    node = build_loop(db, offline_openai.model())
    update = await node({"messages": [HumanMessage(content=QUIZ_USER_MSG)], "user_id": user.id})

    assert len(offline_openai.requests) == 4, "one success plus three failed attempts"
    assert (responder.bound_calls) == 4
    bodies = bound_bodies(offline_openai)
    for body in bodies:
        assert_tool_set(body)
    assert snapshot_at_second == [(1, 1, 1)], (
        "Goal, Topic and Question were all committed before the failing turn"
    )

    probe = fresh()
    try:
        saved = probe.execute(select(Question)).scalar_one()
        persist_result = {"question_id": saved.id, "topic_id": saved.topic_id, "persisted": True}
    finally:
        probe.close()
    PERSIST_ROUND = {
        "calls": [("c_ps", "persist_quiz_question", PERSIST_ARGS)],
        "results": [persist_result],
    }
    # The first request is the bare prefix; the successful tool result and its
    # ids are carried by all three failed attempts of the next invocation.
    assert_agent_request(bodies[0], user_msg=QUIZ_USER_MSG, rounds=[])
    for body in bodies[1:]:
        assert_agent_request(body, user_msg=QUIZ_USER_MSG, rounds=[PERSIST_ROUND])
    assert [request_json(r) for r in offline_openai.requests[1:]] == [bodies[1]] * 3

    assert update["degraded"] is True
    assert update["messages"][0].content == _LLM_FAILED_MSG
    assert "active_quiz_question_id" not in update, "no pointer is fabricated"
    trace = update["agent_trace"]
    assert trace["exit_reason"] == "llm_call_failed"
    assert trace["llm_error"] == EXPECTED_GENERIC_LLM_ERROR
    assert trace["total_iterations"] == 1
    assert trace["total_tool_calls"] == 1
    assert marker not in update["messages"][0].content
    assert marker not in trace["llm_error"]
    assert marker not in json.dumps(update, default=str)

    probe = fresh()
    try:
        questions = list(probe.execute(select(Question)).scalars())
        assert len(questions) == 1, "the persisted Question is not rolled back"
        assert questions[0].prompt == QUESTION_PROMPT
    finally:
        probe.close()


# ===========================================================================
# Q9 -- default loop budget
# ===========================================================================


async def test_Q9_loop_default_budget_six(offline_openai, db):
    """Q9: six logical rounds on the default budget, none of them persisting."""
    user = seed_user(db, "fp-d4b3-q9")

    assert inspect.signature(build_quiz_master_agent).parameters["max_iter"].default == 6, (
        "this case uses the product default; it must still be 6"
    )

    responder = ScriptedResponder(bound=[
        assistant_tool_calls((f"c{i}", "retriever_search", json.dumps({"query": f"q{i}"})))
        for i in range(6)
    ])
    offline_openai.respond(responder)

    node = build_loop(db, offline_openai.model(), max_iter=6)
    update = await node({"messages": [HumanMessage(content=QUIZ_USER_MSG)], "user_id": user.id})

    assert len(offline_openai.requests) == 6, "one HTTP per logical round, no retries"
    bodies = bound_bodies(offline_openai)
    for body in bodies:
        assert_tool_set(body)

    # Round N carries rounds 1..N-1 in full: every round's own query and tool
    # result is declared here and checked on each later request, not just the
    # last one. The synthetic retriever has no chunks, so results are [].
    declared_rounds = [
        {"calls": [(f"c{i}", "retriever_search", {"query": f"q{i}"})], "results": [[]]}
        for i in range(6)
    ]
    for index, body in enumerate(bodies):
        assert_agent_request(body, user_msg=QUIZ_USER_MSG, rounds=declared_rounds[:index])

    trace = update["agent_trace"]
    assert trace["exit_reason"] == "budget_exhausted"
    assert trace["total_iterations"] == 6
    assert trace["total_tool_calls"] == 6
    assert trace["tool_call_breakdown"] == {"retriever_search": 6}
    assert update["degraded"] is True
    assert update["messages"][0].content == _BUDGET_EXHAUSTED_MSG
    assert "active_quiz_question_id" not in update

    probe = db.info["fresh"]()
    try:
        assert count_rows(probe, Question) == 0
        assert count_rows(probe, Goal) == 0
        assert count_rows(probe, Topic) == 0
    finally:
        probe.close()


# ===========================================================================
# Q10 -- cancel the final turn after a successful persist
# ===========================================================================


async def test_Q10_loop_cancelled_after_persist_keeps_question(offline_openai, db):
    """Q10: the Question is already committed when the final body is cancelled."""
    user = seed_user(db, "fp-d4b3-q10")
    fresh = db.info["fresh"]

    gate = asyncio.Event()  # never set: only cancellation can end this body
    gated = offline_openai.body([wait_for(gate), send(b"never-reached")])
    responder = ScriptedResponder(
        bound=[
            assistant_tool_calls(("c_ps", "persist_quiz_question", json.dumps(PERSIST_ARGS))),
            lambda request: httpx.Response(
                200, headers={"content-type": "application/json"}, stream=gated,
            ),
        ]
    )
    offline_openai.respond(responder)

    node = build_loop(db, offline_openai.model())
    task = asyncio.create_task(
        node({"messages": [HumanMessage(content=QUIZ_USER_MSG)], "user_id": user.id})
    )
    try:
        await asyncio.wait_for(gated.started.wait(), timeout=5)

        # R2: before the cancellation, a second session already sees all three
        # committed rows, with this case's declared question payload.
        before = fresh()
        try:
            assert count_rows(before, Goal) == 1
            assert count_rows(before, Topic) == 1
            assert count_rows(before, Question) == 1
            saved = before.execute(select(Question)).scalar_one()
            assert saved.prompt == PERSIST_ARGS["prompt"]
            assert saved.options_json == PERSIST_ARGS["options"]
            assert saved.answer == PERSIST_ARGS["answer"]
            assert saved.explanation == PERSIST_ARGS["explanation"]
            topic_before = before.get(Topic, saved.topic_id)
            assert topic_before is not None and topic_before.name == PERSIST_ARGS["topic"]
            goal_before = before.get(Goal, topic_before.goal_id)
            assert goal_before is not None and goal_before.user_id == user.id
            ids_before = (saved.id, topic_before.id, goal_before.id)
        finally:
            before.close()

        task.cancel()
        with pytest.raises(asyncio.CancelledError) as excinfo:
            await asyncio.wait_for(asyncio.shield(task), timeout=5)
        assert type(excinfo.value) is asyncio.CancelledError
        assert task.cancelled()

        assert len(offline_openai.requests) == 2
        bodies = bound_bodies(offline_openai)
        for body in bodies:
            assert_tool_set(body)
        PERSIST_ROUND = {
            "calls": [("c_ps", "persist_quiz_question", PERSIST_ARGS)],
            "results": [{
                "question_id": ids_before[0],
                "topic_id": ids_before[1],
                "persisted": True,
            }],
        }
        assert_agent_request(bodies[0], user_msg=QUIZ_USER_MSG, rounds=[])
        assert_agent_request(bodies[1], user_msg=QUIZ_USER_MSG, rounds=[PERSIST_ROUND])
        # Observed before the finally and before the fixture closes the SDK
        # clients. Recorded, never asserted.
        print(f"D4b Q10 fixture_pre_close_count={gated.close_count}")
    finally:
        await reclaim_own_task(task)

    # The very same Question, not merely "some row still exists".
    probe = fresh()
    try:
        questions = list(probe.execute(select(Question)).scalars())
        assert len(questions) == 1, "the persisted entity is kept, not rolled back"
        assert questions[0].id == ids_before[0]
        assert questions[0].topic_id == ids_before[1]
        assert questions[0].prompt == PERSIST_ARGS["prompt"]
        assert questions[0].options_json == PERSIST_ARGS["options"]
        assert questions[0].answer == PERSIST_ARGS["answer"]
        assert count_rows(probe, Goal) == 1
        assert count_rows(probe, Topic) == 1
    finally:
        probe.close()


# ===========================================================================
# Self-checks
# ===========================================================================


def test_quiz_sources_are_exact_for_this_checkout():
    """Exact resolved identity for every module this slice exercises."""
    import app.agent.quiz_master as quiz_master_module
    import app.agent.quiz_master_agent as quiz_agent_module
    import app.agent.tools.quiz as quiz_tools_module
    import app.agent.tools.schemas as schemas_module
    import langchain_openai.chat_models.base as openai_adapter
    import openai
    import openai._base_client as openai_base_client

    assert_module_from_backend(quiz_master_module, "app/agent/quiz_master.py")
    assert_module_from_backend(quiz_agent_module, "app/agent/quiz_master_agent.py")
    assert_module_from_backend(quiz_tools_module, "app/agent/tools/quiz.py")
    assert_module_from_backend(schemas_module, "app/agent/tools/schemas.py")
    assert_module_from_backend(sys.modules[__name__], "tests/agent/test_sdk_quiz_contract.py")

    purelib = Path(sysconfig.get_path("purelib")).resolve()
    expected_purelib = Path(
        sys.prefix, "lib",
        f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages",
    ).resolve()
    assert purelib == expected_purelib
    assert Path(openai_adapter.__file__).resolve() == (
        purelib / "langchain_openai" / "chat_models" / "base.py"
    )
    assert Path(openai.__file__).resolve() == purelib / "openai" / "__init__.py"
    assert Path(openai_base_client.__file__).resolve() == (
        purelib / "openai" / "_base_client.py"
    )


def test_four_option_and_answer_rules_are_the_business_layer():
    """The wire schema stays plain; the strict rules live in the model."""
    validated = QuizQuestionPersist(**PERSIST_ARGS)
    assert validated.options == QUESTION_OPTIONS
    assert validated.answer == QUESTION_ANSWER

    # Narrow failures: pydantic's, and each about the field under test.
    with pytest.raises(ValidationError) as short_options:
        QuizQuestionPersist(**THREE_OPTION_ARGS)
    assert type(short_options.value) is ValidationError
    option_locs = [error["loc"] for error in short_options.value.errors()]
    assert option_locs == [("options",)], option_locs

    with pytest.raises(ValidationError) as bad_answer:
        QuizQuestionPersist(**{**PERSIST_ARGS, "answer": "E"})
    assert type(bad_answer.value) is ValidationError
    answer_locs = [error["loc"] for error in bad_answer.value.errors()]
    assert answer_locs == [("answer",)], answer_locs

    options_schema = EXPECTED_TOOL_SCHEMAS[1]["parameters"]["properties"]["options"]
    assert options_schema == {"items": {"type": "string"}, "type": "array"}
    assert EXPECTED_TOOL_SCHEMAS[1]["parameters"]["properties"]["answer"] == {"type": "string"}
    assert "enum" not in EXPECTED_TOOL_SCHEMAS[1]["parameters"]["properties"]["answer"]


# ===========================================================================
# Prompt-oracle mechanism checks (P2).
#
# IMPORTANT: these three cases are NOT SDK, provider or full agent-loop
# evidence. They are pure test logic over synthetic parsed bodies: no node is
# run, no model is called, no production file is edited and no factory is
# patched. The real SDK business cases earlier in this file remain the healthy
# execution path and are not replaced or weakened by anything here.
#
# Each case proves exactly two things, in this order:
#   1. the test-owned literal expectation ACCEPTS the healthy text the product
#      really renders today. The healthy side is read from the frozen
#      production constant named by the task's mutation matrix, imported here
#      and only here -- never from the golden, never from an observed request.
#   2. the same complete comparator REJECTS an in-memory mutant that differs
#      only at that one instruction/context marker, with the roles, the model
#      and every other request field left untouched.
#
# Three green meta cases are not proof that the literals are independent and
# are not offered as such.
# ===========================================================================

_ORACLE_MUTANT = " SYNTHETIC-ORACLE-MUTANT"

_QUIZ_ORACLE_CASES = [
    ("Q-BASE", "You are a quiz question generator for an exam coach."),
    (
        "Q-CONTEXT",
        "GROUND your questions strictly in these source chunks. Do NOT invent facts not",
    ),
    ("Q-AGENT", "You are a study coach quiz generator."),
]


@pytest.mark.parametrize("case_id,marker", _QUIZ_ORACLE_CASES)
def test_oracle_quiz_prompt_accepts_healthy_and_rejects_mutant(case_id, marker):
    """Healthy accepted, in-memory mutant rejected, by this file's own literals."""
    # Imported here and nowhere else: the healthy candidate only.
    from app.agent.quiz_master_agent import _AGENT_SYSTEM_PROMPT
    from app.agent.tools.quiz import _GROUNDED_CONTEXT_TEMPLATE, _QUIZ_PROMPT_BASE

    if case_id == "Q-AGENT":
        healthy = _AGENT_SYSTEM_PROMPT.format(today=TODAY_ISO)
        expected = expected_agent_system_prompt()
        kind = "agent"
    else:
        # Q-BASE and Q-CONTEXT share one non-empty grounded render and differ
        # only in which single marker the mutant is inserted after.
        chunk_lines = "\n".join(
            f"[{i + 1}] {c.get('content', '')}" for i, c in enumerate(SYNTHETIC_CHUNKS)
        )
        healthy = _QUIZ_PROMPT_BASE.format(
            n=1,
            topic_name=QUESTION_TOPIC,
            difficulty="medium",
            context_section=_GROUNDED_CONTEXT_TEMPLATE.format(chunks=chunk_lines),
        )
        expected = expected_generate_prompt(
            topic_name=QUESTION_TOPIC, difficulty="medium", n=1,
            context_chunks=SYNTHETIC_CHUNKS,
        )
        kind = "plain"

    def body_for(text):
        if kind == "agent":
            return {"messages": [
                {"role": "system", "content": text},
                {"role": "user", "content": QUIZ_USER_MSG},
            ]}
        return {"messages": [{"role": "user", "content": text}]}

    def compare(text):
        if kind == "agent":
            assert_agent_request(body_for(text), user_msg=QUIZ_USER_MSG, rounds=[])
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
