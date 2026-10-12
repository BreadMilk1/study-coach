"""Cut ④ — Graph topology with memory_hydrator (entry) + memory_writer (exit).

These tests prove the new graph wiring:
    START → memory_hydrator → router → {tutor|quiz_stub|plan_stub} → judge → memory_writer → END

memory_hydrator/writer are injected via `RunnableConfig.configurable` (same
pattern as judge_llm in P2.1-②) — when absent, the nodes no-op so existing
P2.0/P2.1-①/② tests stay green without any rewrite.
"""
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.agent.graph import build_graph
from app.agent.memory_updater import build_memory_hydrator, build_memory_writer
from app.db.models import Base
from app.db.repositories import (
    GoalRepository,
    MasteryRepository,
    MistakeRepository,
    TopicRepository,
    UserRepository,
)


class StubLLM:
    async def astream(self, messages, **_kwargs):
        yield AIMessageChunk(content="answer")


class StubRetriever:
    def __init__(self, chunks=None):
        self.chunks = chunks or []

    def search(self, query, top_k=5):
        return self.chunks[:top_k]


@pytest.fixture
def session():
    # LangGraph runs nodes in worker threads. in-memory SQLite needs StaticPool +
    # check_same_thread=False so all "connections" share the same backing DB.
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,
    )
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def _seed_user_with_topic(session):
    users = UserRepository(session)
    goals = GoalRepository(session)
    topics = TopicRepository(session)
    user = users.get_or_create("fp-graph")
    goal = goals.create(user_id=user.id, title="X")
    topic = topics.create(goal_id=goal.id, name="HyDE")
    return user, topic


async def test_memory_hydrator_populates_state_when_configured(session):
    user, topic = _seed_user_with_topic(session)
    mastery = MasteryRepository(session)
    mistakes = MistakeRepository(session)
    mastery.upsert(user_id=user.id, topic_id=topic.id, score=0.7)
    hydrator = build_memory_hydrator(mastery_repo=mastery, mistake_repo=mistakes)

    graph = build_graph(retriever=StubRetriever(), llm=StubLLM())

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="What is HyDE?")], "user_id": user.id},
        config={"configurable": {"memory_hydrator": hydrator}},
    )

    assert result.get("mastery_scores") == {"HyDE": pytest.approx(0.7)}
    # Tutor still produced an AIMessage; graph still exits cleanly.
    assert any(isinstance(m, AIMessage) for m in result["messages"])


async def test_memory_writer_drains_pending_mastery_delta_on_tutor_path(session):
    user, topic = _seed_user_with_topic(session)
    mastery = MasteryRepository(session)
    mistakes = MistakeRepository(session)
    writer = build_memory_writer(mastery_repo=mastery, mistake_repo=mistakes)

    graph = build_graph(retriever=StubRetriever(), llm=StubLLM())

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="What is HyDE?")],
            "user_id": user.id,
            "pending_mastery_delta": {topic.id: 0.4},
        },
        config={"configurable": {"memory_writer": writer}},
    )

    assert mastery.get_for_user(user.id) == {"HyDE": pytest.approx(0.4)}
    assert result.get("pending_mastery_delta") == {}


async def test_quiz_stub_path_also_runs_memory_writer(session):
    """Quiz/Plan stub branches must pass through memory_writer too (uniform exit)."""
    user, topic = _seed_user_with_topic(session)
    mastery = MasteryRepository(session)
    mistakes = MistakeRepository(session)
    writer = build_memory_writer(mastery_repo=mastery, mistake_repo=mistakes)

    graph = build_graph(retriever=StubRetriever(), llm=StubLLM())

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="quiz me on RAG")],
            "user_id": user.id,
            "pending_mastery_delta": {topic.id: 0.3},
        },
        config={"configurable": {"memory_writer": writer}},
    )

    assert mastery.get_for_user(user.id) == {"HyDE": pytest.approx(0.3)}
    assert result.get("pending_mastery_delta") == {}
    # Quiz branch still emitted the stub message (P2.1-① contract).
    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert ai_msgs and "P2.1-④" in ai_msgs[-1].content


# ---------------------------------------------------------------------------
# D3 RED — production judge policy across the memory boundary and turns.
#
# Built via `app.api.deps.get_graph` (the production entry). RED stage: the
# legacy graph neutral-passes invalid judge payloads (so the memory writer
# drains pending state on a "pass") and carries judge-derived state
# (retry_count, quiz_action) across turns. Only LLMs/retriever are faked; the
# memory writer runs against a real tmp SQLite session.
# ---------------------------------------------------------------------------

import socket
from datetime import datetime

from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import func, select

from app.agent.planner import build_planner
from app.agent.quiz_master import build_quiz_master
from app.agent.tools.plan import update_study_plan
from app.agent.tools.schemas import Milestone
from app.api.deps import get_graph
from app.db.models import Mistake
from app.db.repositories import (
    GoalRepository,
    PlanRepository,
    QuestionRepository,
)

_UNAVAILABLE_SUFFIX = (
    "\n\n---\n⚠️ Self-check incomplete: this content was kept, but the "
    "quality check could not be completed."
)
_PROSE_VERDICT = "The answer looks well grounded and clearly cited."
_WEAK = (
    '{"relevance":2,"accuracy":2,"citation_quality":2,'
    '"accessibility":2,"example_quality":2,"learner_level_fit":2,'
    '"reasoning":"Ungrounded."}'
)
_PASS = (
    '{"relevance":5,"accuracy":5,"citation_quality":4,'
    '"accessibility":4,"example_quality":5,"learner_level_fit":5,'
    '"reasoning":"Solid."}'
)

_REASONING_MARKER = "D3-PRIVATE-JUDGE-REASONING"
_PASS_WITH_MARKER = (
    '{"relevance":5,"accuracy":5,"citation_quality":4,'
    '"accessibility":4,"example_quality":5,"learner_level_fit":5,'
    f'"reasoning":"{_REASONING_MARKER} ok"}}'
)
_WEAK_WITH_MARKER = (
    '{"relevance":2,"accuracy":2,"citation_quality":2,'
    '"accessibility":2,"example_quality":2,"learner_level_fit":2,'
    f'"reasoning":"{_REASONING_MARKER} ungrounded"}}'
)


class ScriptedJudgeLLM:
    """Judge stub: each script entry is an AIMessage or an exception instance."""

    def __init__(self, script):
        self.script = list(script)
        self.call_count = 0

    async def ainvoke(self, _messages, **_kwargs):
        self.call_count += 1
        item = self.script[min(self.call_count - 1, len(self.script) - 1)]
        if isinstance(item, BaseException):
            raise item
        return item


class MultiAnswerLLM:
    """Tutor LLM consuming one answer per astream call, across turns."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = 0

    async def astream(self, _messages, **_kwargs):
        idx = min(self.calls, len(self.answers) - 1)
        self.calls += 1
        yield AIMessageChunk(content=self.answers[idx])


@pytest.fixture
def deny_network(monkeypatch):
    """Fail immediately on any real DNS / outbound socket attempt."""

    def deny(*_args, **_kwargs):
        raise AssertionError("D3 RED test attempted real network access")

    monkeypatch.setattr(socket, "getaddrinfo", deny)
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", deny)


def test_network_guard_covers_real_socket_surface(deny_network):
    probe = socket.socket()
    try:
        with pytest.raises(AssertionError):
            socket.getaddrinfo("localhost", 80)
        with pytest.raises(AssertionError):
            probe.connect(("127.0.0.1", 80))
        with pytest.raises(AssertionError):
            probe.connect_ex(("127.0.0.1", 80))
        with pytest.raises(AssertionError):
            socket.create_connection(("127.0.0.1", 80))
    finally:
        probe.close()


def _seed_question(session, topic):
    return QuestionRepository(session).create(
        topic_id=topic.id,
        prompt="What does HyDE rewrite?",
        options_json=["A) Queries", "B) Documents", "C) Embeddings", "D) Answers"],
        answer="A",
        explanation="HyDE rewrites the user query.",
    )


def _mistake_count(session) -> int:
    return int(session.scalar(select(func.count()).select_from(Mistake)) or 0)


def _legal_pending_mistake(question) -> dict:
    return {
        "question_id": question.id,
        "user_answer": "B",
        "srs_due_at": datetime(2030, 1, 1),
        "srs_interval_days": 3,
        "srs_ease": 2.5,
    }


async def test_production_judge_unavailable_skips_memory_writer_and_clears_pending(
    session, deny_network
):
    user, topic = _seed_user_with_topic(session)
    question = _seed_question(session, topic)
    mastery = MasteryRepository(session)
    writer = build_memory_writer(
        mastery_repo=mastery, mistake_repo=MistakeRepository(session)
    )
    judge = ScriptedJudgeLLM([AIMessage(content=_PROSE_VERDICT)])
    graph = get_graph(retriever=StubRetriever(), llm=StubLLM(), checkpointer=None)

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="What is HyDE?")],
            "user_id": user.id,
            "pending_mastery_delta": {topic.id: 0.4},
            "pending_mistake": _legal_pending_mistake(question),
        },
        config={
            "configurable": {
                "memory_writer": writer,
                "judge_llm": judge,
            }
        },
    )

    # A judge fault must not execute the MemoryWriter: neither the mastery
    # delta nor the legally-queued mistake may land.
    assert mastery.get_for_user(user.id) == {}
    assert _mistake_count(session) == 0
    # ...and the pending state is cleared, not left to leak into a later turn.
    assert result.get("pending_mastery_delta") == {}
    assert result.get("pending_mistake") is None
    assert result.get("judge_score") is None
    assert result.get("degraded", False) is True
    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert ai_msgs[-1].content == "answer" + _UNAVAILABLE_SUFFIX


async def test_production_unavailable_then_clean_turn_leaves_no_stale_pending(
    session, deny_network
):
    user, topic = _seed_user_with_topic(session)
    question = _seed_question(session, topic)
    mastery = MasteryRepository(session)
    writer = build_memory_writer(
        mastery_repo=mastery, mistake_repo=MistakeRepository(session)
    )
    judge = ScriptedJudgeLLM([AIMessage(content=_PROSE_VERDICT), AIMessage(content=_PASS)])
    llm = MultiAnswerLLM(["answer one", "answer two"])
    graph = get_graph(
        retriever=StubRetriever(), llm=llm, checkpointer=InMemorySaver()
    )
    config = {
        "configurable": {
            "thread_id": "d3-no-leak",
            "memory_writer": writer,
            "judge_llm": judge,
        }
    }

    first = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="What is HyDE?")],
            "user_id": user.id,
            "pending_mastery_delta": {topic.id: 0.4},
            "pending_mistake": _legal_pending_mistake(question),
        },
        config=config,
    )
    assert mastery.get_for_user(user.id) == {}
    assert _mistake_count(session) == 0
    assert first.get("pending_mastery_delta") == {}

    second = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="What is BM25?")],
            "user_id": user.id,
        },
        config=config,
    )
    # The next turn's valid pass must not resurrect or drain stale pending data.
    assert mastery.get_for_user(user.id) == {}
    assert _mistake_count(session) == 0
    assert second.get("judge_score") is not None


async def test_production_new_turn_resets_retry_budget(deny_network):
    llm = MultiAnswerLLM(
        ["weak turn1", "better turn1", "weak t2a", "weak t2b", "weak t2c"]
    )
    judge = ScriptedJudgeLLM([
        AIMessage(content=_WEAK),
        AIMessage(content=_PASS),
        AIMessage(content=_WEAK),
        AIMessage(content=_WEAK),
        AIMessage(content=_WEAK),
    ])
    graph = get_graph(
        retriever=StubRetriever(), llm=llm, checkpointer=InMemorySaver()
    )
    config = {"configurable": {"thread_id": "d3-reset", "judge_llm": judge}}

    await graph.ainvoke(
        {"messages": [HumanMessage(content="Explain HyDE")]},
        config=config,
    )
    assert llm.calls == 2  # weak -> retry -> pass inside turn 1

    result2 = await graph.ainvoke(
        {"messages": [HumanMessage(content="Explain BM25")]},
        config=config,
    )

    # Turn 2 must start from a reset retry budget: weak x3 = initial + 2 retries.
    assert llm.calls == 5
    assert result2.get("retry_count", 0) == 2
    assert result2.get("degraded", False) is True


async def test_production_new_turn_clears_stale_quiz_action_before_judging(deny_network):
    """A stale GRADE marker from a previous turn must not skip the next quiz
    turn's judge (the new turn's consumer re-writes the phase)."""
    judge = ScriptedJudgeLLM([AIMessage(content=_PROSE_VERDICT)])

    async def grade_stub(_state):
        return {
            "messages": [AIMessage(content="graded outcome")],
            "citations": [],
            "quiz_action": "grade",
            "active_quiz_question_id": None,
        }

    async def silent_quiz_stub(_state):
        # Writes no quiz_action phase of its own.
        return {
            "messages": [AIMessage(content="fresh quiz question")],
            "citations": [],
        }

    graph = get_graph(
        retriever=StubRetriever(), llm=StubLLM(), checkpointer=InMemorySaver()
    )
    first_config = {
        "configurable": {
            "thread_id": "d3-stale-phase",
            "judge_llm": judge,
            "quiz_master": grade_stub,
        }
    }
    first = await graph.ainvoke(
        {"messages": [HumanMessage(content="quiz me on HyDE")]},
        config=first_config,
    )
    # Deterministic GRADE turns stay explicitly skipped (no judge call).
    assert judge.call_count == 0

    second = await graph.ainvoke(
        {"messages": [HumanMessage(content="quiz me on BM25")]},
        config={
            "configurable": {
                "thread_id": "d3-stale-phase",
                "judge_llm": judge,
                "quiz_master": silent_quiz_stub,
            }
        },
    )

    # The new turn must judge its candidate instead of honoring last turn's
    # GRADE marker: with an unavailable judge that means kept content + suffix.
    assert judge.call_count == 1
    assert second.get("judge_score") is None
    ai_msgs = [m for m in second["messages"] if isinstance(m, AIMessage)]
    assert ai_msgs[-1].content == "fresh quiz question" + _UNAVAILABLE_SUFFIX


async def test_production_new_turn_clears_stale_plan_action_before_judging(deny_network):
    """A stale CHECK-IN marker from a previous turn must not skip the next plan
    turn's judge (the new turn's consumer re-writes the phase)."""
    judge = ScriptedJudgeLLM([AIMessage(content=_PROSE_VERDICT)])

    async def check_in_stub(_state):
        return {
            "messages": [AIMessage(content="checked in")],
            "citations": [],
            "plan_action": "check_in",
            "active_plan_id": "p-1",
        }

    async def silent_plan_stub(_state):
        # Writes no plan_action phase of its own.
        return {
            "messages": [AIMessage(content="fresh plan")],
            "citations": [],
        }

    graph = get_graph(
        retriever=StubRetriever(), llm=StubLLM(), checkpointer=InMemorySaver()
    )
    first = await graph.ainvoke(
        {"messages": [HumanMessage(content="进度怎么样了")]},
        config={
            "configurable": {
                "thread_id": "d3-stale-plan",
                "judge_llm": judge,
                "planner": check_in_stub,
            }
        },
    )
    # Deterministic CHECK-IN turns stay explicitly skipped (no judge call).
    assert judge.call_count == 0

    second = await graph.ainvoke(
        {"messages": [HumanMessage(content="帮我做学习计划 on BM25")]},
        config={
            "configurable": {
                "thread_id": "d3-stale-plan",
                "judge_llm": judge,
                "planner": silent_plan_stub,
            }
        },
    )

    assert judge.call_count == 1
    assert second.get("judge_score") is None
    ai_msgs = [m for m in second["messages"] if isinstance(m, AIMessage)]
    assert ai_msgs[-1].content == "fresh plan" + _UNAVAILABLE_SUFFIX


async def test_legacy_build_graph_quiz_grade_skip_keeps_1_0_and_drains_memory(
    session, deny_network
):
    """Legacy control: the legacy graph mints 1.0 on deterministic GRADE turns
    and still drains pending through the MemoryWriter. The production graph
    must not (see the production skip tests below)."""
    user, topic = _seed_user_with_topic(session)
    judge = ScriptedJudgeLLM([AIMessage(content=_PROSE_VERDICT)])

    async def grade_stub(_state):
        return {
            "messages": [AIMessage(content="graded outcome")],
            "citations": [],
            "quiz_action": "grade",
            "active_quiz_question_id": None,
        }

    mastery = MasteryRepository(session)
    writer = build_memory_writer(
        mastery_repo=mastery, mistake_repo=MistakeRepository(session)
    )
    graph = build_graph(retriever=StubRetriever(), llm=StubLLM())

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="quiz me on HyDE")],
            "user_id": user.id,
            "pending_mastery_delta": {topic.id: 0.5},
        },
        config={
            "configurable": {
                "memory_writer": writer,
                "judge_llm": judge,
                "quiz_master": grade_stub,
            }
        },
    )

    assert judge.call_count == 0
    assert result.get("judge_score") == 1.0
    assert mastery.get_for_user(user.id) == {"HyDE": pytest.approx(0.5)}


async def test_production_quiz_grade_skip_scores_none_and_writer_still_runs(
    session, deny_network
):
    """A real deterministic quiz GRADE turn is explicitly skipped in production
    (no judge call, no fabricated 1.0) while the deterministic mastery update
    and the MemoryWriter drain both still run."""
    user, topic = _seed_user_with_topic(session)
    question = _seed_question(session, topic)
    mastery = MasteryRepository(session)
    quiz_master = build_quiz_master(
        llm=StubLLM(),
        topic_repo=TopicRepository(session),
        question_repo=QuestionRepository(session),
        mistake_repo=MistakeRepository(session),
        mastery_repo=mastery,
        goal_repo=GoalRepository(session),
        retriever=StubRetriever(),
    )
    writer = build_memory_writer(
        mastery_repo=mastery, mistake_repo=MistakeRepository(session)
    )
    judge = ScriptedJudgeLLM([AIMessage(content=_PROSE_VERDICT)])
    graph = get_graph(retriever=StubRetriever(), llm=StubLLM(), checkpointer=None)

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="A")],
            "user_id": user.id,
            "active_quiz_question_id": question.id,
            "pending_mastery_delta": {topic.id: 0.25},
        },
        config={
            "configurable": {
                "quiz_master": quiz_master,
                "judge_llm": judge,
                "memory_writer": writer,
            }
        },
    )

    assert judge.call_count == 0
    assert result.get("judge_score") is None
    # Deterministic grade (+0.1 for the correct answer) plus the writer drain
    # (0.25) both landed.
    assert mastery.get_for_user(user.id) == {"HyDE": pytest.approx(0.35)}
    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert ai_msgs[-1].content.startswith("✓ Correct!")


class _SilentCheckInLLM:
    """Planner LLM stub returning an empty body: the deterministic CHECK-IN
    keeps the existing plan (Auto-adjust skipped) and writes its phase."""

    async def ainvoke(self, _messages, **_kwargs):
        return AIMessage(content="")


async def test_production_plan_check_in_skip_scores_none_and_writer_still_runs(
    session, deny_network
):
    """A real deterministic plan CHECK-IN turn is explicitly skipped in
    production (no judge call, no fabricated 1.0) while the MemoryWriter drain
    still runs."""
    user, topic = _seed_user_with_topic(session)
    plan_repo = PlanRepository(session)
    # The planner resolves the plan via the first active goal; seed it there.
    goal = GoalRepository(session).list_active_for_user(user.id)[0]
    patch = update_study_plan(
        goal_id=goal.id,
        milestones=[
            Milestone(title="Read HyDE", due_at="2026-05-30", done=False, topic="HyDE")
        ],
        plan_repo=plan_repo,
    )
    planner = build_planner(
        llm=_SilentCheckInLLM(),
        plan_repo=plan_repo,
        goal_repo=GoalRepository(session),
        mastery_repo=MasteryRepository(session),
        mistake_repo=MistakeRepository(session),
        retriever=None,
    )
    mastery = MasteryRepository(session)
    writer = build_memory_writer(
        mastery_repo=mastery, mistake_repo=MistakeRepository(session)
    )
    judge = ScriptedJudgeLLM([AIMessage(content=_PROSE_VERDICT)])
    graph = get_graph(retriever=StubRetriever(), llm=StubLLM(), checkpointer=None)

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="进度怎么样了")],
            "user_id": user.id,
            "active_plan_id": patch.plan_id,
            "pending_mastery_delta": {topic.id: 0.2},
        },
        config={
            "configurable": {
                "planner": planner,
                "judge_llm": judge,
                "memory_writer": writer,
            }
        },
    )

    assert judge.call_count == 0
    assert result.get("judge_score") is None
    assert mastery.get_for_user(user.id) == {"HyDE": pytest.approx(0.2)}
    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert ai_msgs[-1].content.startswith("📊 **Progress Check-in**")


async def test_production_unavailable_replaces_latest_candidate_without_duplicate(
    deny_network,
):
    """The unavailable update reuses the latest candidate's message slot: the
    final state carries exactly one assistant message whose content is the
    candidate plus the fixed warning — no duplicated candidate message."""
    llm = MultiAnswerLLM(["only candidate"])
    judge = ScriptedJudgeLLM([AIMessage(content=_PROSE_VERDICT)])
    graph = get_graph(retriever=StubRetriever(), llm=llm, checkpointer=None)

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="What is HyDE?")]},
        config={"configurable": {"judge_llm": judge}},
    )

    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert [m.content for m in ai_msgs] == ["only candidate" + _UNAVAILABLE_SUFFIX]
    assert result.get("judge_score") is None


async def test_production_weak_then_fault_checkpoint_concatenates_attempts(
    deny_network,
):
    """Weak verdict -> retry -> judge fault: the checkpointed turn keeps both
    streamed attempts in order and replaces only the latest candidate with the
    warning — no lost attempt, no duplicated candidate, no regeneration.

    The tutor candidates are observed directly from the real graph's
    `astream(stream_mode="updates")` — their actual message ids and public
    metadata fields — and matched against the final checkpoint state. The
    current Tutor candidates carry empty response_metadata/additional_kwargs;
    only that current shape is asserted (no non-empty provider metadata is
    claimed or tested).
    """
    llm = MultiAnswerLLM(["first attempt", "second answer"])
    judge = ScriptedJudgeLLM(
        [AIMessage(content=_WEAK), TimeoutError("judge socket marker")]
    )
    graph = get_graph(
        retriever=StubRetriever(), llm=llm, checkpointer=InMemorySaver()
    )
    config = {"configurable": {"thread_id": "d3-concat", "judge_llm": judge}}

    observed_candidates = []
    async for update in graph.astream(
        {"messages": [HumanMessage(content="What is HyDE?")]},
        config=config,
        stream_mode="updates",
    ):
        for message in (update.get("tutor") or {}).get("messages", []):
            if isinstance(message, AIMessage):
                observed_candidates.append({
                    "id": message.id,
                    "content": message.content,
                    "response_metadata": dict(
                        getattr(message, "response_metadata", {}) or {}
                    ),
                    "additional_kwargs": dict(
                        getattr(message, "additional_kwargs", {}) or {}
                    ),
                })

    # The judge fault must complete the turn, not interrupt the stream.
    first, second = observed_candidates
    assert first["id"] != second["id"]
    # Current Tutor candidates: empty metadata shape, observed directly.
    assert first["response_metadata"] == {} and first["additional_kwargs"] == {}
    assert second["response_metadata"] == {} and second["additional_kwargs"] == {}

    final_state = (await graph.aget_state(config)).values
    assert llm.calls == 2
    ai_msgs = [m for m in final_state["messages"] if isinstance(m, AIMessage)]
    # Both attempts stay in order; the latest candidate is replaced in place
    # with the warning under its original id — no third message.
    assert [m.content for m in ai_msgs] == [
        "first attempt",
        "second answer" + _UNAVAILABLE_SUFFIX,
    ]
    assert [m.id for m in ai_msgs] == [first["id"], second["id"]]
    assert ai_msgs[0].response_metadata == first["response_metadata"]
    assert ai_msgs[0].additional_kwargs == first["additional_kwargs"]
    assert ai_msgs[1].response_metadata == second["response_metadata"]
    assert ai_msgs[1].additional_kwargs == second["additional_kwargs"]
    assert final_state.get("judge_score") is None
    assert final_state.get("degraded", False) is True


async def test_production_valid_free_reasoning_absent_from_all_checkpoints(deny_network):
    """The Judge's free reasoning must not land in any checkpoint's
    `judge_reasoning` state field — not the final state, not any prior step.

    The router clears `judge_reasoning` on every new turn; the judge node must
    not write the model's free reasoning back. Walking `aget_state_history`
    proves the marker never reaches a saved snapshot, not just the final one.
    """
    judge = ScriptedJudgeLLM([AIMessage(content=_PASS_WITH_MARKER)])
    graph = get_graph(
        retriever=StubRetriever(),
        llm=StubLLM(),
        checkpointer=InMemorySaver(),
    )
    config = {
        "configurable": {
            "thread_id": "d3-no-free-reasoning",
            "judge_llm": judge,
        }
    }

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="What is HyDE?")]},
        config=config,
    )

    assert result.get("judge_status") == "valid"
    # The final state carries no marker: the field is empty per contract.
    assert _REASONING_MARKER not in (result.get("judge_reasoning") or "")
    # No prior checkpoint stored the marker either.
    leaked = []
    async for snapshot in graph.aget_state_history(config):
        values = snapshot.values or {}
        if _REASONING_MARKER in (values.get("judge_reasoning") or ""):
            leaked.append(snapshot.metadata.get("step"))
    assert leaked == [], (
        f"Free reasoning leaked into checkpoints: {leaked}"
    )


async def test_production_weak_retry_free_reasoning_absent_from_all_checkpoints(
    deny_network,
):
    """Even on the weak→retry→pass branch the model reasoning stays out.

    The prior bug surfaced the free reasoning not just in the final state but
    in the early-step checkpoint created when the judge decided to retry.
    """
    judge = ScriptedJudgeLLM(
        [AIMessage(content=_WEAK_WITH_MARKER), AIMessage(content=_PASS_WITH_MARKER)]
    )
    graph = get_graph(
        retriever=StubRetriever(),
        llm=MultiAnswerLLM(["first attempt", "second attempt"]),
        checkpointer=InMemorySaver(),
    )
    config = {
        "configurable": {
            "thread_id": "d3-no-free-reasoning-retry",
            "judge_llm": judge,
        }
    }

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="Explain HyDE")]},
        config=config,
    )

    assert result.get("judge_status") == "valid"
    leaked = []
    async for snapshot in graph.aget_state_history(config):
        values = snapshot.values or {}
        if _REASONING_MARKER in (values.get("judge_reasoning") or ""):
            leaked.append(snapshot.metadata.get("step"))
    assert leaked == [], (
        f"Free reasoning leaked into checkpoints: {leaked}"
    )


async def test_production_weak_exhausted_free_reasoning_absent_from_all_checkpoints(
    deny_network,
):
    """Weak with the budget exhausted (degrade path) still must not persist
    the free reasoning. The current bug also leaks the marker here because
    the degrade branch also writes `judge_reasoning` from the result.
    """
    judge = ScriptedJudgeLLM([
        AIMessage(content=_WEAK_WITH_MARKER),
        AIMessage(content=_WEAK_WITH_MARKER),
        AIMessage(content=_WEAK_WITH_MARKER),
    ])
    graph = get_graph(
        retriever=StubRetriever(),
        llm=MultiAnswerLLM(["a1", "a2", "a3"]),
        checkpointer=InMemorySaver(),
    )
    config = {
        "configurable": {
            "thread_id": "d3-no-free-reasoning-exhausted",
            "judge_llm": judge,
        }
    }

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="Explain HyDE")]},
        config=config,
    )

    assert result.get("degraded", False) is True
    leaked = []
    async for snapshot in graph.aget_state_history(config):
        values = snapshot.values or {}
        if _REASONING_MARKER in (values.get("judge_reasoning") or ""):
            leaked.append(snapshot.metadata.get("step"))
    assert leaked == [], (
        f"Free reasoning leaked into checkpoints: {leaked}"
    )
