"""Judge Guard graph integration tests (P2.1-② RED-2).

Wires Judge node after Tutor: pass / retry / degrade / no-judge paths.

The judge_llm is injected via LangGraph RunnableConfig (configurable={"judge_llm": ...})
so build_graph signature stays small and baseline test_graph tests keep passing
(without a judge_llm in config, the judge node short-circuits to pass).
"""

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from app.agent.graph import build_graph
from app.agent.tutor_attempt import TutorAttemptEngine


class RecordingTutorLLM:
    """Tutor LLM that records every prompt it sees, useful for retry-hint tests."""

    def __init__(self, response_texts: list[str]):
        # One response per invocation (retries consume successive entries).
        self.responses = list(response_texts)
        self.prompts: list[str] = []
        self.call_count = 0

    async def astream(self, messages, **_kwargs):
        self.prompts.append(messages[-1].content if messages else "")
        idx = min(self.call_count, len(self.responses) - 1)
        self.call_count += 1
        yield AIMessageChunk(content=self.responses[idx])


class StubJudgeLLM:
    """Judge LLM that returns a pre-baked JSON verdict on every call."""

    def __init__(self, payloads: list[str]):
        # One payload per call (so retry loop sees different scores if desired).
        self.payloads = list(payloads)
        self.call_count = 0

    async def ainvoke(self, messages, **_kwargs):
        idx = min(self.call_count, len(self.payloads) - 1)
        self.call_count += 1
        return AIMessage(content=self.payloads[idx])


class StubRetriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks

    def search(self, query: str, top_k: int = 5):
        return self.chunks[:top_k]


class RecordingTutorAttemptEngine:
    def __init__(self):
        self.calls = 0
        self.delegate = TutorAttemptEngine()

    async def answer(self, **kwargs):
        self.calls += 1
        return await self.delegate.answer(**kwargs)


_PASS = (
    '{"relevance":5,"accuracy":5,"citation_quality":4,'
    '"accessibility":4,"example_quality":5,"learner_level_fit":5,'
    '"reasoning":"Well grounded and well paced."}'
)
_WEAK = (
    '{"relevance":2,"accuracy":2,"citation_quality":2,'
    '"accessibility":2,"example_quality":2,"learner_level_fit":2,'
    '"reasoning":"Ungrounded and over-jargon."}'
)

_CHUNKS = [
    {"chunk_id": "a:1:0", "content": "HyDE rewrites queries.",
     "source": "a.pdf", "page": 1, "score": 0.9},
]


@pytest.mark.asyncio
async def test_judge_pass_path_keeps_tutor_answer_unchanged():
    retriever = StubRetriever(_CHUNKS)
    llm = RecordingTutorLLM(["HyDE is a query rewriting technique."])
    judge_llm = StubJudgeLLM([_PASS])
    graph = build_graph(retriever=retriever, llm=llm)

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="What is HyDE?")]},
        config={"configurable": {"judge_llm": judge_llm}},
    )

    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert ai_msgs[-1].content == "HyDE is a query rewriting technique."
    assert result.get("judge_score", 0.0) >= 0.6
    assert result.get("retry_count", 0) == 0
    assert result.get("degraded", False) is False
    assert llm.call_count == 1  # no retry
    assert judge_llm.call_count == 1


@pytest.mark.asyncio
async def test_judge_weak_path_retries_tutor_up_to_2_times():
    retriever = StubRetriever(_CHUNKS)
    # All three tutor attempts produce the same weak answer (worst case).
    llm = RecordingTutorLLM(["weak answer"] * 3)
    # Judge ranks every attempt weak.
    judge_llm = StubJudgeLLM([_WEAK, _WEAK, _WEAK])
    graph = build_graph(retriever=retriever, llm=llm)

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="Explain BM25")]},
        config={"configurable": {"judge_llm": judge_llm}},
    )

    # Initial attempt + 2 retries = 3 tutor calls
    assert llm.call_count == 3
    # Judge ran on each tutor attempt
    assert judge_llm.call_count == 3
    # Retry budget exhausted -> degrade
    assert result.get("retry_count", 0) == 2
    assert result.get("degraded", False) is True


@pytest.mark.asyncio
async def test_judge_degraded_answer_carries_disclaimer():
    retriever = StubRetriever(_CHUNKS)
    llm = RecordingTutorLLM(["raw tutor answer"] * 3)
    judge_llm = StubJudgeLLM([_WEAK, _WEAK, _WEAK])
    graph = build_graph(retriever=retriever, llm=llm)

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="anything")]},
        config={"configurable": {"judge_llm": judge_llm}},
    )

    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
    final = ai_msgs[-1].content
    assert "raw tutor answer" in final, "degraded answer must still surface tutor text"
    lowered = final.lower()
    assert "self-check" in lowered or "low" in lowered or "⚠" in final, (
        f"disclaimer marker missing from degraded answer: {final[:200]}"
    )


@pytest.mark.asyncio
async def test_judge_pass_after_one_retry_stops_loop():
    retriever = StubRetriever(_CHUNKS)
    llm = RecordingTutorLLM(["first attempt", "second attempt"])
    # First attempt weak, retry passes.
    judge_llm = StubJudgeLLM([_WEAK, _PASS])
    graph = build_graph(retriever=retriever, llm=llm)

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="q")]},
        config={"configurable": {"judge_llm": judge_llm}},
    )

    assert llm.call_count == 2
    assert judge_llm.call_count == 2
    assert result.get("retry_count", 0) == 1
    assert result.get("degraded", False) is False
    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert ai_msgs[-1].content == "second attempt"


@pytest.mark.asyncio
async def test_judge_retry_hint_passes_previous_score_and_weak_dims_to_tutor():
    retriever = StubRetriever(_CHUNKS)
    llm = RecordingTutorLLM(["weak", "improved"])
    judge_llm = StubJudgeLLM([_WEAK, _PASS])
    graph = build_graph(retriever=retriever, llm=llm)

    await graph.ainvoke(
        {"messages": [HumanMessage(content="q")]},
        config={"configurable": {"judge_llm": judge_llm}},
    )

    assert len(llm.prompts) == 2
    first_prompt, retry_prompt = llm.prompts
    # First call is plain build_prompt — no retry hint
    assert "previous" not in first_prompt.lower()
    # Retry call must carry score + weak dims so the LLM can self-correct
    rp_lower = retry_prompt.lower()
    assert "previous" in rp_lower
    assert "score" in rp_lower or "rating" in rp_lower
    # At least one weak dim name appears
    assert any(d in retry_prompt for d in (
        "accuracy", "citation_quality", "relevance",
        "accessibility", "example_quality", "learner_level_fit",
    ))


@pytest.mark.asyncio
async def test_judge_first_pass_success_executes_one_tutor_attempt():
    retriever = StubRetriever(_CHUNKS)
    llm = RecordingTutorLLM(["grounded answer"])
    judge_llm = StubJudgeLLM([_PASS])
    attempt_engine = RecordingTutorAttemptEngine()
    graph = build_graph(
        retriever=retriever,
        llm=llm,
        tutor_attempt_engine=attempt_engine,
    )

    await graph.ainvoke(
        {"messages": [HumanMessage(content="What is HyDE?")]},
        config={"configurable": {"judge_llm": judge_llm}},
    )

    assert attempt_engine.calls == 1


@pytest.mark.asyncio
async def test_judge_weak_path_executes_at_most_three_tutor_attempts():
    retriever = StubRetriever(_CHUNKS)
    llm = RecordingTutorLLM(["weak answer"] * 3)
    judge_llm = StubJudgeLLM([_WEAK, _WEAK, _WEAK])
    attempt_engine = RecordingTutorAttemptEngine()
    graph = build_graph(
        retriever=retriever,
        llm=llm,
        tutor_attempt_engine=attempt_engine,
    )

    await graph.ainvoke(
        {"messages": [HumanMessage(content="Explain BM25")]},
        config={"configurable": {"judge_llm": judge_llm}},
    )

    assert attempt_engine.calls == 3


# ---------------------------------------------------------------------------
# D3 RED — production judge policy through the real production entry point.
#
# These tests build the graph via `app.api.deps.get_graph`, the construction
# path the production API uses. RED stage: get_graph still builds the legacy
# graph, so every production-policy expectation below fails against legacy
# behavior (neutral 0.6 fallback on invalid payloads, stream truncation on
# judge request errors, reasoning-bearing retry hints, no completion-signal
# gate). Only the LLM and retriever are faked; no network access is allowed.
# ---------------------------------------------------------------------------

import socket

from app.api.deps import get_graph

_UNAVAILABLE_SUFFIX = (
    "\n\n---\n⚠️ Self-check incomplete: this content was kept, but the "
    "quality check could not be completed."
)
_PROSE_VERDICT = "The answer looks well grounded and clearly cited."
_REASONING_MARKER = "HIDDEN-JUDGE-REASONING"
_WEAK_WITH_REASONING = (
    '{"relevance":2,"accuracy":2,"citation_quality":2,'
    '"accessibility":2,"example_quality":2,"learner_level_fit":2,'
    f'"reasoning":"{_REASONING_MARKER} ungrounded claims"}}'
)
_MISSING_DIM_JSON = (
    '{"relevance":5,"accuracy":5,"citation_quality":4,'
    '"example_quality":5,"learner_level_fit":5}'
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


def _assert_unavailable(result, *, base_answer, llm, judge):
    """The D3 unavailable contract for one tutor candidate."""
    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert ai_msgs[-1].content == base_answer + _UNAVAILABLE_SUFFIX
    assert result.get("judge_score") is None
    assert not result.get("weak_dims")
    assert not result.get("judge_reasoning")
    assert result.get("degraded", False) is True
    # A judge fault is not a retry signal: exactly one candidate, one judge call.
    assert llm.call_count == 1
    assert judge.call_count == 1
    assert result.get("retry_count", 0) == 0


async def test_production_get_graph_judge_prose_payload_is_unavailable_not_neutral_pass(
    deny_network,
):
    retriever = StubRetriever(_CHUNKS)
    llm = RecordingTutorLLM(["HyDE is a query rewriting technique."])
    judge = ScriptedJudgeLLM([AIMessage(content=_PROSE_VERDICT)])
    graph = get_graph(retriever=retriever, llm=llm, checkpointer=None)

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="What is HyDE?")]},
        config={"configurable": {"judge_llm": judge}},
    )

    _assert_unavailable(
        result, base_answer="HyDE is a query rewriting technique.", llm=llm, judge=judge
    )


async def test_production_get_graph_judge_missing_dimension_is_unavailable(deny_network):
    retriever = StubRetriever(_CHUNKS)
    llm = RecordingTutorLLM(["answer with a missing dim"])
    judge = ScriptedJudgeLLM([AIMessage(content=_MISSING_DIM_JSON)])
    graph = get_graph(retriever=retriever, llm=llm, checkpointer=None)

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="q")]},
        config={"configurable": {"judge_llm": judge}},
    )

    _assert_unavailable(
        result, base_answer="answer with a missing dim", llm=llm, judge=judge
    )


async def test_production_get_graph_judge_request_failure_keeps_content_and_completes(
    deny_network,
):
    retriever = StubRetriever(_CHUNKS)
    llm = RecordingTutorLLM(["answer during judge outage"])
    judge = ScriptedJudgeLLM([TimeoutError("judge socket marker")])
    graph = get_graph(retriever=retriever, llm=llm, checkpointer=None)

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="q")]},
        config={"configurable": {"judge_llm": judge}},
    )

    _assert_unavailable(
        result, base_answer="answer during judge outage", llm=llm, judge=judge
    )


async def test_production_get_graph_judge_failure_does_not_regenerate(deny_network):
    retriever = StubRetriever(_CHUNKS)
    llm = RecordingTutorLLM(["only candidate"])
    judge = ScriptedJudgeLLM([TimeoutError("judge socket marker")])
    graph = get_graph(retriever=retriever, llm=llm, checkpointer=None)

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="q")]},
        config={"configurable": {"judge_llm": judge}},
    )

    assert llm.call_count == 1
    assert judge.call_count == 1
    assert result.get("retry_count", 0) == 0
    assert result.get("degraded", False) is True


async def test_production_get_graph_negative_finish_reason_blocks_scoring(deny_network):
    retriever = StubRetriever(_CHUNKS)
    llm = RecordingTutorLLM(["truncated but well scored answer"])
    judge = ScriptedJudgeLLM(
        [AIMessage(content=_PASS, response_metadata={"finish_reason": "length"})]
    )
    graph = get_graph(retriever=retriever, llm=llm, checkpointer=None)

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="q")]},
        config={"configurable": {"judge_llm": judge}},
    )

    _assert_unavailable(
        result, base_answer="truncated but well scored answer", llm=llm, judge=judge
    )


async def test_production_get_graph_judge_tool_call_blocks_scoring(deny_network):
    retriever = StubRetriever(_CHUNKS)
    llm = RecordingTutorLLM(["answer with a stray tool call"])
    judge = ScriptedJudgeLLM([
        AIMessage(
            content=_PASS,
            tool_calls=[{"name": "lookup", "args": {}, "id": "c1"}],
        )
    ])
    graph = get_graph(retriever=retriever, llm=llm, checkpointer=None)

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="q")]},
        config={"configurable": {"judge_llm": judge}},
    )

    _assert_unavailable(
        result, base_answer="answer with a stray tool call", llm=llm, judge=judge
    )


async def test_production_get_graph_valid_weak_retry_hint_excludes_judge_reasoning(
    deny_network,
):
    retriever = StubRetriever(_CHUNKS)
    llm = RecordingTutorLLM(["weak answer", "improved answer"])
    judge = ScriptedJudgeLLM([AIMessage(content=_WEAK_WITH_REASONING), AIMessage(_PASS)])
    graph = get_graph(retriever=retriever, llm=llm, checkpointer=None)

    await graph.ainvoke(
        {"messages": [HumanMessage(content="q")]},
        config={"configurable": {"judge_llm": judge}},
    )

    assert len(llm.prompts) == 2
    retry_prompt = llm.prompts[1]
    # The verified score and fixed dimension names stay in the hint...
    assert "previous" in retry_prompt.lower()
    # ...as the normalized 0..1 score (0.40), never the raw 2.0 rubric value.
    assert "0.40" in retry_prompt
    assert "2.0" not in retry_prompt
    assert any(d in retry_prompt for d in (
        "accuracy", "citation_quality", "relevance",
        "accessibility", "example_quality", "learner_level_fit",
    ))
    # ...but the judge's free reasoning never re-enters the next prompt.
    assert _REASONING_MARKER not in retry_prompt


async def test_production_get_graph_judge_program_error_still_propagates(deny_network):
    """A judge-side program bug is not an availability signal (control)."""
    retriever = StubRetriever(_CHUNKS)
    llm = RecordingTutorLLM(["answer"])
    judge = ScriptedJudgeLLM([RuntimeError("judge program bug marker")])
    graph = get_graph(retriever=retriever, llm=llm, checkpointer=None)

    with pytest.raises(RuntimeError):
        await graph.ainvoke(
            {"messages": [HumanMessage(content="q")]},
            config={"configurable": {"judge_llm": judge}},
        )


async def test_production_get_graph_judge_cancellation_still_propagates(deny_network):
    """Cancellation is a BaseException and must never become unavailable (control)."""
    import asyncio

    retriever = StubRetriever(_CHUNKS)
    llm = RecordingTutorLLM(["answer"])
    judge = ScriptedJudgeLLM([asyncio.CancelledError()])
    graph = get_graph(retriever=retriever, llm=llm, checkpointer=None)

    with pytest.raises(asyncio.CancelledError):
        await graph.ainvoke(
            {"messages": [HumanMessage(content="q")]},
            config={"configurable": {"judge_llm": judge}},
        )


async def test_legacy_build_graph_neutral_fallback_contract_unchanged(deny_network):
    """Legacy control: `build_graph` (default policy) keeps the neutral fallback.

    The RED above comes from the production entry point, not from a broken
    legacy path; this control must stay green through GREEN.
    """
    retriever = StubRetriever(_CHUNKS)
    llm = RecordingTutorLLM(["legacy answer"])
    judge = ScriptedJudgeLLM([AIMessage(content=_PROSE_VERDICT)])
    graph = build_graph(retriever=retriever, llm=llm)

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="q")]},
        config={"configurable": {"judge_llm": judge}},
    )

    assert result.get("judge_score", 0.0) == pytest.approx(0.6)
    assert result.get("degraded", False) is False
    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert ai_msgs[-1].content == "legacy answer"
    assert _UNAVAILABLE_SUFFIX not in ai_msgs[-1].content
