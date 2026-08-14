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
