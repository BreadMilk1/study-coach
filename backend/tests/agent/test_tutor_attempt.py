from dataclasses import dataclass
import asyncio
import threading

import pytest
from langchain_core.messages import AIMessageChunk

from app.agent.prompt import (
    SYSTEM_INSTRUCTION,
    TutorPromptTemplate,
    build_citations,
    build_prompt,
    format_context,
)
from app.agent.tutor_attempt import (
    TutorAttemptConfig,
    TutorAttemptEngine,
    TutorEventSinkError,
)


CHUNK_A = {
    "chunk_id": "rrf:1:0",
    "content": "Reciprocal rank fusion combines ranked lists.",
    "source": "retrieval.pdf",
    "page": 1,
    "score": 0.95,
}
CHUNK_B = {
    "chunk_id": "rrf:2:0",
    "content": "RRF gives higher weight to items near the top.",
    "source": "retrieval.pdf",
    "page": 2,
    "score": 0.82,
}


class FakeRetriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = list(chunks)
        self.search_calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        self.search_calls.append((query, top_k))
        return list(self.chunks[:top_k])


class FakeStreamingLLM:
    def __init__(self, token_sequence: list[str], *, usage: dict | None = None):
        self.token_sequence = list(token_sequence)
        self.usage = usage
        self.prompts: list[str] = []

    async def astream(self, messages, **_kwargs):
        self.prompts.append(messages[-1].content if messages else "")
        for index, text in enumerate(self.token_sequence):
            chunk = AIMessageChunk(content=text)
            if self.usage is not None and index == len(self.token_sequence) - 1:
                chunk.usage_metadata = dict(self.usage)
            yield chunk


class RaisingStreamingLLM:
    async def astream(self, _messages, **_kwargs):
        raise RuntimeError("model unavailable")
        yield  # pragma: no cover


class BlockingRetriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = list(chunks)
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.search_calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        self.search_calls.append((query, top_k))
        self.started.set()
        try:
            self.release.wait(timeout=1)
            return list(self.chunks[:top_k])
        finally:
            self.finished.set()


class BlockingStreamingLLM:
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def astream(self, _messages, **_kwargs):
        self.started.set()
        await self.release.wait()
        yield AIMessageChunk(content="late answer")


class CancellationResistantStreamingLLM:
    def __init__(self):
        self.started = asyncio.Event()
        self.cancellation_seen = asyncio.Event()
        self.release = asyncio.Event()
        self.finished = asyncio.Event()

    async def astream(self, _messages, **_kwargs):
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancellation_seen.set()
            await self.release.wait()
            yield AIMessageChunk(content="late answer")
        finally:
            self.finished.set()


class SelectiveFailingSink:
    def __init__(self, *, event_type: str, status: str | None = None):
        self.event_type = event_type
        self.status = status
        self.events: list[dict] = []

    def __call__(self, event: dict) -> None:
        self.events.append(dict(event))
        if event.get("type") == self.event_type and (
            self.status is None or event.get("status") == self.status
        ):
            raise RuntimeError("event sink failed")


@dataclass
class RecordingSink:
    events: list[dict]

    def __init__(self):
        self.events = []

    def __call__(self, event: dict) -> None:
        self.events.append(dict(event))

    def of_type(self, event_type: str) -> list[dict]:
        return [event for event in self.events if event.get("type") == event_type]


@pytest.mark.asyncio
async def test_tutor_attempt_returns_exact_candidate_and_streams_tokens():
    retriever = FakeRetriever([CHUNK_A, CHUNK_B])
    sink = RecordingSink()
    llm = FakeStreamingLLM(
        ["Reciprocal ", "rank fusion [1]."],
        usage={"input_tokens": 11, "output_tokens": 4, "total_tokens": 15},
    )

    candidate = await TutorAttemptEngine().answer(
        question="What is RRF?",
        retriever=retriever,
        llm=llm,
        prompt_template=TutorPromptTemplate.production_v2(),
        event_sink=sink,
        attempt_config=TutorAttemptConfig(
            top_k=5,
            retrieval_seconds=5,
            generation_seconds=55,
        ),
    )

    assert retriever.search_calls == [("What is RRF?", 5)]
    assert candidate.answer == "Reciprocal rank fusion [1]."
    assert candidate.evidence == [CHUNK_A, CHUNK_B]
    assert candidate.citations == build_citations([CHUNK_A, CHUNK_B])
    assert candidate.formatted_context == format_context([CHUNK_A, CHUNK_B])
    assert candidate.usage == {
        "input_tokens": 11,
        "output_tokens": 4,
        "total_tokens": 15,
    }
    assert [event["text"] for event in sink.of_type("token")] == [
        "Reciprocal ",
        "rank fusion [1].",
    ]
    assert sink.of_type("citations") == [
        {"type": "citations", "citations": candidate.citations}
    ]
    assert sink.of_type("budget") == [
        {"type": "budget", "stage": "retrieval", "limit_seconds": 5},
        {"type": "budget", "stage": "generation", "limit_seconds": 55},
    ]
    citation_event = sink.of_type("citations")[0]
    generation_budget_event = sink.of_type("budget")[1]
    first_token_event = sink.of_type("token")[0]
    assert sink.events.index(citation_event) < sink.events.index(generation_budget_event)
    assert sink.events.index(generation_budget_event) < sink.events.index(first_token_event)
    assert candidate.trace
    assert sink.of_type("trace") == candidate.trace
    assert llm.prompts == [build_prompt("What is RRF?", [CHUNK_A, CHUNK_B])]


@pytest.mark.asyncio
async def test_tutor_attempt_preserves_empty_retrieval_as_valid_candidate():
    retriever = FakeRetriever([])
    sink = RecordingSink()

    candidate = await TutorAttemptEngine().answer(
        question="What is outside the corpus?",
        retriever=retriever,
        llm=FakeStreamingLLM(["I don't know."]),
        prompt_template=TutorPromptTemplate.production_v2(),
        event_sink=sink,
        attempt_config=TutorAttemptConfig.production_default(),
    )

    assert retriever.search_calls == [("What is outside the corpus?", 5)]
    assert candidate.answer == "I don't know."
    assert candidate.evidence == []
    assert candidate.citations == []
    assert candidate.formatted_context == "(no relevant sources retrieved)"
    assert sink.of_type("citations") == [{"type": "citations", "citations": []}]


@pytest.mark.asyncio
async def test_tutor_attempt_propagates_llm_exception_and_records_trace_event():
    sink = RecordingSink()

    with pytest.raises(RuntimeError, match="model unavailable"):
        await TutorAttemptEngine().answer(
            question="What is RRF?",
            retriever=FakeRetriever([CHUNK_A]),
            llm=RaisingStreamingLLM(),
            prompt_template=TutorPromptTemplate.production_v2(),
            event_sink=sink,
            attempt_config=TutorAttemptConfig.production_default(),
        )

    assert any(
        event.get("step") == "generation" and event.get("error") == "RuntimeError"
        for event in sink.of_type("trace")
    )


@pytest.mark.asyncio
async def test_tutor_attempt_marks_usage_unavailable_instead_of_zero():
    sink = RecordingSink()

    candidate = await TutorAttemptEngine().answer(
        question="What is RRF?",
        retriever=FakeRetriever([CHUNK_A]),
        llm=FakeStreamingLLM(["RRF is ranked fusion."]),
        prompt_template=TutorPromptTemplate.production_v2(),
        event_sink=sink,
        attempt_config=TutorAttemptConfig.production_default(),
    )

    assert candidate.usage == "unavailable"
    assert candidate.usage != {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


@pytest.mark.asyncio
async def test_tutor_attempt_does_not_finalize_after_retrieval_deadline():
    retriever = BlockingRetriever([CHUNK_A])
    llm = FakeStreamingLLM(["answer"])
    sink = RecordingSink()
    task = asyncio.create_task(
        TutorAttemptEngine().answer(
            question="What is RRF?",
            retriever=retriever,
            llm=llm,
            prompt_template=TutorPromptTemplate.production_v2(),
            event_sink=sink,
            attempt_config=TutorAttemptConfig(
                top_k=5,
                retrieval_seconds=0.01,
                generation_seconds=55,
            ),
        )
    )
    await asyncio.to_thread(retriever.started.wait, 1)
    try:
        with pytest.raises(asyncio.TimeoutError):
            await task
    finally:
        retriever.release.set()
    assert await asyncio.to_thread(retriever.finished.wait, 1)
    await asyncio.sleep(0)

    assert task.done()
    assert isinstance(task.exception(), asyncio.TimeoutError)
    assert not sink.of_type("token")
    assert not sink.of_type("citations")
    assert llm.prompts == []
    assert not any(
        event.get("step") == "retrieval" and event.get("status") == "completed"
        for event in sink.of_type("trace")
    )
    assert not any(event.get("step") == "generation" for event in sink.of_type("trace"))
    assert any(
        event.get("step") == "retrieval"
        and event.get("status") == "failed"
        and event.get("error") == "TimeoutError"
        for event in sink.of_type("trace")
    )


@pytest.mark.asyncio
async def test_tutor_attempt_does_not_finalize_after_generation_deadline():
    llm = BlockingStreamingLLM()
    sink = RecordingSink()
    task = asyncio.create_task(
        TutorAttemptEngine().answer(
            question="What is RRF?",
            retriever=FakeRetriever([CHUNK_A]),
            llm=llm,
            prompt_template=TutorPromptTemplate.production_v2(),
            event_sink=sink,
            attempt_config=TutorAttemptConfig(
                top_k=5,
                retrieval_seconds=5,
                generation_seconds=0.01,
            ),
        )
    )
    await llm.started.wait()
    with pytest.raises(asyncio.TimeoutError):
        await task

    assert not sink.of_type("token")
    assert any(
        event.get("step") == "generation"
        and event.get("status") == "failed"
        and event.get("error") == "TimeoutError"
        for event in sink.of_type("trace")
    )


@pytest.mark.asyncio
async def test_tutor_attempt_hard_generation_deadline_does_not_wait_for_cancel_resistant_llm():
    llm = CancellationResistantStreamingLLM()
    sink = RecordingSink()
    task = asyncio.create_task(
        TutorAttemptEngine().answer(
            question="What is RRF?",
            retriever=FakeRetriever([CHUNK_A]),
            llm=llm,
            prompt_template=TutorPromptTemplate.production_v2(),
            event_sink=sink,
            attempt_config=TutorAttemptConfig(
                top_k=5,
                retrieval_seconds=5,
                generation_seconds=0.01,
            ),
        )
    )
    await llm.started.wait()

    done, _ = await asyncio.wait({task}, timeout=0.2)
    try:
        assert task in done, "generation deadline did not finish the engine promptly"
        assert isinstance(task.exception(), asyncio.TimeoutError)
    finally:
        llm.release.set()
        try:
            await asyncio.wait_for(llm.finished.wait(), timeout=1)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)

    assert llm.cancellation_seen.is_set()
    assert not sink.of_type("token")
    assert not any(
        event.get("step") == "generation" and event.get("status") == "completed"
        for event in sink.of_type("trace")
    )


@pytest.mark.asyncio
async def test_tutor_attempt_parent_cancellation_does_not_wait_for_cancel_resistant_llm():
    llm = CancellationResistantStreamingLLM()
    sink = RecordingSink()
    task = asyncio.create_task(
        TutorAttemptEngine().answer(
            question="What is RRF?",
            retriever=FakeRetriever([CHUNK_A]),
            llm=llm,
            prompt_template=TutorPromptTemplate.production_v2(),
            event_sink=sink,
            attempt_config=TutorAttemptConfig.production_default(),
        )
    )
    await llm.started.wait()
    task.cancel()

    done, _ = await asyncio.wait({task}, timeout=0.2)
    finished_promptly = task in done
    try:
        if finished_promptly:
            assert task.cancelled()
    finally:
        llm.release.set()
        try:
            await asyncio.wait_for(llm.finished.wait(), timeout=1)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)

    assert finished_promptly, "parent cancellation waited for the child stream"
    assert task.cancelled()
    assert llm.cancellation_seen.is_set()
    assert not sink.of_type("token")
    assert not any(
        event.get("step") == "generation" and event.get("status") == "completed"
        for event in sink.of_type("trace")
    )


@pytest.mark.asyncio
async def test_tutor_attempt_raises_dedicated_error_when_token_sink_fails():
    sink = SelectiveFailingSink(event_type="token")

    with pytest.raises(Exception) as exc_info:
        await TutorAttemptEngine().answer(
            question="What is RRF?",
            retriever=FakeRetriever([CHUNK_A]),
            llm=FakeStreamingLLM(["answer"]),
            prompt_template=TutorPromptTemplate.production_v2(),
            event_sink=sink,
            attempt_config=TutorAttemptConfig.production_default(),
        )

    assert isinstance(exc_info.value, TutorEventSinkError)
    assert isinstance(exc_info.value.__cause__, RuntimeError)


@pytest.mark.asyncio
async def test_tutor_attempt_preserves_original_llm_error_when_failed_trace_sink_fails():
    sink = SelectiveFailingSink(event_type="trace", status="failed")

    with pytest.raises(RuntimeError, match="model unavailable"):
        await TutorAttemptEngine().answer(
            question="What is RRF?",
            retriever=FakeRetriever([CHUNK_A]),
            llm=RaisingStreamingLLM(),
            prompt_template=TutorPromptTemplate.production_v2(),
            event_sink=sink,
            attempt_config=TutorAttemptConfig.production_default(),
        )


def test_production_v2_prompt_template_is_byte_for_byte_compatible():
    expected_prompt = (
        b"You are a study coach answering questions based ONLY on the provided sources. "
        b"Cite each fact you use with [N] referring to the source list. "
        b"If the sources do not contain the answer, say you don't know "
        b"\xe2\x80\x94 do not fabricate.\n\n"
        b"Sources:\n"
        b"[1] retrieval.pdf p.1: Reciprocal rank fusion combines ranked lists.\n\n"
        b"[2] retrieval.pdf p.2: RRF gives higher weight to items near the top.\n\n"
        b"Question: What is RRF?"
    )
    assert SYSTEM_INSTRUCTION.encode("utf-8") == expected_prompt.split(b"\n\nSources:")[0]
    assert (
        TutorPromptTemplate.production_v2()
        .render("What is RRF?", [CHUNK_A, CHUNK_B])
        .encode("utf-8")
        == expected_prompt
    )
    assert build_prompt("What is RRF?", [CHUNK_A, CHUNK_B]).encode("utf-8") == expected_prompt
