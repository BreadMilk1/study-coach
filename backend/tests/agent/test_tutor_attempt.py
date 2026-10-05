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


# ---------------------------------------------------------------------------
# Batch C — content-block text boundary (strict per-chunk extraction + no-body
# generation failure).
# ---------------------------------------------------------------------------

# Batch C publishes these as fixed, payload-free rejection text. The literals
# are duplicated on purpose: a consumer bug must stay observable even when the
# shared helper module is absent.
_MALFORMED_CONTENT_MESSAGE = "LLM response content is not a supported text shape"
_EMPTY_CONTENT_MESSAGE = "LLM response text has no non-whitespace body"
_SENTINEL = "SENTINEL-DO-NOT-LEAK"


class BlockStreamingLLM:
    """Streams one chunk per content value; a chunk may be text blocks or a list."""

    def __init__(self, contents: list, *, usage: dict | None = None, usage_index: int | None = None):
        self.contents = list(contents)
        self.usage = usage
        self.usage_index = usage_index
        self.prompts: list[str] = []

    async def astream(self, messages, **_kwargs):
        self.prompts.append(messages[-1].content if messages else "")
        for index, content in enumerate(self.contents):
            chunk = AIMessageChunk(content=content)
            attach = self.usage_index if self.usage_index is not None else index
            if self.usage is not None and index == attach:
                chunk.usage_metadata = dict(self.usage)
            yield chunk


def _completed_generation_events(sink) -> list[dict]:
    return [
        event
        for event in sink.of_type("trace")
        if event.get("step") == "generation" and event.get("status") == "completed"
    ]


def _failed_generation_events(sink) -> list[dict]:
    return [
        event
        for event in sink.of_type("trace")
        if event.get("step") == "generation" and event.get("status") == "failed"
    ]


@pytest.mark.asyncio
async def test_tutor_attempt_extracts_text_blocks_and_streams_str_tokens():
    sink = RecordingSink()
    llm = BlockStreamingLLM(
        [
            [{"type": "thinking", "thinking": _SENTINEL}],
            [{"type": "text", "text": "Reciprocal "}, "rank "],
            [{"type": "text", "text": "fusion [1]."}],
        ]
    )

    candidate = await TutorAttemptEngine().answer(
        question="What is RRF?",
        retriever=FakeRetriever([CHUNK_A, CHUNK_B]),
        llm=llm,
        prompt_template=TutorPromptTemplate.production_v2(),
        event_sink=sink,
        attempt_config=TutorAttemptConfig.production_default(),
    )

    tokens = sink.of_type("token")
    # One token per chunk that carries text; blocks inside a chunk concatenate.
    assert [event["text"] for event in tokens] == ["Reciprocal rank ", "fusion [1]."]
    assert all(isinstance(event["text"], str) for event in tokens)
    assert candidate.answer == "Reciprocal rank fusion [1]."
    assert _SENTINEL not in candidate.answer
    assert candidate.citations == build_citations([CHUNK_A, CHUNK_B])
    assert candidate.evidence == [CHUNK_A, CHUNK_B]
    assert len(_completed_generation_events(sink)) == 1


@pytest.mark.asyncio
async def test_tutor_attempt_does_not_fail_early_on_text_free_chunks():
    sink = RecordingSink()
    llm = BlockStreamingLLM(
        [
            [],
            [{"type": "reasoning", "text": _SENTINEL}],
            [{"type": "text", "text": ""}],
            [{"type": "text", "text": "answer"}],
        ]
    )

    candidate = await TutorAttemptEngine().answer(
        question="What is RRF?",
        retriever=FakeRetriever([CHUNK_A]),
        llm=llm,
        prompt_template=TutorPromptTemplate.production_v2(),
        event_sink=sink,
        attempt_config=TutorAttemptConfig.production_default(),
    )

    assert [event["text"] for event in sink.of_type("token")] == ["answer"]
    assert candidate.answer == "answer"
    assert len(_completed_generation_events(sink)) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("usage_index", [0, 1, 2])
async def test_tutor_attempt_keeps_usage_from_text_free_chunks(usage_index):
    """Usage metadata survives regardless of whether its chunk carried text."""
    sink = RecordingSink()
    contents = [
        [{"type": "thinking", "thinking": _SENTINEL}],
        [{"type": "text", "text": "answer"}],
        [{"type": "text", "text": ""}],
    ]
    llm = BlockStreamingLLM(
        contents,
        usage={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
        usage_index=usage_index,
    )

    candidate = await TutorAttemptEngine().answer(
        question="What is RRF?",
        retriever=FakeRetriever([CHUNK_A]),
        llm=llm,
        prompt_template=TutorPromptTemplate.production_v2(),
        event_sink=sink,
        attempt_config=TutorAttemptConfig.production_default(),
    )

    assert candidate.usage == {
        "input_tokens": 7,
        "output_tokens": 3,
        "total_tokens": 10,
    }
    assert [event["text"] for event in sink.of_type("token")] == ["answer"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "contents",
    [
        [],
        [""],
        [[]],
        [[{"type": "thinking", "thinking": _SENTINEL}]],
        [[{"type": "tool_use", "id": "t1", "name": "x", "input": {}}]],
        [[{"type": "text", "text": ""}]],
        [[{"type": "text", "text": " \n "}]],
        ["   \n\t "],
    ],
)
async def test_tutor_attempt_fails_generation_without_any_body(contents):
    sink = RecordingSink()

    with pytest.raises(ValueError) as exc_info:
        await TutorAttemptEngine().answer(
            question="What is RRF?",
            retriever=FakeRetriever([CHUNK_A]),
            llm=BlockStreamingLLM(contents),
            prompt_template=TutorPromptTemplate.production_v2(),
            event_sink=sink,
            attempt_config=TutorAttemptConfig.production_default(),
        )

    assert str(exc_info.value) == _EMPTY_CONTENT_MESSAGE
    assert _SENTINEL not in str(exc_info.value)
    assert _completed_generation_events(sink) == []
    assert [
        event["error"] for event in _failed_generation_events(sink)
    ] == ["ValueError"]


@pytest.mark.asyncio
async def test_tutor_attempt_skips_tokens_for_chunks_with_no_text():
    sink = RecordingSink()
    llm = BlockStreamingLLM(
        [
            [],
            [""],
            [{"type": "tool_result", "text": _SENTINEL}],
        ]
    )

    with pytest.raises(ValueError) as exc_info:
        await TutorAttemptEngine().answer(
            question="What is RRF?",
            retriever=FakeRetriever([CHUNK_A]),
            llm=llm,
            prompt_template=TutorPromptTemplate.production_v2(),
            event_sink=sink,
            attempt_config=TutorAttemptConfig.production_default(),
        )

    assert str(exc_info.value) == _EMPTY_CONTENT_MESSAGE
    assert sink.of_type("token") == []


@pytest.mark.asyncio
async def test_tutor_attempt_preserves_whitespace_tokens_and_still_fails_without_body():
    """Whitespace is part of the streamed text, not a body: it is forwarded
    unchanged and the completed response still has no usable answer."""
    sink = RecordingSink()
    llm = BlockStreamingLLM([["  "], [{"type": "text", "text": "\n"}]])

    with pytest.raises(ValueError) as exc_info:
        await TutorAttemptEngine().answer(
            question="What is RRF?",
            retriever=FakeRetriever([CHUNK_A]),
            llm=llm,
            prompt_template=TutorPromptTemplate.production_v2(),
            event_sink=sink,
            attempt_config=TutorAttemptConfig.production_default(),
        )

    assert str(exc_info.value) == _EMPTY_CONTENT_MESSAGE
    assert [event["text"] for event in sink.of_type("token")] == ["  ", "\n"]
    assert _completed_generation_events(sink) == []


@pytest.mark.asyncio
async def test_tutor_attempt_fails_on_malformed_chunk_after_tokens_without_candidate():
    """Already-sent tokens cannot be withdrawn; the attempt still must not
    produce a Candidate or a completed generation trace."""
    sink = RecordingSink()
    llm = BlockStreamingLLM(
        [
            [{"type": "text", "text": "Hello "}],
            [{"type": "unknown_vendor_block", "text": _SENTINEL}],
            [{"type": "text", "text": "world"}],
        ]
    )

    with pytest.raises(ValueError) as exc_info:
        await TutorAttemptEngine().answer(
            question="What is RRF?",
            retriever=FakeRetriever([CHUNK_A]),
            llm=llm,
            prompt_template=TutorPromptTemplate.production_v2(),
            event_sink=sink,
            attempt_config=TutorAttemptConfig.production_default(),
        )

    assert str(exc_info.value) == _MALFORMED_CONTENT_MESSAGE
    assert _SENTINEL not in str(exc_info.value)
    # The token that already reached the client stays; the late one never does.
    assert [event["text"] for event in sink.of_type("token")] == ["Hello "]
    assert _completed_generation_events(sink) == []
    assert [event["error"] for event in _failed_generation_events(sink)] == ["ValueError"]


@pytest.mark.asyncio
async def test_tutor_attempt_rejects_malformed_chunk_before_any_token():
    sink = RecordingSink()
    llm = BlockStreamingLLM([[{"type": "non_standard", "value": _SENTINEL}]])

    with pytest.raises(ValueError) as exc_info:
        await TutorAttemptEngine().answer(
            question="What is RRF?",
            retriever=FakeRetriever([CHUNK_A]),
            llm=llm,
            prompt_template=TutorPromptTemplate.production_v2(),
            event_sink=sink,
            attempt_config=TutorAttemptConfig.production_default(),
        )

    assert str(exc_info.value) == _MALFORMED_CONTENT_MESSAGE
    assert not sink.of_type("token")
    assert _completed_generation_events(sink) == []
    # Retrieval already completed: the rejection belongs to the generation stage.
    assert any(
        event.get("step") == "retrieval" and event.get("status") == "completed"
        for event in sink.of_type("trace")
    )
