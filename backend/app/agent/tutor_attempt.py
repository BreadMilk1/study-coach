"""A single graph-free Tutor retrieval and generation attempt."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable, Literal, Protocol

from langchain_core.messages import HumanMessage

from .prompt import TutorPromptTemplate, build_citations, format_context


class RetrieverLike(Protocol):
    def search(self, query: str, top_k: int) -> list[dict]: ...


class StreamingLLMLike(Protocol):
    def astream(self, messages): ...


EventSink = Callable[[dict], None]


class TutorEventSinkError(RuntimeError):
    """The configured event sink rejected a Tutor attempt event."""


@dataclass(frozen=True)
class TutorAttemptConfig:
    top_k: int = 5
    retrieval_seconds: float | None = None
    generation_seconds: float | None = None

    @classmethod
    def production_default(cls) -> "TutorAttemptConfig":
        """Return production-compatible settings without forced deadlines."""
        return cls(top_k=5, retrieval_seconds=None, generation_seconds=None)


@dataclass(frozen=True)
class TutorCandidate:
    answer: str
    citations: list[dict]
    evidence: list[dict]
    formatted_context: str
    usage: dict[str, int] | Literal["unavailable"]
    trace: list[dict]


class TutorAttemptEngine:
    """Execute one retrieval, prompt render, and streaming generation."""

    async def answer(
        self,
        *,
        question: str,
        retriever: RetrieverLike,
        llm: StreamingLLMLike,
        prompt_template: TutorPromptTemplate,
        event_sink: EventSink | None,
        attempt_config: TutorAttemptConfig,
    ) -> TutorCandidate:
        trace: list[dict] = []

        def emit(event: dict) -> None:
            if event_sink is not None:
                try:
                    event_sink(dict(event))
                except TutorEventSinkError:
                    raise
                except Exception as exc:
                    raise TutorEventSinkError(
                        f"event sink failed for {event.get('type', 'unknown')} event"
                    ) from exc

        def trace_event(**fields) -> None:
            event = {"type": "trace", **fields}
            trace.append(event)
            emit(event)

        def failed_trace(step: str, exc: BaseException) -> None:
            event = {
                "type": "trace",
                "step": step,
                "status": "failed",
                "error": type(exc).__name__,
            }
            trace.append(event)
            try:
                emit(event)
            except TutorEventSinkError:
                # Preserve the operation failure when the diagnostic event is
                # itself rejected by the sink.
                pass

        emit({
            "type": "budget",
            "stage": "retrieval",
            "limit_seconds": attempt_config.retrieval_seconds,
        })
        trace_event(step="retrieval", status="started")
        try:
            search = lambda: retriever.search(question, top_k=attempt_config.top_k)
            if attempt_config.retrieval_seconds is None:
                evidence = search()
            else:
                evidence = await asyncio.wait_for(
                    asyncio.to_thread(search),
                    timeout=attempt_config.retrieval_seconds,
                )
            evidence = list(evidence or [])
        except TutorEventSinkError:
            raise
        except Exception as exc:
            failed_trace("retrieval", exc)
            raise

        citations = build_citations(evidence)
        formatted_context = format_context(evidence)
        trace_event(
            step="retrieval",
            status="completed",
            chunks_count=len(evidence),
        )
        emit({"type": "citations", "citations": citations})

        prompt = prompt_template.render(question, evidence)
        emit({
            "type": "budget",
            "stage": "generation",
            "limit_seconds": attempt_config.generation_seconds,
        })
        trace_event(step="generation", status="started")

        parts: list[str] = []
        usage: dict[str, int] | None = None
        generation_open = True
        loop = asyncio.get_running_loop()

        async def stream_answer() -> None:
            nonlocal usage
            async for chunk in llm.astream([HumanMessage(content=prompt)]):
                text = getattr(chunk, "content", "") or ""
                within_deadline = (
                    generation_deadline is None or loop.time() < generation_deadline
                )
                if generation_open and within_deadline and text:
                    emit({"type": "token", "text": text})
                    parts.append(text)
                chunk_usage = _extract_usage(chunk)
                if generation_open and within_deadline and chunk_usage is not None:
                    usage = chunk_usage

        generation_deadline = (
            None
            if attempt_config.generation_seconds is None
            else loop.time() + attempt_config.generation_seconds
        )

        def consume_finished_stream(task: asyncio.Task) -> None:
            try:
                task.result()
            except BaseException:
                # The parent owns the immediate error path; this callback
                # prevents a cancellation-resistant child from becoming an
                # unhandled task exception after the deadline path returns.
                pass

        stream_task = asyncio.create_task(stream_answer())
        stream_task.add_done_callback(consume_finished_stream)
        try:
            done, _ = await asyncio.wait(
                {stream_task},
                timeout=attempt_config.generation_seconds,
            )
            if stream_task not in done:
                generation_open = False
                stream_task.cancel()
                timeout_error = asyncio.TimeoutError()
                failed_trace("generation", timeout_error)
                raise timeout_error
            stream_task.result()
        except asyncio.CancelledError:
            generation_open = False
            stream_task.cancel()
            raise
        except TutorEventSinkError:
            generation_open = False
            raise
        except asyncio.TimeoutError:
            raise
        except Exception as exc:
            generation_open = False
            failed_trace("generation", exc)
            raise

        answer_text = "".join(parts)
        resolved_usage: dict[str, int] | Literal["unavailable"] = (
            usage if usage is not None else "unavailable"
        )
        trace_event(
            step="generation",
            status="completed",
            answer_length=len(answer_text),
            usage=resolved_usage,
        )

        return TutorCandidate(
            answer=answer_text,
            citations=citations,
            evidence=evidence,
            formatted_context=formatted_context,
            usage=resolved_usage,
            trace=list(trace),
        )


def _extract_usage(chunk) -> dict[str, int] | None:
    usage = getattr(chunk, "usage_metadata", None) or {}
    if not usage:
        metadata = getattr(chunk, "response_metadata", None) or {}
        usage = metadata.get("token_usage") or metadata.get("usage") or {}
    if not usage:
        return None

    aliases = {
        "input_tokens": ("input_tokens", "prompt_tokens"),
        "output_tokens": ("output_tokens", "completion_tokens"),
        "total_tokens": ("total_tokens",),
    }
    normalized: dict[str, int] = {}
    for target, keys in aliases.items():
        for key in keys:
            value = usage.get(key)
            if value is not None:
                normalized[target] = int(value)
                break
    if "total_tokens" not in normalized and {
        "input_tokens",
        "output_tokens",
    } <= normalized.keys():
        normalized["total_tokens"] = (
            normalized["input_tokens"] + normalized["output_tokens"]
        )
    return normalized or None
