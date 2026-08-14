"""Isolated one-attempt boundary for Learning Run evaluation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping

from app.agent.prompt import TutorPromptTemplate
from app.agent.tutor_attempt import TutorAttemptConfig, TutorCandidate

from .contracts import ResolvedRunDefinition
from .corpus import (
    CorpusMaterializerBusyError,
    CorpusMaterializerController,
    CorpusMaterializerLease,
)


@dataclass(frozen=True)
class RunnerOperationalError(RuntimeError):
    """A safe, typed failure emitted by the isolated tutor boundary."""

    stage: str
    code: str
    sanitized_message: str
    retryable: bool

    def __str__(self) -> str:
        return self.sanitized_message


_RETRIEVAL_PROFILES: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        "hybrid-rrf-v1": MappingProxyType(
            {"version": "hybrid-rrf-v1", "top_k": 5}
        )
    }
)


def _safe_message(prefix: str, exc: BaseException | None = None) -> str:
    if exc is None:
        return prefix
    return f"{prefix} ({type(exc).__name__})"


def _emit(events: Any, event: Mapping[str, Any]) -> None:
    if events is None:
        return
    payload = dict(event)
    if callable(events):
        events(payload)
        return
    append = getattr(events, "append", None)
    if callable(append):
        append(payload)
        return
    emit = getattr(events, "emit", None)
    if callable(emit):
        emit(payload)
        return
    raise TypeError("events must be callable or appendable")


def resolved_retrieval_config(
    definition: ResolvedRunDefinition,
    *,
    retrieval_profiles: Mapping[str, Mapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    """Resolve one explicit retrieval policy shared by manifest and attempt."""

    if retrieval_profiles is not None:
        raise TypeError("retrieval_profiles injection is forbidden; use the frozen resolver")
    profiles = _RETRIEVAL_PROFILES
    controls = getattr(definition, "variant_controls", {})
    version = controls.get("retrieval_config_version") if isinstance(controls, Mapping) else None
    profile = profiles.get(str(version)) if version is not None else None
    if not isinstance(profile, Mapping) or type(profile.get("top_k")) is not int:
        raise RunnerOperationalError(
            stage="manifest",
            code="manifest_invalid",
            sanitized_message="retrieval profile is unknown",
            retryable=False,
        )
    top_k = int(profile["top_k"])
    if top_k <= 0:
        raise RunnerOperationalError(
            stage="manifest",
            code="manifest_invalid",
            sanitized_message="retrieval profile top_k is invalid",
            retryable=False,
        )
    return {"version": str(profile.get("version", version)), "top_k": top_k}


class TutorRunner:
    """Load a frozen corpus and execute exactly one TutorAttempt."""

    def __init__(
        self,
        *,
        corpus_loader: Any,
        attempt_engine: Any,
        retrieval_profiles: Mapping[str, Mapping[str, Any]] | None = None,
        materializer_controller: CorpusMaterializerController | None = None,
    ) -> None:
        if corpus_loader is None or not callable(getattr(corpus_loader, "load", None)):
            raise TypeError("corpus_loader must provide load")
        if attempt_engine is None or not callable(getattr(attempt_engine, "answer", None)):
            raise TypeError("attempt_engine must provide answer")
        if retrieval_profiles is not None:
            raise TypeError(
                "retrieval_profiles injection is forbidden; use the frozen resolver"
            )
        self.corpus_loader = corpus_loader
        self.attempt_engine = attempt_engine
        self.materializer_controller = materializer_controller or CorpusMaterializerController(corpus_loader)

    async def run(
        self,
        *,
        definition: ResolvedRunDefinition,
        llm: Any,
        events: Any = None,
    ) -> TutorCandidate:
        loop = asyncio.get_running_loop()
        budget = getattr(definition, "budget", {})
        try:
            retrieval_limit = float(budget["retrieval_preflight_seconds"])
        except (KeyError, TypeError, ValueError):
            raise RunnerOperationalError(
                stage="retrieval",
                code="retriever_error",
                sanitized_message="retrieval preflight deadline is invalid",
                retryable=True,
            )
        if not math.isfinite(retrieval_limit) or retrieval_limit <= 0:
            raise RunnerOperationalError(
                stage="retrieval",
                code="retriever_error",
                sanitized_message="retrieval preflight deadline is invalid",
                retryable=True,
            )
        preflight_started = loop.time()
        retrieval = resolved_retrieval_config(
            definition,
        )
        corpus = getattr(definition, "corpus", None)
        if corpus is None:
            raise RunnerOperationalError(
                stage="manifest",
                code="manifest_invalid",
                sanitized_message="resolved definition has no corpus",
                retryable=False,
            )
        try:
            corpus.validate_hashes()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise RunnerOperationalError(
                stage="corpus",
                code="corpus_mismatch",
                sanitized_message=_safe_message("corpus snapshot validation failed", exc),
                retryable=False,
            ) from exc

        deadline = preflight_started + retrieval_limit
        if loop.time() >= deadline:
            raise RunnerOperationalError(
                stage="retrieval",
                code="retriever_error",
                sanitized_message="retrieval preflight deadline exceeded",
                retryable=True,
            )
        lease: CorpusMaterializerLease | None = None
        try:
            lease = await self.materializer_controller.acquire(
                snapshot=corpus,
                deadline=deadline,
            )
            retriever = lease.retriever
        except asyncio.CancelledError:
            raise
        except CorpusMaterializerBusyError as exc:
            raise RunnerOperationalError(
                stage="corpus",
                code="corpus_unavailable",
                sanitized_message="isolated corpus materializer is busy",
                retryable=True,
            ) from exc
        except (asyncio.TimeoutError, TimeoutError) as exc:
            raise RunnerOperationalError(
                stage="retrieval",
                code="retriever_error",
                sanitized_message="retrieval preflight deadline exceeded",
                retryable=True,
            ) from exc
        except Exception as exc:
            raise RunnerOperationalError(
                stage="corpus",
                code="corpus_unavailable",
                sanitized_message=_safe_message("isolated corpus unavailable", exc),
                retryable=True,
            ) from exc

        remaining = deadline - loop.time()
        if remaining <= 0:
            if lease is not None:
                lease.release()
                lease = None
            raise RunnerOperationalError(
                stage="retrieval",
                code="retriever_error",
                sanitized_message="retrieval preflight deadline exceeded",
                retryable=True,
            )

        stage = "retrieval"
        forward_error: BaseException | None = None

        def forward(event: Mapping[str, Any]) -> None:
            nonlocal stage, forward_error
            if isinstance(event, Mapping):
                event_type = event.get("type")
                candidate_stage = event.get("stage")
                trace_stage = event.get("step")
                if isinstance(candidate_stage, str) and candidate_stage in {"retrieval", "generation"}:
                    stage = candidate_stage
                if isinstance(trace_stage, str) and trace_stage in {"retrieval", "generation"}:
                    stage = trace_stage
                if event_type == "trace" and event.get("status") == "failed":
                    stage = str(trace_stage or candidate_stage or stage)
            try:
                _emit(events, event)
            except BaseException as exc:
                forward_error = exc
                raise

        try:
            config = TutorAttemptConfig(
                top_k=int(retrieval["top_k"]),
                retrieval_seconds=remaining,
                generation_seconds=float(budget["tutor_seconds"]),
            )
            prompt = TutorPromptTemplate(
                version=definition.prompt.version,
                system_instruction=definition.prompt.text,
            )
            result = await self.attempt_engine.answer(
                question=definition.task.question,
                retriever=retriever,
                llm=llm,
                prompt_template=prompt,
                event_sink=forward,
                attempt_config=config,
            )
        except asyncio.CancelledError:
            raise
        except RunnerOperationalError:
            raise
        except Exception as exc:
            error_stage = stage if stage in {"retrieval", "generation"} else "retrieval"
            if error_stage == "generation":
                if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
                    code = "generation_timeout"
                    retryable = True
                    message = _safe_message("tutor generation timed out", exc)
                else:
                    code = "model_unavailable"
                    retryable = True
                    message = _safe_message("tutor model unavailable", exc)
            else:
                code = "retriever_error"
                retryable = True
                message = _safe_message("tutor retrieval failed", exc)
            if forward_error is not None or type(exc).__name__ == "TutorEventSinkError":
                error_stage = "events"
                code = "harness_internal_error"
                retryable = False
                message = _safe_message("event forwarding failed", exc)
            raise RunnerOperationalError(
                stage=error_stage,
                code=code,
                sanitized_message=message,
                retryable=retryable,
            ) from exc
        finally:
            if lease is not None:
                # Release is an ownership handoff.  The controller schedules
                # blocking retriever cleanup on its dedicated worker and never
                # makes this attempt wait for the operation lock.
                lease.release()

        if not isinstance(result, TutorCandidate):
            raise RunnerOperationalError(
                stage="internal",
                code="harness_internal_error",
                sanitized_message="tutor attempt returned an invalid candidate",
                retryable=False,
            )
        return result


__all__ = ["RunnerOperationalError", "TutorRunner", "resolved_retrieval_config"]
