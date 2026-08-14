"""Application orchestration for one isolated Learning Run."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol

from .contracts import (
    CandidateArtifact,
    ResolvedRunDefinition,
    RunManifest,
    ScorerExecutionDraft,
    ScoreSetResultDraft,
    as_plain,
)
from .repositories import (
    ChecksumMismatchError,
    EvalExecutionClaimRepository,
    EvalExecutionControlRepository,
    EvalRunRepository,
    EvalScoreSetRepository,
    EvalScorerExecutionRepository,
    EvaluationBusyError,
    EvaluationUnavailableError,
    RepositoryNotFoundError,
)
from .runner import RunnerOperationalError
from .scoring import ScoringService, derive_score_set


class RunRequestError(ValueError):
    """A request/configuration error that must not create a Run row."""

    def __init__(
        self,
        code: str,
        sanitized_message: str,
        *,
        fields: tuple[str, ...] = (),
    ):
        self.code = str(code)
        self.sanitized_message = str(sanitized_message)
        self.message = self.sanitized_message
        self.fields = tuple(str(field) for field in fields)
        super().__init__(self.sanitized_message)


class Clock(Protocol):
    def monotonic(self) -> float:
        ...


class SystemClock:
    def monotonic(self) -> float:
        return time.monotonic()


class CancellationToken:
    """Process-local cancellation intent for one attached PreparedRun."""

    __slots__ = ("_cancelled", "_callbacks")

    def __init__(self) -> None:
        self._cancelled = False
        self._callbacks: list[Callable[[], None]] = []

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def cancel(self) -> None:
        if self._cancelled:
            return
        self._cancelled = True
        for callback in tuple(self._callbacks):
            try:
                callback()
            except BaseException:
                pass

    def add_cancel_callback(self, callback: Callable[[], None]) -> None:
        if not callable(callback):
            raise TypeError("cancellation callback must be callable")
        self._callbacks.append(callback)
        if self._cancelled:
            try:
                callback()
            except BaseException:
                pass


@dataclass(frozen=True)
class EvalModelConnection:
    """Request-memory-only model handles and their safe configuration identity."""

    tutor_provider: str
    tutor_model: str
    tutor_parameters: Mapping[str, Any]
    tutor_llm: Any = field(repr=False, compare=False)
    scorer_provider: str
    scorer_model: str
    scorer_parameters: Mapping[str, Any]
    scorer_llm: Any = field(repr=False, compare=False)
    connection_fingerprint: str

    def __post_init__(self) -> None:
        import re

        for name in ("tutor_provider", "tutor_model", "scorer_provider", "scorer_model"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("tutor_parameters", "scorer_parameters"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping")
            object.__setattr__(self, name, MappingProxyType(dict(as_plain(value))))
        if self.tutor_llm is None or self.scorer_llm is None:
            raise ValueError("tutor_llm and scorer_llm are required")
        if not isinstance(self.connection_fingerprint, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.connection_fingerprint
        ):
            raise ValueError("connection_fingerprint must be 64 lowercase hex characters")


@dataclass(frozen=True)
class RunServiceResult:
    run: Any
    score_set: Any | None
    executions: tuple[Any, ...] = ()


@dataclass(frozen=True)
class PreparedRun:
    """Immutable definition/connection/claim prepared before streaming."""

    definition: ResolvedRunDefinition
    connection: EvalModelConnection
    manifest: RunManifest
    run: Any
    started: float
    cancellation: CancellationToken = field(default_factory=CancellationToken, compare=False)


@dataclass(frozen=True)
class PreparedRescore:
    """Claimed historical ScoreSet ready for attached scoring."""

    run: Any
    score_set: Any
    artifact: CandidateArtifact
    bundle: Any
    task: Any
    connection: EvalModelConnection
    cancellation: CancellationToken = field(default_factory=CancellationToken, compare=False)


_TIMEOUT = object()
_CANCEL_HANDOFF_TIMEOUT = 0.05


class _CooperativeCancellation(BaseException):
    """Internal stage-boundary signal; never escapes the public service API."""


def _safe_exception_message(prefix: str, exc: BaseException | None = None) -> str:
    if exc is None:
        return prefix
    return f"{prefix} ({type(exc).__name__})"


def _event(events: Any, payload: Mapping[str, Any]) -> None:
    if events is None:
        return
    value = dict(payload)
    if callable(events):
        events(value)
        return
    append = getattr(events, "append", None)
    if callable(append):
        append(value)
        return
    emit = getattr(events, "emit", None)
    if callable(emit):
        emit(value)
        return
    raise TypeError("events must be callable or appendable")


async def _hard_await(
    awaitable: Any,
    timeout: float | None,
    *,
    on_timeout: Callable[[], None] | None = None,
    on_cancel: Callable[[], None] | None = None,
) -> Any:
    """Wait with a hard return deadline and consume late child failures."""

    task = asyncio.ensure_future(awaitable)

    def consume(done: asyncio.Future[Any]) -> None:
        try:
            done.exception()
        except BaseException:
            pass

    task.add_done_callback(consume)
    try:
        if timeout is None:
            return await task
        done, _ = await asyncio.wait({task}, timeout=max(0.0, timeout))
        if task in done:
            return task.result()
        if on_timeout is not None:
            on_timeout()
        task.cancel()
        task.add_done_callback(consume)
        return _TIMEOUT
    except asyncio.CancelledError:
        if on_cancel is not None:
            try:
                on_cancel()
            except BaseException:
                # Preserve the parent cancellation even if a gate hook has
                # an unexpected failure; the attached boundary still performs
                # best-effort durable cleanup below.
                pass
        task.cancel()
        done, _ = await asyncio.wait({task}, timeout=_CANCEL_HANDOFF_TIMEOUT)
        if task not in done:
            # A cancellation-resistant child remains owned by its existing
            # runner/materializer lease.  Its done callback consumes any late
            # exception while the attached request propagates immediately.
            task.add_done_callback(consume)
        raise


def _variant_parameters(controls: Mapping[str, Any]) -> Mapping[str, Any]:
    value = controls.get("parameters")
    return value if isinstance(value, Mapping) else {}


class RunService:
    """Resolve, execute once, freeze, score and terminally persist one Run."""

    def __init__(
        self,
        *,
        registry: Any,
        tutor_runner: Any,
        runs: EvalRunRepository,
        score_sets: EvalScoreSetRepository,
        scorer_executions: EvalScorerExecutionRepository,
        clock: Clock | None = None,
        code_revision: str = "unknown",
        scoring_service_factory: Callable[..., Any] | None = None,
        claim_repository: Any | None = None,
    ) -> None:
        self.registry = registry
        self.tutor_runner = tutor_runner
        self.runs = runs
        self.score_sets = score_sets
        self.scorer_executions = scorer_executions
        self.clock = clock or SystemClock()
        self.code_revision = str(code_revision)
        self.scoring_service_factory = scoring_service_factory
        self.claim_repository = claim_repository

    def _validate_connection(
        self,
        definition: ResolvedRunDefinition,
        connection: EvalModelConnection,
    ) -> None:
        if not isinstance(connection, EvalModelConnection):
            raise RunRequestError(
                "evaluation_config_mismatch",
                "evaluation connection is invalid",
                fields=("connection",),
            )
        controls = definition.variant_controls
        expected_tutor = {
            "provider": controls.get("provider"),
            "model": controls.get("model"),
            "parameters": as_plain(_variant_parameters(controls)),
        }
        actual_tutor = {
            "provider": connection.tutor_provider,
            "model": connection.tutor_model,
            "parameters": as_plain(connection.tutor_parameters),
        }
        if actual_tutor != expected_tutor:
            fields = tuple(
                field
                for field in ("provider", "model", "parameters")
                if actual_tutor[field] != expected_tutor[field]
            )
            raise RunRequestError(
                "evaluation_config_mismatch",
                "tutor evaluation configuration does not match",
                fields=fields,
            )
        scorer_config = as_plain(definition.scorer.model_config)
        expected_scorer = {
            "provider": scorer_config.get("provider"),
            "model": scorer_config.get("model"),
            "parameters": {
                key: value
                for key, value in scorer_config.items()
                if key not in {"provider", "model"}
            },
        }
        actual_scorer = {
            "provider": connection.scorer_provider,
            "model": connection.scorer_model,
            "parameters": as_plain(connection.scorer_parameters),
        }
        if actual_scorer != expected_scorer:
            fields = tuple(
                f"scorer_{field}"
                for field in ("provider", "model", "parameters")
                if actual_scorer[field] != expected_scorer[field]
            )
            raise RunRequestError(
                "evaluation_config_mismatch",
                "scorer evaluation configuration does not match",
                fields=fields,
            )

    def _manifest(
        self,
        definition: ResolvedRunDefinition,
        connection: EvalModelConnection,
    ) -> RunManifest:
        from .runner import resolved_retrieval_config

        controls = definition.variant_controls
        retrieval = resolved_retrieval_config(definition)
        task_snapshot = definition.task.to_dict()
        corpus_snapshot = {
            "schema_version": definition.corpus.schema_version,
            "snapshot_id": definition.corpus.snapshot_id,
            "version": definition.corpus.snapshot_version,
            "aggregate_hash": definition.corpus.aggregate_hash,
            "definition_hash": definition.corpus.definition_hash,
            "chunking_config_version": definition.corpus.chunking_config_version,
            "embedding_config_version": definition.corpus.embedding_config_version,
            "retrieval_config_version": definition.corpus.retrieval_config_version,
            "reranker_config_version": definition.corpus.reranker_config_version,
            "chunk_hashes": [chunk.content_hash for chunk in definition.corpus.chunks],
        }
        scorer_snapshot = definition.scorer.payload()
        scorer_snapshot["definition_hash"] = definition.scorer.definition_hash
        return RunManifest(
            experiment_id=definition.experiment.experiment_id,
            task_case_id=definition.task.task_case_id,
            task_case_version=definition.task.task_case_version,
            variant_id=definition.variant_id,
            run_profile=definition.experiment.run_profile,
            task_snapshot=task_snapshot,
            prompt_text=definition.prompt.text,
            corpus_snapshot=corpus_snapshot,
            scorer_snapshot=scorer_snapshot,
            connection_fingerprint=connection.connection_fingerprint,
            corpus_snapshot_id=definition.corpus.snapshot_id,
            corpus_snapshot_version=definition.corpus.snapshot_version,
            corpus_snapshot_hash=definition.corpus.aggregate_hash,
            prompt_version=definition.prompt.version,
            prompt_hash=definition.prompt.content_hash,
            scorer_bundle_version=definition.scorer.version,
            scorer_bundle_hash=definition.scorer.definition_hash,
            provider=str(controls["provider"]),
            model=str(controls["model"]),
            model_parameters=_variant_parameters(controls),
            retrieval_config=retrieval,
            reranker_config={"version": controls.get("reranker_config_version")},
            chunking_config_version=str(controls.get("chunking_config_version")),
            embedding_config_version=str(controls.get("embedding_config_version")),
            budget=as_plain(definition.budget),
            runtime_judge=definition.runtime_judge,
            runner_version="learning-runner-v1",
            schema_version=definition.experiment.schema_version,
            code_revision=self.code_revision,
            seed=None,
        )

    def _artifact(
        self,
        candidate: Any,
        definition: ResolvedRunDefinition,
        started: float,
    ) -> CandidateArtifact:
        if not hasattr(candidate, "answer"):
            raise TypeError("tutor runner returned an invalid candidate")
        spent = max(0.0, self.clock.monotonic() - started)
        budget = {
            "retrieval_preflight_seconds": definition.budget["retrieval_preflight_seconds"],
            "tutor_seconds": definition.budget["tutor_seconds"],
            "hybrid_scoring_seconds": definition.budget["hybrid_scoring_seconds"],
            "total_seconds": definition.budget["total_seconds"],
            "tutor_spent_seconds": spent,
        }
        return CandidateArtifact(
            answer=str(candidate.answer),
            citations=tuple(candidate.citations or ()),
            exact_evidence=tuple(candidate.evidence or ()),
            formatted_context=str(candidate.formatted_context or ""),
            usage=candidate.usage if candidate.usage is not None else "unavailable",
            trace=tuple(candidate.trace or ()),
            budget=budget,
        )

    def _spent_budget(self, started: float, definition: ResolvedRunDefinition) -> Mapping[str, Any]:
        return {
            "total_limit_seconds": definition.budget.get("total_seconds"),
            "elapsed_seconds": max(0.0, self.clock.monotonic() - started),
        }

    def cancel_prepared(self, prepared: PreparedRun) -> Any:
        """Set local intent and durable terminal state for one attached run."""

        prepared.cancellation.cancel()
        return EvalExecutionControlRepository(self.runs.session).cancel_run(prepared.run.id)

    def _local_cancel_requested(self, prepared: PreparedRun) -> bool:
        return prepared.cancellation.cancelled

    def _durable_cancel_requested(self, prepared: PreparedRun) -> bool:
        if self._local_cancel_requested(prepared):
            return True
        try:
            return EvalExecutionControlRepository(self.runs.session).is_cancelled(
                prepared.run.id
            )
        except RepositoryNotFoundError:
            return True

    def _cancelled_result(self, prepared: PreparedRun, score_set: Any | None = None) -> RunServiceResult:
        run = EvalExecutionControlRepository(self.runs.session).cancel_run(prepared.run.id)
        executions: tuple[Any, ...] = ()
        if score_set is not None:
            score_set = EvalScoreSetRepository(self.runs.session).get_verified(score_set.id)
            executions = tuple(self.scorer_executions.list_verified(score_set.id))
        return RunServiceResult(run=run, score_set=score_set, executions=executions)

    def _finish_failure(
        self,
        run: Any,
        *,
        outcome: str,
        code: str,
        message: str,
        stage: str,
        retryable: bool,
        started: float,
        definition: ResolvedRunDefinition,
        score_set: Any | None = None,
    ) -> RunServiceResult:
        final = self.runs.finalize_failure(
            run.id,
            outcome=outcome,
            error_code=code,
            sanitized_message=message,
            stage=stage,
            retryable=retryable,
            spent_budget=self._spent_budget(started, definition),
        )
        executions: tuple[Any, ...] = ()
        if score_set is not None:
            try:
                executions = tuple(self.scorer_executions.list_verified(score_set.id))
            except Exception:
                executions = ()
        return RunServiceResult(run=final, score_set=score_set, executions=executions)

    def prepare(
        self,
        *,
        experiment_id: str,
        task_case_id: str,
        variant_id: str,
        run_profile: str,
        connection: EvalModelConnection,
    ) -> PreparedRun:
        try:
            definition = self.registry.resolve_run(
                experiment_id=experiment_id,
                task_case_id=task_case_id,
                variant_id=variant_id,
                run_profile=run_profile,
            )
        except RunRequestError:
            raise
        except Exception as exc:
            raise RunRequestError("manifest_invalid", _safe_exception_message("learning run definition is invalid", exc)) from exc
        try:
            self._validate_connection(definition, connection)
            manifest = self._manifest(definition, connection)
        except RunRequestError:
            raise
        except Exception as exc:
            raise RunRequestError("manifest_invalid", _safe_exception_message("learning run manifest is invalid", exc)) from exc

        try:
            claim_repository = self.claim_repository or EvalExecutionClaimRepository(
                self.runs.session
            )
            run = claim_repository.claim_run(
                manifest=manifest,
                manifest_hash=manifest.compute_hash(),
            )
        except Exception as exc:
            if hasattr(exc, "active_entity_id") and hasattr(exc, "active_kind"):
                raise
            if isinstance(exc, EvaluationUnavailableError):
                raise RunRequestError(
                    "evaluation_unavailable",
                    "evaluation storage is unavailable",
                ) from None
            raise RunRequestError(
                "harness_internal_error",
                "learning run could not start",
            ) from None

        return PreparedRun(
            definition=definition,
            connection=connection,
            manifest=manifest,
            run=run,
            started=self.clock.monotonic(),
        )

    async def execute_prepared(
        self,
        prepared: PreparedRun,
        events: Any = None,
    ) -> RunServiceResult:
        """Public attached-execution boundary.

        Cooperative cancellation returns a durable cancelled result.  A real
        task cancellation always performs best-effort cleanup and re-raises
        the original ``CancelledError`` so the caller can distinguish an
        aborted stream from a normal terminal event.
        """

        try:
            return await self._execute_prepared_inner(prepared, events)
        except asyncio.CancelledError:
            prepared.cancellation.cancel()
            try:
                EvalExecutionControlRepository(self.runs.session).cancel_run(prepared.run.id)
            except Exception:
                pass
            raise

    async def _execute_prepared_inner(
        self,
        prepared: PreparedRun,
        events: Any = None,
    ) -> RunServiceResult:
        definition = prepared.definition
        connection = prepared.connection
        run = prepared.run
        started = prepared.started
        total_limit = float(definition.budget["total_seconds"])
        runner_events_open = True
        callback_open = True

        def close_cancellation_gates() -> None:
            nonlocal runner_events_open, callback_open
            prepared.cancellation.cancel()
            runner_events_open = False
            callback_open = False

        prepared.cancellation.add_cancel_callback(close_cancellation_gates)

        if self._durable_cancel_requested(prepared):
            return self._cancelled_result(prepared)

        def forward_runner_event(payload: Mapping[str, Any]) -> None:
            if self._local_cancel_requested(prepared):
                raise _CooperativeCancellation()
            if runner_events_open:
                _event(events, payload)

        def close_runner_events() -> None:
            nonlocal runner_events_open
            runner_events_open = False

        tutor_stage_active = False
        try:
            _event(events, {"type": "run_created", "run_id": run.id})
            _event(events, {"type": "stage_started", "stage": "tutor"})
            tutor_stage_active = True
            runner_result = await _hard_await(
                self.tutor_runner.run(
                    definition=definition,
                    llm=connection.tutor_llm,
                    events=forward_runner_event,
                ),
                total_limit,
                on_timeout=close_runner_events,
                on_cancel=close_cancellation_gates,
            )
            if runner_result is not _TIMEOUT:
                tutor_stage_active = False
                _event(events, {"type": "stage_completed", "stage": "tutor"})
        except _CooperativeCancellation:
            runner_events_open = False
            return self._cancelled_result(prepared)
        except asyncio.CancelledError:
            runner_events_open = False
            raise
        except Exception as exc:
            runner_events_open = False
            if isinstance(exc, RunnerOperationalError):
                outcome = "timed_out" if exc.code == "generation_timeout" else (
                    "budget_exceeded" if exc.code == "budget_exceeded" else "system_failed"
                )
                return self._finish_failure(
                    run,
                    outcome=outcome,
                    code=exc.code,
                    message=exc.sanitized_message,
                    stage=exc.stage,
                    retryable=exc.retryable,
                    started=started,
                    definition=definition,
                )
            return self._finish_failure(
                run,
                outcome="system_failed",
                code="harness_internal_error",
                message=_safe_exception_message("tutor runner failed", exc),
                stage="tutor" if tutor_stage_active else "events",
                retryable=False,
                started=started,
                definition=definition,
            )
        if runner_result is _TIMEOUT:
            runner_events_open = False
            return self._finish_failure(
                run,
                outcome="budget_exceeded",
                code="budget_exceeded",
                message="total wall budget exceeded",
                stage="tutor",
                retryable=False,
                started=started,
                definition=definition,
            )
        runner_events_open = False

        if self._durable_cancel_requested(prepared):
            return self._cancelled_result(prepared)

        # The outer wall budget also covers the boundary between the runner
        # result and the immutable Candidate CAS.  A child may finish just as
        # the hard wait expires, so never freeze an artifact after the limit.
        if self.clock.monotonic() - started >= total_limit:
            return self._finish_failure(
                run,
                outcome="budget_exceeded",
                code="budget_exceeded",
                message="total wall budget exceeded",
                stage="candidate",
                retryable=False,
                started=started,
                definition=definition,
            )

        try:
            if self._durable_cancel_requested(prepared):
                return self._cancelled_result(prepared)
            artifact = self._artifact(runner_result, definition, started)
            run = self.runs.freeze_candidate(
                run.id,
                artifact,
                artifact_hash=artifact.compute_hash(),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return self._finish_failure(
                run,
                outcome="system_failed",
                code="harness_internal_error",
                message=_safe_exception_message("candidate could not be frozen", exc),
                stage="candidate",
                retryable=False,
                started=started,
                definition=definition,
            )

        score_set = None
        callback_error: BaseException | None = None
        appended_count = 0
        success_persisted = False
        persisted_drafts: list[ScorerExecutionDraft] = []

        def finalize_success_and_emit(score_set_value: Any) -> tuple[Any, tuple[Any, ...]]:
            nonlocal run, success_persisted
            run = self.runs.finalize_success(run.id)
            success_persisted = True
            for event in (
                {
                    "type": "score_set_finished",
                    "score_set_id": score_set_value.id,
                    "status": score_set_value.status,
                    "quality_verdict": score_set_value.quality_verdict,
                },
                {
                    "type": "run_finished",
                    "run_id": run.id,
                    "lifecycle": run.lifecycle,
                    "outcome": run.outcome,
                },
            ):
                try:
                    _event(events, event)
                except Exception:
                    pass
            try:
                executions_value = tuple(self.scorer_executions.list_verified(score_set_value.id))
            except Exception:
                # The Run/ScoreSet terminal CAS is durable and has already
                # been published.  A projection read failure must not rewrite
                # that terminal state or manufacture a failed run event.
                executions_value = ()
            return run, executions_value

        try:
            if self._durable_cancel_requested(prepared):
                return self._cancelled_result(prepared)
            score_set = self.score_sets.create(
                run_id=run.id,
                scorer_bundle=definition.scorer,
                artifact_input_hash=run.artifact_hash,
            )
            score_set = self.score_sets.claim_running(score_set.id)
            if self._durable_cancel_requested(prepared):
                return self._cancelled_result(prepared, score_set)
            _event(events, {"type": "score_set_created", "score_set_id": score_set.id})

            def on_execution(draft: ScorerExecutionDraft) -> None:
                nonlocal appended_count, callback_error, callback_open
                if not callback_open:
                    return
                if self._local_cancel_requested(prepared):
                    callback_open = False
                    raise _CooperativeCancellation()
                try:
                    self.scorer_executions.append_draft(score_set.id, draft)
                    appended_count += 1
                    persisted_drafts.append(draft)
                    event_type = "scorer_completed" if draft.status != "failed" else "scorer_failed"
                    _event(
                        events,
                        {
                            "type": event_type,
                            "score_set_id": score_set.id,
                            "scorer_id": draft.scorer_id or draft.component_id,
                            "status": draft.status,
                            "error_code": draft.error_code,
                        },
                    )
                except BaseException as exc:
                    callback_error = exc
                    raise

            def persist_timeout_draft(draft: ScorerExecutionDraft) -> None:
                nonlocal appended_count
                if self._local_cancel_requested(prepared):
                    raise _CooperativeCancellation()
                self.scorer_executions.append_draft(score_set.id, draft)
                appended_count += 1
                persisted_drafts.append(draft)
                _event(
                    events,
                    {
                        "type": "scorer_failed",
                        "score_set_id": score_set.id,
                        "scorer_id": draft.scorer_id,
                        "status": "failed",
                        "error_code": draft.error_code,
                    },
                )

            def ordered_drafts() -> tuple[ScorerExecutionDraft, ...]:
                by_component = {draft.component_id: draft for draft in persisted_drafts}
                return tuple(
                    by_component[component.component_id]
                    for component in definition.scorer.components
                    if component.component_id in by_component
                )

            def derive_persisted_score_set() -> ScoreSetResultDraft:
                return derive_score_set(
                    task=definition.task,
                    scorer_bundle=definition.scorer,
                    drafts=ordered_drafts(),
                    input_hash=run.artifact_hash,
                )

            def append_missing_timeout_drafts(error_code: str, error_message: str) -> None:
                existing = {draft.component_id for draft in persisted_drafts}
                for component in definition.scorer.components:
                    if component.component_id in existing:
                        continue
                    persist_timeout_draft(
                        ScorerExecutionDraft(
                            component_id=component.component_id,
                            component_version=component.version,
                            scorer_id=component.component_id,
                            scorer_version=component.version,
                            status="failed",
                            input_hash=run.artifact_hash,
                            output=None,
                            error_code=error_code,
                            error_message=error_message,
                            latency_ms=None,
                            usage=None,
                            findings=(),
                        )
                    )

            def close_callback() -> None:
                nonlocal callback_open
                callback_open = False

            hybrid_limit = float(definition.budget["hybrid_scoring_seconds"])
            scoring = (
                self.scoring_service_factory(
                    connection.scorer_llm,
                    timeout_seconds=hybrid_limit,
                )
                if self.scoring_service_factory is not None
                else ScoringService(
                    connection.scorer_llm,
                    timeout_seconds=hybrid_limit,
                )
            )
            elapsed = max(0.0, self.clock.monotonic() - started)
            remaining = max(0.0, total_limit - elapsed)
            stage_timeout = min(remaining, hybrid_limit)
            score_result = await _hard_await(
                scoring.score(
                    task=definition.task,
                    candidate=artifact,
                    scorer_bundle=definition.scorer,
                    on_execution=on_execution,
                ),
                stage_timeout,
                on_timeout=close_callback,
                on_cancel=close_cancellation_gates,
            )
            callback_open = False
            if self._durable_cancel_requested(prepared):
                return self._cancelled_result(prepared, score_set)
            if score_result is _TIMEOUT:
                if hybrid_limit < remaining:
                    # The scoring stage has its own hard wall even when the
                    # enclosing Run still has budget.  ScoringService may be
                    # cancellation-resistant, so persist one explicit failed
                    # semantic component before terminalizing the ScoreSet.
                    append_missing_timeout_drafts("scorer_timeout", "scorer timed out")
                    derived = derive_persisted_score_set()
                    score_set = self.score_sets.finalize_once(
                        score_set.id,
                        status=derived.status,
                        quality_verdict=derived.verdict,
                        aggregate_scores=derived.aggregate_scores,
                        findings=derived.findings,
                        error_code=derived.error_code,
                        sanitized_message=derived.error_message,
                    )
                    run, executions = finalize_success_and_emit(score_set)
                    return RunServiceResult(run=run, score_set=score_set, executions=executions)
                append_missing_timeout_drafts("budget_exceeded", "total wall budget exceeded during scoring")
                derived = derive_persisted_score_set()
                score_set = self.score_sets.finalize_once(
                    score_set.id,
                    status=derived.status,
                    quality_verdict=derived.verdict,
                    aggregate_scores=derived.aggregate_scores,
                    findings=derived.findings,
                    error_code=derived.error_code,
                    sanitized_message=derived.error_message,
                )
                return self._finish_failure(
                    run,
                    outcome="budget_exceeded",
                    code="budget_exceeded",
                    message="total wall budget exceeded during scoring",
                    stage="scoring",
                    retryable=False,
                    started=started,
                    definition=definition,
                    score_set=score_set,
                )
            if callback_error is not None:
                raise callback_error
            if not isinstance(score_result, ScoreSetResultDraft):
                raise TypeError("scoring service returned an invalid result")
            if self.clock.monotonic() - started >= total_limit:
                callback_open = False
                append_missing_timeout_drafts("budget_exceeded", "total wall budget exceeded during scoring")
                derived = derive_persisted_score_set()
                score_set = self.score_sets.finalize_once(
                    score_set.id,
                    status=derived.status,
                    quality_verdict=derived.verdict,
                    aggregate_scores=derived.aggregate_scores,
                    findings=derived.findings,
                    error_code=derived.error_code,
                    sanitized_message=derived.error_message,
                )
                return self._finish_failure(
                    run,
                    outcome="budget_exceeded",
                    code="budget_exceeded",
                    message="total wall budget exceeded during scoring",
                    stage="scoring",
                    retryable=False,
                    started=started,
                    definition=definition,
                    score_set=score_set,
                )
            derived = derive_persisted_score_set()
            if self._durable_cancel_requested(prepared):
                return self._cancelled_result(prepared, score_set)
            score_set = self.score_sets.finalize_once(
                score_set.id,
                status=derived.status,
                quality_verdict=derived.verdict,
                aggregate_scores=derived.aggregate_scores,
                findings=derived.findings,
                error_code=derived.error_code,
                sanitized_message=derived.error_message,
            )
            run, executions = finalize_success_and_emit(score_set)
            return RunServiceResult(run=run, score_set=score_set, executions=executions)
        except _CooperativeCancellation:
            callback_open = False
            return self._cancelled_result(prepared, score_set)
        except asyncio.CancelledError:
            callback_open = False
            raise
        except Exception as exc:
            callback_open = False
            if self._durable_cancel_requested(prepared):
                return self._cancelled_result(prepared, score_set)
            if success_persisted:
                raise
            score_terminalization_error: BaseException | None = None
            if score_set is not None and score_set.status == "running":
                try:
                    status = "partial" if appended_count else "failed"
                    score_set = self.score_sets.finalize_once(
                        score_set.id,
                        status=status,
                        quality_verdict="inconclusive",
                        aggregate_scores={},
                        findings=[],
                        error_code="harness_internal_error",
                        sanitized_message="scoring orchestration failed",
                    )
                except BaseException as terminal_exc:
                    # A best-effort ScoreSet terminalization failure must not
                    # leave the parent Run running.  Preserve the exception
                    # for the caller after the Run's own CAS terminalization.
                    score_terminalization_error = terminal_exc
            result = self._finish_failure(
                run,
                outcome="system_failed",
                code="harness_internal_error",
                message=_safe_exception_message("scoring orchestration failed", exc),
                stage="scoring",
                retryable=False,
                started=started,
                definition=definition,
                score_set=score_set,
            )
            if score_terminalization_error is not None:
                raise score_terminalization_error
            return result

    async def run(
        self,
        *,
        experiment_id: str,
        task_case_id: str,
        variant_id: str,
        run_profile: str,
        connection: EvalModelConnection,
        events: Any = None,
    ) -> RunServiceResult:
        """Compatibility wrapper around the typed prepare/execute phases."""
        prepared = self.prepare(
            experiment_id=experiment_id,
            task_case_id=task_case_id,
            variant_id=variant_id,
            run_profile=run_profile,
            connection=connection,
        )
        return await self.execute_prepared(prepared, events)

    def prepare_rescore(
        self,
        *,
        run_id: str,
        scorer_version: str,
        connection: EvalModelConnection,
    ) -> PreparedRescore:
        """Claim a historical ScoreSet before the attached stream starts."""

        try:
            run = self.runs.get_verified(run_id)
        except RepositoryNotFoundError as exc:
            raise RunRequestError("manifest_invalid", "evaluation run was not found") from exc
        except ChecksumMismatchError as exc:
            raise RunRequestError("manifest_invalid", "frozen artifact hash mismatch") from exc
        if run.lifecycle != "finished" or run.artifact_hash is None or not run.candidate_artifact_json:
            raise RunRequestError("manifest_invalid", "run is not a finished frozen candidate")
        artifact = CandidateArtifact.from_dict(run.candidate_artifact_json)
        if artifact.compute_hash() != run.artifact_hash:
            raise RunRequestError("manifest_invalid", "frozen artifact hash mismatch")

        try:
            bundle = self.registry.scorer_for(scorer_version)
            document = self.registry.scorer_document(scorer_version)
            task_snapshot = (run.manifest_json or {}).get("task_snapshot")
            if isinstance(task_snapshot, Mapping) and task_snapshot:
                from .contracts import TaskCase

                task = TaskCase.from_dict(task_snapshot)
            else:
                task = self.registry.task_cases[run.task_case_id]
        except Exception as exc:
            raise RunRequestError("manifest_invalid", "scorer version is not in the registry") from exc

        scorer_config = as_plain(bundle.model_config)
        expected_scorer = {
            "provider": scorer_config.get("provider"),
            "model": scorer_config.get("model"),
            "parameters": {
                key: value
                for key, value in scorer_config.items()
                if key not in {"provider", "model"}
            },
        }
        actual_scorer = {
            "provider": connection.scorer_provider,
            "model": connection.scorer_model,
            "parameters": as_plain(connection.scorer_parameters),
        }
        if actual_scorer != expected_scorer:
            raise RunRequestError(
                "evaluation_config_mismatch",
                "scorer evaluation configuration does not match",
            )

        claim = self.claim_repository or EvalExecutionClaimRepository(self.runs.session)
        score_set = claim.claim_score_set(
            run_id=run.id,
            artifact_input_hash=run.artifact_hash,
            scorer_bundle=bundle,
            scorer_snapshot=document,
            scorer_definition_hash=bundle.definition_hash,
        )
        return PreparedRescore(
            run=run,
            score_set=score_set,
            artifact=artifact,
            bundle=bundle,
            task=task,
            connection=connection,
        )

    async def execute_rescore(self, prepared: PreparedRescore, events: Any = None) -> Any:
        """Score a claimed historical ScoreSet and emit stream events."""

        score_set = prepared.score_set
        _event(events, {"type": "score_set_created", "score_set_id": score_set.id})
        try:
            return await self._execute_rescore_inner(prepared, events)
        except asyncio.CancelledError:
            prepared.cancellation.cancel()
            try:
                self.score_sets.cancel_once(score_set.id)
            except Exception:
                pass
            raise
        except Exception:
            try:
                self.score_sets.finalize_once(
                    score_set.id,
                    status="failed",
                    quality_verdict="inconclusive",
                    error_code="harness_internal_error",
                    sanitized_message="historical rescore failed",
                )
            except Exception:
                pass
            raise

    async def _execute_rescore_inner(self, prepared: PreparedRescore, events: Any) -> Any:

        score_set = prepared.score_set
        if prepared.cancellation.cancelled:
            return self.score_sets.cancel_once(score_set.id)

        def on_execution(draft: ScorerExecutionDraft) -> None:
            if prepared.cancellation.cancelled:
                raise _CooperativeCancellation()
            self.scorer_executions.append_draft(score_set.id, draft)
            _event(
                events,
                {
                    "type": "scorer_completed" if draft.status == "success" else "scorer_failed",
                    "score_set_id": score_set.id,
                    "scorer_id": draft.scorer_id,
                    "status": draft.status,
                    "error_code": draft.error_code,
                },
            )

        hybrid_limit = float(self.registry.experiment.budget["hybrid_scoring_seconds"])
        scoring = (
            self.scoring_service_factory(
                prepared.connection.scorer_llm, timeout_seconds=hybrid_limit
            )
            if self.scoring_service_factory is not None
            else ScoringService(
                prepared.connection.scorer_llm, timeout_seconds=hybrid_limit
            )
        )
        try:
            score_result = await scoring.score(
                task=prepared.task,
                candidate=prepared.artifact,
                scorer_bundle=prepared.bundle,
                on_execution=on_execution,
            )
        except _CooperativeCancellation:
            return self.score_sets.cancel_once(score_set.id)
        score_set = self.score_sets.finalize_once(
            score_set.id,
            status=score_result.status,
            quality_verdict=score_result.verdict,
            aggregate_scores=score_result.aggregate_scores,
            findings=score_result.findings,
            error_code=score_result.error_code,
            sanitized_message=score_result.error_message,
        )
        _event(
            events,
            {
                "type": "score_set_finished",
                "score_set_id": score_set.id,
                "status": score_set.status,
                "quality_verdict": score_set.quality_verdict,
            },
        )
        return score_set

    async def rescore(
        self,
        *,
        run_id: str,
        scorer_version: str,
        connection: EvalModelConnection,
        events: Any = None,
    ) -> Any:
        prepared = self.prepare_rescore(
            run_id=run_id,
            scorer_version=scorer_version,
            connection=connection,
        )
        return await self.execute_rescore(prepared, events)


__all__ = [
    "Clock",
    "EvalModelConnection",
    "PreparedRescore",
    "PreparedRun",
    "RunRequestError",
    "RunService",
    "RunServiceResult",
    "SystemClock",
]
