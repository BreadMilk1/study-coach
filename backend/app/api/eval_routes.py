"""Authenticated local-only Learning Run API."""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.db.session import get_eval_session
from app.eval.learning_run.repositories import (
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
from app.eval.learning_run.compare import compare_score_sets
from app.eval.learning_run.service import (
    EvalModelConnection,
    RunRequestError,
    RunService,
)
from app.eval.learning_run.registry import RegistryError, TaskRegistry

from .deps import require_existing_user, require_local_eval_mode
from .eval_schemas import (
    CompareResponse,
    EvalErrorDetail,
    ExperimentSummary,
    RescoreStreamRequest,
    RunDetail,
    RunStreamRequest,
    RunSummary,
    ScoreSetDetail,
    ScoreSetSummary,
    ScorerExecutionDetail,
)


eval_router = APIRouter(
    prefix="/api/eval",
    dependencies=[Depends(require_existing_user), Depends(require_local_eval_mode)],
)

_APPROVED_EVENTS = {
    "run_created",
    "stage_started",
    "stage_completed",
    "score_set_created",
    "scorer_completed",
    "scorer_failed",
    "score_set_finished",
    "run_finished",
}
_CANCEL_HANDOFF_TIMEOUT = 0.05


def _error(
    status: int,
    code: str,
    message: str,
    *,
    fields: tuple[str, ...] = (),
    active_entity_id: str | None = None,
    active_kind: str | None = None,
) -> HTTPException:
    detail = EvalErrorDetail(
        code=code,
        message=message,
        fields=fields,
        active_entity_id=active_entity_id,
        active_kind=active_kind,
    )
    return HTTPException(
        status_code=status,
        detail=detail.model_dump(exclude_none=True, exclude_defaults=True),
    )


def _canonical_endpoint(base_url: str | None, provider: str) -> str:
    if not base_url:
        return f"provider-default:{provider.lower().strip()}"
    value = str(base_url).strip()
    parsed = urlsplit(value)
    if parsed.scheme and parsed.netloc:
        path = parsed.path.rstrip("/")
        return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, "", ""))
    return value.rstrip("/").lower()


def connection_fingerprint(provider: str, base_url: str | None) -> str:
    identity = f"{provider.strip().lower()}|{_canonical_endpoint(base_url, provider)}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _default_connection_factory(
    *,
    provider: str,
    model: str,
    variant_id: str,
    api_key: str | None,
    base_url: str | None,
    registry: TaskRegistry,
) -> EvalModelConnection:
    from app.llm.provider import LLMConfig, get_chat_model

    try:
        controls = registry.experiment.variants[variant_id]
    except KeyError as exc:
        raise RunRequestError("manifest_invalid", "learning run variant is invalid") from exc
    scorer_config = dict(registry.scorer.model_config)
    scorer_provider = str(scorer_config["provider"])
    scorer_model = str(scorer_config["model"])
    if scorer_provider != provider or scorer_model != model:
        # The service emits the stable mismatch fields; do not silently use a
        # second request connection for a different frozen scorer model.
        raise RunRequestError(
            "evaluation_config_mismatch",
            "evaluation model connection does not match",
            fields=("provider", "model"),
        )
    tutor_parameters = dict(controls["parameters"])
    scorer_parameters = {
        key: value for key, value in scorer_config.items() if key not in {"provider", "model"}
    }
    tutor_llm = get_chat_model(
        LLMConfig(provider=provider, model=model, api_key=api_key, base_url=base_url),
        **tutor_parameters,
    )
    scorer_llm = get_chat_model(
        LLMConfig(provider=scorer_provider, model=scorer_model, api_key=api_key, base_url=base_url),
        **scorer_parameters,
    )
    return EvalModelConnection(
        tutor_provider=provider,
        tutor_model=model,
        tutor_parameters=tutor_parameters,
        tutor_llm=tutor_llm,
        scorer_provider=scorer_provider,
        scorer_model=scorer_model,
        scorer_parameters=scorer_parameters,
        scorer_llm=scorer_llm,
        connection_fingerprint=connection_fingerprint(provider, base_url),
    )


def _registry(request: Request) -> TaskRegistry:
    value = getattr(request.app.state, "eval_registry", None)
    if value is None:
        value = TaskRegistry.load_default()
        request.app.state.eval_registry = value
    return value


def _service(request: Request, session: Session, registry: TaskRegistry) -> RunService:
    factory = getattr(request.app.state, "eval_service_factory", None)
    if callable(factory):
        return factory(
            request=request,
            session=session,
            registry=registry,
            claim_repository=EvalExecutionClaimRepository(session),
        )
    return RunService(
        registry=registry,
        tutor_runner=request.app.state.eval_tutor_runner,
        runs=EvalRunRepository(session),
        score_sets=EvalScoreSetRepository(session),
        scorer_executions=EvalScorerExecutionRepository(session),
        claim_repository=EvalExecutionClaimRepository(session),
    )


def _build_connection(
    request: Request,
    registry: TaskRegistry,
    *,
    variant_id: str,
    provider: str | None,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
) -> EvalModelConnection:
    provider_value = (provider or "ollama").strip().lower()
    model_value = (model or "").strip()
    factory = getattr(request.app.state, "eval_connection_factory", None)
    if factory is None:
        factory = _default_connection_factory
    try:
        connection = factory(
            provider=provider_value,
            model=model_value,
            variant_id=variant_id,
            api_key=api_key,
            base_url=base_url,
            registry=registry,
        )
    except RunRequestError:
        raise
    except ValueError:
        raise RunRequestError("evaluation_config_mismatch", "evaluation connection is invalid") from None
    except Exception:
        raise RunRequestError("model_unavailable", "evaluation model is unavailable") from None
    if not isinstance(connection, EvalModelConnection):
        raise RunRequestError("evaluation_config_mismatch", "evaluation connection is invalid")
    return connection


def _score_summary(row: Any) -> ScoreSetSummary:
    return ScoreSetSummary(
        score_set_id=row.id,
        scorer_id=row.scorer_id,
        scorer_version=row.scorer_version,
        scorer_definition_hash=getattr(row, "scorer_definition_hash", None),
        status=row.status,
        quality_verdict=row.quality_verdict,
        aggregate_scores=row.aggregate_scores_json,
        created_at=row.created_at,
        finished_at=row.finished_at,
    )


def _run_summary(row: Any, score_sets: list[Any]) -> RunSummary:
    latest = score_sets[-1] if score_sets else None
    return RunSummary(
        run_id=row.id,
        experiment_id=row.experiment_id,
        suite_execution_id=row.suite_execution_id,
        task_case_id=row.task_case_id,
        variant_id=row.variant_id,
        run_profile=row.run_profile,
        lifecycle=row.lifecycle,
        outcome=row.outcome,
        latest_score_set=_score_summary(latest) if latest is not None else None,
        created_at=row.created_at,
        started_at=row.started_at,
        finished_at=row.finished_at,
    )


def _score_detail(row: Any) -> ScoreSetDetail:
    return ScoreSetDetail(
        **_score_summary(row).model_dump(),
        artifact_input_hash=row.artifact_input_hash,
        operational_error_code=row.operational_error_code,
        operational_error_message=row.operational_error_message,
        findings=row.findings_json,
    )


def _execution_detail(row: Any) -> ScorerExecutionDetail:
    return ScorerExecutionDetail(
        execution_id=row.id,
        score_set_id=row.score_set_id,
        scorer_id=row.scorer_id,
        scorer_version=row.scorer_version,
        status=row.status,
        input_hash=row.input_hash,
        output=row.output_json,
        operational_error_code=row.operational_error_code,
        operational_error_message=row.operational_error_message,
        latency_ms=row.latency_ms,
        usage=row.usage_json,
        created_at=row.created_at,
    )


@eval_router.get("/experiments", response_model=list[ExperimentSummary])
def list_experiments(request: Request) -> list[ExperimentSummary]:
    registry = _registry(request)
    counts: dict[str, int] = {}
    for case in registry.task_cases.values():
        counts[case.case_type] = counts.get(case.case_type, 0) + 1
    variants = tuple(
        {"variant_id": variant_id, "prompt_version": str(controls["prompt_version"])}
        for variant_id, controls in registry.experiment.variants.items()
    )
    return [
        ExperimentSummary(
            experiment_id=registry.experiment.experiment_id,
            task_family="Tutor Grounded QA",
            experiment_axes=tuple(registry.experiment.experiment_axes),
            variants=variants,
            case_counts=counts,
            run_profile=registry.experiment.run_profile,
            budgets=dict(registry.experiment.budget),
        )
    ]


@eval_router.get("/runs", response_model=list[RunSummary])
def list_runs(
    session: Session = Depends(get_eval_session),
) -> list[RunSummary]:
    try:
        repo = EvalRunRepository(session)
        rows = repo.list_verified()
        score_repo = EvalScoreSetRepository(session)
        return [_run_summary(row, score_repo.list_verified(row.id)) for row in rows]
    except ChecksumMismatchError:
        raise _error(
            500,
            "evaluation_integrity_error",
            "evaluation artifact integrity could not be verified",
        ) from None
    except EvaluationUnavailableError:
        raise _error(
            503,
            "evaluation_unavailable",
            "evaluation storage is unavailable",
        ) from None


@eval_router.get("/runs/{run_id}", response_model=RunDetail)
def get_run_detail(
    run_id: str,
    session: Session = Depends(get_eval_session),
) -> RunDetail:
    try:
        run = EvalRunRepository(session).get_verified(run_id)
        score_sets = EvalScoreSetRepository(session).list_verified(run.id)
        executions = [
            execution
            for score_set in score_sets
            for execution in EvalScorerExecutionRepository(session).list_verified(score_set.id)
        ]
    except RepositoryNotFoundError:
        raise _error(404, "evaluation_not_found", "evaluation run was not found") from None
    except ChecksumMismatchError:
        raise _error(
            500,
            "evaluation_integrity_error",
            "evaluation artifact integrity could not be verified",
        ) from None
    return RunDetail(
        summary=_run_summary(run, score_sets),
        manifest=run.manifest_json,
        candidate_artifact=run.candidate_artifact_json,
        score_sets=[_score_detail(row) for row in score_sets],
        scorer_executions=[_execution_detail(row) for row in executions],
        operational_error=run.operational_error_json,
    )


@eval_router.post("/runs/{run_id}/cancel", response_model=RunSummary)
def cancel_run(
    run_id: str,
    session: Session = Depends(get_eval_session),
) -> RunSummary:
    """Durably cancel an active Run (idempotently) and return its summary."""

    try:
        run = EvalExecutionControlRepository(session).cancel_run(run_id)
        score_sets = EvalScoreSetRepository(session).list_verified(run.id)
        return _run_summary(run, score_sets)
    except RepositoryNotFoundError:
        raise _error(404, "evaluation_not_found", "evaluation run was not found") from None
    except ChecksumMismatchError:
        raise _error(
            500,
            "evaluation_integrity_error",
            "evaluation artifact integrity could not be verified",
        ) from None
    except EvaluationUnavailableError:
        raise _error(
            503,
            "evaluation_unavailable",
            "evaluation storage is unavailable",
        ) from None


@eval_router.post("/score-sets/{score_set_id}/cancel", response_model=ScoreSetSummary)
def cancel_score_set(
    score_set_id: str,
    session: Session = Depends(get_eval_session),
) -> ScoreSetSummary:
    try:
        row = EvalScoreSetRepository(session).cancel_once(score_set_id)
        return _score_summary(row)
    except RepositoryNotFoundError:
        raise _error(404, "evaluation_not_found", "evaluation score set was not found") from None
    except ChecksumMismatchError:
        raise _error(
            500,
            "evaluation_integrity_error",
            "evaluation artifact integrity could not be verified",
        ) from None


@eval_router.get("/compare", response_model=CompareResponse)
def compare_runs(
    left: str,
    right: str,
    session: Session = Depends(get_eval_session),
) -> CompareResponse:
    try:
        runs = EvalRunRepository(session)
        scores = EvalScoreSetRepository(session)
        left_run = runs.get_verified(left)
        right_run = runs.get_verified(right)
        left_sets = scores.list_verified(left_run.id)
        right_sets = scores.list_verified(right_run.id)
    except RepositoryNotFoundError:
        raise _error(404, "evaluation_not_found", "evaluation run was not found") from None
    except ChecksumMismatchError:
        raise _error(
            500,
            "evaluation_integrity_error",
            "evaluation artifact integrity could not be verified",
        ) from None
    left_score = left_sets[-1] if left_sets else None
    right_score = right_sets[-1] if right_sets else None
    payload = compare_score_sets(
        {
            "run_id": left_run.id,
            "variant_id": left_run.variant_id,
            "manifest": left_run.manifest_json,
            "artifact": left_run.candidate_artifact_json,
            "score_set": None
            if left_score is None
            else {
                "scorer_id": left_score.scorer_id,
                "scorer_version": left_score.scorer_version,
                "aggregate_scores": left_score.aggregate_scores_json,
            },
        },
        {
            "run_id": right_run.id,
            "variant_id": right_run.variant_id,
            "manifest": right_run.manifest_json,
            "artifact": right_run.candidate_artifact_json,
            "score_set": None
            if right_score is None
            else {
                "scorer_id": right_score.scorer_id,
                "scorer_version": right_score.scorer_version,
                "aggregate_scores": right_score.aggregate_scores_json,
            },
        },
    )
    return CompareResponse(**payload)


@eval_router.post("/runs/{run_id}/rescore/stream")
async def stream_rescore(
    run_id: str,
    request: Request,
    payload: RescoreStreamRequest,
    session: Session = Depends(get_eval_session),
    x_provider: str | None = Header(None),
    x_model: str | None = Header(None),
    x_api_key: str | None = Header(None),
    x_base_url: str | None = Header(None),
) -> StreamingResponse:
    registry = _registry(request)
    try:
        existing = EvalRunRepository(session).get_verified(run_id)
        connection = _build_connection(
            request,
            registry,
            provider=x_provider,
            model=x_model,
            api_key=x_api_key,
            base_url=x_base_url,
            variant_id=existing.variant_id,
        )
        service = _service(request, session, registry)
    except RepositoryNotFoundError:
        raise _error(404, "evaluation_not_found", "evaluation run was not found") from None
    except ChecksumMismatchError:
        raise _error(
            500,
            "evaluation_integrity_error",
            "evaluation artifact integrity could not be verified",
        ) from None
    except RunRequestError as exc:
        raise _error(422, exc.code, exc.sanitized_message, fields=exc.fields) from None

    async def body():
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        def emit(event: Mapping[str, Any]) -> None:
            adapted = _adapt_event(event, run_id)
            if adapted is not None:
                queue.put_nowait(adapted)

        try:
            score_set = await service.rescore(
                run_id=run_id,
                scorer_version=payload.scorer_version,
                connection=connection,
                events=emit,
            )
            queue.put_nowait(
                {
                    "schema_version": "eval-api-v1",
                    "type": "score_set_finished",
                    "run_id": run_id,
                    "score_set_id": score_set.id,
                    "status": score_set.status,
                    "quality_verdict": score_set.quality_verdict,
                }
            )
        except RunRequestError as exc:
            queue.put_nowait(
                {
                    "schema_version": "eval-api-v1",
                    "type": "run_finished",
                    "run_id": run_id,
                    "lifecycle": "finished",
                    "outcome": "system_failed",
                    "error_code": exc.code,
                }
            )
        except Exception:
            queue.put_nowait(
                {
                    "schema_version": "eval-api-v1",
                    "type": "run_finished",
                    "run_id": run_id,
                    "lifecycle": "finished",
                    "outcome": "system_failed",
                    "error_code": "harness_internal_error",
                }
            )
        queue.put_nowait(None)
        while True:
            item = await queue.get()
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(body(), media_type="text/event-stream")


def _adapt_event(payload: Mapping[str, Any], run_id: str) -> dict[str, Any] | None:
    event_type = payload.get("type")
    if event_type not in _APPROVED_EVENTS:
        return None
    event: dict[str, Any] = {"schema_version": "eval-api-v1", "type": event_type, "run_id": run_id}
    if event_type in {"stage_started", "stage_completed"}:
        event["stage"] = str(payload.get("stage", "unknown"))
    elif event_type in {"score_set_created"}:
        event["score_set_id"] = str(payload["score_set_id"])
    elif event_type in {"scorer_completed", "scorer_failed"}:
        event["score_set_id"] = str(payload["score_set_id"])
        event["scorer_id"] = str(payload.get("scorer_id", "unknown"))
        event["status"] = "failed" if event_type == "scorer_failed" else str(payload.get("status", "success"))
        if event_type == "scorer_failed" and payload.get("error_code"):
            event["error_code"] = str(payload["error_code"])
    elif event_type == "score_set_finished":
        event["score_set_id"] = str(payload["score_set_id"])
        event["status"] = str(payload.get("status", "completed"))
        event["quality_verdict"] = str(payload.get("quality_verdict", "inconclusive"))
    elif event_type == "run_finished":
        event["lifecycle"] = str(payload.get("lifecycle", "finished"))
        event["outcome"] = payload.get("outcome")
        if payload.get("error_code"):
            event["error_code"] = str(payload["error_code"])
    return event


def _durable_terminal_event(run: Any) -> dict[str, Any] | None:
    lifecycle = getattr(run, "lifecycle", None)
    outcome = getattr(run, "outcome", None)
    if lifecycle == "finished" and outcome is None:
        return None
    if lifecycle not in {"finished", "cancelled"}:
        return None
    event: dict[str, Any] = {
        "schema_version": "eval-api-v1",
        "type": "run_finished",
        "run_id": str(run.id),
        "lifecycle": lifecycle,
        "outcome": outcome,
    }
    operational_error = getattr(run, "operational_error_json", None)
    if isinstance(operational_error, Mapping) and operational_error.get("code"):
        event["error_code"] = str(operational_error["code"])
    return event


@eval_router.post("/runs/stream")
async def stream_run(
    request: Request,
    payload: RunStreamRequest,
    session: Session = Depends(get_eval_session),
    x_provider: str | None = Header(None),
    x_model: str | None = Header(None),
    x_api_key: str | None = Header(None),
    x_base_url: str | None = Header(None),
) -> StreamingResponse:
    registry = _registry(request)
    try:
        connection = _build_connection(
            request,
            registry,
            provider=x_provider,
            model=x_model,
            api_key=x_api_key,
            base_url=x_base_url,
            variant_id=payload.variant_id,
        )
        service = _service(request, session, registry)
        prepared = service.prepare(
            experiment_id=payload.experiment_id,
            task_case_id=payload.task_case_id,
            variant_id=payload.variant_id,
            run_profile=payload.run_profile,
            connection=connection,
        )
    except EvaluationBusyError as exc:
        raise _error(
            409,
            "evaluation_busy",
            "another evaluation is already running",
            active_entity_id=exc.active_entity_id,
            active_kind=exc.active_kind,
        ) from None
    except EvaluationUnavailableError:
        raise _error(
            503,
            "evaluation_unavailable",
            "evaluation storage is unavailable",
        ) from None
    except RunRequestError as exc:
        status = {
            "evaluation_config_mismatch": 409,
            "manifest_invalid": 422,
            "model_unavailable": 503,
            "evaluation_unavailable": 503,
            "harness_internal_error": 500,
        }.get(exc.code, 500)
        raise _error(status, exc.code, exc.sanitized_message, fields=exc.fields) from None
    except (RegistryError, KeyError, ValueError):
        raise _error(422, "manifest_invalid", "learning run definition is invalid") from None

    async def body():
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        terminal_sent = False

        def emit(payload: Mapping[str, Any]) -> None:
            nonlocal terminal_sent
            event = _adapt_event(payload, prepared.run.id)
            if event is None:
                return
            if event["type"] == "run_created":
                if getattr(emit, "run_created_sent", False):
                    return
                setattr(emit, "run_created_sent", True)
            elif not getattr(emit, "run_created_sent", False):
                return
            if event["type"] == "run_finished":
                if terminal_sent:
                    return
                terminal_sent = True
            queue.put_nowait(event)

        task = asyncio.create_task(service.execute_prepared(prepared, emit))
        result = None
        execution_error: BaseException | None = None
        get_event: asyncio.Task[Any] | None = None
        try:
            while True:
                get_event = asyncio.create_task(queue.get())
                done, _ = await asyncio.wait(
                    {get_event, task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if get_event in done:
                    yield f"data: {json.dumps(get_event.result(), separators=(',', ':'))}\n\n"
                    get_event = None
                    continue
                get_event.cancel()
                try:
                    await get_event
                except BaseException:
                    pass
                get_event = None
                try:
                    result = task.result()
                except BaseException as exc:
                    execution_error = exc
                while not queue.empty():
                    yield f"data: {json.dumps(queue.get_nowait(), separators=(',', ':'))}\n\n"
                break
        except asyncio.CancelledError:
            raise
        finally:
            # ``aclose()`` injects GeneratorExit while a consumer cancellation
            # injects CancelledError; both paths share the same durable
            # cleanup, but a normally completed task is never rewritten.
            if get_event is not None:
                if not get_event.done():
                    get_event.cancel()
                try:
                    await get_event
                except BaseException:
                    pass
            needs_durable_cancel = (
                not task.done()
                or isinstance(execution_error, asyncio.CancelledError)
            )
            if needs_durable_cancel:
                token = getattr(prepared, "cancellation", None)
                if token is not None and callable(getattr(token, "cancel", None)):
                    token.cancel()
                cancel_prepared = getattr(service, "cancel_prepared", None)
                try:
                    if callable(cancel_prepared):
                        cancel_prepared(prepared)
                    else:
                        EvalExecutionControlRepository(session).cancel_run(prepared.run.id)
                except Exception:
                    # The task is still cancelled/awaited below.  A durable
                    # endpoint remains available to reconcile a DB failure.
                    pass
                if not task.done():
                    task.cancel()
            if task.done():
                try:
                    await asyncio.shield(task)
                except BaseException:
                    pass
            else:
                try:
                    done, _ = await asyncio.wait(
                        {task}, timeout=_CANCEL_HANDOFF_TIMEOUT
                    )
                except BaseException:
                    done = set()
                if task in done:
                    try:
                        task.result()
                    except BaseException:
                        pass
                else:
                    def consume_late_task(done_task: asyncio.Future[Any]) -> None:
                        try:
                            done_task.exception()
                        except BaseException:
                            pass

                    task.add_done_callback(consume_late_task)
        if not terminal_sent:
            if not getattr(emit, "run_created_sent", False):
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "schema_version": "eval-api-v1",
                            "type": "run_created",
                            "run_id": prepared.run.id,
                        },
                        separators=(",", ":"),
                    )
                    + "\n\n"
                )
            if execution_error is not None:
                durable_event = None
                try:
                    durable_run = EvalRunRepository(session).get_verified(prepared.run.id)
                    durable_event = _durable_terminal_event(durable_run)
                except Exception:
                    durable_event = None
                event = durable_event or {
                    "schema_version": "eval-api-v1",
                    "type": "run_finished",
                    "run_id": prepared.run.id,
                    "lifecycle": "finished",
                    "outcome": "system_failed",
                    "error_code": "harness_internal_error",
                }
            else:
                run = getattr(result, "run", prepared.run)
                event = {
                    "schema_version": "eval-api-v1",
                    "type": "run_finished",
                    "run_id": prepared.run.id,
                    "lifecycle": getattr(run, "lifecycle", "finished"),
                    "outcome": getattr(run, "outcome", None),
                }
                operational_error = getattr(run, "operational_error_json", None)
                if isinstance(operational_error, Mapping) and operational_error.get("code"):
                    event["error_code"] = str(operational_error["code"])
            yield f"data: {json.dumps(event, separators=(',', ':'))}\n\n"

    return StreamingResponse(body(), media_type="text/event-stream")


__all__ = ["connection_fingerprint", "eval_router"]
