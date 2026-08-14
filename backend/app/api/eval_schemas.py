"""Strict, versioned DTOs for the local Learning Run API."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class RunStreamRequest(StrictModel):
    experiment_id: str
    task_case_id: str
    variant_id: str
    run_profile: Literal["evaluation"]


class EvalErrorDetail(StrictModel):
    code: str
    message: str
    fields: tuple[str, ...] = ()
    active_entity_id: str | None = None
    active_kind: Literal["run", "score_set"] | None = None


class VariantSummary(StrictModel):
    variant_id: str
    prompt_version: str


class ExperimentSummary(StrictModel):
    experiment_id: str
    task_family: str
    experiment_axes: tuple[str, ...]
    variants: tuple[VariantSummary, ...]
    case_counts: dict[str, int]
    run_profile: str
    budgets: dict[str, int]


class ScoreSetSummary(StrictModel):
    score_set_id: str
    scorer_id: str
    scorer_version: str
    status: Literal["pending", "running", "completed", "partial", "failed", "cancelled"]
    quality_verdict: Literal["pass", "fail", "inconclusive", "not_evaluated"]
    aggregate_scores: dict[str, object] | None = None
    created_at: datetime
    finished_at: datetime | None = None


class RunSummary(StrictModel):
    run_id: str
    experiment_id: str
    suite_execution_id: str | None
    task_case_id: str
    variant_id: str
    run_profile: str
    lifecycle: Literal["queued", "running", "finished", "cancelled"]
    outcome: Literal["success", "system_failed", "timed_out", "budget_exceeded"] | None
    latest_score_set: ScoreSetSummary | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class ScorerExecutionDetail(StrictModel):
    execution_id: str
    score_set_id: str
    scorer_id: str
    scorer_version: str
    status: Literal["success", "failed", "skipped"]
    input_hash: str
    output: dict | list | None = None
    operational_error_code: str | None = None
    operational_error_message: str | None = None
    latency_ms: int | None = None
    usage: dict | None = None
    created_at: datetime


class ScoreSetDetail(ScoreSetSummary):
    artifact_input_hash: str
    operational_error_code: str | None = None
    operational_error_message: str | None = None
    findings: list | dict | None = None


class RunDetail(StrictModel):
    summary: RunSummary
    manifest: dict
    candidate_artifact: dict | None
    score_sets: list[ScoreSetDetail]
    scorer_executions: list[ScorerExecutionDetail]
    operational_error: dict | None


class CompareRunRef(StrictModel):
    run_id: str
    variant_id: str


class ScorerBundleRef(StrictModel):
    scorer_id: str
    version: str


class CompareResponse(StrictModel):
    compatibility: Literal["controlled", "informational", "incompatible"]
    reasons: list[str]
    left: CompareRunRef
    right: CompareRunRef
    scorer_bundle: ScorerBundleRef
    delta: dict | None
    scope: Literal["case", "suite"]


class EvalEventBase(StrictModel):
    schema_version: Literal["eval-api-v1"]
    type: str
    run_id: str


class RunCreatedEvent(EvalEventBase):
    type: Literal["run_created"]


class StageStartedEvent(EvalEventBase):
    type: Literal["stage_started"]
    stage: str


class StageCompletedEvent(EvalEventBase):
    type: Literal["stage_completed"]
    stage: str


class ScoreSetCreatedEvent(EvalEventBase):
    type: Literal["score_set_created"]
    score_set_id: str


class ScorerCompletedEvent(EvalEventBase):
    type: Literal["scorer_completed"]
    score_set_id: str
    scorer_id: str
    status: Literal["success", "skipped"]


class ScorerFailedEvent(EvalEventBase):
    type: Literal["scorer_failed"]
    score_set_id: str
    scorer_id: str
    status: Literal["failed"]
    error_code: str | None = None


class ScoreSetFinishedEvent(EvalEventBase):
    type: Literal["score_set_finished"]
    score_set_id: str
    status: Literal["completed", "partial", "failed", "cancelled"]
    quality_verdict: Literal["pass", "fail", "inconclusive", "not_evaluated"]


class RunFinishedEvent(EvalEventBase):
    type: Literal["run_finished"]
    lifecycle: Literal["finished", "cancelled"]
    outcome: Literal["success", "system_failed", "timed_out", "budget_exceeded"] | None = None
    error_code: str | None = None


EvalEvent = Annotated[
    RunCreatedEvent
    | StageStartedEvent
    | StageCompletedEvent
    | ScoreSetCreatedEvent
    | ScorerCompletedEvent
    | ScorerFailedEvent
    | ScoreSetFinishedEvent
    | RunFinishedEvent,
    Field(discriminator="type"),
]


__all__ = [
    "CompareResponse",
    "EvalErrorDetail",
    "EvalEvent",
    "ExperimentSummary",
    "RunDetail",
    "RunStreamRequest",
    "RunSummary",
]
