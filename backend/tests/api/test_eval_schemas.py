"""Task 6 API v1 contract fixtures and strict DTO tests."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
from pydantic import TypeAdapter


EXAMPLES = Path(__file__).resolve().parents[3] / "contracts" / "eval-api-v1" / "examples"


def _schema_api():
    try:
        from app.api.eval_schemas import (
            CompareResponse,
            EvalEvent,
            EvalErrorDetail,
            RunDetail,
            RunStreamRequest,
        )
    except ModuleNotFoundError as exc:  # intentional Task 6 RED guard
        pytest.fail(f"eval API schema module is missing: {exc}", pytrace=False)
    return CompareResponse, EvalEvent, EvalErrorDetail, RunDetail, RunStreamRequest


def test_run_request_is_strict_and_forbids_client_owned_definition_fields():
    _, _, _, _, RunStreamRequest = _schema_api()

    request = RunStreamRequest(
        experiment_id="tutor-prompt-regression-v1",
        task_case_id="tgqa-004",
        variant_id="tutor-v3",
        run_profile="evaluation",
    )
    assert request.model_dump() == {
        "experiment_id": "tutor-prompt-regression-v1",
        "task_case_id": "tgqa-004",
        "variant_id": "tutor-v3",
        "run_profile": "evaluation",
    }
    with pytest.raises(Exception):
        RunStreamRequest(
            experiment_id="tutor-prompt-regression-v1",
            task_case_id="tgqa-004",
            variant_id="tutor-v3",
            run_profile="evaluation",
            prompt="client-injected",
        )
    with pytest.raises(Exception):
        RunStreamRequest(
            experiment_id=123,
            task_case_id="tgqa-004",
            variant_id="tutor-v3",
            run_profile="evaluation",
        )


def test_stream_example_lines_validate_against_discriminated_event_union():
    _, EvalEvent, _, _, _ = _schema_api()
    path = EXAMPLES / "run-stream.jsonl"
    assert path.is_file()
    adapter = TypeAdapter(EvalEvent)
    events = [adapter.validate_json(line) for line in path.read_text().splitlines() if line.strip()]
    assert events
    assert events[0].type == "run_created"
    assert events[0].run_id
    assert sum(event.type == "run_finished" for event in events) == 1
    assert all(event.schema_version == "eval-api-v1" for event in events)


def test_detail_compare_and_busy_examples_are_json_payloads_without_secrets():
    CompareResponse, _, EvalErrorDetail, RunDetail, _ = _schema_api()
    detail = json.loads((EXAMPLES / "run-detail.json").read_text())
    compare = json.loads((EXAMPLES / "compare-controlled.json").read_text())
    busy = json.loads((EXAMPLES / "evaluation-busy.json").read_text())
    RunDetail.model_validate_json((EXAMPLES / "run-detail.json").read_text())
    CompareResponse.model_validate(compare)
    error = EvalErrorDetail.model_validate_json((EXAMPLES / "evaluation-busy.json").read_text())
    assert error.code == "evaluation_busy"
    assert error.active_kind in {"run", "score_set"}
    dumped = json.dumps([detail, compare, busy]).lower()
    for secret in ("authorization", "api_key", "secret-value", "https://"):
        assert secret not in dumped


def test_response_dtos_reject_unknown_db_lifecycle_and_score_status_values():
    from app.api.eval_schemas import RunSummary, ScoreSetSummary

    now = datetime(2026, 8, 12)
    with pytest.raises(Exception):
        RunSummary(
            run_id="run-1",
            experiment_id="experiment-1",
            suite_execution_id=None,
            task_case_id="case-1",
            variant_id="tutor-v3",
            run_profile="evaluation",
            lifecycle="unknown",
            outcome=None,
            latest_score_set=None,
            created_at=now,
            started_at=now,
            finished_at=None,
        )
    with pytest.raises(Exception):
        RunSummary(
            run_id="run-1",
            experiment_id="experiment-1",
            suite_execution_id=None,
            task_case_id="case-1",
            variant_id="tutor-v3",
            run_profile="evaluation",
            lifecycle="finished",
            outcome="unknown",
            latest_score_set=None,
            created_at=now,
            started_at=now,
            finished_at=now,
        )
    with pytest.raises(Exception):
        ScoreSetSummary(
            score_set_id="score-1",
            scorer_id="scorer",
            scorer_version="v1",
            status="unknown",
            quality_verdict="pass",
            created_at=now,
        )
    with pytest.raises(Exception):
        ScoreSetSummary(
            score_set_id="score-1",
            scorer_id="scorer",
            scorer_version="v1",
            status="completed",
            quality_verdict="unknown",
            created_at=now,
        )


def test_scorer_event_dtos_enforce_disjoint_status_literals():
    from app.api.eval_schemas import (
        ScoreSetFinishedEvent,
        ScorerCompletedEvent,
        ScorerFailedEvent,
    )

    common = {
        "schema_version": "eval-api-v1",
        "run_id": "run-1",
        "score_set_id": "score-1",
        "scorer_id": "scorer",
    }
    with pytest.raises(Exception):
        ScorerCompletedEvent(**common, type="scorer_completed", status="failed")
    with pytest.raises(Exception):
        ScorerFailedEvent(**common, type="scorer_failed", status="skipped")
    for non_terminal_status in ("pending", "running"):
        with pytest.raises(Exception):
            ScoreSetFinishedEvent(
                schema_version="eval-api-v1",
                type="score_set_finished",
                run_id="run-1",
                score_set_id="score-1",
                status=non_terminal_status,
                quality_verdict="not_evaluated",
            )
