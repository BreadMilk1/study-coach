"""Strict TDD contract tests for the Learning Run hybrid scorer.

The scorer is intentionally tested at its public boundary.  In particular,
these tests use injected fake chat models and frozen registry artifacts; they
never construct a provider model or use the production Judge parser.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import FrozenInstanceError, replace
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from app.db.models import Base
from app.eval.learning_run.contracts import (
    CandidateArtifact,
    CalibrationTaskInput,
    RunManifest,
    ScorerBundle,
    ScorerComponent,
    ScorerExecutionDraft,
    ScoreSetResultDraft,
    TaskCase,
    canonical_hash,
)
from app.eval.learning_run.registry import TaskRegistry
from app.eval.learning_run.repositories import (
    EvalRunRepository,
    EvalScoreSetRepository,
    EvalScorerExecutionRepository,
)


def _scoring_api():
    """Turn a missing Task 4 surface into an explicit feature RED."""

    try:
        from app.eval.learning_run.scoring import (
            ScoringService,
            derive_score_set,
            derive_verdict,
            parse_rubric_output,
        )
    except ModuleNotFoundError as exc:  # pragma: no cover - intentional RED guard
        pytest.fail(f"learning-run scoring module is missing: {exc}", pytrace=False)
    return ScoringService, derive_score_set, derive_verdict, parse_rubric_output


REGISTRY = TaskRegistry.load_default()
SCORER = REGISTRY.scorer
ANSWERABLE = REGISTRY.task_cases["tgqa-001"]
MULTI_EVIDENCE = REGISTRY.task_cases["tgqa-009"]
REFUSAL = REGISTRY.task_cases["tgqa-004"]


def _artifact(
    *,
    answer: str = "RRF combines ranked lists [1].",
    citations: tuple[dict[str, Any], ...] | None = None,
    exact_evidence: tuple[dict[str, Any], ...] | None = None,
    usage: Any = "unavailable",
) -> CandidateArtifact:
    evidence = exact_evidence if exact_evidence is not None else (
        {
            "chunk_id": "tgqa-c01-rrf",
            "content": "RRF combines several ranked lists.",
            "source": "learning-run-notes.md",
            "page": 1,
        },
    )
    return CandidateArtifact(
        answer=answer,
        citations=citations
        if citations is not None
        else (
            {
                "chunk_id": evidence[0]["chunk_id"],
                "source": evidence[0]["source"],
                "page": evidence[0]["page"],
                "span_start": 0,
                "span_end": len(evidence[0]["content"]),
            },
        ),
        exact_evidence=evidence,
        formatted_context="[1] learning-run-notes.md p.1: RRF combines several ranked lists.",
        usage=usage,
        trace=({"stage": "tutor", "event": "complete"},),
        budget={"total_seconds": 1},
    )


def _refusal_artifact(*, answer: str = "I don't know; the sources do not contain that fact."):
    return _artifact(answer=answer, citations=(), exact_evidence=())


def _response(payload: Any, *, usage_metadata: Any = None, response_metadata: Any = None):
    return SimpleNamespace(
        content=json.dumps(payload) if not isinstance(payload, str) else payload,
        usage_metadata=usage_metadata,
        response_metadata=response_metadata,
    )


class FakeLLM:
    def __init__(self, response: Any = None, *, error: BaseException | None = None):
        self.response = response
        self.error = error
        self.calls: list[Any] = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        return self.response


def _valid_scores(task: TaskCase | CalibrationTaskInput, *, value: int = 4):
    return {dimension: value for dimension in task.required_dimensions}


def _rubric_payload(task, *, value: int = 4, findings=None, reasoning="clear"):
    payload = _valid_scores(task, value=value)
    payload.update({"reasoning": reasoning, "findings": findings or []})
    return payload


def _task_for_calibration(candidate):
    return candidate.task


def _manifest_for_scoring() -> RunManifest:
    return RunManifest(
        experiment_id="experiment-scoring",
        task_case_id="tgqa-001",
        task_case_version="1",
        variant_id="tutor-v2",
        run_profile="evaluation",
        task_snapshot={"id": "tgqa-001", "version": "1", "question": "question"},
        prompt_text="frozen prompt",
        corpus_snapshot={"snapshot_id": "corpus-scoring", "version": "1", "aggregate_hash": "c" * 64},
        scorer_snapshot={"id": "hybrid", "version": "v1", "hash": "s" * 64},
        connection_fingerprint="d" * 64,
        corpus_snapshot_id="corpus-scoring",
        corpus_snapshot_version="1",
        corpus_snapshot_hash="c" * 64,
        prompt_version="tutor-v2",
        prompt_hash="p" * 64,
        scorer_bundle_version="hybrid-v1",
        scorer_bundle_hash="s" * 64,
        provider="test-provider",
        model="test-model",
        model_parameters={"temperature": 0, "max_tokens": 32},
        retrieval_config={"top_k": 5},
        reranker_config={"version": "test-reranker-v1"},
        chunking_config_version="chunking-v1",
        embedding_config_version="embedding-v1",
        budget={"total_seconds": 1},
        runtime_judge=False,
        runner_version="runner-v1",
        schema_version="learning-run-test-v1",
        code_revision="test-revision",
    )


@pytest.fixture
def scoring_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def _running_score_set(session, artifact: CandidateArtifact):
    manifest = _manifest_for_scoring()
    run_repo = EvalRunRepository(session)
    run = run_repo.create(
        id="run-scoring-identity",
        experiment_id=manifest.experiment_id,
        task_case_id=manifest.task_case_id,
        task_case_version=manifest.task_case_version,
        variant_id=manifest.variant_id,
        run_profile=manifest.run_profile,
        manifest=manifest,
        manifest_hash=manifest.compute_hash(),
    )
    running = run_repo.claim_running(run.id)
    finished = run_repo.finalize_candidate(
        running.id,
        expected_lifecycle="running",
        candidate_artifact=artifact,
        artifact_hash=artifact.compute_hash(),
    )
    score_repo = EvalScoreSetRepository(session)
    score_set = score_repo.create(
        run_id=finished.id,
        scorer_bundle=SCORER,
        artifact_input_hash=artifact.compute_hash(),
        id="score-set-scoring-identity",
    )
    score_repo.claim_running(score_set.id)
    return score_set


@pytest.mark.parametrize(
    ("case_type", "scores", "hard_findings", "failed_scorers", "verdict"),
    [
        ("answerable", {"groundedness": 4, "citation_entailment": 4, "coverage": 4}, [], [], "pass"),
        ("answerable", {"groundedness": 5, "citation_entailment": 3, "coverage": 5}, [], [], "fail"),
        ("expected_refusal", {"refusal_appropriateness": 4, "unsupported_claims": 4}, [], [], "pass"),
        ("answerable", {"groundedness": 5, "citation_entailment": 5, "coverage": 5}, ["citation_invalid"], [], "fail"),
        ("answerable", {}, [], ["scorer_parse_error"], "inconclusive"),
    ],
)
def test_verdict_policy_truth_table(case_type, scores, hard_findings, failed_scorers, verdict):
    _, _, derive_verdict, _ = _scoring_api()
    result = derive_verdict(
        case_type=case_type,
        dimension_scores=scores,
        hard_findings=hard_findings,
        failed_scorers=failed_scorers,
    )
    assert result.verdict == verdict


def test_noncritical_finding_is_visible_without_blocking_pass():
    _, _, derive_verdict, _ = _scoring_api()
    result = derive_verdict(
        case_type="answerable",
        dimension_scores={"groundedness": 4, "citation_entailment": 4, "coverage": 4},
        hard_findings=[],
        failed_scorers=[],
        findings=[{"code": "incomplete_answer", "severity": "noncritical"}],
    )
    assert result.verdict == "pass"
    assert any(item["code"] == "incomplete_answer" for item in result.findings)


def test_unknown_case_and_invalid_dimension_fail_closed():
    _, _, derive_verdict, _ = _scoring_api()
    assert derive_verdict(
        case_type="unknown",
        dimension_scores={"groundedness": 5},
        hard_findings=[],
        failed_scorers=[],
    ).verdict != "pass"
    assert derive_verdict(
        case_type="answerable",
        dimension_scores={"groundedness": True, "citation_entailment": 4, "coverage": 4},
        hard_findings=[],
        failed_scorers=[],
    ).verdict != "pass"


def test_contract_drafts_are_deep_frozen_and_reject_negative_latency():
    from app.eval.learning_run.contracts import ScorerExecutionDraft, ScoreSetResultDraft

    execution = ScorerExecutionDraft(
        component_id="component",
        component_version="v1",
        scorer_id="component",
        scorer_version="v1",
        status="success",
        input_hash="a" * 64,
        output={"nested": {"values": [1]}},
        findings=({"code": "retrieval_empty", "severity": "noncritical"},),
        latency_ms=0,
        usage=None,
    )
    assert isinstance(execution.output, MappingProxyType)
    assert isinstance(execution.output["nested"], MappingProxyType)
    assert isinstance(execution.output["nested"]["values"], tuple)
    with pytest.raises((TypeError, FrozenInstanceError)):
        execution.output["nested"]["values"] = ()
    with pytest.raises(ValueError, match="latency"):
        ScorerExecutionDraft(
            component_id="component",
            component_version="v1",
            scorer_id="component",
            scorer_version="v1",
            status="success",
            input_hash="a" * 64,
            latency_ms=-1,
        )
    result = ScoreSetResultDraft(
        status="completed",
        verdict="pass",
        aggregate_scores={"groundedness": 4},
        findings=({"code": "incomplete_answer", "severity": "noncritical"},),
        executions=(execution,),
    )
    assert isinstance(result.aggregate_scores, MappingProxyType)
    assert result.executions == (execution,)


def _draft_base(**overrides):
    values = {
        "component_id": "component",
        "component_version": "v1",
        "scorer_id": "component",
        "scorer_version": "v1",
        "status": "success",
        "input_hash": "a" * 64,
        "output": {"ok": True},
        "error_code": None,
        "error_message": None,
        "latency_ms": 0,
        "usage": None,
        "findings": (),
    }
    values.update(overrides)
    return ScorerExecutionDraft(**values)


@pytest.mark.parametrize(
    "overrides",
    [
        {"status": "success", "error_code": "harness_internal_error", "error_message": "bad"},
        {"status": "failed", "error_code": None, "error_message": None, "output": None},
        {"status": "failed", "error_code": "harness_internal_error", "error_message": "bad", "output": {"score": 4}},
        {"status": "failed", "error_code": "harness_internal_error", "error_message": "bad", "output": None, "usage": {"input_tokens": 1}},
        {"status": "failed", "error_code": "harness_internal_error", "error_message": "bad", "output": None, "findings": ({"code": "unsupported_claim", "severity": "critical"},)},
        {"status": "skipped", "error_code": "harness_internal_error", "error_message": "bad"},
        {"status": "failed", "error_code": "harness_internal_error", "error_message": None, "output": None},
        {"status": "failed", "error_code": None, "error_message": "bad", "output": None},
        {"status": "failed", "error_code": "not-a-stable-code", "error_message": "bad", "output": None},
    ],
)
def test_scorer_execution_draft_rejects_inconsistent_operational_fields(overrides):
    with pytest.raises((TypeError, ValueError)):
        _draft_base(**overrides)


@pytest.mark.parametrize(
    "usage",
    [
        {"Authorization": "Bearer secret"},
        {"api_key": "secret"},
        {"unknown": 1},
        {"input_tokens": -1},
        {"input_tokens": True},
        {"input_tokens": "1"},
        {"prompt_tokens": 1},
        {"input_tokens": 1, "Authorization": "Bearer secret"},
        {"input_tokens": 1, "output_tokens": -1},
        {"input_tokens": 1, "total_tokens": True},
        {"input_tokens": 1, "output_tokens": "2"},
    ],
)
def test_scorer_execution_draft_rejects_unsanitized_or_noncanonical_usage(usage):
    with pytest.raises((TypeError, ValueError)):
        _draft_base(usage=usage)


def test_scorer_execution_draft_accepts_only_frozen_canonical_usage():
    execution = _draft_base(
        usage={"input_tokens": 11, "output_tokens": 13, "total_tokens": 24}
    )
    assert execution.usage == {
        "input_tokens": 11,
        "output_tokens": 13,
        "total_tokens": 24,
    }
    assert isinstance(execution.usage, MappingProxyType)
    with pytest.raises(TypeError):
        execution.usage["input_tokens"] = 99  # type: ignore[index]
    assert execution.to_dict()["usage"] == {
        "input_tokens": 11,
        "output_tokens": 13,
        "total_tokens": 24,
    }


@pytest.mark.parametrize(
    "overrides",
    [
        {"status": "skipped", "output": {"applicable": False}, "usage": {"input_tokens": 1}},
        {
            "status": "skipped",
            "output": {"applicable": False},
            "findings": ({"code": "retrieval_empty", "severity": "noncritical"},),
        },
    ],
)
def test_skipped_scorer_execution_draft_cannot_contain_usage_or_findings(overrides):
    with pytest.raises((TypeError, ValueError)):
        _draft_base(**overrides)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"status": "completed", "verdict": "pass", "error_code": "harness_internal_error", "error_message": "bad"},
        {"status": "partial", "verdict": "pass"},
        {"status": "failed", "verdict": "pass"},
        {"status": "partial", "verdict": "inconclusive", "error_code": "harness_internal_error"},
        {"status": "failed", "verdict": "inconclusive", "error_message": "bad"},
    ],
)
def test_score_set_result_draft_rejects_inconsistent_operational_fields(kwargs):
    with pytest.raises((TypeError, ValueError)):
        ScoreSetResultDraft(**kwargs)


@pytest.mark.asyncio
async def test_deterministic_components_callback_once_in_frozen_order():
    ScoringService, _, _, _ = _scoring_api()
    service = ScoringService(FakeLLM(_response(_rubric_payload(ANSWERABLE))))
    seen = []
    result = await service.score(
        task=ANSWERABLE,
        candidate=_artifact(),
        scorer_bundle=SCORER,
        on_execution=seen.append,
    )
    expected = [component.component_id for component in SCORER.components]
    assert [item.component_id for item in seen] == expected
    assert [item.component_id for item in result.executions] == expected
    assert len(seen) == len(expected) == 4
    assert all(item.input_hash == canonical_hash(_artifact().to_dict()) for item in seen)


@pytest.mark.asyncio
async def test_callback_exception_is_not_swallowed():
    ScoringService, _, _, _ = _scoring_api()
    service = ScoringService(FakeLLM(_response(_rubric_payload(ANSWERABLE))))

    def callback(_):
        raise RuntimeError("callback failure")

    with pytest.raises(RuntimeError, match="callback failure"):
        await service.score(
            task=ANSWERABLE,
            candidate=_artifact(),
            scorer_bundle=SCORER,
            on_execution=callback,
        )


@pytest.mark.asyncio
async def test_scoring_drafts_append_once_with_frozen_component_identities(scoring_session):
    ScoringService, _, _, _ = _scoring_api()
    artifact = _artifact()
    seen = []
    result = await ScoringService(FakeLLM(_response(_rubric_payload(ANSWERABLE)))).score(
        task=ANSWERABLE,
        candidate=artifact,
        scorer_bundle=SCORER,
        on_execution=seen.append,
    )
    score_set = _running_score_set(scoring_session, artifact)
    execution_repo = EvalScorerExecutionRepository(scoring_session)
    persisted = []
    for draft in result.executions:
        persisted.append(
            execution_repo.append(
                score_set_id=score_set.id,
                scorer_id=draft.scorer_id,
                scorer_version=draft.scorer_version,
                status=draft.status,
                input_hash=draft.input_hash,
                output=draft.output,
                error_code=draft.error_code,
                sanitized_message=draft.error_message,
                latency_ms=draft.latency_ms,
                usage=draft.usage,
            )
        )
    assert len(persisted) == len(SCORER.components) == 4
    assert [(row.scorer_id, row.scorer_version) for row in persisted] == [
        (component.component_id, component.version) for component in SCORER.components
    ]


async def _valid_scored_result():
    ScoringService, _, _, _ = _scoring_api()
    artifact = _artifact()
    result = await ScoringService(FakeLLM(_response(_rubric_payload(ANSWERABLE)))).score(
        task=ANSWERABLE,
        candidate=artifact,
        scorer_bundle=SCORER,
        on_execution=lambda _: None,
    )
    return artifact, result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_component_id",
        "duplicate_component",
        "out_of_order",
        "wrong_component_version",
        "wrong_scorer_version",
        "wrong_scorer_id",
        "all_same_input_hash",
        "wrong_expected_input_hash",
    ],
)
async def test_derive_score_set_rejects_identity_order_and_hash_mutations(mutation):
    _, derive_score_set, _, _ = _scoring_api()
    artifact, result = await _valid_scored_result()
    drafts = list(result.executions)
    if mutation == "wrong_component_id":
        drafts[0] = replace(drafts[0], component_id="not-a-frozen-component")
    elif mutation == "duplicate_component":
        drafts[1] = replace(drafts[1], component_id=drafts[0].component_id)
    elif mutation == "out_of_order":
        drafts[0], drafts[1] = drafts[1], drafts[0]
    elif mutation == "wrong_component_version":
        drafts[0] = replace(drafts[0], component_version="v999")
    elif mutation == "wrong_scorer_version":
        drafts[0] = replace(drafts[0], scorer_version="hybrid-v999")
    elif mutation == "wrong_scorer_id":
        drafts[0] = replace(drafts[0], scorer_id="hybrid")
    elif mutation == "all_same_input_hash":
        drafts = [replace(draft, input_hash="b" * 64) for draft in drafts]
    else:
        assert mutation == "wrong_expected_input_hash"

    expected_hash = "c" * 64 if mutation == "wrong_expected_input_hash" else artifact.compute_hash()
    derived = derive_score_set(
        task=ANSWERABLE,
        scorer_bundle=SCORER,
        executions=tuple(drafts),
        input_hash=expected_hash,
    )
    assert derived.status == "failed"
    assert derived.verdict == "inconclusive"
    assert derived.aggregate_scores == {}
    assert len(derived.executions) == len(SCORER.components)


@pytest.mark.asyncio
async def test_derive_score_set_only_llm_success_can_supply_dimension_scores():
    _, derive_score_set, _, _ = _scoring_api()
    artifact, result = await _valid_scored_result()
    drafts = list(result.executions)
    retrieval = drafts[0]
    llm_index = next(
        index
        for index, draft in enumerate(drafts)
        if draft.component_id == "grounded-quality-rubric"
    )
    drafts[0] = replace(
        retrieval,
        output={"groundedness": 5, "citation_entailment": 5, "coverage": 5},
    )
    drafts[llm_index] = replace(
        drafts[llm_index],
        output={"groundedness": 4, "reasoning": "partial", "findings": []},
    )
    derived = derive_score_set(
        task=ANSWERABLE,
        scorer_bundle=SCORER,
        executions=tuple(drafts),
        input_hash=artifact.compute_hash(),
    )
    assert derived.aggregate_scores == {"groundedness": 4}
    assert derived.verdict == "inconclusive"


@pytest.mark.asyncio
async def test_expected_refusal_observed_requires_explicit_refusal_language():
    ScoringService, _, _, _ = _scoring_api()
    fabricated = _artifact(
        answer="The requested fact is definitely 42.",
        citations=(),
        exact_evidence=(),
    )
    seen = []
    await ScoringService(FakeLLM(_response(_rubric_payload(REFUSAL)))).score(
        task=REFUSAL,
        candidate=fabricated,
        scorer_bundle=SCORER,
        on_execution=seen.append,
    )
    observation = next(
        draft for draft in seen if draft.component_id == "expected-refusal-observation"
    )
    assert observation.findings == ()

    seen.clear()
    await ScoringService(FakeLLM(_response(_rubric_payload(REFUSAL)))).score(
        task=REFUSAL,
        candidate=_refusal_artifact(),
        scorer_bundle=SCORER,
        on_execution=seen.append,
    )
    observation = next(
        draft for draft in seen if draft.component_id == "expected-refusal-observation"
    )
    assert [finding["code"] for finding in observation.findings] == [
        "expected_refusal_observed"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("answer", "observed"),
    [
        ("These two options cannot both be correct; the answer is blue.", False),
        ("The method cannot guarantee convergence, so use RRF.", False),
        ("I cannot answer from the provided evidence.", True),
        ("I don't know; the provided sources do not contain that fact.", True),
    ],
)
async def test_refusal_observation_avoids_generic_cannot_phrases(answer, observed):
    ScoringService, _, _, _ = _scoring_api()
    seen = []
    await ScoringService(FakeLLM(_response(_rubric_payload(REFUSAL)))).score(
        task=REFUSAL,
        candidate=_artifact(answer=answer, citations=(), exact_evidence=()),
        scorer_bundle=SCORER,
        on_execution=seen.append,
    )
    observation = next(
        draft for draft in seen if draft.component_id == "expected-refusal-observation"
    )
    assert ("expected_refusal_observed" in {finding["code"] for finding in observation.findings}) is observed


@pytest.mark.asyncio
async def test_refusal_observation_uses_answer_text_even_when_retrieval_is_populated():
    ScoringService, _, _, _ = _scoring_api()
    seen = []
    await ScoringService(FakeLLM(_response(_rubric_payload(REFUSAL)))).score(
        task=REFUSAL,
        candidate=_artifact(
            answer="I don't know; the provided sources do not contain that fact."
        ),
        scorer_bundle=SCORER,
        on_execution=seen.append,
    )
    observation = next(
        draft for draft in seen if draft.component_id == "expected-refusal-observation"
    )
    assert "expected_refusal_observed" in {finding["code"] for finding in observation.findings}


@pytest.mark.asyncio
@pytest.mark.parametrize("span", [(0, 0), (5, 2), (0, 10_000)])
async def test_citation_span_requires_positive_in_bounds_interval(span):
    ScoringService, _, _, _ = _scoring_api()
    evidence = {
        "chunk_id": "tgqa-c01-rrf",
        "content": "RRF combines several ranked lists.",
        "source": "learning-run-notes.md",
        "page": 1,
    }
    candidate = _artifact(
        citations=(
            {
                "chunk_id": evidence["chunk_id"],
                "source": evidence["source"],
                "page": evidence["page"],
                "span_start": span[0],
                "span_end": span[1],
            },
        ),
        exact_evidence=(evidence,),
    )
    result = await ScoringService(FakeLLM(_response(_rubric_payload(ANSWERABLE)))).score(
        task=ANSWERABLE,
        candidate=candidate,
        scorer_bundle=SCORER,
        on_execution=lambda _: None,
    )
    citation = next(
        draft for draft in result.executions if draft.component_id == "citation-integrity"
    )
    assert any(finding["code"] == "citation_invalid" for finding in citation.findings)


@pytest.mark.asyncio
async def test_retrieval_empty_is_quality_finding_not_operational_failure():
    ScoringService, _, _, _ = _scoring_api()
    service = ScoringService(FakeLLM(_response(_rubric_payload(REFUSAL))))
    seen = []
    result = await service.score(
        task=REFUSAL,
        candidate=_refusal_artifact(),
        scorer_bundle=SCORER,
        on_execution=seen.append,
    )
    retrieval = next(item for item in seen if item.component_id == "retrieval-integrity")
    assert retrieval.status == "success"
    assert retrieval.error_code is None
    assert any(item["code"] == "retrieval_empty" for item in retrieval.findings)


@pytest.mark.asyncio
async def test_citation_numbering_maps_prompt_markers_to_used_sources_only():
    ScoringService, _, _, _ = _scoring_api()
    evidence = tuple(
        {
            "chunk_id": f"chunk-{index}",
            "content": f"chunk {index} content here.",
            "source": "learning-run-notes.md",
            "page": 1,
        }
        for index in range(1, 6)
    )
    citations = tuple(
        {
            "chunk_id": item["chunk_id"],
            "source": item["source"],
            "page": item["page"],
            "span_start": 0,
            "span_end": len(item["content"]),
        }
        for item in evidence
    )
    formatted = "\n".join(
        f"[{index}] learning-run-notes.md p.1: {item['content']}"
        for index, item in enumerate(evidence, start=1)
    )
    candidate = CandidateArtifact(
        answer="RRF combines ranked lists [N1].",
        citations=citations,
        exact_evidence=evidence,
        formatted_context=formatted,
        usage="unavailable",
        trace=({"stage": "tutor", "event": "complete"},),
        budget={"total_seconds": 1},
    )
    result = await ScoringService(FakeLLM(_response(_rubric_payload(ANSWERABLE)))).score(
        task=ANSWERABLE,
        candidate=candidate,
        scorer_bundle=SCORER,
        on_execution=lambda _: None,
    )
    citation = next(
        item for item in result.executions if item.component_id == "citation-integrity"
    )
    assert citation.status == "success"
    assert [
        item for item in citation.findings if item.get("check") == "citation_number"
    ] == []


@pytest.mark.asyncio
async def test_citation_numbering_rejects_markers_outside_the_source_list():
    ScoringService, _, _, _ = _scoring_api()
    result = await ScoringService(FakeLLM(_response(_rubric_payload(ANSWERABLE)))).score(
        task=ANSWERABLE,
        candidate=_artifact(answer="RRF combines ranked lists [N9]."),
        scorer_bundle=SCORER,
        on_execution=lambda _: None,
    )
    citation = next(
        item for item in result.executions if item.component_id == "citation-integrity"
    )
    assert any(item.get("check") == "citation_number" for item in citation.findings)


@pytest.mark.asyncio
async def test_citation_integrity_checks_membership_numbering_and_span():
    ScoringService, _, _, _ = _scoring_api()
    candidate = _artifact(
        answer="RRF combines ranked lists [2].",
        citations=(
            {
                "chunk_id": "not-an-evidence-chunk",
                "source": "wrong.md",
                "page": 99,
                "span_start": 99,
                "span_end": 100,
            },
        ),
    )
    service = ScoringService(FakeLLM(_response(_rubric_payload(ANSWERABLE))))
    result = await service.score(
        task=ANSWERABLE,
        candidate=candidate,
        scorer_bundle=SCORER,
        on_execution=lambda _: None,
    )
    citation = next(item for item in result.executions if item.component_id == "citation-integrity")
    assert citation.status == "success"
    assert {item["code"] for item in citation.findings} >= {"citation_invalid"}
    assert result.verdict == "fail"


@pytest.mark.asyncio
async def test_expected_refusal_observation_skips_for_non_refusal_and_never_hard_fails():
    ScoringService, _, _, _ = _scoring_api()
    seen = []
    service = ScoringService(FakeLLM(_response(_rubric_payload(ANSWERABLE))))
    await service.score(
        task=ANSWERABLE,
        candidate=_artifact(),
        scorer_bundle=SCORER,
        on_execution=seen.append,
    )
    skipped = next(item for item in seen if item.component_id == "expected-refusal-observation")
    assert skipped.status == "skipped"
    assert skipped.findings == ()

    seen.clear()
    service = ScoringService(FakeLLM(_response(_rubric_payload(REFUSAL))))
    await service.score(
        task=REFUSAL,
        candidate=_refusal_artifact(),
        scorer_bundle=SCORER,
        on_execution=seen.append,
    )
    observed = next(item for item in seen if item.component_id == "expected-refusal-observation")
    finding = next(item for item in observed.findings if item["code"] == "expected_refusal_observed")
    assert finding["severity"] == "noncritical"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        "not-json",
        {"groundedness": 4, "citation_entailment": 4, "reasoning": "ok", "findings": []},
        {"groundedness": 4, "citation_entailment": 4, "coverage": 4, "reasoning": "ok", "findings": [], "extra": 1},
        {"groundedness": 6, "citation_entailment": 4, "coverage": 4, "reasoning": "ok", "findings": []},
        {"groundedness": True, "citation_entailment": 4, "coverage": 4, "reasoning": "ok", "findings": []},
        {"groundedness": 4, "citation_entailment": 4, "coverage": 4, "reasoning": "ok", "findings": [{"code": "bad"}]},
    ],
)
async def test_strict_parser_rejects_malformed_missing_extra_out_of_range_bool_and_bad_findings(payload):
    ScoringService, _, _, _ = _scoring_api()
    service = ScoringService(FakeLLM(_response(payload)))
    seen = []
    result = await service.score(
        task=ANSWERABLE,
        candidate=_artifact(),
        scorer_bundle=SCORER,
        on_execution=seen.append,
    )
    llm = next(item for item in seen if item.component_id == "grounded-quality-rubric")
    assert llm.status == "failed"
    assert llm.error_code == "scorer_parse_error"
    assert llm.output is None
    assert llm.error_message in {"scorer output malformed", "scorer output schema invalid"}
    assert result.verdict == "inconclusive"


@pytest.mark.asyncio
async def test_parser_accepts_one_object_with_valid_findings_and_rejects_trailing_json():
    ScoringService, _, _, parse_rubric_output = _scoring_api()
    valid = _rubric_payload(
        ANSWERABLE,
        findings=[{"code": "incomplete_answer", "severity": "noncritical", "message": "minor gap"}],
    )
    parsed = parse_rubric_output(json.dumps(valid), ANSWERABLE.required_dimensions)
    assert parsed["groundedness"] == 4
    assert parsed["findings"][0]["code"] == "incomplete_answer"
    service = ScoringService(FakeLLM(_response(json.dumps(valid) + json.dumps(valid))))
    seen = []
    result = await service.score(
        task=ANSWERABLE,
        candidate=_artifact(),
        scorer_bundle=SCORER,
        on_execution=seen.append,
    )
    llm = next(item for item in seen if item.component_id == "grounded-quality-rubric")
    assert llm.status == "failed"
    assert result.verdict == "inconclusive"


def test_parser_rejects_oversized_raw_reasoning_messages_findings_and_surrogates():
    _, _, _, parse_rubric_output = _scoring_api()
    from app.eval.learning_run.scoring import RubricParseError

    valid = _rubric_payload(ANSWERABLE)
    oversized_raw = json.dumps({"reasoning": "x" * 2_000_000})
    with pytest.raises(RubricParseError):
        parse_rubric_output(oversized_raw, ANSWERABLE.required_dimensions)

    reasoning = dict(valid, reasoning="x" * 9_000)
    with pytest.raises(RubricParseError):
        parse_rubric_output(json.dumps(reasoning), ANSWERABLE.required_dimensions)

    message = dict(
        valid,
        findings=[
            {"code": "incomplete_answer", "severity": "noncritical", "message": "x" * 3_000}
        ],
    )
    with pytest.raises(RubricParseError):
        parse_rubric_output(json.dumps(message), ANSWERABLE.required_dimensions)

    too_many_findings = dict(
        valid,
        findings=[
            {"code": "incomplete_answer", "severity": "noncritical", "message": "gap"}
            for _ in range(21)
        ],
    )
    with pytest.raises(RubricParseError):
        parse_rubric_output(json.dumps(too_many_findings), ANSWERABLE.required_dimensions)

    surrogate = dict(valid, reasoning="bad\ud800")
    with pytest.raises(RubricParseError):
        parse_rubric_output(json.dumps(surrogate), ANSWERABLE.required_dimensions)


@pytest.mark.parametrize("opening, closing", [("[", "]"), ("{", "}")])
def test_parser_rejects_deep_nested_json_with_rubric_parse_error(opening, closing):
    _, _, _, parse_rubric_output = _scoring_api()
    from app.eval.learning_run.scoring import RubricParseError

    nested = opening * 1_200 + ("0" if opening == "[" else '"nested":0') + closing * 1_200
    raw = json.dumps(_rubric_payload(ANSWERABLE)).replace("\"groundedness\": 4", f'"groundedness": {nested}')
    assert len(raw.encode("utf-8")) < 64 * 1024
    with pytest.raises(RubricParseError):
        parse_rubric_output(raw, ANSWERABLE.required_dimensions)


@pytest.mark.asyncio
@pytest.mark.parametrize("opening, closing", [("[", "]"), ("{", "}")])
async def test_service_marks_deep_nested_json_as_scorer_parse_error(opening, closing):
    ScoringService, _, _, _ = _scoring_api()
    nested = opening * 1_200 + ("0" if opening == "[" else '"nested":0') + closing * 1_200
    raw = json.dumps(_rubric_payload(ANSWERABLE)).replace("\"groundedness\": 4", f'"groundedness": {nested}')
    seen = []
    result = await ScoringService(FakeLLM(_response(raw))).score(
        task=ANSWERABLE,
        candidate=_artifact(),
        scorer_bundle=SCORER,
        on_execution=seen.append,
    )
    llm = next(item for item in seen if item.component_id == "grounded-quality-rubric")
    assert llm.status == "failed"
    assert llm.error_code == "scorer_parse_error"
    assert result.verdict == "inconclusive"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "error_code"),
    [
        (asyncio.TimeoutError(), "scorer_timeout"),
        (RuntimeError("Authorization: Bearer secret-value"), "model_unavailable"),
    ],
)
async def test_timeout_and_exception_are_failed_without_sensitive_error_details(error, error_code):
    ScoringService, _, _, _ = _scoring_api()
    seen = []
    result = await ScoringService(FakeLLM(error=error)).score(
        task=ANSWERABLE,
        candidate=_artifact(),
        scorer_bundle=SCORER,
        on_execution=seen.append,
    )
    llm = next(item for item in seen if item.component_id == "grounded-quality-rubric")
    assert llm.status == "failed"
    assert llm.error_code == error_code
    assert "secret-value" not in (llm.error_message or "")
    assert "Traceback" not in (llm.error_message or "")
    assert result.verdict == "inconclusive"


@pytest.mark.asyncio
async def test_local_prompt_and_usage_exceptions_are_harness_errors(monkeypatch):
    from app.eval.learning_run import scoring as scoring_module

    ScoringService, _, _, _ = _scoring_api()
    seen = []
    original_build_prompt = scoring_module._build_prompt
    monkeypatch.setattr(scoring_module, "_build_prompt", lambda **_: (_ for _ in ()).throw(RuntimeError("local prompt")))
    result = await ScoringService(FakeLLM(_response(_rubric_payload(ANSWERABLE)))).score(
        task=ANSWERABLE,
        candidate=_artifact(),
        scorer_bundle=SCORER,
        on_execution=seen.append,
    )
    llm = next(item for item in seen if item.component_id == "grounded-quality-rubric")
    assert llm.error_code == "harness_internal_error"
    assert result.verdict == "inconclusive"

    monkeypatch.setattr(scoring_module, "_build_prompt", original_build_prompt)
    monkeypatch.setattr(scoring_module, "_extract_usage", lambda _: (_ for _ in ()).throw(RuntimeError("local usage")))
    seen.clear()
    result = await ScoringService(FakeLLM(_response(_rubric_payload(ANSWERABLE)))).score(
        task=ANSWERABLE,
        candidate=_artifact(),
        scorer_bundle=SCORER,
        on_execution=seen.append,
    )
    llm = next(item for item in seen if item.component_id == "grounded-quality-rubric")
    assert llm.error_code == "harness_internal_error"
    assert result.verdict == "inconclusive"


@pytest.mark.asyncio
async def test_cancelled_model_invocation_propagates():
    ScoringService, _, _, _ = _scoring_api()
    with pytest.raises(asyncio.CancelledError):
        await ScoringService(FakeLLM(error=asyncio.CancelledError())).score(
            task=ANSWERABLE,
            candidate=_artifact(),
            scorer_bundle=SCORER,
            on_execution=lambda _: None,
        )


@pytest.mark.asyncio
async def test_usage_metadata_priority_and_unavailable_is_none_not_zero():
    ScoringService, _, _, _ = _scoring_api()
    seen = []
    response = _response(
        _rubric_payload(ANSWERABLE),
        usage_metadata={"input_tokens": 11, "output_tokens": 4},
        response_metadata={"token_usage": {"input_tokens": 99}},
    )
    await ScoringService(FakeLLM(response)).score(
        task=ANSWERABLE,
        candidate=_artifact(),
        scorer_bundle=SCORER,
        on_execution=seen.append,
    )
    llm = next(item for item in seen if item.component_id == "grounded-quality-rubric")
    assert llm.usage == {"input_tokens": 11, "output_tokens": 4}
    seen.clear()
    response = _response(_rubric_payload(ANSWERABLE))
    await ScoringService(FakeLLM(response)).score(
        task=ANSWERABLE,
        candidate=_artifact(),
        scorer_bundle=SCORER,
        on_execution=seen.append,
    )
    llm = next(item for item in seen if item.component_id == "grounded-quality-rubric")
    assert llm.usage is None
    assert llm.usage != {"input_tokens": 0, "output_tokens": 0}


@pytest.mark.asyncio
async def test_usage_is_private_shape_filtered_and_aliases_are_canonicalized():
    ScoringService, _, _, _ = _scoring_api()
    response = _response(
        _rubric_payload(ANSWERABLE),
        usage_metadata={
            "input_tokens": 11,
            "prompt_tokens": 99,
            "output_tokens": -1,
            "completion_tokens": 13,
            "total_tokens": 15,
            "Authorization": "Bearer secret",
            "api_key": "sk-secret",
            "accessToken": "access-secret",
            "clientSecret": "client-secret",
            "unknown": {"secret": "nested"},
        },
    )
    seen = []
    await ScoringService(FakeLLM(response)).score(
        task=ANSWERABLE,
        candidate=_artifact(),
        scorer_bundle=SCORER,
        on_execution=seen.append,
    )
    llm = next(item for item in seen if item.component_id == "grounded-quality-rubric")
    assert llm.usage == {
        "input_tokens": 11,
        "output_tokens": 13,
        "total_tokens": 15,
    }
    serialized = llm.to_dict()
    assert serialized["usage"] == {
        "input_tokens": 11,
        "output_tokens": 13,
        "total_tokens": 15,
    }
    assert "secret" not in json.dumps(serialized, ensure_ascii=True)


@pytest.mark.asyncio
async def test_usage_falls_back_to_response_metadata_and_rejects_invalid_shapes():
    ScoringService, _, _, _ = _scoring_api()
    response = _response(
        _rubric_payload(ANSWERABLE),
        usage_metadata={"unknown": "drop-me", "api_key": "secret"},
        response_metadata={
            "token_usage": {
                "prompt_tokens": 7,
                "completion_tokens": 8,
                "total_tokens": True,
                "Authorization": "Bearer secret",
            }
        },
    )
    seen = []
    await ScoringService(FakeLLM(response)).score(
        task=ANSWERABLE,
        candidate=_artifact(),
        scorer_bundle=SCORER,
        on_execution=seen.append,
    )
    llm = next(item for item in seen if item.component_id == "grounded-quality-rubric")
    assert llm.usage == {"input_tokens": 7, "output_tokens": 8}

    response = _response(
        _rubric_payload(ANSWERABLE),
        usage_metadata={
            "input_tokens": -1,
            "output_tokens": False,
            "total_tokens": "15",
            "Authorization": "Bearer secret",
        },
        response_metadata=object(),
    )
    seen.clear()
    await ScoringService(FakeLLM(response)).score(
        task=ANSWERABLE,
        candidate=_artifact(),
        scorer_bundle=SCORER,
        on_execution=seen.append,
    )
    llm = next(item for item in seen if item.component_id == "grounded-quality-rubric")
    assert llm.usage is None
    assert "input_tokens\": 0" not in json.dumps(llm.to_dict())


@pytest.mark.asyncio
async def test_sanitized_usage_can_append_to_repository_without_secrets(scoring_session):
    ScoringService, _, _, _ = _scoring_api()
    artifact = _artifact()
    response = _response(
        _rubric_payload(ANSWERABLE),
        usage_metadata={
            "input_tokens": 4,
            "output_tokens": 5,
            "Authorization": "Bearer secret",
            "api_key": "secret",
        },
    )
    seen = []
    result = await ScoringService(FakeLLM(response)).score(
        task=ANSWERABLE,
        candidate=artifact,
        scorer_bundle=SCORER,
        on_execution=seen.append,
    )
    score_set = _running_score_set(scoring_session, artifact)
    llm = next(item for item in result.executions if item.component_id == "grounded-quality-rubric")
    row = EvalScorerExecutionRepository(scoring_session).append(
        score_set_id=score_set.id,
        scorer_id=llm.scorer_id,
        scorer_version=llm.scorer_version,
        status=llm.status,
        input_hash=llm.input_hash,
        output=llm.output,
        latency_ms=llm.latency_ms,
        usage=llm.usage,
    )
    assert row.usage_json == {"input_tokens": 4, "output_tokens": 5}
    assert "secret" not in json.dumps(llm.to_dict(), ensure_ascii=True)


@pytest.mark.asyncio
async def test_llm_prompt_contains_only_allowed_frozen_inputs():
    ScoringService, _, _, _ = _scoring_api()
    fake = FakeLLM(_response(_rubric_payload(ANSWERABLE)))
    await ScoringService(fake).score(
        task=ANSWERABLE,
        candidate=_artifact(),
        scorer_bundle=SCORER,
        on_execution=lambda _: None,
    )
    prompt = fake.calls[-1][0].content
    assert ANSWERABLE.question in prompt
    assert ANSWERABLE.expected_behavior in prompt
    assert "RRF combines several ranked lists." in prompt
    assert "learning-run-notes.md" in prompt
    assert "required_dimensions" in prompt
    assert "groundedness" in prompt
    assert "reasoning" in prompt
    assert "findings" in prompt
    assert "code" in prompt
    assert "severity" in prompt
    assert "message" in prompt
    assert "retrieval_empty" in prompt
    assert "critical" in prompt
    assert "noncritical" in prompt
    assert "expected_answer" not in prompt
    assert "manual_rationale" not in prompt
    assert "budget" not in prompt
    assert "trace" not in prompt


def test_derive_score_set_uses_only_emitted_drafts_and_fail_closed_status():
    _, derive_score_set, _, _ = _scoring_api()
    from app.eval.learning_run.contracts import ScorerExecutionDraft

    input_hash = "a" * 64
    drafts = (
        ScorerExecutionDraft(
            component_id="retrieval-integrity",
            component_version="v1",
            scorer_id="retrieval-integrity",
            scorer_version="v1",
            status="success",
            input_hash=input_hash,
            output={"evidence_count": 1},
            latency_ms=0,
            usage=None,
        ),
        ScorerExecutionDraft(
            component_id="citation-integrity",
            component_version="v1",
            scorer_id="citation-integrity",
            scorer_version="v1",
            status="success",
            input_hash=input_hash,
            output={"valid": True},
            latency_ms=0,
            usage=None,
        ),
    )
    result = derive_score_set(
        task=ANSWERABLE,
        scorer_bundle=SCORER,
        executions=drafts,
        input_hash=input_hash,
    )
    assert result.status in {"partial", "failed"}
    assert result.verdict == "inconclusive"


@pytest.mark.asyncio
async def test_no_provider_or_ollama_is_constructed():
    ScoringService, _, _, _ = _scoring_api()
    fake = FakeLLM(_response(_rubric_payload(ANSWERABLE)))
    service = ScoringService(fake)
    assert service.llm is fake
    await service.score(
        task=ANSWERABLE,
        candidate=_artifact(),
        scorer_bundle=SCORER,
        on_execution=lambda _: None,
    )
    assert len(fake.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("variant", ["extra_llm", "missing_rubric", "duplicate_rubric"])
async def test_scoring_requires_one_frozen_semantic_llm_producer(variant):
    ScoringService, _, _, _ = _scoring_api()
    rubric = next(
        component
        for component in SCORER.components
        if component.component_id == "grounded-quality-rubric"
    )
    if variant == "extra_llm":
        components = SCORER.components + (
            ScorerComponent(
                component_id="alternate-quality-rubric",
                version="v1",
                kind="llm",
                config=rubric.config,
            ),
        )
    elif variant == "missing_rubric":
        components = tuple(
            component
            for component in SCORER.components
            if component.component_id != "grounded-quality-rubric"
        )
    else:
        components = SCORER.components + (
            ScorerComponent(
                component_id="grounded-quality-rubric",
                version="v2",
                kind="llm",
                config=rubric.config,
            ),
        )
    bundle = replace(SCORER, components=components)
    fake = FakeLLM(_response(_rubric_payload(ANSWERABLE, value=5)))
    seen = []
    result = await ScoringService(fake).score(
        task=ANSWERABLE,
        candidate=_artifact(),
        scorer_bundle=bundle,
        on_execution=seen.append,
    )
    assert result.status == "failed"
    assert result.verdict == "inconclusive"
    assert result.aggregate_scores == {}
    assert len(seen) == len(components)
    assert all(draft.status == "failed" for draft in seen)
    assert fake.calls == []


def test_calibration_artifacts_have_expected_manual_truth_and_disjoint_ids():
    registry = TaskRegistry.load_default()
    assert set(registry.calibration_case_ids).isdisjoint(registry.task_cases)
    assert [candidate.manual_expected_verdict for candidate in registry.calibration_candidates] == [
        "pass",
        "fail",
        "fail",
        "pass",
        "fail",
    ]
    for candidate in registry.calibration_candidates:
        assert canonical_hash(candidate.artifact.to_dict()) == candidate.artifact_hash


@pytest.mark.asyncio
async def test_five_calibration_anchors_match_manual_labels_with_frozen_llm_outputs():
    ScoringService, _, _, _ = _scoring_api()
    outputs = {
        "cal-001": {"groundedness": 4, "citation_entailment": 4, "coverage": 4},
        "cal-002": {"groundedness": 2, "citation_entailment": 2, "coverage": 2},
        "cal-003": {"groundedness": 4, "citation_entailment": 4, "coverage": 3},
        "cal-004": {"refusal_appropriateness": 5, "unsupported_claims": 5},
        "cal-005": {"groundedness": 1, "citation_entailment": 1, "coverage": 1},
    }
    for candidate in REGISTRY.calibration_candidates:
        payload = dict(outputs[candidate.candidate_id])
        payload.update({"reasoning": "frozen calibration output", "findings": []})
        result = await ScoringService(FakeLLM(_response(payload))).score(
            task=_task_for_calibration(candidate),
            candidate=candidate.artifact,
            scorer_bundle=SCORER,
            on_execution=lambda _: None,
        )
        assert result.verdict == candidate.manual_expected_verdict


@pytest.mark.asyncio
async def test_hybrid_v2_calibration_anchors_match_versioned_labels():
    ScoringService, _, _, _ = _scoring_api()
    v2 = REGISTRY.scorer_for("hybrid-v2")
    labels = {
        item["id"]: item["manual_expected_verdict"]
        for item in json.loads(
            (
                REGISTRY.definitions_path / "calibration" / "hybrid-v2-labels.json"
            ).read_text(encoding="utf-8")
        )["labels"]
    }
    outputs = {
        "cal-001": {"groundedness": 4, "citation_entailment": 4, "coverage": 4},
        "cal-002": {"groundedness": 2, "citation_entailment": 2, "coverage": 2},
        "cal-003": {"groundedness": 4, "citation_entailment": 4, "coverage": 3},
        "cal-004": {"refusal_appropriateness": 5, "unsupported_claims": 5},
        "cal-005": {"groundedness": 1, "citation_entailment": 1, "coverage": 1},
    }
    for candidate in REGISTRY.calibration_candidates:
        payload = dict(outputs[candidate.candidate_id])
        payload.update({"reasoning": "frozen hybrid-v2 calibration", "findings": []})
        result = await ScoringService(FakeLLM(_response(payload))).score(
            task=_task_for_calibration(candidate),
            candidate=candidate.artifact,
            scorer_bundle=v2,
            on_execution=lambda _: None,
        )
        assert result.verdict == labels[candidate.candidate_id]
