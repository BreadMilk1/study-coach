"""Stable release contract for the Learning Run suite evidence."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session

from app.agent.prompt import SYSTEM_INSTRUCTION
from app.db.models import Base, EvalRun, EvalScoreSet, EvalScorerExecution
from app.eval.learning_run.contracts import (
    ScorerExecutionDraft,
    TaskCase,
    canonical_hash,
    hash_without_field,
)
from app.eval.learning_run.registry import TaskRegistry
from app.eval.learning_run.scoring import derive_score_set


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
FIXTURE_PATH = (
    BACKEND_ROOT
    / "app"
    / "eval"
    / "learning_run"
    / "fixtures"
    / "tutor-prompt-regression-v1.jsonl"
)
REFUSAL_CASE_IDS = frozenset({"tgqa-004", "tgqa-008", "tgqa-012"})
SECRET_PATTERN = re.compile(
    r"authorization\s*:|api[_-]?key|sk-[A-Za-z0-9]{8,}|https?://[^\s\"']+",
    re.IGNORECASE,
)
MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
VERDICT_RANK = {
    "pass": 3,
    "inconclusive": 2,
    "not_evaluated": 1,
    "fail": 0,
}


def _load_script(name: str):
    path = BACKEND_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _registry_chunk_ids(registry: TaskRegistry) -> set[str]:
    return {chunk.chunk_id for chunk in registry.corpus.chunks}


def test_release_definitions_cover_frozen_suite_and_versions():
    registry = TaskRegistry.load_default()
    assert registry.case_type_counts == {
        "answerable": 6,
        "multi_evidence": 3,
        "expected_refusal": 3,
    }
    assert registry.experiment_axes == ("prompt_version",)
    assert registry.prompt("tutor-v2").text == SYSTEM_INSTRUCTION
    assert registry.calibration_case_ids.isdisjoint(registry.task_case_ids)


def _draft_from_execution(row: EvalScorerExecution) -> ScorerExecutionDraft:
    output = row.output_json
    findings: tuple = ()
    payload = output
    if isinstance(output, dict) and ("result" in output or "findings" in output):
        payload = output.get("result")
        raw_findings = output.get("findings") or ()
        findings = tuple(raw_findings) if isinstance(raw_findings, list) else ()
    return ScorerExecutionDraft(
        component_id=row.scorer_id,
        component_version=row.scorer_version,
        scorer_id=row.scorer_id,
        scorer_version=row.scorer_version,
        status=row.status,
        input_hash=row.input_hash,
        output=payload,
        error_code=row.operational_error_code,
        error_message=row.operational_error_message,
        latency_ms=row.latency_ms,
        usage=row.usage_json,
        findings=findings,
    )


_UNGROUNDED_KNOWLEDGE_MARKERS = (
    "general study knowledge",
    "general knowledge",
    "from general study",
)


def _finding_codes(score_set: EvalScoreSet) -> set[str]:
    raw = score_set.findings_json or []
    if isinstance(raw, dict):
        raw = raw.get("findings") or []
    if not isinstance(raw, list):
        return set()
    return {
        str(item.get("code"))
        for item in raw
        if isinstance(item, dict) and item.get("code")
    }


def _answer_text(run: EvalRun) -> str:
    artifact = run.candidate_artifact_json or {}
    if not isinstance(artifact, dict):
        return ""
    return str(artifact.get("answer") or "")


def _leaks_ungrounded_knowledge(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in _UNGROUNDED_KNOWLEDGE_MARKERS)


def _is_score_regression(baseline: EvalScoreSet, candidate: EvalScoreSet) -> bool:
    if VERDICT_RANK.get(candidate.quality_verdict, 0) < VERDICT_RANK.get(
        baseline.quality_verdict, 0
    ):
        return True
    left = baseline.aggregate_scores_json or {}
    right = candidate.aggregate_scores_json or {}
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    return any(
        isinstance(left.get(key), int)
        and isinstance(right.get(key), int)
        and right[key] < left[key]
        for key in set(left) | set(right)
    )


def _is_refusal_axis_regression(
    *,
    registry: TaskRegistry,
    baseline_run: EvalRun,
    baseline: EvalScoreSet,
    candidate_run: EvalRun,
    candidate: EvalScoreSet,
) -> bool:
    case = registry.task_cases.get(baseline_run.task_case_id)
    if case is None or case.case_type != "expected_refusal":
        return False
    if "expected_refusal_observed" not in _finding_codes(baseline):
        return False
    if "expected_refusal_observed" in _finding_codes(candidate):
        return False
    return _leaks_ungrounded_knowledge(_answer_text(candidate_run))


def _is_regression(
    *,
    registry: TaskRegistry,
    baseline_run: EvalRun,
    baseline: EvalScoreSet,
    candidate_run: EvalRun,
    candidate: EvalScoreSet,
) -> bool:
    return _is_score_regression(baseline, candidate) or _is_refusal_axis_regression(
        registry=registry,
        baseline_run=baseline_run,
        baseline=baseline,
        candidate_run=candidate_run,
        candidate=candidate,
    )


def test_curated_fixture_imports_complete_auditable_suite(tmp_path: Path):
    assert FIXTURE_PATH.is_file(), "committed curated fixture is missing"
    registry = TaskRegistry.load_default()
    raw = FIXTURE_PATH.read_text(encoding="utf-8")
    assert SECRET_PATTERN.search(raw) is None

    importer = _load_script("import_learning_run_suite.py")
    db_path = tmp_path / "learning-run-fixture-check.db"
    exit_code = importer.main(
        [str(FIXTURE_PATH), "--database-url", f"sqlite:///{db_path}"]
    )
    assert exit_code == 0
    assert db_path.is_file()

    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    chunk_ids = _registry_chunk_ids(registry)
    with Session(engine) as session:
        runs = list(session.scalars(select(EvalRun)).all())
        score_sets = list(session.scalars(select(EvalScoreSet)).all())
        executions = list(session.scalars(select(EvalScorerExecution)).all())
        expected_pairs = {
            (case_id, variant_id)
            for case_id in registry.task_case_ids
            for variant_id in registry.experiment.variants
        }
        assert {(run.task_case_id, run.variant_id) for run in runs} == expected_pairs
        assert len(runs) == 24
        refusal_pairs = {
            (run.task_case_id, run.variant_id)
            for run in runs
            if run.task_case_id in REFUSAL_CASE_IDS
        }
        assert refusal_pairs == {
            (case_id, variant_id)
            for case_id in REFUSAL_CASE_IDS
            for variant_id in registry.experiment.variants
        }

        by_run = {run.id: run for run in runs}
        for run in runs:
            assert canonical_hash(run.manifest_json) == run.manifest_hash
            if run.candidate_artifact_json is not None:
                assert canonical_hash(run.candidate_artifact_json) == run.artifact_hash
            manifest = run.manifest_json
            assert manifest["corpus_snapshot_hash"] == registry.corpus.aggregate_hash
            assert manifest["corpus_snapshot_id"] == registry.corpus.snapshot_id
            artifact = run.candidate_artifact_json or {}
            for collection_name in ("citations", "exact_evidence"):
                for item in artifact.get(collection_name) or ():
                    if isinstance(item, dict) and item.get("chunk_id"):
                        assert item["chunk_id"] in chunk_ids

        sets_by_run: dict[str, list[EvalScoreSet]] = {}
        for score_set in score_sets:
            sets_by_run.setdefault(score_set.run_id, []).append(score_set)
            assert hash_without_field(
                score_set.scorer_snapshot_json, "definition_hash"
            ) == score_set.scorer_definition_hash
            parent = by_run[score_set.run_id]
            if parent.artifact_hash:
                assert score_set.artifact_input_hash == parent.artifact_hash

        assert any(
            {item.scorer_version for item in items} >= {"hybrid-v1", "hybrid-v2"}
            for items in sets_by_run.values()
        )

        baseline_runs = {
            run.task_case_id: run for run in runs if run.variant_id == "tutor-v2"
        }
        candidate_runs = {
            run.task_case_id: run for run in runs if run.variant_id == "tutor-v3"
        }
        baseline = {
            case_id: score_set
            for case_id, run in baseline_runs.items()
            for score_set in sets_by_run.get(run.id, ())
            if score_set.scorer_version == "hybrid-v1"
        }
        candidate = {
            case_id: score_set
            for case_id, run in candidate_runs.items()
            for score_set in sets_by_run.get(run.id, ())
            if score_set.scorer_version == "hybrid-v1"
        }
        regressions = [
            case_id
            for case_id in baseline
            if case_id in candidate
            and _is_regression(
                registry=registry,
                baseline_run=baseline_runs[case_id],
                baseline=baseline[case_id],
                candidate_run=candidate_runs[case_id],
                candidate=candidate[case_id],
            )
        ]
        assert regressions, (
            "curated fixture has no tutor-v3 hybrid-v1 score regression or "
            "expected-refusal finding leak against tutor-v2"
        )

        executions_by_set: dict[str, list[EvalScorerExecution]] = {}
        for execution in executions:
            executions_by_set.setdefault(execution.score_set_id, []).append(execution)
        for score_set in score_sets:
            rows = executions_by_set.get(score_set.id, [])
            bundle = registry.scorer_for(score_set.scorer_version)
            assert len(rows) == len(bundle.components)
            parent = by_run[score_set.run_id]
            task = TaskCase.from_dict(parent.manifest_json["task_snapshot"])
            rebuilt = derive_score_set(
                task,
                bundle,
                [_draft_from_execution(row) for row in rows],
                input_hash=score_set.artifact_input_hash,
            )
            assert rebuilt.verdict == score_set.quality_verdict


def _markdown_targets(path: Path) -> list[Path]:
    text = path.read_text(encoding="utf-8")
    targets: list[Path] = []
    for raw in MARKDOWN_LINK.findall(text):
        target = raw.split("#", 1)[0].split("?", 1)[0].strip()
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        resolved = (path.parent / target).resolve()
        targets.append(resolved)
    return targets


def test_release_docs_and_cli_contracts():
    registry = TaskRegistry.load_default()
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    eval_doc = (REPO_ROOT / "docs" / "EVAL.md").read_text(encoding="utf-8")
    architecture = (REPO_ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    demo = (REPO_ROOT / "docs" / "DEMO.md").read_text(encoding="utf-8")
    roadmap = (REPO_ROOT / "docs" / "ROADMAP.md").read_text(encoding="utf-8")
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    context = (REPO_ROOT / "CONTEXT.md").read_text(encoding="utf-8")
    adr = (
        REPO_ROOT / "docs" / "adr" / "0001-share-tutor-attempt-not-production-orchestration.md"
    ).read_text(encoding="utf-8")

    assert "12-case" in readme or "12 cases" in readme
    assert "Run Lab" in readme
    assert "tutor-prompt-regression-v1" in eval_doc
    assert "ollama" in eval_doc.lower()
    assert "llama3.2" in eval_doc
    assert "TutorAttemptEngine" in architecture
    assert "ScoreSet" in architecture
    assert "single-worker" in architecture or "single worker" in architecture
    assert "import_learning_run_suite.py" in demo
    assert "fixtures/tutor-prompt-regression-v1.jsonl" in demo
    assert "--database-url" in demo
    assert "background queue" in roadmap.lower() or "background worker" in roadmap.lower()
    assert "Plan/Quiz" in roadmap or "Plan / Quiz" in roadmap
    assert "TutorAttempt" in context
    assert "TutorAttemptEngine" in adr
    assert "backend/app/eval/**/output/" in gitignore
    assert "learning_run/fixtures" not in gitignore

    for path in (
        REPO_ROOT / "README.md",
        REPO_ROOT / "docs" / "EVAL.md",
        REPO_ROOT / "docs" / "ARCHITECTURE.md",
        REPO_ROOT / "docs" / "DEMO.md",
    ):
        for target in _markdown_targets(path):
            assert target.exists(), f"broken link in {path.name}: {target}"

    importer = _load_script("import_learning_run_suite.py")
    curator = _load_script("curate_learning_run_fixture.py")
    assert "--database-url" in importer.build_parser().format_help()
    assert hasattr(curator, "curate")

    assert registry.experiment.schema_version == "learning-run-v1"
    assert set(registry.scorers) >= {"hybrid-v1", "hybrid-v2"}
    assert set(registry.prompts) == {"tutor-v2", "tutor-v3"}
    assert registry.experiment.variants["tutor-v2"]["provider"] == "ollama"
    assert registry.experiment.variants["tutor-v2"]["model"] == "llama3.2"
