"""Fail-closed curator for a redistributable Learning Run suite fixture."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

from app.eval.learning_run.contracts import canonical_hash, hash_without_field
from app.eval.learning_run.registry import TaskRegistry


SECRET_PATTERN = re.compile(
    r"authorization\s*:|api[_-]?key|sk-[A-Za-z0-9]{8,}|https?://[^\s\"']+",
    re.IGNORECASE,
)
RUN_FIELDS = (
    "id",
    "experiment_id",
    "task_case_id",
    "task_case_version",
    "variant_id",
    "run_profile",
    "lifecycle",
    "outcome",
    "suite_execution_id",
    "manifest",
    "manifest_hash",
    "candidate_artifact",
    "artifact_hash",
    "operational_error_json",
)
SCORE_SET_FIELDS = (
    "id",
    "scorer_id",
    "scorer_version",
    "scorer_snapshot",
    "scorer_definition_hash",
    "artifact_input_hash",
    "status",
    "quality_verdict",
    "aggregate_scores",
    "findings",
    "operational_error_code",
    "operational_error_message",
)
EXECUTION_FIELDS = (
    "id",
    "score_set_id",
    "scorer_id",
    "scorer_version",
    "status",
    "input_hash",
    "output",
    "error_code",
    "error_message",
    "latency_ms",
    "usage",
)


class CurateError(ValueError):
    """Raised when raw suite evidence cannot be published."""


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CurateError(f"{label} must be an object")
    return dict(value)


def _pick(payload: Mapping[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {key: payload[key] for key in fields if key in payload}


def _parse_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CurateError(f"invalid JSONL at line {line_number}: {exc}") from exc
        if not isinstance(payload, dict):
            raise CurateError(f"invalid JSONL at line {line_number}: expected object")
        records.append(payload)
    if not records:
        raise CurateError("raw suite export is empty")
    return records


def _reject_secrets(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_secrets(item, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_secrets(item, path=f"{path}[{index}]")
        return
    if isinstance(value, str) and SECRET_PATTERN.search(value):
        raise CurateError(f"secret or private endpoint leaked at {path}")


def _chunk_ids(registry: TaskRegistry) -> set[str]:
    return {chunk.chunk_id for chunk in registry.corpus.chunks}


def _validate_record(
    record: Mapping[str, Any],
    *,
    registry: TaskRegistry,
    chunk_ids: set[str],
) -> dict[str, Any]:
    raw_run = _require_mapping(record.get("run"), "run")
    task_case_id = str(raw_run.get("task_case_id") or "")
    variant_id = str(raw_run.get("variant_id") or "")
    if raw_run.get("experiment_id") != registry.experiment.experiment_id:
        raise CurateError(f"{task_case_id}/{variant_id} experiment is not in the registry")
    if task_case_id not in registry.task_cases:
        raise CurateError(f"unknown task case: {task_case_id}")
    if variant_id not in registry.experiment.variants:
        raise CurateError(f"unknown variant: {variant_id}")
    expected_version = registry.task_cases[task_case_id].task_case_version
    if str(raw_run.get("task_case_version") or "") != expected_version:
        raise CurateError(f"{task_case_id} task case version mismatch")
    manifest = raw_run.get("manifest") or raw_run.get("manifest_json")
    if not isinstance(manifest, Mapping):
        raise CurateError(f"{task_case_id}/{variant_id} manifest is invalid")
    if canonical_hash(manifest) != str(raw_run.get("manifest_hash") or ""):
        raise CurateError(f"{task_case_id}/{variant_id} manifest hash mismatch")
    if manifest.get("corpus_snapshot_hash") != registry.corpus.aggregate_hash:
        raise CurateError(f"{task_case_id}/{variant_id} uses a non-registry corpus")
    if manifest.get("corpus_snapshot_id") != registry.corpus.snapshot_id:
        raise CurateError(f"{task_case_id}/{variant_id} uses a non-registry corpus")
    artifact = raw_run.get("candidate_artifact") or raw_run.get("candidate_artifact_json")
    artifact_hash = raw_run.get("artifact_hash")
    if not isinstance(artifact, Mapping) or not artifact_hash:
        raise CurateError(f"{task_case_id}/{variant_id} is missing a frozen artifact")
    if canonical_hash(artifact) != str(artifact_hash):
        raise CurateError(f"{task_case_id}/{variant_id} artifact hash mismatch")
    for collection_name in ("citations", "exact_evidence"):
        for item in artifact.get(collection_name) or ():
            if isinstance(item, Mapping) and item.get("chunk_id") not in chunk_ids:
                raise CurateError(
                    f"{task_case_id}/{variant_id} cites a non-registry chunk"
                )

    score_sets: list[dict[str, Any]] = []
    dual_versions: set[str] = set()
    for raw_score in record.get("score_sets") or ():
        score = _require_mapping(raw_score, "score_set")
        snapshot = score.get("scorer_snapshot") or score.get("scorer_snapshot_json")
        definition_hash = str(score.get("scorer_definition_hash") or "")
        if not isinstance(snapshot, Mapping) or not definition_hash:
            raise CurateError(f"{task_case_id}/{variant_id} score set is missing a snapshot")
        if hash_without_field(snapshot, "definition_hash") != definition_hash:
            raise CurateError(f"{task_case_id}/{variant_id} score set hash mismatch")
        version = str(score.get("scorer_version") or "")
        if version not in registry.scorers:
            raise CurateError(f"{task_case_id}/{variant_id} unknown scorer {version}")
        if str(score.get("artifact_input_hash") or "") != str(artifact_hash):
            raise CurateError(f"{task_case_id}/{variant_id} score set artifact hash mismatch")
        dual_versions.add(version)
        cleaned = _pick(score, SCORE_SET_FIELDS)
        cleaned["scorer_snapshot"] = snapshot
        cleaned["scorer_definition_hash"] = definition_hash
        cleaned["scorer_version"] = version
        if "aggregate_scores" not in cleaned and "aggregate_scores_json" in score:
            cleaned["aggregate_scores"] = score["aggregate_scores_json"]
        if "findings" not in cleaned and "findings_json" in score:
            cleaned["findings"] = score["findings_json"]
        score_sets.append(cleaned)
    if not score_sets:
        raise CurateError(f"{task_case_id}/{variant_id} has no score sets")

    executions: list[dict[str, Any]] = []
    for raw_execution in record.get("executions") or ():
        execution = _require_mapping(raw_execution, "execution")
        cleaned = _pick(execution, EXECUTION_FIELDS)
        if "output" not in cleaned and "output_json" in execution:
            cleaned["output"] = execution["output_json"]
        if "error_code" not in cleaned and "operational_error_code" in execution:
            cleaned["error_code"] = execution["operational_error_code"]
        if "error_message" not in cleaned and "operational_error_message" in execution:
            cleaned["error_message"] = execution["operational_error_message"]
        if "usage" not in cleaned and "usage_json" in execution:
            cleaned["usage"] = execution["usage_json"]
        executions.append(cleaned)

    executions_by_set: dict[str, int] = {}
    for execution in executions:
        parent = str(execution.get("score_set_id") or "")
        executions_by_set[parent] = executions_by_set.get(parent, 0) + 1
    for score_set in score_sets:
        expected = len(registry.scorer_for(str(score_set["scorer_version"])).components)
        actual = executions_by_set.get(str(score_set.get("id") or ""), 0)
        if actual != expected:
            raise CurateError(
                f"{task_case_id}/{variant_id} score set {score_set.get('id')} is incomplete"
            )

    cleaned_run = _pick(raw_run, RUN_FIELDS)
    cleaned_run["manifest"] = dict(manifest)
    cleaned_run["candidate_artifact"] = dict(artifact)
    cleaned_run["artifact_hash"] = str(artifact_hash)
    payload = {
        "run": cleaned_run,
        "score_sets": score_sets,
        "executions": executions,
    }
    _reject_secrets(payload, path=f"{task_case_id}/{variant_id}")
    payload["_dual_versions"] = dual_versions
    return payload


def curate(source: Path, destination: Path) -> int:
    registry = TaskRegistry.load_default()
    chunk_ids = _chunk_ids(registry)
    cleaned: list[dict[str, Any]] = []
    pairs: list[tuple[str, str]] = []
    has_dual = False
    for record in _parse_jsonl(source):
        item = _validate_record(record, registry=registry, chunk_ids=chunk_ids)
        versions = item.pop("_dual_versions")
        if versions >= {"hybrid-v1", "hybrid-v2"}:
            has_dual = True
        pairs.append((item["run"]["task_case_id"], item["run"]["variant_id"]))
        cleaned.append(item)
    if len(pairs) != len(set(pairs)):
        raise CurateError("raw suite export contains duplicate case/variant pairs")
    expected = {
        (case_id, variant_id)
        for case_id in registry.task_case_ids
        for variant_id in registry.experiment.variants
    }
    if set(pairs) != expected:
        raise CurateError("raw suite export does not cover the frozen 12x2 matrix")
    if not has_dual:
        raise CurateError("raw suite export has no hybrid-v1/hybrid-v2 historical pair")
    cleaned.sort(
        key=lambda item: (item["run"]["task_case_id"], item["run"]["variant_id"], item["run"]["id"])
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        for item in cleaned
    ]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(cleaned)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Curate a redistributable Learning Run fixture")
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args(argv)
    try:
        count = curate(args.source, args.destination)
    except CurateError as exc:
        print(f"curate failed: {exc}", file=sys.stderr)
        return 1
    print(f"curated {count} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
