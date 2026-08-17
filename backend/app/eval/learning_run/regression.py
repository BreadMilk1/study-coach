"""Suite regression detection shared by the release contract and Run Lab."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


VERDICT_RANK = {
    "pass": 3,
    "inconclusive": 2,
    "not_evaluated": 1,
    "fail": 0,
}
UNGROUNDED_KNOWLEDGE_MARKERS = (
    "general study knowledge",
    "general knowledge",
    "from general study",
)
SUITE_SCORER_VERSION = "hybrid-v1"
BASELINE_VARIANT = "tutor-v2"
CANDIDATE_VARIANT = "tutor-v3"


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _field(obj: Any, *names: str) -> Any:
    mapping = _mapping(obj)
    for name in names:
        if mapping is not None and name in mapping:
            return mapping[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def finding_codes(score_set: Any) -> set[str]:
    raw = _field(score_set, "findings_json", "findings") or []
    if isinstance(raw, Mapping):
        raw = raw.get("findings") or []
    if not isinstance(raw, list):
        return set()
    return {
        str(item.get("code"))
        for item in raw
        if isinstance(item, Mapping) and item.get("code")
    }


def answer_text(run: Any) -> str:
    artifact = _field(run, "candidate_artifact_json", "candidate_artifact") or {}
    mapping = _mapping(artifact)
    if mapping is None:
        return ""
    return str(mapping.get("answer") or "")


def leaks_ungrounded_knowledge(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in UNGROUNDED_KNOWLEDGE_MARKERS)


def pick_score_set(score_sets: Sequence[Any], scorer_version: str) -> Any | None:
    matched = [
        item
        for item in score_sets
        if str(_field(item, "scorer_version") or "") == scorer_version
    ]
    return matched[-1] if matched else None


def is_score_regression(baseline: Any, candidate: Any) -> bool:
    if VERDICT_RANK.get(str(_field(candidate, "quality_verdict") or ""), 0) < VERDICT_RANK.get(
        str(_field(baseline, "quality_verdict") or ""),
        0,
    ):
        return True
    left = _field(baseline, "aggregate_scores_json", "aggregate_scores") or {}
    right = _field(candidate, "aggregate_scores_json", "aggregate_scores") or {}
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return False
    return any(
        isinstance(left.get(key), int)
        and isinstance(right.get(key), int)
        and right[key] < left[key]
        for key in set(left) | set(right)
    )


def is_refusal_axis_regression(
    *,
    registry: Any,
    baseline_run: Any,
    baseline: Any,
    candidate_run: Any,
    candidate: Any,
) -> bool:
    case_id = str(_field(baseline_run, "task_case_id") or "")
    cases = getattr(registry, "task_cases", {})
    case = cases.get(case_id) if isinstance(cases, Mapping) else None
    if case is None or getattr(case, "case_type", None) != "expected_refusal":
        return False
    if "expected_refusal_observed" not in finding_codes(baseline):
        return False
    if "expected_refusal_observed" in finding_codes(candidate):
        return False
    return leaks_ungrounded_knowledge(answer_text(candidate_run))


def is_regression(
    *,
    registry: Any,
    baseline_run: Any,
    baseline: Any,
    candidate_run: Any,
    candidate: Any,
) -> bool:
    return is_score_regression(baseline, candidate) or is_refusal_axis_regression(
        registry=registry,
        baseline_run=baseline_run,
        baseline=baseline,
        candidate_run=candidate_run,
        candidate=candidate,
    )


def suite_regression_case_ids(
    registry: Any,
    runs: Sequence[Any],
    score_sets_by_run: Mapping[str, Sequence[Any]],
    *,
    scorer_version: str = SUITE_SCORER_VERSION,
) -> tuple[str, ...]:
    baseline_runs: dict[str, Any] = {}
    candidate_runs: dict[str, Any] = {}
    for run in runs:
        case_id = str(_field(run, "task_case_id") or "")
        variant_id = str(_field(run, "variant_id") or "")
        if not case_id:
            continue
        if variant_id == BASELINE_VARIANT:
            baseline_runs[case_id] = run
        elif variant_id == CANDIDATE_VARIANT:
            candidate_runs[case_id] = run

    found: list[str] = []
    for case_id, baseline_run in baseline_runs.items():
        candidate_run = candidate_runs.get(case_id)
        if candidate_run is None:
            continue
        baseline = pick_score_set(
            score_sets_by_run.get(str(_field(baseline_run, "id") or ""), ()),
            scorer_version,
        )
        candidate = pick_score_set(
            score_sets_by_run.get(str(_field(candidate_run, "id") or ""), ()),
            scorer_version,
        )
        if baseline is None or candidate is None:
            continue
        if is_regression(
            registry=registry,
            baseline_run=baseline_run,
            baseline=baseline,
            candidate_run=candidate_run,
            candidate=candidate,
        ):
            found.append(case_id)
    return tuple(found)


__all__ = [
    "BASELINE_VARIANT",
    "CANDIDATE_VARIANT",
    "SUITE_SCORER_VERSION",
    "UNGROUNDED_KNOWLEDGE_MARKERS",
    "VERDICT_RANK",
    "answer_text",
    "finding_codes",
    "is_refusal_axis_regression",
    "is_regression",
    "is_score_regression",
    "leaks_ungrounded_knowledge",
    "pick_score_set",
    "suite_regression_case_ids",
]
