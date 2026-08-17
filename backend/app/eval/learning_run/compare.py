"""Pure compatibility comparison for two frozen Learning Run ScoreSets."""

from __future__ import annotations

from typing import Any, Mapping


_ALIGN_FIELDS = (
    "task_case_id",
    "task_case_version",
    "corpus_snapshot_hash",
)
_DECLARED_AXES_DEFAULT = ("prompt_version",)
_AXIS_DEPENDENT = {
    "prompt_version": frozenset(
        {"prompt_version", "prompt_text", "prompt_hash", "variant_id"}
    ),
}
_BOOKKEEPING_FIELDS = frozenset(
    {
        "experiment_id",
        "task_snapshot",
        "connection_fingerprint",
        "manifest_hash",
        "runner_version",
        "code_revision",
        "seed",
        "corpus_snapshot",
        "scorer_snapshot",
    }
)


def _manifest(side: Mapping[str, Any]) -> Mapping[str, Any]:
    value = side.get("manifest")
    return value if isinstance(value, Mapping) else {}


CANDIDATE_ARTIFACT_SCHEMA = "candidate-artifact-v1"


def _artifact_schema(side: Mapping[str, Any]) -> str:
    artifact = side.get("artifact")
    if not isinstance(artifact, Mapping):
        return ""
    return str(artifact.get("schema_version") or CANDIDATE_ARTIFACT_SCHEMA)


def _score_set(side: Mapping[str, Any]) -> Mapping[str, Any]:
    value = side.get("score_set")
    return value if isinstance(value, Mapping) else {}


def _undeclared_diffs(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    axes: tuple[str, ...],
) -> tuple[str, ...]:
    axis_fields: set[str] = set()
    for axis in axes:
        axis_fields |= set(_AXIS_DEPENDENT.get(axis, {axis}))
    ignored = set(_ALIGN_FIELDS) | axis_fields | _BOOKKEEPING_FIELDS | {
        "variant_id",
        "experiment_axes",
        "schema_version",
    }
    keys = (set(left) | set(right)) - ignored
    diffs: list[str] = []
    for key in sorted(keys):
        if left.get(key) != right.get(key):
            diffs.append(f"undeclared config differs: {key}")
    return tuple(diffs)


def compare_score_sets(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare two frozen run/score-set snapshots without writing copy as suite quality."""

    left_manifest = _manifest(left)
    right_manifest = _manifest(right)
    reasons: list[str] = []

    for field in _ALIGN_FIELDS:
        if left_manifest.get(field) != right_manifest.get(field):
            reasons.append(f"cannot align {field}")
    if _artifact_schema(left) != _artifact_schema(right) or not _artifact_schema(left):
        reasons.append("cannot align artifact schema")

    if reasons:
        return {
            "compatibility": "incompatible",
            "reasons": reasons,
            "left": {"run_id": left.get("run_id"), "variant_id": left.get("variant_id")},
            "right": {"run_id": right.get("run_id"), "variant_id": right.get("variant_id")},
            "scorer_bundle": {
                "scorer_id": _score_set(left).get("scorer_id") or "hybrid",
                "version": _score_set(left).get("scorer_version") or "",
            },
            "delta": None,
            "scope": "case",
            "caption": "case delta",
            "rescore_required": False,
        }

    axes = tuple(
        left_manifest.get("experiment_axes")
        or right_manifest.get("experiment_axes")
        or _DECLARED_AXES_DEFAULT
    )
    undeclared = _undeclared_diffs(left_manifest, right_manifest, axes=axes)
    compatibility = "informational" if undeclared else "controlled"
    reasons.extend(undeclared)

    left_scorer = str(_score_set(left).get("scorer_version") or "")
    right_scorer = str(_score_set(right).get("scorer_version") or "")
    rescore_required = left_scorer != right_scorer
    delta = None
    if not rescore_required and compatibility == "controlled":
        left_scores = _score_set(left).get("aggregate_scores") or {}
        right_scores = _score_set(right).get("aggregate_scores") or {}
        if isinstance(left_scores, Mapping) and isinstance(right_scores, Mapping):
            delta = {}
            for key in sorted(set(left_scores) | set(right_scores)):
                left_value = left_scores.get(key)
                right_value = right_scores.get(key)
                if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
                    delta[key] = {
                        "left": left_value,
                        "right": right_value,
                        "delta": right_value - left_value,
                    }

    return {
        "compatibility": compatibility,
        "reasons": reasons,
        "left": {"run_id": left.get("run_id"), "variant_id": left.get("variant_id")},
        "right": {"run_id": right.get("run_id"), "variant_id": right.get("variant_id")},
        "scorer_bundle": {
            "scorer_id": _score_set(left).get("scorer_id") or "hybrid",
            "version": left_scorer,
        },
        "delta": delta,
        "scope": "case",
        "caption": "case delta",
        "rescore_required": rescore_required,
    }


__all__ = ["CANDIDATE_ARTIFACT_SCHEMA", "compare_score_sets"]
