"""Pure compatibility comparison for two frozen Learning Run ScoreSets."""

from __future__ import annotations

from typing import Any, Mapping

from app.agent.tutor_attempt import TUTOR_USAGE_CONTRACT

from .contracts import hash_without_field, is_dimension_score


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

# The usage contracts this comparison understands. Declaring one of these on
# both sides is comparable; anything else (a different, unknown, or malformed
# value) is informational because the two runs did not measure usage the same
# way. A pair where neither side declares a contract is legacy and keeps the
# original quality comparison untouched.
_SUPPORTED_USAGE_CONTRACTS = frozenset({TUTOR_USAGE_CONTRACT})


def _usage_contract_reason(
    left_manifest: Mapping[str, Any],
    right_manifest: Mapping[str, Any],
) -> str | None:
    """Return a reason when two manifests cannot share one usage contract.

    ``None`` means the pair is comparable on usage: either both manifests
    genuinely omit the key (a legacy pair, whose historical usage semantics are
    deliberately not asserted to match), or both declare the same supported
    contract. Every other combination -- one-sided, differing, unknown, or an
    invalid declaration -- is informational with an explicit reason. The gate is
    independent of ``experiment_axes`` and of ``runner_version``/``code_revision``,
    so declaring the field as an axis cannot bypass it.
    """
    left_declared = "usage_contract" in left_manifest
    right_declared = "usage_contract" in right_manifest
    if not left_declared and not right_declared:
        return None

    left_value = left_manifest.get("usage_contract")
    right_value = right_manifest.get("usage_contract")

    def _well_formed(value: Any) -> bool:
        return isinstance(value, str) and value != ""

    if (left_declared and not _well_formed(left_value)) or (
        right_declared and not _well_formed(right_value)
    ):
        return "invalid usage contract declaration; usage semantics are not comparable"
    if not left_declared or not right_declared:
        return "usage contract declared on one side only; usage semantics are not comparable"
    if left_value != right_value:
        return (
            f"usage contracts differ ({left_value} vs {right_value}); "
            "usage semantics are not comparable"
        )
    if left_value not in _SUPPORTED_USAGE_CONTRACTS:
        return f"unsupported usage contract ({left_value}); usage semantics are not comparable"
    return None


def _manifest(side: Mapping[str, Any]) -> Mapping[str, Any]:
    value = side.get("manifest")
    return value if isinstance(value, Mapping) else {}


CANDIDATE_ARTIFACT_SCHEMA = "candidate-artifact-v1"


def _artifact_schema(side: Mapping[str, Any]) -> tuple[str, bool]:
    """Return the artifact schema and whether it had to be inferred.

    A missing `schema_version` is read as v1: imported fixtures predate the
    field and `CandidateArtifact.to_dict()` still omits it, because writing it
    would change `artifact_hash`. The inference is reported back so a
    `controlled` verdict never hides the assumption -- a genuinely newer
    artifact that simply forgot the field would otherwise compare as v1 with
    no signal at all.
    """
    artifact = side.get("artifact")
    if not isinstance(artifact, Mapping):
        return "", False
    declared = artifact.get("schema_version")
    if declared:
        return str(declared), False
    return CANDIDATE_ARTIFACT_SCHEMA, True


def _score_set(side: Mapping[str, Any]) -> Mapping[str, Any]:
    value = side.get("score_set")
    return value if isinstance(value, Mapping) else {}


def scorer_identity(
    score_set: Mapping[str, Any],
) -> tuple[str, str, str] | None:
    """Return a complete, self-consistent scorer identity or ``None``.

    The identity is ``(scorer_id, scorer_version, scorer_definition_hash)``.  It
    is only returned when all three are non-empty, the frozen scorer snapshot
    names the same id and version, and the snapshot still hashes to the stored
    definition hash.  A snapshot's own ``definition_hash`` field is never trusted
    by itself; the hash is recomputed from the rest of the snapshot.  Both the
    nine-key projection and the complete document are accepted, because both
    carry ``scorer_id``/``version`` and recompute to the stored hash under the
    same rule.  This reads only the frozen snapshot -- it never consults today's
    registry to back-infer a historical identity.
    """

    scorer_id = score_set.get("scorer_id")
    scorer_version = score_set.get("scorer_version")
    scorer_definition_hash = score_set.get("scorer_definition_hash")
    if not (
        isinstance(scorer_id, str)
        and scorer_id
        and isinstance(scorer_version, str)
        and scorer_version
        and isinstance(scorer_definition_hash, str)
        and scorer_definition_hash
    ):
        return None
    snapshot = score_set.get("scorer_snapshot")
    if not isinstance(snapshot, Mapping) or not snapshot:
        return None
    if str(snapshot.get("scorer_id") or "") != scorer_id:
        return None
    if str(snapshot.get("version") or "") != scorer_version:
        return None
    if hash_without_field(snapshot, "definition_hash") != scorer_definition_hash:
        return None
    return (scorer_id, scorer_version, scorer_definition_hash)


def _scorer_identity_reason(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> str | None:
    """Return a reason when two sides cannot share one scorer identity.

    ``None`` means both sides carry a complete, self-consistent identity and the
    two are equal.  A side with no score set, an empty ``{}`` score set, or a
    score set whose identity is missing or inconsistent is informational -- two
    equal-but-empty identities never count as a match.  Same-version/different-
    hash is informational; a version difference additionally raises
    ``rescore_required`` in the caller.
    """

    left_identity = scorer_identity(_score_set(left))
    right_identity = scorer_identity(_score_set(right))
    if left_identity is not None and left_identity == right_identity:
        return None
    if left_identity is None or right_identity is None:
        return (
            "scorer identity is missing or not self-consistent on one or both "
            "sides; scorer versions are not comparable"
        )
    if left_identity[1] != right_identity[1]:
        return (
            f"scorer versions differ ({left_identity[1]} vs {right_identity[1]}); "
            "a re-score is required"
        )
    return (
        f"scorer definition hash differs at the same version ({left_identity[1]}); "
        "scorer definitions are not comparable"
    )


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
        "usage_contract",
    }
    keys = (set(left) | set(right)) - ignored
    diffs: list[str] = []
    for key in sorted(keys):
        if left.get(key) != right.get(key):
            diffs.append(f"undeclared config differs: {key}")
    return tuple(diffs)


def _align_present(value: Any) -> bool:
    """Whether an alignment field is present and non-empty on one side.

    A missing or empty value must never align by matching the other side's
    missing/empty value: two ``None``s are not evidence the two runs share a
    task, version or corpus.
    """

    if value is None:
        return False
    if isinstance(value, str):
        return value != ""
    return True


def _resolve_axes(
    left_manifest: Mapping[str, Any],
    right_manifest: Mapping[str, Any],
) -> tuple[tuple[str, ...], str | None]:
    """Resolve the declared experiment axes for one comparison.

    ``RunManifest.payload()`` does not write ``experiment_axes``, so a real
    frozen manifest carries no such key.  When neither side declares the field we
    keep the historical single-axis default (``prompt_version``) exactly as
    before; this is a legacy compatibility default, not a schema guarantee that
    every run was actually controlled on that axis.

    When either side does declare the field, both must declare the same legal
    single ``prompt_version`` axis.  A one-sided, malformed, differing, empty or
    extended declaration is informational, so a declaration can never widen the
    controlled surface or let an imported axis bypass the other gates.
    """

    left_declared = "experiment_axes" in left_manifest
    right_declared = "experiment_axes" in right_manifest
    if not left_declared and not right_declared:
        return _DECLARED_AXES_DEFAULT, None

    def _parse(value: Any) -> tuple[str, ...] | None:
        if not isinstance(value, (list, tuple)):
            return None
        return tuple(value)

    left_axes = _parse(left_manifest.get("experiment_axes")) if left_declared else None
    right_axes = _parse(right_manifest.get("experiment_axes")) if right_declared else None
    if not left_declared or not right_declared:
        return _DECLARED_AXES_DEFAULT, (
            "experiment axes declared on one side only; comparison is informational"
        )
    if left_axes is None or right_axes is None:
        return _DECLARED_AXES_DEFAULT, (
            "experiment axes are malformed; comparison is informational"
        )
    if left_axes != right_axes:
        return _DECLARED_AXES_DEFAULT, (
            "experiment axes differ; comparison is informational"
        )
    if left_axes != _DECLARED_AXES_DEFAULT:
        return _DECLARED_AXES_DEFAULT, (
            "unsupported experiment axes; comparison is informational"
        )
    return left_axes, None


def compare_score_sets(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare two frozen run/score-set snapshots without writing copy as suite quality."""

    left_manifest = _manifest(left)
    right_manifest = _manifest(right)
    reasons: list[str] = []

    left_schema, left_inferred = _artifact_schema(left)
    right_schema, right_inferred = _artifact_schema(right)
    # Kept out of `reasons` until the verdict is decided: a non-empty `reasons`
    # below means incompatible, and an inferred schema is not a blocking reason.
    notes: list[str] = []
    if left_inferred or right_inferred:
        notes.append(f"artifact schema not declared; assumed {CANDIDATE_ARTIFACT_SCHEMA}")

    for field in _ALIGN_FIELDS:
        left_value = left_manifest.get(field)
        right_value = right_manifest.get(field)
        if (
            not _align_present(left_value)
            or not _align_present(right_value)
            or left_value != right_value
        ):
            reasons.append(f"cannot align {field}")
    if left_schema != right_schema or not left_schema:
        reasons.append("cannot align artifact schema")

    if reasons:
        return {
            "compatibility": "incompatible",
            "reasons": reasons + notes,
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

    axes, axes_reason = _resolve_axes(left_manifest, right_manifest)
    undeclared = _undeclared_diffs(left_manifest, right_manifest, axes=axes)
    compatibility = "informational" if undeclared else "controlled"
    reasons.extend(undeclared)
    if axes_reason is not None:
        compatibility = "informational"
        reasons.append(axes_reason)

    usage_reason = _usage_contract_reason(left_manifest, right_manifest)
    if usage_reason is not None:
        compatibility = "informational"
        reasons.append(usage_reason)

    identity_reason = _scorer_identity_reason(left, right)
    if identity_reason is not None:
        compatibility = "informational"
        reasons.append(identity_reason)

    reasons.extend(notes)

    left_scorer = str(_score_set(left).get("scorer_version") or "")
    right_scorer = str(_score_set(right).get("scorer_version") or "")
    rescore_required = left_scorer != right_scorer
    delta = None
    if not rescore_required and compatibility == "controlled":
        left_scores = _score_set(left).get("aggregate_scores")
        right_scores = _score_set(right).get("aggregate_scores")
        if not isinstance(left_scores, Mapping) or not isinstance(right_scores, Mapping):
            compatibility = "informational"
            reasons.append(
                "aggregate scores are not comparable (not a mapping); "
                "delta is informational"
            )
        else:
            left_keys = {
                key for key, value in left_scores.items() if is_dimension_score(value)
            }
            right_keys = {
                key for key, value in right_scores.items() if is_dimension_score(value)
            }
            if not left_keys or not right_keys or left_keys != right_keys:
                compatibility = "informational"
                reasons.append(
                    "aggregate score dimensions are not comparable "
                    "(empty or differing dimension sets); delta is informational"
                )
            else:
                candidate_delta: dict[str, Any] = {}
                not_subtractable = False
                for key in sorted(left_keys):
                    left_value = left_scores[key]
                    right_value = right_scores[key]
                    try:
                        difference = right_value - left_value
                    except OverflowError:
                        # Comparable does not imply subtractable: subtracting an
                        # oversized int against a float coerces to float and
                        # raises. Do not emit a partial or empty delta as if it
                        # were controlled; downgrade the whole comparison.
                        not_subtractable = True
                        break
                    candidate_delta[key] = {
                        "left": left_value,
                        "right": right_value,
                        "delta": difference,
                    }
                if not_subtractable:
                    compatibility = "informational"
                    reasons.append(
                        "aggregate score dimensions are not subtractable; "
                        "delta is informational"
                    )
                else:
                    delta = candidate_delta

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


__all__ = [
    "CANDIDATE_ARTIFACT_SCHEMA",
    "compare_score_sets",
    "scorer_identity",
]
