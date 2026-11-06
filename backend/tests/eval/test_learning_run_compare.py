"""Pure compatibility tests for controlled Learning Run comparison."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.eval.learning_run.compare import compare_score_sets, scorer_identity
from app.eval.learning_run.contracts import hash_without_field
from app.eval.learning_run.registry import TaskRegistry


FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "app"
    / "eval"
    / "learning_run"
    / "fixtures"
    / "tutor-prompt-regression-v1.jsonl"
)

_REGISTRY = TaskRegistry.load_default()

# The committed hybrid-v1 nine-key projection hash, computed independently with
# the standard library. Distinct from the full document hash.
HYBRID_V1_PROJECTION_HASH = (
    "80ab0249b27ff284601d5eb2da43d6442d935015ce0d9d0c062765a74fa46a60"
)


def _identity(version: str) -> dict:
    """A real, self-consistent frozen scorer identity for a registry version.

    The definition hash is read from the committed scorer document, not invented,
    so the identity gate is exercised with the same frozen document a live run
    would carry rather than a fake constant.
    """

    document = dict(_REGISTRY.scorer_document(version))
    return {
        "scorer_id": document["scorer_id"],
        "scorer_version": version,
        "scorer_definition_hash": document["definition_hash"],
        "scorer_snapshot": document,
    }


def _side(
    *,
    run_id: str,
    variant_id: str = "tutor-v2",
    prompt_version: str = "tutor-v2",
    task_case_id: str = "tgqa-001",
    task_case_version: str = "1",
    corpus_hash: str = "c" * 64,
    artifact_schema: str = "candidate-artifact-v1",
    scorer_version: str = "hybrid-v1",
    score: int = 4,
    identity: bool = True,
) -> dict:
    score_set = {
        "scorer_id": "hybrid",
        "scorer_version": scorer_version,
        "aggregate_scores": {"groundedness": score},
    }
    if identity:
        score_set.update(_identity(scorer_version))
    return {
        "run_id": run_id,
        "variant_id": variant_id,
        "manifest": {
            "task_case_id": task_case_id,
            "task_case_version": task_case_version,
            "variant_id": variant_id,
            "prompt_version": prompt_version,
            "corpus_snapshot_hash": corpus_hash,
            "experiment_axes": ("prompt_version",),
            "schema_version": "learning-run-v1",
        },
        "artifact": {"schema_version": artifact_schema, "answer": f"answer-{run_id}"},
        "score_set": score_set,
    }


def test_controlled_compare_allows_prompt_axis_case_delta_only():
    result = compare_score_sets(
        _side(run_id="left", variant_id="tutor-v2", prompt_version="tutor-v2", score=4),
        _side(run_id="right", variant_id="tutor-v3", prompt_version="tutor-v3", score=2),
    )

    assert result["compatibility"] == "controlled"
    assert result["scope"] == "case"
    assert result["rescore_required"] is False
    assert result["delta"] == {
        "groundedness": {"left": 4, "right": 2, "delta": -2},
    }
    assert "case delta" in result["caption"]
    assert "suite" not in result["caption"]
    assert "quality" not in result["caption"]


def test_full_run_manifest_prompt_axis_stays_controlled():
    left = _side(run_id="left", variant_id="tutor-v2", prompt_version="tutor-v2")
    right = _side(run_id="right", variant_id="tutor-v3", prompt_version="tutor-v3", score=1)
    left["manifest"].update(
        {
            "prompt_text": "tutor v2 prompt",
            "prompt_hash": "a" * 64,
            "provider": "ollama",
            "model": "llama3.2",
            "connection_fingerprint": "b" * 64,
            "manifest_hash": "c" * 64,
            "code_revision": "left",
        }
    )
    right["manifest"].update(
        {
            "prompt_text": "tutor v3 prompt",
            "prompt_hash": "d" * 64,
            "provider": "ollama",
            "model": "llama3.2",
            "connection_fingerprint": "e" * 64,
            "manifest_hash": "f" * 64,
            "code_revision": "right",
        }
    )

    result = compare_score_sets(left, right)

    assert result["compatibility"] == "controlled"
    assert result["delta"]["groundedness"]["delta"] == -3


def test_informational_compare_when_undeclared_config_differs():
    left = _side(run_id="left")
    right = _side(run_id="right", variant_id="tutor-v3", prompt_version="tutor-v3")
    right["manifest"]["model"] = "other-model"

    result = compare_score_sets(left, right)

    assert result["compatibility"] == "informational"
    assert result["delta"] is None
    assert result["scope"] == "case"
    assert any("undeclared" in reason or "model" in reason for reason in result["reasons"])


def test_missing_artifact_cannot_align():
    left = _side(run_id="left")
    right = _side(run_id="right", variant_id="tutor-v3", prompt_version="tutor-v3")
    right["artifact"] = None

    result = compare_score_sets(left, right)

    assert result["compatibility"] == "incompatible"
    assert "cannot align artifact schema" in result["reasons"]


def test_curated_fixture_empty_scores_are_informational_without_schema_version():
    """The committed fixture's empty aggregate scores cannot form a controlled delta.

    The fixture is not edited: its hybrid-v1 ScoreSets carry no comparable
    dimensions, so the new dimension gate must report informational rather than
    a controlled empty delta. The artifact-schema inference is still exercised
    because the fixture predates the field.
    """
    by_pair: dict[tuple[str, str], dict] = {}
    for raw in FIXTURE_PATH.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        record = json.loads(raw)
        run = record["run"]
        by_pair[(run["task_case_id"], run["variant_id"])] = record

    left_record = by_pair[("tgqa-008", "tutor-v2")]
    right_record = by_pair[("tgqa-008", "tutor-v3")]
    left_run = left_record["run"]
    right_run = right_record["run"]
    assert "schema_version" not in left_run["candidate_artifact"]
    assert "schema_version" not in right_run["candidate_artifact"]
    left_score = next(
        item
        for item in left_record["score_sets"]
        if item["scorer_version"] == "hybrid-v1"
    )
    right_score = next(
        item
        for item in right_record["score_sets"]
        if item["scorer_version"] == "hybrid-v1"
    )
    # The fixture's frozen scores are empty; the gate must not be relaxed to
    # keep a controlled verdict for a fixture that carries no dimensions.
    assert left_score["aggregate_scores"] == {}
    assert right_score["aggregate_scores"] == {}

    result = compare_score_sets(
        {
            "run_id": left_run["id"],
            "variant_id": left_run["variant_id"],
            "manifest": left_run["manifest"],
            "artifact": left_run["candidate_artifact"],
            "score_set": {
                "scorer_id": left_score["scorer_id"],
                "scorer_version": left_score["scorer_version"],
                "scorer_definition_hash": left_score["scorer_definition_hash"],
                "scorer_snapshot": left_score["scorer_snapshot"],
                "aggregate_scores": left_score["aggregate_scores"],
            },
        },
        {
            "run_id": right_run["id"],
            "variant_id": right_run["variant_id"],
            "manifest": right_run["manifest"],
            "artifact": right_run["candidate_artifact"],
            "score_set": {
                "scorer_id": right_score["scorer_id"],
                "scorer_version": right_score["scorer_version"],
                "scorer_definition_hash": right_score["scorer_definition_hash"],
                "scorer_snapshot": right_score["scorer_snapshot"],
                "aggregate_scores": right_score["aggregate_scores"],
            },
        },
    )

    assert result["compatibility"] == "informational"
    assert result["delta"] is None
    assert result["rescore_required"] is False
    assert result["scope"] == "case"
    assert any("dimension" in reason for reason in result["reasons"])


def test_incompatible_when_task_or_corpus_or_artifact_schema_cannot_align():
    left = _side(run_id="left")
    right = _side(run_id="right", task_case_id="tgqa-002", corpus_hash="d" * 64)
    right["artifact"]["schema_version"] = "candidate-artifact-v0"

    result = compare_score_sets(left, right)

    assert result["compatibility"] == "incompatible"
    assert result["delta"] is None
    assert result["rescore_required"] is False
    assert result["reasons"]


def test_inferred_artifact_schema_is_reported_without_changing_compatibility():
    """Defaulting a missing `schema_version` must not be silent.

    Imported fixtures predate the field and `to_dict()` deliberately still
    omits it, because writing it would change `artifact_hash`. Treating the
    absence as `candidate-artifact-v1` is therefore correct, but a reader
    seeing `controlled` deserves to know an assumption was applied -- a
    genuinely v2 artifact that forgot the field would compare as v1.
    """
    left = _side(run_id="left")
    right = _side(run_id="right", variant_id="tutor-v3", prompt_version="tutor-v3")
    del left["artifact"]["schema_version"]
    del right["artifact"]["schema_version"]

    result = compare_score_sets(left, right)

    assert result["compatibility"] == "controlled"
    assert any("candidate-artifact-v1" in reason for reason in result["reasons"])


def test_declared_artifact_schema_reports_no_inference():
    left = _side(run_id="left")
    right = _side(run_id="right", variant_id="tutor-v3", prompt_version="tutor-v3")

    result = compare_score_sets(left, right)

    assert result["compatibility"] == "controlled"
    assert result["reasons"] == []



def test_different_scorer_version_requires_rescore_and_hides_score_delta():
    result = compare_score_sets(
        _side(run_id="left", scorer_version="hybrid-v1", score=5),
        _side(
            run_id="right",
            variant_id="tutor-v3",
            prompt_version="tutor-v3",
            scorer_version="hybrid-v2",
            score=1,
        ),
    )

    assert result["rescore_required"] is True
    assert result["delta"] is None
    assert result["compatibility"] in {"controlled", "informational"}


def test_single_case_copy_cannot_claim_suite_or_general_quality():
    result = compare_score_sets(
        _side(run_id="a"),
        _side(run_id="b", prompt_version="tutor-v3", variant_id="tutor-v3"),
    )
    assert result["caption"] == "case delta"
    assert "suite" not in result["caption"]
    assert "quality" not in result["caption"]


def test_overflowing_dimension_downgrades_the_whole_comparison_to_informational():
    """Comparable does not imply subtractable.

    `is_dimension_score` answers "can these be compared", and comparison is
    safe for an oversized int (`10**1000 < 4.5` is False, no raise). Building a
    delta needs more: `10**1000 - 4.5` coerces the int to float and raises
    OverflowError, so a mixed pair from the unvalidated import path would turn
    `/api/eval/compare` into a 500. A partial or empty controlled delta would be
    a wrong answer, so the whole comparison is downgraded to informational with
    a None delta rather than emitting a hole.
    """
    import json as _json

    left = _side(run_id="left")
    right = _side(run_id="right", variant_id="tutor-v3", prompt_version="tutor-v3")
    left["score_set"]["aggregate_scores"] = {"groundedness": 4.5}
    right["score_set"]["aggregate_scores"] = {"groundedness": 10**1000}

    result = compare_score_sets(left, right)

    assert result["compatibility"] == "informational"
    assert result["delta"] is None
    assert any("not subtractable" in reason for reason in result["reasons"])
    # The original safety purpose is unchanged: no exception, no NaN/Infinity.
    _json.dumps(result, allow_nan=False)


# ---------------------------------------------------------------------------
# D5c-2 — the dedicated usage_contract gate.
#
# Both sides genuinely omitting the key is a legacy pair and keeps the original
# quality comparison. Both declaring the same supported contract is comparable.
# Every other combination is informational with an explicit reason and a None
# delta, and it cannot be bypassed by declaring the field as an experiment axis.
# ---------------------------------------------------------------------------

_MISSING = object()


def test_both_legacy_manifests_keep_the_original_quality_comparison():
    """Neither manifest declares usage_contract: no gate, no added note."""
    left = _side(run_id="left", variant_id="tutor-v2", prompt_version="tutor-v2", score=4)
    right = _side(run_id="right", variant_id="tutor-v3", prompt_version="tutor-v3", score=2)

    result = compare_score_sets(left, right)

    assert result["compatibility"] == "controlled"
    assert result["reasons"] == []
    assert result["delta"] == {"groundedness": {"left": 4, "right": 2, "delta": -2}}


def test_both_new_manifests_declare_the_same_supported_contract():
    left = _side(run_id="left", variant_id="tutor-v2", prompt_version="tutor-v2", score=4)
    right = _side(run_id="right", variant_id="tutor-v3", prompt_version="tutor-v3", score=2)
    left["manifest"]["usage_contract"] = "tutor-chunk-sum-v1"
    right["manifest"]["usage_contract"] = "tutor-chunk-sum-v1"

    result = compare_score_sets(left, right)

    assert result["compatibility"] == "controlled"
    assert result["reasons"] == []
    assert result["delta"] == {"groundedness": {"left": 4, "right": 2, "delta": -2}}


@pytest.mark.parametrize(
    "left_value,right_value,fragment",
    [
        pytest.param("tutor-chunk-sum-v1", _MISSING, "one side only", id="one-sided-old-new"),
        pytest.param("tutor-chunk-sum-v1", "tutor-latest-chunk-v1", "differ", id="different"),
        pytest.param("tutor-unknown-v9", "tutor-unknown-v9", "unsupported", id="same-unknown"),
        pytest.param("", "tutor-chunk-sum-v1", "invalid", id="empty"),
        pytest.param(None, "tutor-chunk-sum-v1", "invalid", id="null"),
        pytest.param(123, "tutor-chunk-sum-v1", "invalid", id="non-string"),
    ],
)
def test_uncomparable_usage_contracts_are_informational(
    left_value, right_value, fragment
):
    left = _side(run_id="left", variant_id="tutor-v2", prompt_version="tutor-v2")
    right = _side(run_id="right", variant_id="tutor-v3", prompt_version="tutor-v3")
    if left_value is not _MISSING:
        left["manifest"]["usage_contract"] = left_value
    if right_value is not _MISSING:
        right["manifest"]["usage_contract"] = right_value

    result = compare_score_sets(left, right)

    assert result["compatibility"] == "informational"
    assert result["delta"] is None
    assert any(fragment in reason for reason in result["reasons"])
    # The dedicated gate reports once; the generic undeclared diff must not
    # double-report the same field.
    assert not any(
        "undeclared config differs: usage_contract" in reason
        for reason in result["reasons"]
    )


def test_usage_contract_cannot_be_bypassed_by_declaring_it_as_an_axis():
    left = _side(run_id="left", variant_id="tutor-v2", prompt_version="tutor-v2")
    right = _side(run_id="right", variant_id="tutor-v3", prompt_version="tutor-v3")
    left["manifest"]["usage_contract"] = "tutor-chunk-sum-v1"
    left["manifest"]["experiment_axes"] = ("prompt_version", "usage_contract")
    right["manifest"]["experiment_axes"] = ("prompt_version", "usage_contract")

    result = compare_score_sets(left, right)

    assert result["compatibility"] == "informational"
    assert result["delta"] is None
    assert any("usage contract" in reason for reason in result["reasons"])


def test_scorer_version_difference_still_requires_rescore_under_one_contract():
    left = _side(run_id="left", scorer_version="hybrid-v1", score=5)
    right = _side(
        run_id="right",
        variant_id="tutor-v3",
        prompt_version="tutor-v3",
        scorer_version="hybrid-v2",
        score=1,
    )
    left["manifest"]["usage_contract"] = "tutor-chunk-sum-v1"
    right["manifest"]["usage_contract"] = "tutor-chunk-sum-v1"

    result = compare_score_sets(left, right)

    # A different scorer version is a different scorer identity, so the D5d
    # identity gate downgrades the pair to informational even though the usage
    # contract matches; the version difference still requires a re-score.
    assert result["compatibility"] == "informational"
    assert result["rescore_required"] is True
    assert result["delta"] is None
    assert any("versions differ" in reason for reason in result["reasons"])


def test_other_undeclared_config_difference_still_reports_under_one_contract():
    left = _side(run_id="left")
    right = _side(run_id="right", variant_id="tutor-v3", prompt_version="tutor-v3")
    left["manifest"]["usage_contract"] = "tutor-chunk-sum-v1"
    right["manifest"]["usage_contract"] = "tutor-chunk-sum-v1"
    right["manifest"]["model"] = "other-model"

    result = compare_score_sets(left, right)

    assert result["compatibility"] == "informational"
    assert any(
        "undeclared" in reason or "model" in reason for reason in result["reasons"]
    )


# ---------------------------------------------------------------------------
# D5d-1 -- the dedicated scorer identity gate.
#
# A controlled comparison needs both sides to carry a complete, self-consistent
# (scorer_id, scorer_version, scorer_definition_hash) and for the two to be
# equal. Missing or inconsistent identities, and same-version/different-hash
# pairs, are informational with a None delta; a version difference additionally
# sets rescore_required. The gate reads only the frozen snapshot and never
# back-infers a historical identity from today's registry.
# ---------------------------------------------------------------------------


def test_scorer_identity_is_complete_and_self_consistent_for_both_shapes():
    document = dict(_REGISTRY.scorer_document("hybrid-v1"))
    assert scorer_identity(
        {
            "scorer_id": "hybrid",
            "scorer_version": "hybrid-v1",
            "scorer_definition_hash": document["definition_hash"],
            "scorer_snapshot": document,
        }
    ) == ("hybrid", "hybrid-v1", document["definition_hash"])

    # The nine-key projection is the historical shape and must recompute to the
    # stored hash under the same rule. The expected hash is the independently
    # computed fixed value, not the production helper echoed back.
    projection = {
        key: value
        for key, value in document.items()
        if key not in ("schema_version", "definition_hash")
    }
    assert hash_without_field(projection, "definition_hash") == HYBRID_V1_PROJECTION_HASH
    assert scorer_identity(
        {
            "scorer_id": "hybrid",
            "scorer_version": "hybrid-v1",
            "scorer_definition_hash": HYBRID_V1_PROJECTION_HASH,
            "scorer_snapshot": projection,
        }
    ) == ("hybrid", "hybrid-v1", HYBRID_V1_PROJECTION_HASH)

    # A snapshot whose recomputed hash does not match the stored hash is not a
    # trusted identity, even though it declares its own definition_hash.
    assert (
        scorer_identity(
            {
                "scorer_id": "hybrid",
                "scorer_version": "hybrid-v1",
                "scorer_definition_hash": "0" * 64,
                "scorer_snapshot": document,
            }
        )
        is None
    )


def test_same_self_consistent_identity_stays_controlled():
    result = compare_score_sets(
        _side(run_id="left", variant_id="tutor-v2", prompt_version="tutor-v2", score=4),
        _side(run_id="right", variant_id="tutor-v3", prompt_version="tutor-v3", score=2),
    )

    assert result["compatibility"] == "controlled"
    assert result["rescore_required"] is False
    assert result["delta"] == {"groundedness": {"left": 4, "right": 2, "delta": -2}}


def test_missing_identity_on_both_sides_is_not_controlled():
    """Equal-but-empty identities must not read as a match."""
    left = _side(run_id="left", identity=False)
    right = _side(run_id="right", variant_id="tutor-v3", prompt_version="tutor-v3", identity=False)

    result = compare_score_sets(left, right)

    assert result["compatibility"] == "informational"
    assert result["delta"] is None
    assert result["rescore_required"] is False
    assert any("scorer identity" in reason for reason in result["reasons"])


def test_one_sided_identity_is_informational():
    left = _side(run_id="left")
    right = _side(run_id="right", variant_id="tutor-v3", prompt_version="tutor-v3", identity=False)

    result = compare_score_sets(left, right)

    assert result["compatibility"] == "informational"
    assert result["delta"] is None
    assert any("scorer identity" in reason for reason in result["reasons"])


def test_same_version_different_definition_hash_is_informational():
    left = _side(run_id="left", score=4)
    right = _side(run_id="right", variant_id="tutor-v3", prompt_version="tutor-v3", score=2)

    # A self-consistent identity at the same version but a different definition.
    other = dict(_REGISTRY.scorer_document("hybrid-v1"))
    other["parser_version"] = "json-rubric-v1-tampered"
    other["definition_hash"] = hash_without_field(other, "definition_hash")
    right["score_set"].update(
        {
            "scorer_definition_hash": other["definition_hash"],
            "scorer_snapshot": other,
        }
    )

    result = compare_score_sets(left, right)

    assert result["compatibility"] == "informational"
    assert result["delta"] is None
    assert result["rescore_required"] is False
    assert any("definition hash differs" in reason for reason in result["reasons"])


def test_different_version_identity_is_informational_and_requires_rescore():
    left = _side(run_id="left", scorer_version="hybrid-v1", score=5)
    right = _side(
        run_id="right",
        variant_id="tutor-v3",
        prompt_version="tutor-v3",
        scorer_version="hybrid-v2",
        score=1,
    )

    result = compare_score_sets(left, right)

    assert result["compatibility"] == "informational"
    assert result["delta"] is None
    assert result["rescore_required"] is True
    assert any("versions differ" in reason for reason in result["reasons"])


def test_scorer_identity_rejects_a_snapshot_that_does_not_name_the_row_identity():
    """The snapshot's own id/version must match the row, not just its hash."""
    document = dict(_REGISTRY.scorer_document("hybrid-v1"))

    wrong_id = {**document, "scorer_id": "not-hybrid"}
    wrong_id_hash = hash_without_field(wrong_id, "definition_hash")
    assert (
        scorer_identity(
            {
                "scorer_id": "hybrid",
                "scorer_version": "hybrid-v1",
                "scorer_definition_hash": wrong_id_hash,
                "scorer_snapshot": wrong_id,
            }
        )
        is None
    )

    wrong_version = {**document, "version": "hybrid-v9"}
    wrong_version_hash = hash_without_field(wrong_version, "definition_hash")
    assert (
        scorer_identity(
            {
                "scorer_id": "hybrid",
                "scorer_version": "hybrid-v1",
                "scorer_definition_hash": wrong_version_hash,
                "scorer_snapshot": wrong_version,
            }
        )
        is None
    )

    # A snapshot missing the naming fields cannot be tied to the row identity,
    # even when its recomputed hash still matches the stored definition hash.
    for missing_field in ("scorer_id", "version"):
        partial = {key: value for key, value in document.items() if key != missing_field}
        partial_hash = hash_without_field(partial, "definition_hash")
        assert (
            scorer_identity(
                {
                    "scorer_id": "hybrid",
                    "scorer_version": "hybrid-v1",
                    "scorer_definition_hash": partial_hash,
                    "scorer_snapshot": partial,
                }
            )
            is None
        )


def test_no_score_set_on_both_sides_is_informational():
    left = _side(run_id="left")
    right = _side(run_id="right", variant_id="tutor-v3", prompt_version="tutor-v3")
    left["score_set"] = None
    right["score_set"] = None

    result = compare_score_sets(left, right)

    assert result["compatibility"] == "informational"
    assert result["delta"] is None
    assert result["rescore_required"] is False
    assert any("scorer identity" in reason for reason in result["reasons"])


def test_one_sided_missing_score_set_is_informational():
    left = _side(run_id="left")
    right = _side(run_id="right", variant_id="tutor-v3", prompt_version="tutor-v3")
    right["score_set"] = None

    result = compare_score_sets(left, right)

    assert result["compatibility"] == "informational"
    assert result["delta"] is None
    assert any("scorer identity" in reason for reason in result["reasons"])


def test_empty_score_set_mapping_is_informational():
    left = _side(run_id="left")
    right = _side(run_id="right", variant_id="tutor-v3", prompt_version="tutor-v3")
    right["score_set"] = {}

    result = compare_score_sets(left, right)

    assert result["compatibility"] == "informational"
    assert result["delta"] is None
    assert any("scorer identity" in reason for reason in result["reasons"])


# ---------------------------------------------------------------------------
# D5d batch C -- alignment presence, comparable dimension sets and the
# declared-axis gate.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["task_case_id", "task_case_version", "corpus_snapshot_hash"])
@pytest.mark.parametrize("side", ["left", "right", "both"])
def test_missing_align_field_never_aligns_by_mutual_absence(field, side):
    left = _side(run_id="left")
    right = _side(run_id="right", variant_id="tutor-v3", prompt_version="tutor-v3")
    if side in ("left", "both"):
        del left["manifest"][field]
    if side in ("right", "both"):
        del right["manifest"][field]

    result = compare_score_sets(left, right)

    assert result["compatibility"] == "incompatible"
    assert result["delta"] is None
    assert f"cannot align {field}" in result["reasons"]


@pytest.mark.parametrize("field", ["task_case_id", "task_case_version", "corpus_snapshot_hash"])
def test_empty_align_field_is_incompatible(field):
    left = _side(run_id="left")
    right = _side(run_id="right", variant_id="tutor-v3", prompt_version="tutor-v3")
    left["manifest"][field] = ""

    result = compare_score_sets(left, right)

    assert result["compatibility"] == "incompatible"
    assert result["delta"] is None
    assert f"cannot align {field}" in result["reasons"]


def test_empty_dimension_sets_are_informational():
    left = _side(run_id="left")
    right = _side(run_id="right", variant_id="tutor-v3", prompt_version="tutor-v3")
    left["score_set"]["aggregate_scores"] = {}
    right["score_set"]["aggregate_scores"] = {}

    result = compare_score_sets(left, right)

    assert result["compatibility"] == "informational"
    assert result["delta"] is None
    assert any("dimension" in reason for reason in result["reasons"])


def test_asymmetric_dimension_sets_are_informational():
    left = _side(run_id="left")
    right = _side(run_id="right", variant_id="tutor-v3", prompt_version="tutor-v3")
    left["score_set"]["aggregate_scores"] = {"groundedness": 4}
    right["score_set"]["aggregate_scores"] = {"groundedness": 4, "coverage": 3}

    result = compare_score_sets(left, right)

    assert result["compatibility"] == "informational"
    assert result["delta"] is None
    assert any("dimension" in reason for reason in result["reasons"])


def test_non_mapping_or_missing_aggregate_scores_are_informational():
    left = _side(run_id="left")
    right = _side(run_id="right", variant_id="tutor-v3", prompt_version="tutor-v3")
    left["score_set"]["aggregate_scores"] = [4]

    result = compare_score_sets(left, right)

    assert result["compatibility"] == "informational"
    assert result["delta"] is None
    assert any("not a mapping" in reason for reason in result["reasons"])

    del left["score_set"]["aggregate_scores"]
    missing = compare_score_sets(left, right)
    assert missing["compatibility"] == "informational"
    assert missing["delta"] is None


def test_both_missing_axes_keep_the_legacy_prompt_default():
    left = _side(run_id="left")
    right = _side(run_id="right", variant_id="tutor-v3", prompt_version="tutor-v3")
    del left["manifest"]["experiment_axes"]
    del right["manifest"]["experiment_axes"]

    result = compare_score_sets(left, right)

    assert result["compatibility"] == "controlled"
    assert result["delta"] == {"groundedness": {"left": 4, "right": 4, "delta": 0}}


def test_explicit_matching_prompt_axis_stays_controlled():
    left = _side(run_id="left")
    right = _side(run_id="right", variant_id="tutor-v3", prompt_version="tutor-v3")
    left["manifest"]["experiment_axes"] = ["prompt_version"]
    right["manifest"]["experiment_axes"] = ("prompt_version",)

    result = compare_score_sets(left, right)

    assert result["compatibility"] == "controlled"


@pytest.mark.parametrize(
    "left_axes,right_axes",
    [
        pytest.param(("prompt_version",), None, id="one-sided"),
        pytest.param(
            ("prompt_version",), ("prompt_version", "model"), id="differing"
        ),
        pytest.param(
            ("prompt_version", "model"), ("prompt_version", "model"), id="extended"
        ),
        pytest.param((), (), id="empty"),
        pytest.param("prompt_version", "prompt_version", id="malformed-string"),
        pytest.param(5, 5, id="malformed-int"),
        pytest.param(None, ("prompt_version",), id="one-sided-right"),
    ],
)
def test_nonconforming_declared_axes_are_informational(left_axes, right_axes):
    left = _side(run_id="left")
    right = _side(run_id="right", variant_id="tutor-v3", prompt_version="tutor-v3")
    if left_axes is None:
        del left["manifest"]["experiment_axes"]
    else:
        left["manifest"]["experiment_axes"] = left_axes
    if right_axes is None:
        del right["manifest"]["experiment_axes"]
    else:
        right["manifest"]["experiment_axes"] = right_axes

    result = compare_score_sets(left, right)

    assert result["compatibility"] == "informational"
    assert result["delta"] is None
    assert any("experiment axes" in reason for reason in result["reasons"])


def test_declared_axis_cannot_bypass_the_usage_gate():
    left = _side(run_id="left")
    right = _side(run_id="right", variant_id="tutor-v3", prompt_version="tutor-v3")
    left["manifest"]["usage_contract"] = "tutor-chunk-sum-v1"
    left["manifest"]["experiment_axes"] = ("prompt_version",)
    right["manifest"]["experiment_axes"] = ("prompt_version",)

    result = compare_score_sets(left, right)

    assert result["compatibility"] == "informational"
    assert result["delta"] is None
    assert any("usage contract" in reason for reason in result["reasons"])
