"""Pure compatibility tests for controlled Learning Run comparison."""

from __future__ import annotations

from app.eval.learning_run.compare import compare_score_sets


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
) -> dict:
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
        "score_set": {
            "scorer_id": "hybrid",
            "scorer_version": scorer_version,
            "aggregate_scores": {"groundedness": score},
        },
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


def test_informational_compare_when_undeclared_config_differs():
    left = _side(run_id="left")
    right = _side(run_id="right", variant_id="tutor-v3", prompt_version="tutor-v3")
    right["manifest"]["model"] = "other-model"

    result = compare_score_sets(left, right)

    assert result["compatibility"] == "informational"
    assert result["delta"] is None
    assert result["scope"] == "case"
    assert any("undeclared" in reason or "model" in reason for reason in result["reasons"])


def test_incompatible_when_task_or_corpus_or_artifact_schema_cannot_align():
    left = _side(run_id="left")
    right = _side(run_id="right", task_case_id="tgqa-002", corpus_hash="d" * 64)
    right["artifact"]["schema_version"] = "candidate-artifact-v0"

    result = compare_score_sets(left, right)

    assert result["compatibility"] == "incompatible"
    assert result["delta"] is None
    assert result["rescore_required"] is False
    assert result["reasons"]


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
