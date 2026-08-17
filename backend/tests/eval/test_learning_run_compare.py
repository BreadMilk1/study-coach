"""Pure compatibility tests for controlled Learning Run comparison."""

from __future__ import annotations

import json
from pathlib import Path

from app.eval.learning_run.compare import compare_score_sets


FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "app"
    / "eval"
    / "learning_run"
    / "fixtures"
    / "tutor-prompt-regression-v1.jsonl"
)


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


def test_curated_fixture_prompt_axis_pair_is_controlled_without_schema_version():
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

    result = compare_score_sets(
        {
            "run_id": left_run["id"],
            "variant_id": left_run["variant_id"],
            "manifest": left_run["manifest"],
            "artifact": left_run["candidate_artifact"],
            "score_set": {
                "scorer_id": left_score["scorer_id"],
                "scorer_version": left_score["scorer_version"],
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
                "aggregate_scores": right_score["aggregate_scores"],
            },
        },
    )

    assert result["compatibility"] == "controlled"
    assert result["rescore_required"] is False
    assert result["scope"] == "case"


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
