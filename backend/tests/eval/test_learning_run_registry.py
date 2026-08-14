"""Contract tests for the versioned Learning Run Registry.

These tests intentionally exercise the public registry boundary.  Clients may
submit identifiers only; all executable definitions and integrity hashes come
from the version-controlled registry.
"""

import hashlib
import json
import os
import shutil
import typing
from pathlib import Path

import pytest

from app.agent.prompt import SYSTEM_INSTRUCTION
from app.eval.learning_run import contracts as contracts_module
from app.eval.learning_run.contracts import (
    canonical_hash,
    canonical_json_bytes,
    ExperimentDefinition,
    ResolvedRunDefinition,
    RunManifest,
    ScorerBundle,
    TaskCase,
)
from app.eval.learning_run.registry import RegistryError, TaskRegistry


def test_registry_resolves_ids_and_freezes_evaluation_contract():
    registry = TaskRegistry.load_default()

    resolved = registry.resolve_run(
        experiment_id="tutor-prompt-regression-v1",
        task_case_id="tgqa-004",
        variant_id="tutor-v3",
        run_profile="evaluation",
    )

    assert resolved.experiment_axes == ("prompt_version",)
    assert resolved.runtime_judge is False
    assert resolved.task.task_case_id == "tgqa-004"
    assert resolved.task.case_type == "expected_refusal"
    assert resolved.prompt.version == "tutor-v3"
    assert resolved.experiment.run_profile == "evaluation"


def test_registry_has_exactly_twelve_cases_with_required_distribution():
    registry = TaskRegistry.load_default()

    assert len(registry.task_cases) == 12
    assert {
        case.case_type for case in registry.task_cases.values()
    } == {"answerable", "multi_evidence", "expected_refusal"}
    assert sum(c.case_type == "answerable" for c in registry.task_cases.values()) == 6
    assert sum(c.case_type == "multi_evidence" for c in registry.task_cases.values()) == 3
    assert sum(c.case_type == "expected_refusal" for c in registry.task_cases.values()) == 3
    assert set(registry.task_cases).isdisjoint(registry.calibration_case_ids)


def test_default_prompt_is_byte_exact_production_v2_and_candidate_differs():
    registry = TaskRegistry.load_default()

    assert registry.prompts["tutor-v2"].text == SYSTEM_INSTRUCTION
    assert registry.prompts["tutor-v2"].text.encode("utf-8") == SYSTEM_INSTRUCTION.encode(
        "utf-8"
    )
    assert registry.prompts["tutor-v2"].text == registry.production_prompt.text
    assert registry.prompts["tutor-v3"].text != registry.prompts["tutor-v2"].text
    assert not registry.prompts["tutor-v2"].text.endswith("\n")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prompt_text", "client supplied prompt"),
        ("prompt_path", "/tmp/client-prompt.txt"),
        ("expected_answer", "client supplied golden answer"),
        ("profile", "production-fidelity"),
        ("retrieval", {"top_k": 99}),
        ("model", "client-model"),
        ("provider", "client-provider"),
        ("parameters", {"temperature": 1.0}),
    ],
)
def test_resolve_run_rejects_every_client_override(field, value):
    registry = TaskRegistry.load_default()
    kwargs = {
        "experiment_id": "tutor-prompt-regression-v1",
        "task_case_id": "tgqa-004",
        "variant_id": "tutor-v3",
        "run_profile": "evaluation",
        field: value,
    }

    with pytest.raises(RegistryError, match="override|client"):
        registry.resolve_run(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "experiment_id": "unknown-experiment",
            "task_case_id": "tgqa-004",
            "variant_id": "tutor-v3",
            "run_profile": "evaluation",
        },
        {
            "experiment_id": "tutor-prompt-regression-v1",
            "task_case_id": "unknown-case",
            "variant_id": "tutor-v3",
            "run_profile": "evaluation",
        },
        {
            "experiment_id": "tutor-prompt-regression-v1",
            "task_case_id": "tgqa-004",
            "variant_id": "unknown-variant",
            "run_profile": "evaluation",
        },
        {
            "experiment_id": "tutor-prompt-regression-v1",
            "task_case_id": "tgqa-004",
            "variant_id": "tutor-v3",
            "run_profile": "production-fidelity",
        },
    ],
)
def test_registry_rejects_unknown_ids_and_non_evaluation_profile(kwargs):
    registry = TaskRegistry.load_default()

    with pytest.raises(RegistryError):
        registry.resolve_run(**kwargs)


def test_canonical_hash_is_compact_sorted_utf8_and_shared_for_payloads():
    payload = {"z": ["é", 2], "a": {"b": True, "a": None}}
    expected = b'{"a":{"a":null,"b":true},"z":["\xc3\xa9",2]}'

    assert canonical_json_bytes(payload) == expected
    assert canonical_hash(payload) == hashlib.sha256(expected).hexdigest()
    assert canonical_hash({"a": payload["a"], "z": payload["z"]}) == canonical_hash(
        payload
    )


def test_duplicate_case_ids_are_rejected_before_registry_can_load(tmp_path):
    source = Path(TaskRegistry.default_definitions_path())
    shutil.copytree(source, tmp_path, dirs_exist_ok=True)
    cases_path = tmp_path / "task_cases.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    cases["cases"].append(dict(cases["cases"][0]))
    cases_path.write_text(
        json.dumps(cases, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    with pytest.raises(RegistryError, match="duplicate|hash"):
        TaskRegistry.from_directory(tmp_path)


@pytest.mark.parametrize(
    "relative_path",
    [
        "task_cases.json",
        "prompts/tutor-v3.txt",
        "scorers/hybrid-v1.json",
    ],
)
def test_definition_hash_mismatch_is_rejected(tmp_path, relative_path):
    source = Path(TaskRegistry.default_definitions_path())
    shutil.copytree(source, tmp_path, dirs_exist_ok=True)
    target = tmp_path / relative_path
    if target.suffix == ".json":
        payload = json.loads(target.read_text(encoding="utf-8"))
        if "cases" in payload:
            payload["cases"][0]["question"] += " tampered"
        else:
            payload["tampered"] = True
        target.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
    else:
        target.write_text(
            target.read_text(encoding="utf-8") + " tampered",
            encoding="utf-8",
        )

    with pytest.raises(RegistryError, match="hash|integrity|mismatch"):
        TaskRegistry.from_directory(tmp_path)


def test_resolved_run_contains_frozen_references_and_fixed_controls():
    registry = TaskRegistry.load_default()
    resolved = registry.resolve_run(
        experiment_id="tutor-prompt-regression-v1",
        task_case_id="tgqa-004",
        variant_id="tutor-v2",
        run_profile="evaluation",
    )

    assert resolved.corpus.snapshot_id == registry.corpus.snapshot_id
    assert resolved.scorer.version == "hybrid-v1"
    assert resolved.runtime_judge is False
    assert resolved.budget == {
        "retrieval_preflight_seconds": 5,
        "tutor_seconds": 55,
        "hybrid_scoring_seconds": 25,
        "total_seconds": 90,
    }
    assert resolved.variant_id == "tutor-v2"
    assert resolved.variant_controls["prompt_version"] == "tutor-v2"
    assert "tutor-v3" not in resolved.variant_controls
    assert resolved.variant_controls["embedding_config_version"] == (
        resolved.corpus.embedding_config_version
    )


def test_registry_exposes_typed_calibration_artifacts_with_manual_labels():
    registry = TaskRegistry.load_default()

    calibration_type = getattr(contracts_module, "CalibrationCandidate", None)
    assert calibration_type is not None
    assert isinstance(registry.calibration_candidates, tuple)
    assert all(isinstance(candidate, calibration_type) for candidate in registry.calibration_candidates)
    assert [candidate.anchor_label for candidate in registry.calibration_candidates] == [
        "pass",
        "fail",
        "borderline",
        "correct_refusal",
        "incorrect_refusal",
    ]
    assert [candidate.manual_expected_verdict for candidate in registry.calibration_candidates] == [
        "pass",
        "fail",
        "fail",
        "pass",
        "fail",
    ]
    for candidate in registry.calibration_candidates:
        assert candidate.artifact_hash == canonical_hash(candidate.artifact.to_dict())
        assert candidate.task.question
        assert candidate.task.expected_behavior
        assert candidate.manual_reason
        assert candidate.artifact.answer
        assert isinstance(candidate.artifact.citations, tuple)
        assert isinstance(candidate.artifact.exact_evidence, tuple)
        assert isinstance(candidate.artifact.trace, tuple)


def test_scorer_bundle_has_distinct_deterministic_and_llm_components_and_anchors():
    registry = TaskRegistry.load_default()

    assert set(registry.scorer.rubric["anchors"]) == {"1", "2", "3", "4", "5"}
    assert len(registry.scorer.components) >= 2
    assert len({component.component_id for component in registry.scorer.components}) == len(
        registry.scorer.components
    )
    assert {component.kind for component in registry.scorer.components} == {
        "deterministic",
        "llm",
    }
    assert registry.scorer.calibration_hash
    controls = registry.experiment.variants
    v2 = dict(controls["tutor-v2"])
    v3 = dict(controls["tutor-v3"])
    v2.pop("prompt_version")
    v3.pop("prompt_version")
    assert v2 == v3


def test_scorer_component_boundaries_keep_semantic_rubric_in_llm_component():
    registry = TaskRegistry.load_default()
    semantic_dimensions = {
        dimension
        for dimensions in registry.scorer.required_dimensions_by_case_type.values()
        for dimension in dimensions
    }
    deterministic_checks = {
        "evidence_membership",
        "retrieval_empty",
        "citation_presence",
        "citation_number",
        "numbering",
        "chunk_id",
        "span",
        "expected_refusal_observed",
    }
    deterministic = [
        component
        for component in registry.scorer.components
        if component.kind == "deterministic"
    ]
    llm = [
        component
        for component in registry.scorer.components
        if component.component_id == "grounded-quality-rubric"
    ]

    assert deterministic
    assert len(llm) == 1
    for component in deterministic:
        checks = set(component.config.get("checks", ()))
        assert not checks & semantic_dimensions
        assert checks <= deterministic_checks
    assert set(llm[0].config["dimensions"]) == semantic_dimensions
    assert llm[0].config["parser_version"] == registry.scorer.parser_version
    assert llm[0].config["model_config"] == registry.scorer.model_config


def test_contract_surface_has_selected_controls_annotation_and_no_prompt_helpers():
    from app.eval.learning_run.contracts import ResolvedRunDefinition, ScorerComponent

    annotation = typing.get_type_hints(ResolvedRunDefinition)["variant_controls"]
    assert str(annotation) == "typing.Mapping[str, typing.Any]"
    assert not hasattr(ScorerComponent, "from_text")
    assert not hasattr(ScorerComponent, "payload")


def _rehash_definition(payload: dict) -> None:
    payload["definition_hash"] = canonical_hash(
        {key: value for key, value in payload.items() if key != "definition_hash"}
    )


def _rehash_experiment_references(root: Path, *, task_hash: str | None = None) -> None:
    experiment_path = root / "experiment.json"
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    if task_hash is not None:
        experiment["task_cases_hash"] = task_hash
    _rehash_definition(experiment)
    experiment_path.write_text(
        json.dumps(experiment, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("target_id", "mutation", "reason"),
    [
        ("tgqa-001", lambda case: case["required_evidence_set"].append("not-a-real-chunk"), "chunk"),
        ("tgqa-001", lambda case: case["required_dimensions"].append("not-a-real-dimension"), "dimension"),
        ("tgqa-001", lambda case: case.update(question=""), "question"),
        ("tgqa-001", lambda case: case.update(manual_rationale=""), "rationale"),
        ("tgqa-001", lambda case: case.update(expected_behavior=""), "behavior"),
        ("tgqa-004", lambda case: case.update(required_evidence_set=["tgqa-c01-rrf"]), "evidence"),
        ("tgqa-001", lambda case: case.update(required_evidence_set=[]), "evidence"),
        ("tgqa-009", lambda case: case.update(required_evidence_set=["tgqa-c03-spaced-practice"]), "evidence"),
        ("tgqa-001", lambda case: case.update(critical_policy={"hard_fail_findings": [], "expected_refusal": True}), "critical"),
    ],
)
def test_task_case_semantics_are_cross_validated_after_rehash(tmp_path, target_id, mutation, reason):
    source = Path(TaskRegistry.default_definitions_path())
    shutil.copytree(source, tmp_path, dirs_exist_ok=True)
    cases_path = tmp_path / "task_cases.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    target = next(case for case in cases["cases"] if case["id"] == target_id)
    mutation(target)
    _rehash_definition(cases)
    cases_path.write_text(
        json.dumps(cases, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    _rehash_experiment_references(tmp_path, task_hash=cases["definition_hash"])

    with pytest.raises(RegistryError, match=reason):
        TaskRegistry.from_directory(tmp_path)


@pytest.mark.parametrize("mutation", ["artifact", "artifact_hash"])
def test_calibration_artifact_or_hash_tamper_is_rejected_even_when_file_rehashed(
    tmp_path, mutation
):
    source = Path(TaskRegistry.default_definitions_path())
    shutil.copytree(source, tmp_path, dirs_exist_ok=True)
    calibration_path = tmp_path / "calibration" / "candidates.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    candidate = calibration["candidates"][0]
    if mutation == "artifact":
        candidate["candidate_artifact"]["answer"] += " tampered"
    else:
        candidate["artifact_hash"] = "0" * 64
    _rehash_definition(calibration)
    calibration_path.write_text(
        json.dumps(calibration, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    with pytest.raises(RegistryError, match="artifact hash"):
        TaskRegistry.from_directory(tmp_path)


def _sample_manifest() -> RunManifest:
    return RunManifest(
        experiment_id="experiment",
        task_case_id="case",
        task_case_version="1",
        variant_id="tutor-v2",
        run_profile="evaluation",
        task_snapshot={"id": "case", "version": "1", "question": "question"},
        prompt_text="frozen prompt",
        corpus_snapshot={"snapshot_id": "corpus", "version": "1", "aggregate_hash": "a" * 64},
        scorer_snapshot={"id": "hybrid", "version": "v1", "hash": "c" * 64},
        connection_fingerprint="d" * 64,
        corpus_snapshot_id="corpus",
        corpus_snapshot_version="1",
        corpus_snapshot_hash="a" * 64,
        prompt_version="tutor-v2",
        prompt_hash="b" * 64,
        scorer_bundle_version="hybrid-v1",
        scorer_bundle_hash="c" * 64,
        provider="fake",
        model="fake-model",
        model_parameters={"temperature": 0},
        retrieval_config={"top_k": 5},
        reranker_config={"version": "reranker-v1"},
        chunking_config_version="chunk-v1",
        embedding_config_version="embed-v1",
        budget={"total_seconds": 90},
        runtime_judge=False,
        runner_version="runner-v1",
        schema_version="learning-run-v1",
        code_revision="revision",
    )


def test_public_frozen_contracts_deep_freeze_nested_mappings_and_hash_payloads():
    registry = TaskRegistry.load_default()
    manifest = _sample_manifest()
    before = manifest.compute_hash()
    with pytest.raises(TypeError):
        manifest.model_parameters["temperature"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        manifest.retrieval_config["top_k"] = 99  # type: ignore[index]
    assert manifest.compute_hash() == before

    task = TaskCase(
        task_case_id="case",
        task_case_version="1",
        question="question",
        case_type="answerable",
        expected_behavior="answer",
        manual_rationale="reason",
        required_evidence_set=["chunk"],
        required_dimensions=["groundedness"],
        critical_policy={"hard_fail_findings": ["unsupported_claim"], "expected_refusal": False},
    )
    assert isinstance(task.required_evidence_set, tuple)
    with pytest.raises(TypeError):
        task.critical_policy["expected_refusal"] = True  # type: ignore[index]

    component_type = getattr(contracts_module, "ScorerComponent", None)
    assert component_type is not None
    component = component_type(component_id="deterministic", version="v1", kind="deterministic")
    scorer = ScorerBundle(
        scorer_id="hybrid",
        version="v1",
        parser_version="parser-v1",
        rubric={"anchors": {"1": "one"}},
        required_dimensions_by_case_type={"answerable": ["groundedness"]},
        verdict_policy={"required_minimum": 4},
        model_config={"model": "fake"},
        components=[component],
    )
    assert isinstance(scorer.components, tuple)
    with pytest.raises(TypeError):
        scorer.model_config["model"] = "other"  # type: ignore[index]

    experiment = ExperimentDefinition(
        experiment_id="experiment",
        version="1",
        experiment_axes=["prompt_version"],
        runtime_judge=False,
        run_profile="evaluation",
        production_default="tutor-v2",
        prompt_hashes={"tutor-v2": "hash"},
        task_cases_hash="task",
        corpus_hash="corpus",
        scorer_hash="scorer",
        budget={"nested": {"seconds": 90}},
        variants={"tutor-v2": {"parameters": {"temperature": 0}}},
        schema_version="schema-v1",
    )
    with pytest.raises(TypeError):
        experiment.variants["tutor-v2"]["parameters"]["temperature"] = 1  # type: ignore[index]

    resolved = ResolvedRunDefinition(
        experiment=experiment,
        task=task,
        prompt=registry.production_prompt,
        corpus=registry.corpus,
        scorer=registry.scorer,
        variant_id="tutor-v2",
        experiment_axes=["prompt_version"],
        runtime_judge=False,
        budget={"nested": {"seconds": 90}},
        variant_controls={"parameters": {"temperature": 0}},
    )
    with pytest.raises(TypeError):
        resolved.variant_controls["parameters"]["temperature"] = 1  # type: ignore[index]

    candidate = registry.calibration_candidates[0]
    with pytest.raises(TypeError):
        candidate.artifact.budget["total_seconds"] = 1  # type: ignore[index]


def test_run_manifest_freezes_corpus_and_embedding_controls():
    manifest = _sample_manifest()
    assert manifest.corpus_snapshot_version == "1"
    assert manifest.corpus_snapshot_hash == "a" * 64
    assert manifest.chunking_config_version == "chunk-v1"
    assert manifest.embedding_config_version == "embed-v1"


def _canonical_definition_hash(payload: dict) -> str:
    return canonical_hash(
        {key: value for key, value in payload.items() if key != "definition_hash"}
    )


def _rehash_definition_tree(root: Path) -> None:
    task_path = root / "task_cases.json"
    corpus_path = root / "corpus.json"
    scorer_path = root / "scorers" / "hybrid-v1.json"
    calibration_path = root / "calibration" / "candidates.json"
    experiment_path = root / "experiment.json"

    task = json.loads(task_path.read_text(encoding="utf-8"))
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    scorer = json.loads(scorer_path.read_text(encoding="utf-8"))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))

    corpus_payload = {
        key: corpus[key]
        for key in (
            "schema_version",
            "snapshot_id",
            "version",
            "chunks",
            "chunking_config_version",
            "embedding_config_version",
            "retrieval_config_version",
            "reranker_config_version",
        )
        if key in corpus
    }
    for chunk in corpus.get("chunks", []):
        if isinstance(chunk, dict) and isinstance(chunk.get("content"), str):
            chunk["content_hash"] = hashlib.sha256(
                chunk["content"].encode("utf-8")
            ).hexdigest()
    corpus_payload["chunks"] = corpus.get("chunks", [])
    corpus["aggregate_hash"] = canonical_hash(corpus_payload)
    task["definition_hash"] = _canonical_definition_hash(task)
    corpus["definition_hash"] = _canonical_definition_hash(corpus)
    for candidate in calibration.get("candidates", []):
        if isinstance(candidate, dict) and isinstance(
            candidate.get("candidate_artifact"), dict
        ):
            candidate["artifact_hash"] = canonical_hash(
                candidate["candidate_artifact"]
            )
    calibration["definition_hash"] = _canonical_definition_hash(calibration)
    scorer["calibration_hash"] = calibration["definition_hash"]
    scorer["definition_hash"] = _canonical_definition_hash(scorer)
    experiment["task_cases_hash"] = task["definition_hash"]
    experiment["corpus_hash"] = corpus["definition_hash"]
    experiment["scorer_hash"] = scorer["definition_hash"]
    experiment["calibration_hash"] = calibration["definition_hash"]
    for controls in experiment["variants"].values():
        controls["corpus_hash"] = corpus["definition_hash"]
        controls["scorer_hash"] = scorer["definition_hash"]
    experiment["definition_hash"] = _canonical_definition_hash(experiment)

    for path, payload in (
        (task_path, task),
        (corpus_path, corpus),
        (scorer_path, scorer),
        (calibration_path, calibration),
        (experiment_path, experiment),
    ):
        path.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )


@pytest.mark.parametrize(
    ("relative_path", "mutation", "error_match"),
    [
        ("experiment.json", lambda payload: payload.update(runtime_judge=0), "type|schema"),
        ("task_cases.json", lambda payload: payload["cases"][0].update(question=123), "type|question"),
        ("experiment.json", lambda payload: payload.update(schema_version="unsupported"), "schema"),
        ("task_cases.json", lambda payload: payload.pop("schema_version"), "schema"),
        ("corpus.json", lambda payload: payload.update(schema_version="unsupported"), "schema"),
        ("scorers/hybrid-v1.json", lambda payload: payload.pop("schema_version"), "schema"),
        ("calibration/candidates.json", lambda payload: payload.update(schema_version="unsupported"), "schema"),
        ("experiment.json", lambda payload: payload.update(budget={"total_seconds": 90}), "budget"),
        ("experiment.json", lambda payload: payload["variants"]["tutor-v2"].update(budget=[]), "budget|type"),
        ("experiment.json", lambda payload: payload["variants"]["tutor-v2"].update(parameters=[]), "parameters|type"),
        ("experiment.json", lambda payload: payload["variants"]["tutor-v2"].update(provider=""), "provider|type"),
        ("experiment.json", lambda payload: payload["variants"]["tutor-v2"].update(model=123), "model|type"),
        ("experiment.json", lambda payload: payload["variants"]["tutor-v2"].update(chunking_config_version=""), "chunking|type"),
        ("experiment.json", lambda payload: payload["variants"]["tutor-v2"].update(embedding_config_version=0), "embedding|type"),
        ("experiment.json", lambda payload: payload["variants"]["tutor-v2"].update(retrieval_config_version=""), "retrieval|type"),
        ("experiment.json", lambda payload: payload["variants"]["tutor-v2"].update(reranker_config_version=False), "reranker|type"),
        ("experiment.json", lambda payload: payload["variants"]["tutor-v2"].update(schema_version="unsupported"), "schema"),
        ("scorers/hybrid-v1.json", lambda payload: payload.update(rubric=[]), "rubric|type"),
        ("scorers/hybrid-v1.json", lambda payload: payload.update(model_config=[]), "model|type"),
        ("scorers/hybrid-v1.json", lambda payload: payload.update(verdict_policy=[]), "verdict|type"),
        ("scorers/hybrid-v1.json", lambda payload: payload.update(required_dimensions_by_case_type=[]), "dimension|type"),
        ("scorers/hybrid-v1.json", lambda payload: payload.update(components=[]), "component"),
        ("scorers/hybrid-v1.json", lambda payload: payload["components"].__setitem__(0, {**payload["components"][0], "config": []}), "config|type"),
        ("corpus.json", lambda payload: payload.update(chunks={}), "chunk|type"),
        ("corpus.json", lambda payload: payload["chunks"][0].update(page=True), "page|type"),
        ("corpus.json", lambda payload: payload["chunks"][0].update(content=123), "content|type"),
        ("calibration/candidates.json", lambda payload: payload["candidates"][0].update(candidate_artifact=[]), "artifact|type"),
        ("calibration/candidates.json", lambda payload: payload["candidates"][0]["candidate_artifact"].update(citations={}), "citation|type"),
        ("calibration/candidates.json", lambda payload: payload["candidates"][0]["candidate_artifact"].update(exact_evidence={}), "evidence|type"),
        ("calibration/candidates.json", lambda payload: payload["candidates"][0]["candidate_artifact"].update(trace={}), "trace|type"),
        ("calibration/candidates.json", lambda payload: payload["candidates"][0]["candidate_artifact"].update(budget=[]), "budget|type"),
    ],
)
def test_registry_rejects_strict_schema_mutations_after_valid_rehash(
    tmp_path, relative_path, mutation, error_match
):
    source = Path(TaskRegistry.default_definitions_path())
    shutil.copytree(source, tmp_path, dirs_exist_ok=True)
    target = tmp_path / relative_path
    payload = json.loads(target.read_text(encoding="utf-8"))
    mutation(payload)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    _rehash_definition_tree(tmp_path)

    with pytest.raises(RegistryError):
        TaskRegistry.from_directory(tmp_path)


@pytest.mark.parametrize(
    ("relative_path", "mutation"),
    [
        (
            "experiment.json",
            lambda payload: payload["variants"]["tutor-v2"]["parameters"].update(
                temperature="0"
            ),
        ),
        (
            "experiment.json",
            lambda payload: payload["variants"]["tutor-v2"]["parameters"].update(
                top_p="1"
            ),
        ),
        (
            "experiment.json",
            lambda payload: payload["variants"]["tutor-v2"]["parameters"].update(
                top_p=True
            ),
        ),
        (
            "experiment.json",
            lambda payload: payload["variants"]["tutor-v2"]["parameters"].update(
                top_p=2
            ),
        ),
        (
            "scorers/hybrid-v1.json",
            lambda payload: payload["model_config"].update(provider=123),
        ),
        (
            "scorers/hybrid-v1.json",
            lambda payload: payload["model_config"].update(model=""),
        ),
        (
            "scorers/hybrid-v1.json",
            lambda payload: payload["model_config"].update(temperature="0"),
        ),
        (
            "scorers/hybrid-v1.json",
            lambda payload: payload["model_config"].update(max_tokens=True),
        ),
        (
            "scorers/hybrid-v1.json",
            lambda payload: payload["verdict_policy"].update(required_minimum=0),
        ),
        (
            "scorers/hybrid-v1.json",
            lambda payload: payload["verdict_policy"].update(
                critical_findings_fail="true"
            ),
        ),
        (
            "scorers/hybrid-v1.json",
            lambda payload: payload["verdict_policy"].pop("fallback_verdict"),
        ),
        (
            "scorers/hybrid-v1.json",
            lambda payload: payload["rubric"]["scale"].update(min=2),
        ),
        (
            "calibration/candidates.json",
            lambda payload: payload["candidates"][0]["candidate_artifact"].update(
                budget={"total_seconds": 90}
            ),
        ),
        (
            "calibration/candidates.json",
            lambda payload: payload["candidates"][0]["candidate_artifact"][
                "budget"
            ].update(total_seconds=False),
        ),
        (
            "calibration/candidates.json",
            lambda payload: payload["candidates"][0]["candidate_artifact"][
                "budget"
            ].update(total_seconds=91),
        ),
        (
            "calibration/candidates.json",
            lambda payload: payload["candidates"][0]["candidate_artifact"].update(
                usage={"prompt_tokens": "1"}
            ),
        ),
        (
            "calibration/candidates.json",
            lambda payload: payload["candidates"][0]["candidate_artifact"].update(
                usage={"prompt_tokens": -1}
            ),
        ),
    ],
)
def test_registry_rejects_nested_scalar_mutations_after_valid_rehash(
    tmp_path, relative_path, mutation
):
    source = Path(TaskRegistry.default_definitions_path())
    shutil.copytree(source, tmp_path, dirs_exist_ok=True)
    target = tmp_path / relative_path
    payload = json.loads(target.read_text(encoding="utf-8"))
    mutation(payload)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    _rehash_definition_tree(tmp_path)

    with pytest.raises(RegistryError):
        TaskRegistry.from_directory(tmp_path)
