"""Version-controlled Task, Prompt, Corpus and Scorer registry."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .contracts import (
    CalibrationCandidate,
    CorpusSnapshot,
    ExperimentDefinition,
    PromptDefinition,
    ResolvedRunDefinition,
    ScorerBundle,
    TaskCase,
    hash_without_field,
)


class RegistryError(ValueError):
    """Raised when a frozen definition or client-selected ID is invalid."""


_CASE_TYPES = {"answerable", "multi_evidence", "expected_refusal"}
_CALIBRATION_LABELS = {
    "pass",
    "fail",
    "borderline",
    "correct_refusal",
    "incorrect_refusal",
}

_ALLOWED_SCHEMAS = {
    "experiment": "learning-run-v1",
    "task_cases": "task-suite-v1",
    "corpus": "corpus-snapshot-v1",
    "scorer": "scorer-bundle-v1",
    "calibration": "scorer-calibration-v1",
}
_BUDGET_KEYS = {
    "retrieval_preflight_seconds",
    "tutor_seconds",
    "hybrid_scoring_seconds",
    "total_seconds",
}


def _schema(payload: Mapping[str, Any], label: str) -> None:
    _require_mapping(payload, label)
    expected = _ALLOWED_SCHEMAS[label]
    if payload.get("schema_version") != expected:
        raise RegistryError(f"{label} schema_version must be {expected}")


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RegistryError(f"{label} must be a mapping")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise RegistryError(f"{label} must be a list")
    return value


def _require_nonempty_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RegistryError(f"{label} must be a non-empty string")
    return value


def _require_exact_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise RegistryError(f"{label} must be a boolean")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise RegistryError(f"{label} must be a positive integer")
    return value


def _require_nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise RegistryError(f"{label} must be a non-negative integer")
    return value


def _require_number(value: Any, label: str, *, minimum: float | None = None) -> int | float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise RegistryError(f"{label} must be a finite number")
    if minimum is not None and value < minimum:
        raise RegistryError(f"{label} must be >= {minimum}")
    return value


def _require_string_list(value: Any, label: str, *, allow_empty: bool = False) -> list[str]:
    values = _require_list(value, label)
    if not allow_empty and not values:
        raise RegistryError(f"{label} must not be empty")
    return [_require_nonempty_str(item, f"{label} item") for item in values]


def _validate_budget(value: Any, label: str) -> Mapping[str, Any]:
    budget = _require_mapping(value, label)
    if set(budget) != _BUDGET_KEYS:
        raise RegistryError(f"{label} keys are invalid")
    for key in _BUDGET_KEYS:
        _require_positive_int(budget[key], f"{label}.{key}")
    return budget


def _validate_variant_parameters(value: Any, label: str) -> Mapping[str, Any]:
    parameters = _require_mapping(value, label)
    if set(parameters) != {"temperature", "top_p"}:
        raise RegistryError(f"{label} keys are invalid")
    _require_number(parameters["temperature"], f"{label}.temperature", minimum=0)
    top_p = _require_number(parameters["top_p"], f"{label}.top_p")
    if not 0 < top_p <= 1:
        raise RegistryError(f"{label}.top_p must be > 0 and <= 1")
    return parameters


def _validate_model_config(value: Any, label: str) -> Mapping[str, Any]:
    model_config = _require_mapping(value, label)
    if set(model_config) != {"provider", "model", "temperature", "max_tokens"}:
        raise RegistryError(f"{label} keys are invalid")
    _require_nonempty_str(model_config["provider"], f"{label}.provider")
    _require_nonempty_str(model_config["model"], f"{label}.model")
    _require_number(model_config["temperature"], f"{label}.temperature", minimum=0)
    _require_positive_int(model_config["max_tokens"], f"{label}.max_tokens")
    return model_config


def _validate_verdict_policy(value: Any, label: str) -> Mapping[str, Any]:
    policy = _require_mapping(value, label)
    expected_keys = {
        "required_minimum",
        "critical_findings_fail",
        "scorer_failure_verdict",
        "fallback_verdict",
        "average_cannot_cover_dimension_below",
    }
    if set(policy) != expected_keys:
        raise RegistryError(f"{label} keys are invalid")
    for key in ("required_minimum", "average_cannot_cover_dimension_below"):
        value = _require_positive_int(policy[key], f"{label}.{key}")
        if value > 5:
            raise RegistryError(f"{label}.{key} must be between 1 and 5")
    _require_exact_bool(policy["critical_findings_fail"], f"{label}.critical_findings_fail")
    if policy["scorer_failure_verdict"] != "inconclusive":
        raise RegistryError(f"{label}.scorer_failure_verdict is invalid")
    if policy["fallback_verdict"] != "never_pass":
        raise RegistryError(f"{label}.fallback_verdict is invalid")
    return policy


def _validate_experiment_raw(payload: Mapping[str, Any]) -> None:
    _schema(payload, "experiment")
    for key in ("experiment_id", "version", "run_profile", "production_default"):
        _require_nonempty_str(payload.get(key), f"experiment.{key}")
    axes = _require_string_list(payload.get("experiment_axes"), "experiment.experiment_axes")
    if axes != ["prompt_version"]:
        raise RegistryError("experiment.experiment_axes must contain prompt_version")
    _require_exact_bool(payload.get("runtime_judge"), "experiment.runtime_judge")
    prompt_hashes = _require_mapping(payload.get("prompt_hashes"), "experiment.prompt_hashes")
    for key, value in prompt_hashes.items():
        _require_nonempty_str(key, "experiment.prompt hash key")
        _require_nonempty_str(value, f"experiment.prompt_hashes.{key}")
    for key in ("task_cases_hash", "corpus_hash", "scorer_hash", "calibration_hash", "definition_hash"):
        _require_nonempty_str(payload.get(key), f"experiment.{key}")
    budget = _validate_budget(payload.get("budget"), "experiment.budget")
    variants = _require_mapping(payload.get("variants"), "experiment.variants")
    if set(variants) != {"tutor-v2", "tutor-v3"}:
        raise RegistryError("experiment.variants must contain tutor-v2 and tutor-v3")
    for variant_id, controls_value in variants.items():
        controls = _require_mapping(controls_value, f"experiment.variants.{variant_id}")
        for key in ("prompt_version", "provider", "model", "chunking_config_version", "embedding_config_version", "retrieval_config_version", "reranker_config_version", "corpus_hash", "scorer_hash", "schema_version"):
            _require_nonempty_str(controls.get(key), f"experiment.variants.{variant_id}.{key}")
        if controls["prompt_version"] != variant_id:
            raise RegistryError(f"variant {variant_id} prompt_version mismatch")
        _validate_variant_parameters(
            controls.get("parameters"), f"experiment.variants.{variant_id}.parameters"
        )
        variant_budget = _validate_budget(controls.get("budget"), f"experiment.variants.{variant_id}.budget")
        if dict(variant_budget) != dict(budget):
            raise RegistryError(f"variant {variant_id} budget mismatch")
        if controls["schema_version"] != _ALLOWED_SCHEMAS["experiment"]:
            raise RegistryError(f"variant {variant_id} schema_version mismatch")


def _validate_task_cases_raw(payload: Mapping[str, Any]) -> None:
    _schema(payload, "task_cases")
    _require_nonempty_str(payload.get("suite_id"), "task_cases.suite_id")
    cases = _require_list(payload.get("cases"), "task_cases.cases")
    for index, value in enumerate(cases):
        case = _require_mapping(value, f"task_cases.cases[{index}]")
        for key in ("id", "version", "question", "expected_behavior", "manual_rationale"):
            _require_nonempty_str(case.get(key), f"task_cases.cases[{index}].{key}")
        case_type = _require_nonempty_str(case.get("type"), f"task_cases.cases[{index}].type")
        if case_type not in _CASE_TYPES:
            raise RegistryError(f"unsupported task case type: {case_type}")
        _require_string_list(case.get("required_evidence_set"), f"task_cases.cases[{index}].required_evidence_set", allow_empty=True)
        _require_string_list(case.get("required_dimensions"), f"task_cases.cases[{index}].required_dimensions")
        policy = _require_mapping(case.get("critical_policy"), f"task_cases.cases[{index}].critical_policy")
        findings = _require_string_list(policy.get("hard_fail_findings"), f"task_cases.cases[{index}].critical_policy.hard_fail_findings")
        if not findings:
            raise RegistryError(f"task_cases.cases[{index}] hard_fail_findings must not be empty")
        _require_exact_bool(policy.get("expected_refusal"), f"task_cases.cases[{index}].critical_policy.expected_refusal")


def _validate_corpus_raw(payload: Mapping[str, Any]) -> None:
    _schema(payload, "corpus")
    for key in ("snapshot_id", "version", "chunking_config_version", "embedding_config_version", "retrieval_config_version", "reranker_config_version", "aggregate_hash", "definition_hash"):
        _require_nonempty_str(payload.get(key), f"corpus.{key}")
    chunks = _require_list(payload.get("chunks"), "corpus.chunks")
    for index, value in enumerate(chunks):
        chunk = _require_mapping(value, f"corpus.chunks[{index}]")
        for key in ("chunk_id", "content", "source", "content_hash"):
            _require_nonempty_str(chunk.get(key), f"corpus.chunks[{index}].{key}")
        if type(chunk.get("page")) is not int:
            raise RegistryError(f"corpus.chunks[{index}].page must be an integer")


def _validate_scorer_raw(payload: Mapping[str, Any]) -> None:
    _schema(payload, "scorer")
    for key in ("scorer_id", "version", "parser_version", "calibration_hash", "definition_hash"):
        _require_nonempty_str(payload.get(key), f"scorer.{key}")
    rubric = _require_mapping(payload.get("rubric"), "scorer.rubric")
    scale = _require_mapping(rubric.get("scale"), "scorer.rubric.scale")
    if set(scale) != {"min", "max"}:
        raise RegistryError("scorer.rubric.scale keys are invalid")
    if type(scale["min"]) is not int or scale["min"] != 1:
        raise RegistryError("scorer.rubric.scale.min must be exactly 1")
    if type(scale["max"]) is not int or scale["max"] != 5:
        raise RegistryError("scorer.rubric.scale.max must be exactly 5")
    anchors = _require_mapping(rubric.get("anchors"), "scorer.rubric.anchors")
    if set(anchors) != {"1", "2", "3", "4", "5"}:
        raise RegistryError("scorer.rubric.anchors must contain 1 through 5")
    for key, value in anchors.items():
        _require_nonempty_str(value, f"scorer.rubric.anchors.{key}")
    dimensions = _require_mapping(payload.get("required_dimensions_by_case_type"), "scorer.required_dimensions_by_case_type")
    for key, value in dimensions.items():
        _require_nonempty_str(key, "scorer dimension case type")
        _require_string_list(value, f"scorer.required_dimensions_by_case_type.{key}")
    _validate_verdict_policy(payload.get("verdict_policy"), "scorer.verdict_policy")
    _validate_model_config(payload.get("model_config"), "scorer.model_config")
    components = _require_list(payload.get("components"), "scorer.components")
    for index, value in enumerate(components):
        component = _require_mapping(value, f"scorer.components[{index}]")
        for key in ("id", "version", "kind"):
            _require_nonempty_str(component.get(key), f"scorer.components[{index}].{key}")
        if component["kind"] not in {"deterministic", "llm"}:
            raise RegistryError(f"scorer.components[{index}].kind unsupported")
        _require_mapping(component.get("config"), f"scorer.components[{index}].config")


def _validate_calibration_raw(
    payload: Mapping[str, Any], *, expected_budget: Mapping[str, Any] | None = None
) -> None:
    _schema(payload, "calibration")
    _require_list(payload.get("candidates"), "calibration.candidates")
    _require_nonempty_str(payload.get("definition_hash"), "calibration.definition_hash")
    for index, value in enumerate(payload["candidates"]):
        candidate = _require_mapping(value, f"calibration.candidates[{index}]")
        for key in ("id", "version", "question", "case_type", "expected_behavior", "artifact_hash", "anchor_label", "manual_expected_verdict", "manual_reason"):
            _require_nonempty_str(candidate.get(key), f"calibration.candidates[{index}].{key}")
        _require_string_list(candidate.get("required_dimensions"), f"calibration.candidates[{index}].required_dimensions")
        policy = _require_mapping(candidate.get("critical_policy"), f"calibration.candidates[{index}].critical_policy")
        _require_string_list(policy.get("hard_fail_findings"), f"calibration.candidates[{index}].critical_policy.hard_fail_findings")
        _require_exact_bool(policy.get("expected_refusal"), f"calibration.candidates[{index}].critical_policy.expected_refusal")
        artifact = _require_mapping(candidate.get("candidate_artifact"), f"calibration.candidates[{index}].candidate_artifact")
        _require_nonempty_str(artifact.get("answer"), f"calibration.candidates[{index}].artifact.answer")
        _require_nonempty_str(artifact.get("formatted_context"), f"calibration.candidates[{index}].artifact.formatted_context")
        for key in ("citations", "exact_evidence", "trace"):
            items = _require_list(artifact.get(key), f"calibration.candidates[{index}].artifact.{key}")
            for item_index, item in enumerate(items):
                _require_mapping(item, f"calibration.candidates[{index}].artifact.{key}[{item_index}]")
        usage = artifact.get("usage")
        if usage != "unavailable":
            usage_mapping = _require_mapping(
                usage, f"calibration.candidates[{index}].artifact.usage"
            )
            for key, token_value in usage_mapping.items():
                _require_nonempty_str(key, f"calibration.candidates[{index}].artifact.usage key")
                _require_nonnegative_int(
                    token_value,
                    f"calibration.candidates[{index}].artifact.usage.{key}",
                )
        budget = _validate_budget(
            artifact.get("budget"), f"calibration.candidates[{index}].artifact.budget"
        )
        if expected_budget is not None and dict(budget) != dict(expected_budget):
            raise RegistryError(
                f"calibration.candidates[{index}] budget must match experiment budget"
            )


def self_validate_components(scorer: ScorerBundle) -> None:
    components = scorer.components
    if not components:
        raise RegistryError("scorer components must not be empty")
    ids = [component.component_id for component in components]
    if len(ids) != len(set(ids)):
        raise RegistryError("scorer component IDs must be unique")
    kinds = {component.kind for component in components}
    if not kinds <= {"deterministic", "llm"}:
        raise RegistryError("scorer component kind is unsupported")
    if "deterministic" not in kinds or "llm" not in kinds:
        raise RegistryError("scorer needs deterministic and llm components")
    semantic_dimensions = {
        dimension
        for dimensions in scorer.required_dimensions_by_case_type.values()
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
    llm_components = [component for component in components if component.kind == "llm"]
    for component in components:
        checks = set(component.config.get("checks", ()))
        if component.kind == "deterministic":
            if checks & semantic_dimensions:
                raise RegistryError(
                    f"deterministic scorer {component.component_id} owns semantic dimensions"
                )
            if not checks <= deterministic_checks:
                raise RegistryError(
                    f"deterministic scorer {component.component_id} has unsupported checks"
                )
    rubric_component = next(
        (component for component in llm_components if component.component_id == "grounded-quality-rubric"),
        None,
    )
    if rubric_component is None:
        raise RegistryError("grounded-quality-rubric llm component is required")
    if set(rubric_component.config.get("dimensions", ())) != semantic_dimensions:
        raise RegistryError("grounded-quality-rubric dimensions must cover scorer dimensions")
    if rubric_component.config.get("parser_version") != scorer.parser_version:
        raise RegistryError("grounded-quality-rubric parser binding mismatch")
    if rubric_component.config.get("model_config") != scorer.model_config:
        raise RegistryError("grounded-quality-rubric model binding mismatch")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RegistryError(f"cannot read definition {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise RegistryError(f"definition {path.name} must contain a JSON object")
    return value


def _verify_definition_hash(payload: Mapping[str, Any], *, label: str) -> str:
    expected = payload.get("definition_hash")
    if not isinstance(expected, str) or not expected:
        raise RegistryError(f"{label} is missing definition_hash")
    actual = hash_without_field(payload, "definition_hash")
    if actual != expected:
        raise RegistryError(f"{label} definition hash mismatch")
    return actual


class TaskRegistry:
    """Read and validate one immutable set of version-controlled definitions."""

    def __init__(
        self,
        *,
        definitions_path: Path,
        experiment: ExperimentDefinition,
        task_cases: Mapping[str, TaskCase],
        corpus: CorpusSnapshot,
        prompts: Mapping[str, PromptDefinition],
        scorer: ScorerBundle,
        calibration_candidates: tuple[CalibrationCandidate, ...],
        scorers: Mapping[str, ScorerBundle] | None = None,
        scorer_documents: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.definitions_path = definitions_path
        self.experiment = experiment
        self.task_cases = MappingProxyType(dict(task_cases))
        self.corpus = corpus
        self.prompts = MappingProxyType(dict(prompts))
        self.scorer = scorer
        self.scorers = MappingProxyType(dict(scorers or {scorer.version: scorer}))
        self.scorer_documents = MappingProxyType(dict(scorer_documents or {}))
        self.calibration_candidates = tuple(calibration_candidates)
        self.calibration_case_ids = frozenset(
            candidate.candidate_id for candidate in self.calibration_candidates
        )

    @staticmethod
    def default_definitions_path() -> Path:
        return Path(__file__).resolve().parent / "definitions"

    @property
    def production_prompt(self) -> PromptDefinition:
        try:
            return self.prompts[self.experiment.production_default]
        except KeyError as exc:
            raise RegistryError("production_default prompt is not registered") from exc

    @property
    def task_case_ids(self) -> frozenset[str]:
        return frozenset(self.task_cases)

    @property
    def case_type_counts(self) -> dict[str, int]:
        return {
            case_type: sum(case.case_type == case_type for case in self.task_cases.values())
            for case_type in ("answerable", "multi_evidence", "expected_refusal")
        }

    @property
    def experiment_axes(self) -> tuple[str, ...]:
        return self.experiment.experiment_axes

    def prompt(self, version: str) -> PromptDefinition:
        try:
            return self.prompts[version]
        except KeyError as exc:
            raise RegistryError(f"unknown prompt version: {version}") from exc

    @classmethod
    def load_default(cls) -> "TaskRegistry":
        return cls.from_directory(cls.default_definitions_path())

    @classmethod
    def from_directory(cls, definitions_path: str | Path) -> "TaskRegistry":
        root = Path(definitions_path)
        if not root.is_dir():
            raise RegistryError(f"definitions directory does not exist: {root}")

        experiment_payload = _read_json(root / "experiment.json")
        task_payload = _read_json(root / "task_cases.json")
        corpus_payload = _read_json(root / "corpus.json")
        scorer_payloads = {
            path.stem: _read_json(path)
            for path in sorted((root / "scorers").glob("*.json"))
        }
        if "hybrid-v1" not in scorer_payloads:
            raise RegistryError("hybrid-v1 scorer bundle is required")
        scorer_payload = scorer_payloads["hybrid-v1"]
        calibration_payload = _read_json(
            root / "calibration" / "candidates.json"
        )

        # Validate raw JSON before any frozen-contract constructor can coerce
        # values (for example ``bool(0)`` or ``str(123)``).  This keeps every
        # malformed definition on the public RegistryError boundary.
        _validate_experiment_raw(experiment_payload)
        _validate_task_cases_raw(task_payload)
        _validate_corpus_raw(corpus_payload)
        for name, payload in scorer_payloads.items():
            _validate_scorer_raw(payload)
            _verify_definition_hash(payload, label=f"scorer.{name}")
        _validate_scorer_raw(scorer_payload)
        _validate_calibration_raw(
            calibration_payload,
            expected_budget=experiment_payload["budget"],
        )

        # Check duplicate IDs before the file-level hash so malformed suites
        # cannot hide a duplicate behind a stale aggregate hash.
        raw_cases = task_payload.get("cases")
        if not isinstance(raw_cases, list):
            raise RegistryError("task_cases cases must be a list")
        case_ids = [str(case.get("id", "")) for case in raw_cases if isinstance(case, dict)]
        if len(case_ids) != len(set(case_ids)):
            raise RegistryError("duplicate task case ID")

        raw_candidates = calibration_payload.get("candidates")
        if not isinstance(raw_candidates, list):
            raise RegistryError("calibration candidates must be a list")
        calibration_ids = [
            str(candidate.get("id", ""))
            for candidate in raw_candidates
            if isinstance(candidate, dict)
        ]
        if len(calibration_ids) != len(set(calibration_ids)):
            raise RegistryError("duplicate calibration candidate ID")

        task_file_hash = _verify_definition_hash(task_payload, label="task_cases")
        corpus_file_hash = _verify_definition_hash(corpus_payload, label="corpus")
        scorer_file_hash = _verify_definition_hash(scorer_payload, label="scorer")
        calibration_file_hash = _verify_definition_hash(
            calibration_payload, label="calibration"
        )
        _verify_definition_hash(experiment_payload, label="experiment")

        try:
            task_cases = {
                case.task_case_id: case
                for case in (TaskCase.from_dict(raw_case) for raw_case in raw_cases)
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise RegistryError(f"invalid task case definition: {exc}") from exc
        if len(task_cases) != 12:
            raise RegistryError("task suite must contain exactly 12 cases")
        distribution = {
            case_type: sum(case.case_type == case_type for case in task_cases.values())
            for case_type in _CASE_TYPES
        }
        if distribution != {
            "answerable": 6,
            "multi_evidence": 3,
            "expected_refusal": 3,
        }:
            raise RegistryError(f"task suite distribution mismatch: {distribution}")
        invalid_types = set(case.case_type for case in task_cases.values()) - _CASE_TYPES
        if invalid_types:
            raise RegistryError(f"unsupported task case type: {sorted(invalid_types)}")
        if "tgqa-004" not in task_cases or task_cases["tgqa-004"].case_type != "expected_refusal":
            raise RegistryError("tgqa-004 must be an expected_refusal case")

        try:
            corpus = CorpusSnapshot.from_dict(corpus_payload)
            corpus.validate_hashes()
        except (KeyError, TypeError, ValueError) as exc:
            raise RegistryError(f"invalid corpus definition: {exc}") from exc

        try:
            scorer = ScorerBundle.from_dict(scorer_payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise RegistryError(f"invalid scorer definition: {exc}") from exc

        scorers = {scorer.version: scorer}
        scorer_documents = {scorer.version: scorer_payload}
        for name, payload in scorer_payloads.items():
            if name == "hybrid-v1":
                continue
            try:
                extra = ScorerBundle.from_dict(payload)
            except (KeyError, TypeError, ValueError) as exc:
                raise RegistryError(f"invalid scorer definition {name}: {exc}") from exc
            self_validate_components(extra)
            scorers[extra.version] = extra
            scorer_documents[extra.version] = payload
            labels_path = root / "calibration" / f"{extra.version}-labels.json"
            if labels_path.exists():
                labels_payload = _read_json(labels_path)
                if labels_payload.get("scorer_version") not in {extra.version, extra.scorer_id}:
                    if str(labels_payload.get("scorer_version") or "") != extra.version:
                        raise RegistryError(f"{extra.version} labels target mismatch")
                _verify_definition_hash(labels_payload, label=f"calibration.{extra.version}")
                if extra.calibration_hash != labels_payload.get("definition_hash"):
                    raise RegistryError(f"{extra.version} calibration hash mismatch")

        self_validate_components(scorer)

        try:
            calibration_candidates = tuple(
                CalibrationCandidate.from_dict(candidate)
                for candidate in raw_candidates
            )
            for candidate in calibration_candidates:
                candidate.validate_hash()
        except (KeyError, TypeError, ValueError) as exc:
            raise RegistryError(f"invalid calibration candidate: {exc}") from exc
        if not calibration_candidates:
            raise RegistryError("calibration candidates must not be empty")

        prompts: dict[str, PromptDefinition] = {}
        for version in ("tutor-v2", "tutor-v3"):
            prompt_path = root / "prompts" / f"{version}.txt"
            try:
                text = prompt_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                raise RegistryError(f"cannot read prompt {version}: {exc}") from exc
            prompts[version] = PromptDefinition.from_text(version, text)

        try:
            experiment = ExperimentDefinition.from_dict(experiment_payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise RegistryError(f"invalid experiment definition: {exc}") from exc

        if experiment.experiment_id != "tutor-prompt-regression-v1":
            raise RegistryError("unexpected experiment ID")
        if experiment.experiment_axes != ("prompt_version",):
            raise RegistryError("experiment_axes must contain only prompt_version")
        if experiment.runtime_judge is not False:
            raise RegistryError("evaluation registry must disable runtime_judge")
        if experiment.run_profile != "evaluation":
            raise RegistryError("registry experiment profile must be evaluation")
        if experiment.production_default != "tutor-v2":
            raise RegistryError("production default must remain tutor-v2")
        if experiment.task_cases_hash != task_file_hash:
            raise RegistryError("experiment task_cases reference hash mismatch")
        if experiment.corpus_hash != corpus_file_hash:
            raise RegistryError("experiment corpus reference hash mismatch")
        if experiment.scorer_hash != scorer_file_hash:
            raise RegistryError("experiment scorer reference hash mismatch")
        if experiment.calibration_hash != calibration_file_hash:
            raise RegistryError("experiment calibration reference hash mismatch")
        if scorer.calibration_hash != calibration_file_hash:
            raise RegistryError("scorer calibration reference hash mismatch")
        if not experiment.prompt_hashes:
            raise RegistryError("experiment has no prompt references")
        for version, prompt in prompts.items():
            expected_prompt_hash = experiment.prompt_hashes.get(version)
            if expected_prompt_hash != prompt.content_hash:
                raise RegistryError(f"experiment prompt reference hash mismatch: {version}")

        variants = dict(experiment.variants)
        if set(variants) != {"tutor-v2", "tutor-v3"}:
            raise RegistryError("experiment must define tutor-v2 and tutor-v3 variants")
        for variant_id, controls in variants.items():
            if controls.get("prompt_version") != variant_id:
                raise RegistryError(f"variant {variant_id} prompt mismatch")
            if controls.get("corpus_hash") != corpus_file_hash:
                raise RegistryError(f"variant {variant_id} corpus mismatch")
            if controls.get("scorer_hash") != scorer_file_hash:
                raise RegistryError(f"variant {variant_id} scorer mismatch")
            expected_versions = {
                "chunking_config_version": corpus.chunking_config_version,
                "embedding_config_version": corpus.embedding_config_version,
                "retrieval_config_version": corpus.retrieval_config_version,
                "reranker_config_version": corpus.reranker_config_version,
            }
            for config_key, expected_version in expected_versions.items():
                if controls.get(config_key) != expected_version:
                    raise RegistryError(
                        f"variant {variant_id} {config_key} mismatch"
                    )
        control_keys = set(variants["tutor-v2"]) | set(variants["tutor-v3"])
        for key in control_keys - {"prompt_version"}:
            if variants["tutor-v2"].get(key) != variants["tutor-v3"].get(key):
                raise RegistryError(f"variants differ outside prompt_version: {key}")

        required_labels = {candidate.anchor_label for candidate in calibration_candidates}
        if not _CALIBRATION_LABELS <= required_labels:
            raise RegistryError("calibration candidates missing required labels")
        calibration_ids = [candidate.candidate_id for candidate in calibration_candidates]
        if set(calibration_ids) & set(task_cases):
            raise RegistryError("calibration IDs overlap regression suite IDs")
        if {
            candidate.task.task_case_id for candidate in calibration_candidates
        } & set(task_cases):
            raise RegistryError("calibration task IDs overlap regression suite IDs")

        evidence_ids = {chunk.chunk_id for chunk in corpus.chunks}
        scorer_dimensions = scorer.required_dimensions_by_case_type
        for case in task_cases.values():
            if not case.question.strip():
                raise RegistryError(f"task case {case.task_case_id} question must be non-empty")
            if not case.expected_behavior.strip():
                raise RegistryError(
                    f"task case {case.task_case_id} behavior must be non-empty"
                )
            if not case.manual_rationale.strip():
                raise RegistryError(
                    f"task case {case.task_case_id} rationale must be non-empty"
                )
            unknown_evidence = set(case.required_evidence_set) - evidence_ids
            if unknown_evidence:
                raise RegistryError(
                    f"task case {case.task_case_id} unknown evidence chunk: {sorted(unknown_evidence)}"
                )
            expected_dimensions = tuple(scorer_dimensions.get(case.case_type, ()))
            if case.required_dimensions != expected_dimensions:
                raise RegistryError(
                    f"task case {case.task_case_id} required dimensions mismatch"
                )
            hard_findings = case.critical_policy.get("hard_fail_findings")
            if (
                not isinstance(hard_findings, (list, tuple))
                or not hard_findings
                or any(not isinstance(finding, str) or not finding.strip() for finding in hard_findings)
            ):
                raise RegistryError(
                    f"task case {case.task_case_id} critical policy must include hard findings"
                )
            expected_refusal = case.critical_policy.get("expected_refusal")
            if not isinstance(expected_refusal, bool) or expected_refusal != (
                case.case_type == "expected_refusal"
            ):
                raise RegistryError(
                    f"task case {case.task_case_id} critical expected_refusal mismatch"
                )
            if case.case_type == "answerable" and not case.required_evidence_set:
                raise RegistryError(
                    f"task case {case.task_case_id} answerable evidence must not be empty"
                )
            if case.case_type == "multi_evidence" and len(case.required_evidence_set) < 2:
                raise RegistryError(
                    f"task case {case.task_case_id} multi_evidence requires at least two chunks"
                )
            if case.case_type == "expected_refusal" and case.required_evidence_set:
                raise RegistryError(
                    f"task case {case.task_case_id} expected_refusal evidence must be empty"
                )

        return cls(
            definitions_path=root,
            experiment=experiment,
            task_cases=task_cases,
            corpus=corpus,
            prompts=prompts,
            scorer=scorer,
            scorers=scorers,
            scorer_documents=scorer_documents,
            calibration_candidates=calibration_candidates,
        )

    def resolve_run(
        self,
        *,
        experiment_id: str,
        task_case_id: str,
        variant_id: str,
        run_profile: str,
        **client_overrides: Any,
    ) -> ResolvedRunDefinition:
        if client_overrides:
            keys = ", ".join(sorted(client_overrides))
            raise RegistryError(f"client overrides are forbidden: {keys}")
        if experiment_id != self.experiment.experiment_id:
            raise RegistryError(f"unknown experiment ID: {experiment_id}")
        if run_profile != self.experiment.run_profile:
            raise RegistryError(f"run_profile must be {self.experiment.run_profile}")
        try:
            task = self.task_cases[task_case_id]
            controls = self.experiment.variants[variant_id]
            prompt_version = str(controls["prompt_version"])
            prompt = self.prompts[prompt_version]
        except KeyError as exc:
            raise RegistryError(f"unknown registry ID: {exc.args[0]}") from exc
        return ResolvedRunDefinition(
            experiment=self.experiment,
            task=task,
            prompt=prompt,
            corpus=self.corpus,
            scorer=self.scorer,
            variant_id=variant_id,
            experiment_axes=self.experiment.experiment_axes,
            runtime_judge=self.experiment.runtime_judge,
            budget=self.experiment.budget,
            variant_controls=controls,
        )

    def scorer_for(self, version: str) -> ScorerBundle:
        try:
            return self.scorers[version]
        except KeyError as exc:
            raise RegistryError(f"unknown scorer version: {version}") from exc

    def scorer_document(self, version: str) -> Mapping[str, Any]:
        try:
            return self.scorer_documents[version]
        except KeyError as exc:
            raise RegistryError(f"unknown scorer version: {version}") from exc

    def validate(self) -> dict[str, Any]:
        distribution = {
            case_type: sum(case.case_type == case_type for case in self.task_cases.values())
            for case_type in ("answerable", "multi_evidence", "expected_refusal")
        }
        return {
            "cases": len(self.task_cases),
            "distribution": distribution,
            "prompt_hashes": {
                version: prompt.content_hash for version, prompt in self.prompts.items()
            },
            "corpus_aggregate_hash": self.corpus.aggregate_hash,
            "calibration_count": len(self.calibration_case_ids),
        }


def _main() -> int:
    parser = argparse.ArgumentParser(description="Validate Learning Run definitions")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if not args.validate:
        parser.error("--validate is required")
    summary = TaskRegistry.load_default().validate()
    distribution = summary["distribution"]
    print(
        "cases={cases} answerable={answerable} multi_evidence={multi_evidence} "
        "expected_refusal={expected_refusal}".format(
            cases=summary["cases"], **distribution
        )
    )
    print(
        "prompt_hashes="
        + " ".join(
            f"{version}:{prompt_hash}"
            for version, prompt_hash in sorted(summary["prompt_hashes"].items())
        )
    )
    print(f"corpus_aggregate_hash={summary['corpus_aggregate_hash']}")
    print(f"calibration_count={summary['calibration_count']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by CLI validation
    raise SystemExit(_main())
