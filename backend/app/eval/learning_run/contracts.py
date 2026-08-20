"""Frozen contracts and canonical hashing for Learning Run definitions.

Definitions are intentionally represented by small frozen dataclasses.  The
registry is the only place that turns version-controlled JSON into these
objects; callers receive immutable values and cannot replace prompt, corpus or
model settings with request data.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, is_dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence


def _freeze(value: Any) -> Any:
    """Recursively freeze JSON-compatible values for frozen contracts."""

    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze(item) for item in value))
    return value


def _plain(value: Any) -> Any:
    """Convert frozen contracts and nested immutable values to JSON values."""

    if is_dataclass(value):
        return {key: _plain(item) for key, item in vars(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def as_plain(value: Any) -> Any:
    """Public conversion used by registry and corpus integrity checks."""

    return _plain(value)


def canonical_json_bytes(value: Any) -> bytes:
    """Return compact, sorted-key UTF-8 JSON bytes for a payload.

    ``ensure_ascii=False`` is deliberate: hashes are over the exact UTF-8
    representation committed in definitions, not an implementation-specific
    escaped representation.
    """

    return json.dumps(
        _plain(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(value: Any) -> str:
    """Hash a canonical payload with SHA-256.

    Hash-bearing JSON objects must pass a payload with their own hash removed;
    ``hash_without_field`` is provided for that common read/write operation.
    """

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def canonical_payload_hash(value: Any, *, exclude_field: str | None = None) -> str:
    """Compatibility helper for callers hashing a payload sans self-reference."""

    if exclude_field is None:
        return canonical_hash(value)
    if not isinstance(value, Mapping):
        raise TypeError("exclude_field requires a mapping payload")
    payload = {key: item for key, item in value.items() if key != exclude_field}
    return canonical_hash(payload)


def hash_without_field(payload: Mapping[str, Any], field_name: str) -> str:
    """Compute a definition hash while excluding its hash field."""

    return canonical_payload_hash(payload, exclude_field=field_name)


def text_hash(text: str) -> str:
    """Hash prompt bytes exactly as stored on disk."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_dimension_score(value: Any) -> bool:
    """Whether a stored aggregate-score value is comparable across ScoreSets.

    Shared by `regression.py` and `compare.py` so the two readers of the same
    numbers cannot drift apart. Every rubric is an integer 1-5 scale today, but
    a scorer that ever averages judges into a float must not blind regression
    detection while compare keeps rendering deltas -- regression is the release
    contract, so divergence there is silence rather than failure.

    `bool` is excluded deliberately: `isinstance(True, int)` is True in Python,
    and a flag is not a rubric score. Scorer output validation stays stricter
    still (`scoring.py` accepts only `int` in range); this is the read path.
    """

    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _tuple_strings(values: Sequence[str] | None) -> tuple[str, ...]:
    return tuple(str(value) for value in (values or ()))


def _freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return _freeze(value or {})


def _freeze_mapping_sequence(
    values: Sequence[Mapping[str, Any]] | None,
) -> tuple[Mapping[str, Any], ...]:
    return tuple(_freeze_mapping(value) for value in (values or ()))


_CANONICAL_USAGE_FIELDS = ("input_tokens", "output_tokens", "total_tokens")
_USAGE_FIELD_ALIASES: Mapping[str, tuple[str, ...]] = {
    "input_tokens": ("input_tokens", "prompt_tokens"),
    "output_tokens": ("output_tokens", "completion_tokens"),
    "total_tokens": ("total_tokens",),
}


def normalize_usage(
    value: Any,
    *,
    allow_aliases: bool = False,
) -> dict[str, int] | None:
    """Return safe token usage fields, or ``None`` when none are usable.

    Provider metadata may use the documented prompt/completion aliases and
    carry arbitrary private keys; direct persisted drafts use the strict
    canonical-only mode (the default) so secrets and invalid values cannot
    enter an immutable contract.
    """

    if not isinstance(value, Mapping):
        return None
    if not allow_aliases:
        if not value or any(key not in _CANONICAL_USAGE_FIELDS for key in value):
            return None
        if any(
            type(token_count) is not int or token_count < 0
            for token_count in value.values()
        ):
            return None
        return {
            field: value[field]
            for field in _CANONICAL_USAGE_FIELDS
            if field in value
        }
    normalized: dict[str, int] = {}
    for canonical, aliases in _USAGE_FIELD_ALIASES.items():
        for key in aliases:
            token_count = value.get(key)
            if type(token_count) is int and token_count >= 0:
                normalized[canonical] = token_count
                break
    return normalized or None


@dataclass(frozen=True)
class TaskCase:
    task_case_id: str
    task_case_version: str
    question: str
    case_type: str
    expected_behavior: str
    manual_rationale: str
    required_evidence_set: tuple[str, ...]
    required_dimensions: tuple[str, ...]
    critical_policy: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "required_evidence_set", _tuple_strings(self.required_evidence_set))
        object.__setattr__(self, "required_dimensions", _tuple_strings(self.required_dimensions))
        object.__setattr__(self, "critical_policy", _freeze_mapping(self.critical_policy))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TaskCase":
        return cls(
            task_case_id=str(payload["id"]),
            task_case_version=str(payload["version"]),
            question=str(payload["question"]),
            case_type=str(payload["type"]),
            expected_behavior=str(payload["expected_behavior"]),
            manual_rationale=str(payload["manual_rationale"]),
            required_evidence_set=_tuple_strings(payload.get("required_evidence_set")),
            required_dimensions=_tuple_strings(payload.get("required_dimensions")),
            critical_policy=_freeze(payload.get("critical_policy", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.task_case_id,
            "version": self.task_case_version,
            "question": self.question,
            "type": self.case_type,
            "expected_behavior": self.expected_behavior,
            "manual_rationale": self.manual_rationale,
            "required_evidence_set": list(self.required_evidence_set),
            "required_dimensions": list(self.required_dimensions),
            "critical_policy": as_plain(self.critical_policy),
        }


@dataclass(frozen=True)
class CorpusChunk:
    chunk_id: str
    content: str
    source: str
    page: int
    content_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "chunk_id", str(self.chunk_id))
        object.__setattr__(self, "content", str(self.content))
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "page", int(self.page))
        object.__setattr__(self, "content_hash", str(self.content_hash))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CorpusChunk":
        return cls(
            chunk_id=str(payload["chunk_id"]),
            content=str(payload["content"]),
            source=str(payload["source"]),
            page=int(payload["page"]),
            content_hash=str(payload["content_hash"]),
        )

    def payload(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "source": self.source,
            "page": self.page,
            "content_hash": self.content_hash,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.payload()


@dataclass(frozen=True)
class CorpusSnapshot:
    schema_version: str
    snapshot_id: str
    snapshot_version: str
    chunks: tuple[CorpusChunk, ...]
    chunking_config_version: str
    embedding_config_version: str
    retrieval_config_version: str
    reranker_config_version: str
    aggregate_hash: str
    definition_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "chunks", tuple(self.chunks))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CorpusSnapshot":
        return cls(
            schema_version=str(payload.get("schema_version", "corpus-snapshot-v1")),
            snapshot_id=str(payload["snapshot_id"]),
            snapshot_version=str(payload["version"]),
            chunks=tuple(CorpusChunk.from_dict(chunk) for chunk in payload["chunks"]),
            chunking_config_version=str(payload["chunking_config_version"]),
            embedding_config_version=str(payload["embedding_config_version"]),
            retrieval_config_version=str(payload["retrieval_config_version"]),
            reranker_config_version=str(payload["reranker_config_version"]),
            aggregate_hash=str(payload["aggregate_hash"]),
            definition_hash=str(payload.get("definition_hash", "")),
        )

    def aggregate_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "version": self.snapshot_version,
            "chunks": [chunk.payload() for chunk in self.chunks],
            "chunking_config_version": self.chunking_config_version,
            "embedding_config_version": self.embedding_config_version,
            "retrieval_config_version": self.retrieval_config_version,
            "reranker_config_version": self.reranker_config_version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.aggregate_payload()
        payload["aggregate_hash"] = self.aggregate_hash
        if self.definition_hash:
            payload["definition_hash"] = self.definition_hash
        return payload

    def validate_hashes(self) -> None:
        for chunk in self.chunks:
            actual = text_hash(chunk.content)
            if actual != chunk.content_hash:
                raise ValueError(f"corpus chunk content hash mismatch: {chunk.chunk_id}")
        actual_aggregate = canonical_hash(self.aggregate_payload())
        if actual_aggregate != self.aggregate_hash:
            raise ValueError("corpus aggregate hash mismatch")
        if self.definition_hash:
            payload = self.to_dict()
            payload.pop("definition_hash", None)
            if canonical_hash(payload) != self.definition_hash:
                raise ValueError("corpus definition hash mismatch")


@dataclass(frozen=True)
class PromptDefinition:
    version: str
    text: str
    content_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "version", str(self.version))
        object.__setattr__(self, "text", str(self.text))
        object.__setattr__(self, "content_hash", str(self.content_hash))

    @classmethod
    def from_text(cls, version: str, text: str) -> "PromptDefinition":
        return cls(version=version, text=text, content_hash=text_hash(text))

    def payload(self) -> dict[str, str]:
        return {"version": self.version, "text": self.text}


@dataclass(frozen=True)
class ScorerComponent:
    component_id: str
    version: str
    kind: str
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "component_id", str(self.component_id))
        object.__setattr__(self, "version", str(self.version))
        object.__setattr__(self, "kind", str(self.kind))
        object.__setattr__(self, "config", _freeze_mapping(self.config))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ScorerComponent":
        return cls(
            component_id=str(payload["id"]),
            version=str(payload["version"]),
            kind=str(payload["kind"]),
            config=payload.get("config", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.component_id,
            "version": self.version,
            "kind": self.kind,
            "config": as_plain(self.config),
        }

@dataclass(frozen=True)
class ScorerBundle:
    scorer_id: str
    version: str
    parser_version: str
    rubric: Mapping[str, Any]
    required_dimensions_by_case_type: Mapping[str, tuple[str, ...]]
    verdict_policy: Mapping[str, Any]
    model_config: Mapping[str, Any]
    components: tuple[ScorerComponent, ...] = field(default_factory=tuple)
    calibration_hash: str = ""
    definition_hash: str = ""

    def __post_init__(self) -> None:
        components = tuple(
            component
            if isinstance(component, ScorerComponent)
            else ScorerComponent.from_dict(component)
            for component in (self.components or ())
        )
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "rubric", _freeze_mapping(self.rubric))
        object.__setattr__(
            self,
            "required_dimensions_by_case_type",
            _freeze(
                {
                    str(case_type): _tuple_strings(values)
                    for case_type, values in (
                        self.required_dimensions_by_case_type or {}
                    ).items()
                }
            ),
        )
        object.__setattr__(self, "verdict_policy", _freeze_mapping(self.verdict_policy))
        object.__setattr__(self, "model_config", _freeze_mapping(self.model_config))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ScorerBundle":
        dimensions = {
            str(case_type): _tuple_strings(values)
            for case_type, values in payload.get(
                "required_dimensions_by_case_type", {}
            ).items()
        }
        return cls(
            scorer_id=str(payload["scorer_id"]),
            version=str(payload["version"]),
            parser_version=str(payload["parser_version"]),
            rubric=_freeze(payload.get("rubric", {})),
            required_dimensions_by_case_type=_freeze(dimensions),
            verdict_policy=_freeze(payload.get("verdict_policy", {})),
            model_config=_freeze(payload.get("model_config", {})),
            components=tuple(
                ScorerComponent.from_dict(component)
                for component in payload.get("components", ())
            ),
            calibration_hash=str(payload.get("calibration_hash", "")),
            definition_hash=str(payload.get("definition_hash", "")),
        )

    def payload(self) -> dict[str, Any]:
        return {
            "scorer_id": self.scorer_id,
            "version": self.version,
            "parser_version": self.parser_version,
            "rubric": as_plain(self.rubric),
            "required_dimensions_by_case_type": as_plain(
                self.required_dimensions_by_case_type
            ),
            "verdict_policy": as_plain(self.verdict_policy),
            "model_config": as_plain(self.model_config),
            "components": [component.to_dict() for component in self.components],
            "calibration_hash": self.calibration_hash,
        }


@dataclass(frozen=True)
class ExperimentDefinition:
    experiment_id: str
    version: str
    experiment_axes: tuple[str, ...]
    runtime_judge: bool
    run_profile: str
    production_default: str
    prompt_hashes: Mapping[str, str]
    task_cases_hash: str
    corpus_hash: str
    scorer_hash: str
    budget: Mapping[str, int]
    variants: Mapping[str, Mapping[str, Any]]
    schema_version: str
    calibration_hash: str = ""
    definition_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "experiment_axes", _tuple_strings(self.experiment_axes))
        object.__setattr__(self, "prompt_hashes", _freeze_mapping(self.prompt_hashes))
        object.__setattr__(self, "budget", _freeze_mapping(self.budget))
        object.__setattr__(self, "variants", _freeze_mapping(self.variants))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExperimentDefinition":
        return cls(
            experiment_id=str(payload["experiment_id"]),
            version=str(payload["version"]),
            experiment_axes=_tuple_strings(payload.get("experiment_axes")),
            runtime_judge=bool(payload["runtime_judge"]),
            run_profile=str(payload["run_profile"]),
            production_default=str(payload["production_default"]),
            prompt_hashes=_freeze(payload.get("prompt_hashes", {})),
            task_cases_hash=str(payload["task_cases_hash"]),
            corpus_hash=str(payload["corpus_hash"]),
            scorer_hash=str(payload["scorer_hash"]),
            budget=_freeze(payload.get("budget", {})),
            variants=_freeze(payload.get("variants", {})),
            schema_version=str(payload.get("schema_version", "learning-run-v1")),
            calibration_hash=str(payload.get("calibration_hash", "")),
            definition_hash=str(payload.get("definition_hash", "")),
        )

    def payload(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "version": self.version,
            "experiment_axes": list(self.experiment_axes),
            "runtime_judge": self.runtime_judge,
            "run_profile": self.run_profile,
            "production_default": self.production_default,
            "prompt_hashes": as_plain(self.prompt_hashes),
            "task_cases_hash": self.task_cases_hash,
            "corpus_hash": self.corpus_hash,
            "scorer_hash": self.scorer_hash,
            "budget": as_plain(self.budget),
            "variants": as_plain(self.variants),
            "schema_version": self.schema_version,
            "calibration_hash": self.calibration_hash,
        }


@dataclass(frozen=True)
class ResolvedRunDefinition:
    experiment: ExperimentDefinition
    task: TaskCase
    prompt: PromptDefinition
    corpus: CorpusSnapshot
    scorer: ScorerBundle
    variant_id: str
    experiment_axes: tuple[str, ...]
    runtime_judge: bool
    budget: Mapping[str, int]
    variant_controls: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "experiment_axes", _tuple_strings(self.experiment_axes))
        object.__setattr__(self, "budget", _freeze_mapping(self.budget))
        object.__setattr__(self, "variant_controls", _freeze_mapping(self.variant_controls))


@dataclass(frozen=True)
class CalibrationTaskInput:
    task_case_id: str
    task_case_version: str
    question: str
    case_type: str
    expected_behavior: str
    required_dimensions: tuple[str, ...]
    critical_policy: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "required_dimensions", _tuple_strings(self.required_dimensions))
        object.__setattr__(self, "critical_policy", _freeze_mapping(self.critical_policy))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CalibrationTaskInput":
        return cls(
            task_case_id=str(payload["id"]),
            task_case_version=str(payload["version"]),
            question=str(payload["question"]),
            case_type=str(payload["case_type"]),
            expected_behavior=str(payload["expected_behavior"]),
            required_dimensions=_tuple_strings(payload.get("required_dimensions")),
            critical_policy=payload.get("critical_policy", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.task_case_id,
            "version": self.task_case_version,
            "question": self.question,
            "case_type": self.case_type,
            "expected_behavior": self.expected_behavior,
            "required_dimensions": list(self.required_dimensions),
            "critical_policy": as_plain(self.critical_policy),
        }


@dataclass(frozen=True)
class CandidateArtifact:
    answer: str
    citations: tuple[Mapping[str, Any], ...]
    exact_evidence: tuple[Mapping[str, Any], ...]
    formatted_context: str
    usage: Mapping[str, Any] | str
    trace: tuple[Mapping[str, Any], ...]
    budget: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "citations", _freeze_mapping_sequence(self.citations))
        object.__setattr__(self, "exact_evidence", _freeze_mapping_sequence(self.exact_evidence))
        object.__setattr__(self, "trace", _freeze_mapping_sequence(self.trace))
        if isinstance(self.usage, Mapping):
            object.__setattr__(self, "usage", _freeze_mapping(self.usage))
        object.__setattr__(self, "budget", _freeze_mapping(self.budget))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateArtifact":
        return cls(
            answer=str(payload["answer"]),
            citations=payload.get("citations", ()),
            exact_evidence=payload.get("exact_evidence", ()),
            formatted_context=str(payload.get("formatted_context", "")),
            usage=payload.get("usage", "unavailable"),
            trace=payload.get("trace", ()),
            budget=payload.get("budget", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": as_plain(self.citations),
            "exact_evidence": as_plain(self.exact_evidence),
            "formatted_context": self.formatted_context,
            "usage": as_plain(self.usage),
            "trace": as_plain(self.trace),
            "budget": as_plain(self.budget),
        }

    def compute_hash(self) -> str:
        return canonical_hash(self.to_dict())


_SCORER_EXECUTION_STATUSES = frozenset({"success", "failed", "skipped"})
_SCORE_SET_STATUSES = frozenset({"completed", "partial", "failed"})
_QUALITY_VERDICTS = frozenset({"pass", "fail", "inconclusive", "not_evaluated"})
_OPERATIONAL_ERROR_CODES = frozenset(
    {
        "manifest_invalid",
        "corpus_unavailable",
        "corpus_mismatch",
        "retriever_error",
        "model_unavailable",
        "generation_timeout",
        "scorer_timeout",
        "scorer_parse_error",
        "budget_exceeded",
        "process_interrupted",
        "harness_internal_error",
    }
)


def _validate_error_message(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("error_message must be a non-empty sanitized string")
    if len(value) > 1000 or "\n" in value or "\r" in value:
        raise ValueError("error_message must be a short single-line string")
    lowered = value.lower()
    if "traceback" in lowered or "authorization:" in lowered or "api_key" in lowered:
        raise ValueError("error_message must not contain sensitive diagnostics")
    return value


def _validate_error_fields(
    *,
    status: str,
    error_code: str | None,
    error_message: str | None,
    output: Any,
    usage: Mapping[str, Any] | None,
    findings: Sequence[Mapping[str, Any]],
) -> None:
    if status == "success":
        if error_code is not None or error_message is not None:
            raise ValueError("successful or skipped execution cannot contain an error")
        return
    if status == "skipped":
        if error_code is not None or error_message is not None:
            raise ValueError("successful or skipped execution cannot contain an error")
        if usage is not None or findings:
            raise ValueError("skipped execution cannot contain usage or findings")
        return
    if status != "failed":
        return
    if error_code is None or error_message is None:
        raise ValueError("failed execution requires an error code and message")
    if error_code not in _OPERATIONAL_ERROR_CODES:
        raise ValueError("failed execution error code is not stable")
    _validate_error_message(error_message)
    if output is not None or usage is not None:
        raise ValueError("failed execution cannot contain output or usage")
    if findings:
        raise ValueError("failed execution cannot contain quality findings")


@dataclass(frozen=True)
class ScorerExecutionDraft:
    """Immutable in-memory contract emitted by one scorer component.

    A draft is deliberately shaped like the persistence repository's append
    boundary.  ``output`` and ``findings`` are recursively frozen so a
    callback cannot mutate the data used to derive the final ScoreSet.
    ``usage`` is nullable: unavailable provider metadata must remain absent,
    never be represented by fabricated zero counts.
    """

    component_id: str
    component_version: str
    scorer_id: str
    scorer_version: str
    status: str
    input_hash: str
    output: Any = None
    error_code: str | None = None
    error_message: str | None = None
    latency_ms: int | None = None
    usage: Mapping[str, Any] | None = None
    findings: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for field_name in (
            "component_id",
            "component_version",
            "scorer_id",
            "scorer_version",
            "input_hash",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be a non-empty string")
        if self.status not in _SCORER_EXECUTION_STATUSES:
            raise ValueError(f"unsupported scorer execution status: {self.status}")
        if self.latency_ms is not None and (
            type(self.latency_ms) is not int or self.latency_ms < 0
        ):
            raise ValueError("latency_ms must be a non-negative integer")
        if self.usage is not None and not isinstance(self.usage, Mapping):
            raise TypeError("usage must be a mapping or None")
        normalized_usage = normalize_usage(self.usage)
        if self.usage is not None and normalized_usage is None:
            raise ValueError("usage must contain only canonical non-negative token counts")
        if self.error_code is not None and (
            not isinstance(self.error_code, str) or not self.error_code
        ):
            raise ValueError("error_code must be a non-empty string or None")
        if self.error_message is not None and not isinstance(self.error_message, str):
            raise TypeError("error_message must be a string or None")
        _validate_error_fields(
            status=self.status,
            error_code=self.error_code,
            error_message=self.error_message,
            output=self.output,
            usage=self.usage,
            findings=self.findings,
        )
        object.__setattr__(self, "output", _freeze(self.output))
        object.__setattr__(
            self,
            "usage",
            _freeze(normalized_usage) if normalized_usage is not None else None,
        )
        frozen_findings: list[Mapping[str, Any]] = []
        for finding in self.findings or ():
            if not isinstance(finding, Mapping):
                raise TypeError("findings must contain mappings")
            frozen_findings.append(_freeze(finding))
        object.__setattr__(self, "findings", tuple(frozen_findings))

    @property
    def error(self) -> Mapping[str, str] | None:
        """Expose a compact error object without introducing mutable state."""

        if self.error_code is None and self.error_message is None:
            return None
        payload: dict[str, str] = {}
        if self.error_code is not None:
            payload["code"] = self.error_code
        if self.error_message is not None:
            payload["message"] = self.error_message
        return MappingProxyType(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "component_version": self.component_version,
            "scorer_id": self.scorer_id,
            "scorer_version": self.scorer_version,
            "status": self.status,
            "input_hash": self.input_hash,
            "output": as_plain(self.output),
            "error_code": self.error_code,
            "error_message": self.error_message,
            "latency_ms": self.latency_ms,
            "usage": as_plain(self.usage),
            "findings": as_plain(self.findings),
        }


@dataclass(frozen=True)
class ScoreSetResultDraft:
    """Immutable aggregate derived solely from emitted scorer executions."""

    status: str
    verdict: str
    aggregate_scores: Mapping[str, int] = field(default_factory=dict)
    findings: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    executions: tuple[ScorerExecutionDraft, ...] = field(default_factory=tuple)
    error_code: str | None = None
    error_message: str | None = None
    input_hash: str = ""

    def __post_init__(self) -> None:
        if self.status not in _SCORE_SET_STATUSES:
            raise ValueError(f"unsupported score set status: {self.status}")
        if self.verdict not in _QUALITY_VERDICTS:
            raise ValueError(f"unsupported quality verdict: {self.verdict}")
        if self.error_code is not None and (
            not isinstance(self.error_code, str) or not self.error_code
        ):
            raise ValueError("error_code must be a non-empty string or None")
        if self.error_message is not None and not isinstance(self.error_message, str):
            raise TypeError("error_message must be a string or None")
        if (self.error_code is None) != (self.error_message is None):
            raise ValueError("score set error code and message must be paired")
        if self.error_code is not None:
            if self.error_code not in _OPERATIONAL_ERROR_CODES:
                raise ValueError("score set error code is not stable")
            _validate_error_message(self.error_message)
        if self.status == "completed" and self.error_code is not None:
            raise ValueError("completed score set cannot contain an operational error")
        if self.status in {"partial", "failed"} and self.verdict == "pass":
            raise ValueError("partial or failed score set cannot pass")
        scores: dict[str, int] = {}
        for dimension, score in (self.aggregate_scores or {}).items():
            if not isinstance(dimension, str) or not dimension:
                raise ValueError("aggregate score dimension must be a non-empty string")
            if type(score) is not int or not 1 <= score <= 5:
                raise ValueError("aggregate score must be an integer between 1 and 5")
            scores[dimension] = score
        object.__setattr__(self, "aggregate_scores", _freeze(scores))
        frozen_findings: list[Mapping[str, Any]] = []
        for finding in self.findings or ():
            if not isinstance(finding, Mapping):
                raise TypeError("findings must contain mappings")
            frozen_findings.append(_freeze(finding))
        object.__setattr__(self, "findings", tuple(frozen_findings))
        frozen_executions: list[ScorerExecutionDraft] = []
        for execution in self.executions or ():
            if not isinstance(execution, ScorerExecutionDraft):
                raise TypeError("executions must contain ScorerExecutionDraft values")
            frozen_executions.append(execution)
        object.__setattr__(self, "executions", tuple(frozen_executions))
        object.__setattr__(self, "input_hash", str(self.input_hash))

    @property
    def error(self) -> Mapping[str, str] | None:
        if self.error_code is None and self.error_message is None:
            return None
        payload: dict[str, str] = {}
        if self.error_code is not None:
            payload["code"] = self.error_code
        if self.error_message is not None:
            payload["message"] = self.error_message
        return MappingProxyType(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "verdict": self.verdict,
            "aggregate_scores": as_plain(self.aggregate_scores),
            "findings": as_plain(self.findings),
            "executions": [execution.to_dict() for execution in self.executions],
            "error_code": self.error_code,
            "error_message": self.error_message,
            "input_hash": self.input_hash,
        }


@dataclass(frozen=True)
class CalibrationCandidate:
    candidate_id: str
    task: CalibrationTaskInput
    artifact: CandidateArtifact
    artifact_hash: str
    anchor_label: str
    manual_expected_verdict: str
    manual_reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.task, CalibrationTaskInput):
            object.__setattr__(self, "task", CalibrationTaskInput.from_dict(self.task))
        if not isinstance(self.artifact, CandidateArtifact):
            object.__setattr__(self, "artifact", CandidateArtifact.from_dict(self.artifact))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CalibrationCandidate":
        return cls(
            candidate_id=str(payload["id"]),
            task=CalibrationTaskInput.from_dict(payload),
            artifact=CandidateArtifact.from_dict(payload["candidate_artifact"]),
            artifact_hash=str(payload["artifact_hash"]),
            anchor_label=str(payload["anchor_label"]),
            manual_expected_verdict=str(payload["manual_expected_verdict"]),
            manual_reason=str(payload["manual_reason"]),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = self.task.to_dict()
        payload.update(
            {
                "id": self.candidate_id,
                "candidate_artifact": self.artifact.to_dict(),
                "artifact_hash": self.artifact_hash,
                "anchor_label": self.anchor_label,
                "manual_expected_verdict": self.manual_expected_verdict,
                "manual_reason": self.manual_reason,
            }
        )
        return payload

    def validate_hash(self) -> None:
        if canonical_hash(self.artifact.to_dict()) != self.artifact_hash:
            raise ValueError(f"calibration artifact hash mismatch: {self.candidate_id}")


@dataclass(frozen=True)
class RunManifest:
    """Frozen execution contract for one historical Learning Run.

    The snapshot fields are intentionally required.  A historical detail view
    must be able to explain the task, prompt, corpus and scorer without
    consulting today's Registry; ``connection_fingerprint`` is the only
    request connection value that crosses the persistence boundary.
    """

    experiment_id: str
    task_case_id: str
    task_case_version: str
    variant_id: str
    run_profile: str
    task_snapshot: Mapping[str, Any]
    prompt_text: str
    corpus_snapshot: Mapping[str, Any]
    scorer_snapshot: Mapping[str, Any]
    connection_fingerprint: str
    corpus_snapshot_id: str
    corpus_snapshot_version: str
    corpus_snapshot_hash: str
    prompt_version: str
    prompt_hash: str
    scorer_bundle_version: str
    scorer_bundle_hash: str
    provider: str
    model: str
    model_parameters: Mapping[str, Any]
    retrieval_config: Mapping[str, Any]
    reranker_config: Mapping[str, Any]
    chunking_config_version: str
    embedding_config_version: str
    budget: Mapping[str, Any]
    runtime_judge: bool
    runner_version: str
    schema_version: str
    code_revision: str
    seed: int | None = None
    manifest_hash: str = field(default="", compare=True)

    def __post_init__(self) -> None:
        for field_name in (
            "task_snapshot",
            "corpus_snapshot",
            "scorer_snapshot",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, Mapping) or not value:
                raise TypeError(f"{field_name} must be a non-empty mapping")
            object.__setattr__(self, field_name, _freeze_mapping(value))
        if not isinstance(self.prompt_text, str) or not self.prompt_text:
            raise ValueError("prompt_text must be a non-empty string")
        if not isinstance(self.connection_fingerprint, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.connection_fingerprint
        ):
            raise ValueError("connection_fingerprint must be 64 lowercase hex characters")
        object.__setattr__(self, "model_parameters", _freeze_mapping(self.model_parameters))
        object.__setattr__(self, "retrieval_config", _freeze_mapping(self.retrieval_config))
        object.__setattr__(self, "reranker_config", _freeze_mapping(self.reranker_config))
        object.__setattr__(self, "budget", _freeze_mapping(self.budget))

    def payload(self) -> dict[str, Any]:
        payload = {
            key: _plain(value)
            for key, value in vars(self).items()
            if key != "manifest_hash"
        }
        return payload

    def compute_hash(self) -> str:
        return canonical_hash(self.payload())
