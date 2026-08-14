"""Hybrid, fail-closed scoring for one frozen Learning Run artifact.

The service is deliberately independent from the production runtime Judge.
It receives a small injected async chat model and emits one immutable draft
for every frozen scorer component, in definition order.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

from langchain_core.messages import HumanMessage

from .contracts import (
    CalibrationTaskInput,
    CandidateArtifact,
    ScorerBundle,
    ScorerComponent,
    ScorerExecutionDraft,
    ScoreSetResultDraft,
    TaskCase,
    as_plain,
    canonical_hash,
    normalize_usage,
)


class AsyncChatModel(Protocol):
    """Minimal model boundary required by the evaluation scorer."""

    async def ainvoke(self, messages: Sequence[HumanMessage]) -> Any:
        ...


_TASK_TYPES = frozenset({"answerable", "multi_evidence", "expected_refusal"})
_DEFAULT_DIMENSIONS: Mapping[str, tuple[str, ...]] = {
    "answerable": ("groundedness", "citation_entailment", "coverage"),
    "multi_evidence": ("groundedness", "citation_entailment", "coverage"),
    "expected_refusal": ("refusal_appropriateness", "unsupported_claims"),
}
_FINDING_CODES = frozenset(
    {
        "retrieval_empty",
        "citation_missing",
        "citation_invalid",
        "unsupported_claim",
        "incomplete_answer",
        "inappropriate_refusal",
        "expected_refusal_observed",
    }
)
_FINDING_SEVERITIES = frozenset({"critical", "noncritical"})
_NUMBER_RE = re.compile(r"\[(\d+)\]")
_MAX_RUBRIC_BYTES = 64 * 1024
_MAX_REASONING_CHARS = 8 * 1024
_MAX_FINDING_MESSAGE_CHARS = 2 * 1024
_MAX_FINDINGS = 20
_REFUSAL_PATTERNS = (
    re.compile(r"\bi\s+don['’]?t\s+know\b", re.IGNORECASE),
    re.compile(
        r"\bi\s+(?:cannot|can't|can\s+not|am\s+unable\s+to)\s+"
        r"(?:answer|determine|verify|provide|confirm|say|tell)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:provided|available)\s+(?:sources?|evidence|context)\s+"
        r"(?:do|does)\s+not\s+(?:contain|include|support)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bno\s+(?:relevant\s+)?(?:sources?|evidence|context)\b", re.IGNORECASE),
)


class RubricParseError(ValueError):
    """Raised when a model response is not the exact frozen rubric schema."""


@dataclass(frozen=True)
class VerdictDecision:
    verdict: str
    findings: tuple[Mapping[str, Any], ...] = ()
    missing_dimensions: tuple[str, ...] = ()
    failed_scorers: tuple[str, ...] = ()


def _required_dimensions(
    *,
    case_type: str,
    required_dimensions: Sequence[str] | None = None,
) -> tuple[str, ...]:
    if case_type not in _TASK_TYPES:
        return ()
    values = tuple(required_dimensions or _DEFAULT_DIMENSIONS[case_type])
    if not values or any(not isinstance(value, str) or not value for value in values):
        return ()
    return values


def _plain_findings(findings: Sequence[Any] | None) -> tuple[Mapping[str, Any], ...]:
    normalized: list[Mapping[str, Any]] = []
    for finding in findings or ():
        if isinstance(finding, Mapping):
            normalized.append({str(key): as_plain(value) for key, value in finding.items()})
        elif isinstance(finding, str) and finding:
            normalized.append({"code": finding, "severity": "critical"})
    return tuple(normalized)


def _finding(
    code: str,
    *,
    severity: str,
    message: str,
    **extra: Any,
) -> Mapping[str, Any]:
    payload: dict[str, Any] = {
        "code": code,
        "severity": severity,
        "message": message,
    }
    payload.update(extra)
    return payload


def _hard_finding_codes(
    *,
    task: TaskCase | CalibrationTaskInput | None,
    findings: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    policy = getattr(task, "critical_policy", {}) if task is not None else {}
    configured = policy.get("hard_fail_findings", ()) if isinstance(policy, Mapping) else ()
    emitted_codes = {
        str(item.get("code"))
        for item in findings
        if isinstance(item, Mapping) and isinstance(item.get("code"), str)
    }
    hard = {str(code) for code in configured if str(code) in emitted_codes}
    # Missing citations are deterministic quality gates for answerable cases
    # even when an older custom policy omitted citation_missing.  Only add the
    # gate when that finding was actually emitted.
    if (
        getattr(task, "case_type", None) in {"answerable", "multi_evidence"}
        and any(item.get("code") == "citation_missing" for item in findings)
    ):
        hard.add("citation_missing")
    for item in findings:
        code = item.get("code")
        if not isinstance(code, str):
            continue
        if item.get("severity") in {"critical", "hard"}:
            hard.add(code)
    return tuple(sorted(hard))


def derive_verdict(
    case_type: str,
    dimension_scores: Mapping[str, Any],
    hard_findings: Sequence[Any] = (),
    failed_scorers: Sequence[Any] = (),
    findings: Sequence[Mapping[str, Any]] = (),
    required_dimensions: Sequence[str] | None = None,
    required_minimum: int = 4,
) -> VerdictDecision:
    """Apply the v1 truth table without averaging away a weak dimension."""

    visible_findings = _plain_findings(findings)
    dimensions = _required_dimensions(
        case_type=case_type,
        required_dimensions=required_dimensions,
    )
    failed = tuple(
        str(item.get("component_id") or item.get("scorer_id") or item)
        if isinstance(item, Mapping)
        else str(item)
        for item in (failed_scorers or ())
    )
    if case_type not in _TASK_TYPES or not dimensions:
        return VerdictDecision(
            verdict="inconclusive",
            findings=visible_findings,
            failed_scorers=failed,
        )
    if type(required_minimum) is not int or not 1 <= required_minimum <= 5:
        return VerdictDecision(
            verdict="inconclusive",
            findings=visible_findings,
            failed_scorers=failed,
        )
    if not isinstance(dimension_scores, Mapping):
        return VerdictDecision(
            verdict="inconclusive",
            findings=visible_findings,
            failed_scorers=failed,
        )

    invalid = False
    missing: list[str] = []
    weak = False
    for dimension in dimensions:
        if dimension not in dimension_scores:
            missing.append(dimension)
            continue
        score = dimension_scores[dimension]
        if type(score) is not int or not 1 <= score <= 5:
            invalid = True
            continue
        if score < required_minimum:
            weak = True
    if any(key not in dimensions for key in dimension_scores):
        invalid = True

    explicit_hard = {
        str(item.get("code"))
        if isinstance(item, Mapping)
        else str(item)
        for item in (hard_findings or ())
    }
    for item in visible_findings:
        if item.get("severity") in {"critical", "hard"}:
            explicit_hard.add(str(item.get("code")))

    if explicit_hard:
        verdict = "fail"
    elif invalid:
        verdict = "inconclusive"
    elif weak:
        verdict = "fail"
    elif missing:
        verdict = "inconclusive"
    elif failed:
        verdict = "inconclusive"
    else:
        verdict = "pass"
    return VerdictDecision(
        verdict=verdict,
        findings=visible_findings,
        missing_dimensions=tuple(missing),
        failed_scorers=failed,
    )


def parse_rubric_output(
    raw: str,
    required_dimensions: Sequence[str],
) -> Mapping[str, Any]:
    """Parse one exact JSON object emitted by the frozen rubric.

    No markdown extraction, partial object recovery, coercion, or score
    defaulting is allowed.  That is what makes a malformed evaluator output a
    visible failed execution rather than an accidental pass.
    """

    if not isinstance(raw, str):
        raise RubricParseError("raw output must be a string")
    try:
        if len(raw.encode("utf-8")) > _MAX_RUBRIC_BYTES:
            raise RubricParseError("output is too large")
    except UnicodeEncodeError as exc:
        raise RubricParseError("output encoding is invalid") from exc
    duplicate_key = False

    def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal duplicate_key
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                duplicate_key = True
            result[key] = value
        return result

    try:
        parsed = json.loads(raw, object_pairs_hook=_object_pairs)
    except (TypeError, ValueError, json.JSONDecodeError, RecursionError) as exc:
        raise RubricParseError("output is not valid JSON") from exc
    if duplicate_key or not isinstance(parsed, dict):
        raise RubricParseError("output must be one JSON object")

    dimensions = tuple(required_dimensions)
    if len(dimensions) != len(set(dimensions)):
        raise RubricParseError("required dimensions are duplicated")
    expected_keys = set(dimensions) | {"reasoning", "findings"}
    if set(parsed) != expected_keys:
        raise RubricParseError("output schema keys are invalid")
    for dimension in dimensions:
        value = parsed.get(dimension)
        if type(value) is not int or not 1 <= value <= 5:
            raise RubricParseError("dimension score is invalid")
    reasoning = parsed.get("reasoning")
    if (
        not isinstance(reasoning, str)
        or not reasoning.strip()
        or len(reasoning) > _MAX_REASONING_CHARS
    ):
        raise RubricParseError("reasoning is invalid")
    try:
        reasoning.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise RubricParseError("reasoning encoding is invalid") from exc
    findings = parsed.get("findings")
    if not isinstance(findings, list) or len(findings) > _MAX_FINDINGS:
        raise RubricParseError("findings must be a list")
    normalized_findings: list[dict[str, Any]] = []
    for item in findings:
        if not isinstance(item, dict) or set(item) != {"code", "severity", "message"}:
            raise RubricParseError("finding schema is invalid")
        code = item.get("code")
        severity = item.get("severity")
        message = item.get("message")
        if (
            not isinstance(code, str)
            or not code
            or code not in _FINDING_CODES
            or severity not in _FINDING_SEVERITIES
            or not isinstance(message, str)
            or not message.strip()
            or len(message) > _MAX_FINDING_MESSAGE_CHARS
        ):
            raise RubricParseError("finding schema is invalid")
        try:
            message.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise RubricParseError("finding message encoding is invalid") from exc
        normalized_findings.append(
            {"code": code, "severity": severity, "message": message.strip()}
        )
    result = {dimension: parsed[dimension] for dimension in dimensions}
    result["reasoning"] = reasoning.strip()
    result["findings"] = normalized_findings
    return result


def _task_dimensions(task: TaskCase | CalibrationTaskInput, bundle: ScorerBundle) -> tuple[str, ...]:
    dimensions = tuple(getattr(task, "required_dimensions", ()) or ())
    configured = tuple(bundle.required_dimensions_by_case_type.get(task.case_type, ()))
    # A frozen TaskCase is authoritative.  A mismatch means the input is not a
    # compatible frozen contract and will be handled as a failed scorer run.
    if not dimensions:
        dimensions = configured
    if configured and dimensions != configured:
        return ()
    return dimensions


def _valid_scorer_bundle_for_scoring(bundle: ScorerBundle) -> bool:
    """Check the frozen hybrid bundle's unique semantic LLM producer."""

    if not isinstance(bundle, ScorerBundle):
        return False
    components = tuple(bundle.components)
    component_ids = tuple(component.component_id for component in components)
    if len(component_ids) != len(set(component_ids)):
        return False
    llm_components = tuple(component for component in components if component.kind == "llm")
    if len(llm_components) != 1:
        return False
    llm_component = llm_components[0]
    if llm_component.component_id != "grounded-quality-rubric":
        return False
    try:
        configured_dimensions = tuple(llm_component.config.get("dimensions", ()))
        if any(
            not isinstance(dimension, str) or not dimension
            for dimension in configured_dimensions
        ):
            return False
        if len(configured_dimensions) != len(set(configured_dimensions)):
            return False
        required_dimensions = {
            str(dimension)
            for dimensions in bundle.required_dimensions_by_case_type.values()
            for dimension in dimensions
        }
        if set(configured_dimensions) != required_dimensions:
            return False
        if llm_component.config.get("parser_version") != bundle.parser_version:
            return False
        if as_plain(llm_component.config.get("model_config", {})) != as_plain(bundle.model_config):
            return False
    except Exception:
        return False
    return True


def _safe_candidate_payload(candidate: CandidateArtifact) -> Mapping[str, Any]:
    # Runtime usage, trace and budget are intentionally excluded from the LLM
    # prompt; only answer/citations/context are candidate quality inputs.
    return {
        "answer": candidate.answer,
        "citations": as_plain(candidate.citations),
        "formatted_context": candidate.formatted_context,
    }


def _build_prompt(
    *,
    task: TaskCase | CalibrationTaskInput,
    candidate: CandidateArtifact,
    scorer_bundle: ScorerBundle,
    dimensions: Sequence[str],
) -> str:
    payload = {
        "question": task.question,
        "expected_behavior": task.expected_behavior,
        "candidate": _safe_candidate_payload(candidate),
        "candidate_exact_evidence": as_plain(candidate.exact_evidence),
        "required_dimensions": list(dimensions),
        "rubric_anchors": as_plain(scorer_bundle.rubric.get("anchors", {})),
    }
    return (
        "Evaluate this frozen Learning Run candidate. Return exactly one JSON "
        "object with exactly these top-level keys: "
        f"{list(dimensions) + ['reasoning', 'findings']}. Each dimension is an "
        "integer from 1 to 5; reasoning is a non-empty string. Each findings "
        "entry must use exactly the schema "
        '{"code": string, "severity": "critical"|"noncritical", "message": string}. '
        f"Allowed finding codes: {sorted(_FINDING_CODES)}. Allowed severities: "
        f"{sorted(_FINDING_SEVERITIES)}. Do not infer facts outside the "
        "provided evidence. Keep reasoning concise (at most 8192 characters), "
        "return at most 20 findings, and keep each finding message at most "
        "2048 characters.\n\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def _extract_usage(response: Any) -> Mapping[str, Any] | None:
    sources: list[Mapping[str, Any]] = []
    usage_metadata = getattr(response, "usage_metadata", None)
    if isinstance(usage_metadata, Mapping):
        sources.append(usage_metadata)
    response_metadata = getattr(response, "response_metadata", None)
    if isinstance(response_metadata, Mapping):
        for key in ("token_usage", "usage"):
            usage = response_metadata.get(key)
            if isinstance(usage, Mapping):
                sources.append(usage)
    for source in sources:
        normalized = normalize_usage(source, allow_aliases=True)
        if normalized:
            return normalized
    return None


def _evidence_content(evidence: Mapping[str, Any]) -> str:
    value = evidence.get("content", evidence.get("text", ""))
    return value if isinstance(value, str) else ""


def _citation_span(citation: Mapping[str, Any]) -> tuple[Any, Any] | None:
    if "span_start" in citation or "span_end" in citation:
        return citation.get("span_start"), citation.get("span_end")
    span = citation.get("span")
    if isinstance(span, Mapping):
        return span.get("start"), span.get("end")
    if isinstance(span, (list, tuple)) and len(span) == 2:
        return span[0], span[1]
    return None


def _critical_for_task(task: TaskCase | CalibrationTaskInput, code: str) -> str:
    policy = getattr(task, "critical_policy", {})
    configured = policy.get("hard_fail_findings", ()) if isinstance(policy, Mapping) else ()
    if code in configured or (code == "citation_missing" and task.case_type in {"answerable", "multi_evidence"}):
        return "critical"
    return "noncritical"


def _retrieval_integrity(
    *,
    task: TaskCase | CalibrationTaskInput,
    candidate: CandidateArtifact,
) -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]]:
    exact = tuple(candidate.exact_evidence)
    findings: list[Mapping[str, Any]] = []
    if not exact:
        findings.append(
            _finding(
                "retrieval_empty",
                severity="noncritical",
                message="no exact evidence was retrieved",
            )
        )
    evidence_ids = {
        str(item.get("chunk_id"))
        for item in exact
        if isinstance(item, Mapping) and item.get("chunk_id") is not None
    }
    for citation in candidate.citations:
        chunk_id = citation.get("chunk_id") if isinstance(citation, Mapping) else None
        if chunk_id is None or str(chunk_id) not in evidence_ids:
            findings.append(
                _finding(
                    "citation_invalid",
                    severity=_critical_for_task(task, "citation_invalid"),
                    message="citation chunk is not present in exact evidence",
                    check="evidence_membership",
                )
            )
    return (
        {
            "evidence_count": len(exact),
            "citation_count": len(candidate.citations),
            "checks": ["evidence_membership", "retrieval_empty"],
        },
        tuple(findings),
    )


def _citation_integrity(
    *,
    task: TaskCase | CalibrationTaskInput,
    candidate: CandidateArtifact,
) -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]]:
    citations = tuple(candidate.citations)
    exact = tuple(candidate.exact_evidence)
    findings: list[Mapping[str, Any]] = []
    if not citations:
        findings.append(
            _finding(
                "citation_missing",
                severity=_critical_for_task(task, "citation_missing"),
                message="answerable candidate has no citations",
            )
        )
        return ({"valid": False, "citation_count": 0}, tuple(findings))

    answer = candidate.answer if isinstance(candidate.answer, str) else ""
    numbers = [int(value) for value in _NUMBER_RE.findall(answer)]
    first_seen: list[int] = []
    for number in numbers:
        if number not in first_seen:
            first_seen.append(number)
    expected_numbers = list(range(1, len(citations) + 1))
    if first_seen != expected_numbers or any(number < 1 or number > len(citations) for number in numbers):
        findings.append(
            _finding(
                "citation_invalid",
                severity=_critical_for_task(task, "citation_invalid"),
                message="answer citation numbering does not match citation tuple order",
                check="citation_number",
            )
        )

    evidence_by_id = {
        str(item.get("chunk_id")): item
        for item in exact
        if isinstance(item, Mapping) and item.get("chunk_id") is not None
    }
    for citation in citations:
        valid = isinstance(citation, Mapping)
        evidence = evidence_by_id.get(str(citation.get("chunk_id"))) if valid else None
        if evidence is None:
            valid = False
        if valid:
            if (
                citation.get("chunk_id") != evidence.get("chunk_id")
                or citation.get("source") != evidence.get("source")
                or type(citation.get("page")) is not int
                or type(evidence.get("page")) is not int
                or citation.get("page") != evidence.get("page")
            ):
                valid = False
            span = _citation_span(citation)
            content = _evidence_content(evidence)
            if span is None:
                valid = False
            else:
                start, end = span
                if (
                    type(start) is not int
                    or type(end) is not int
                    or start < 0
                    or start >= end
                    or end > len(content)
                ):
                    valid = False
                evidence_span = _citation_span(evidence)
                if evidence_span is not None and span != evidence_span:
                    valid = False
        if not valid:
            findings.append(
                _finding(
                    "citation_invalid",
                    severity=_critical_for_task(task, "citation_invalid"),
                    message="citation chunk, source, page, or span does not match exact evidence",
                    check="chunk_id_source_page_span",
                )
            )
    return (
        {"valid": not findings, "citation_count": len(citations)},
        tuple(findings),
    )


def _refusal_observation(
    *,
    task: TaskCase | CalibrationTaskInput,
    candidate: CandidateArtifact,
) -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...], bool]:
    if task.case_type != "expected_refusal":
        return ({"applicable": False}, (), False)
    # This is intentionally observable only.  A conservative explicit
    # refusal expression is required; empty retrieval alone is not enough.
    answer = candidate.answer if isinstance(candidate.answer, str) else ""
    observed = not candidate.exact_evidence and not candidate.citations and any(
        pattern.search(answer) for pattern in _REFUSAL_PATTERNS
    )
    if observed:
        return (
            {"applicable": True, "observed": True},
            (
                _finding(
                    "expected_refusal_observed",
                    severity="noncritical",
                    message="candidate declined with no retrieved evidence",
                ),
            ),
            True,
        )
    return ({"applicable": True, "observed": False}, (), True)


def _component_error(
    *,
    component: ScorerComponent,
    bundle: ScorerBundle,
    input_hash: str,
    error_code: str,
    error_message: str,
    latency_ms: int,
) -> ScorerExecutionDraft:
    return ScorerExecutionDraft(
        component_id=component.component_id,
        component_version=component.version,
        scorer_id=component.component_id,
        scorer_version=component.version,
        status="failed",
        input_hash=input_hash,
        output=None,
        error_code=error_code,
        error_message=error_message,
        latency_ms=max(0, latency_ms),
        usage=None,
        findings=(),
    )


class ScoringService:
    """Run deterministic and injected LLM scorer components exactly once."""

    def __init__(self, llm: AsyncChatModel, *, timeout_seconds: float | None = None) -> None:
        if llm is None or not hasattr(llm, "ainvoke"):
            raise TypeError("llm must provide async ainvoke")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.llm = llm
        self.timeout_seconds = timeout_seconds

    async def _invoke(self, prompt: str) -> Any:
        call = self.llm.ainvoke([HumanMessage(content=prompt)])
        if self.timeout_seconds is None:
            return await call
        return await asyncio.wait_for(call, timeout=self.timeout_seconds)

    async def score(
        self,
        *,
        task: TaskCase | CalibrationTaskInput,
        candidate: CandidateArtifact,
        scorer_bundle: ScorerBundle,
        on_execution: Callable[[ScorerExecutionDraft], None],
    ) -> ScoreSetResultDraft:
        if not callable(on_execution):
            raise TypeError("on_execution must be callable")
        input_hash = canonical_hash(candidate.to_dict()) if isinstance(candidate, CandidateArtifact) else ""
        emitted: list[ScorerExecutionDraft] = []

        def emit(draft: ScorerExecutionDraft) -> None:
            emitted.append(draft)
            # Callback exceptions intentionally propagate to the caller.
            on_execution(draft)

        valid_input = (
            isinstance(task, (TaskCase, CalibrationTaskInput))
            and isinstance(candidate, CandidateArtifact)
            and isinstance(scorer_bundle, ScorerBundle)
            and bool(_task_dimensions(task, scorer_bundle))
            and _valid_scorer_bundle_for_scoring(scorer_bundle)
        )
        if not valid_input:
            for component in getattr(scorer_bundle, "components", ()):
                emit(
                    _component_error(
                        component=component,
                        bundle=scorer_bundle,
                        input_hash=input_hash,
                        error_code="harness_internal_error",
                        error_message="scoring input contract invalid",
                        latency_ms=0,
                    )
                )
            return derive_score_set(
                task=task,
                scorer_bundle=scorer_bundle,
                executions=tuple(emitted),
                input_hash=input_hash,
            )

        dimensions = _task_dimensions(task, scorer_bundle)
        for component in scorer_bundle.components:
            started = time.perf_counter()
            if component.kind == "deterministic":
                if component.component_id == "retrieval-integrity":
                    output, findings = _retrieval_integrity(task=task, candidate=candidate)
                elif component.component_id == "citation-integrity":
                    if task.case_type == "expected_refusal":
                        draft = ScorerExecutionDraft(
                            component_id=component.component_id,
                            component_version=component.version,
                            scorer_id=component.component_id,
                            scorer_version=component.version,
                            status="skipped",
                            input_hash=input_hash,
                            output={"applicable": False},
                            latency_ms=0,
                            usage=None,
                            findings=(),
                        )
                        emit(draft)
                        continue
                    output, findings = _citation_integrity(task=task, candidate=candidate)
                elif component.component_id == "expected-refusal-observation":
                    output, findings, _ = _refusal_observation(task=task, candidate=candidate)
                    if task.case_type != "expected_refusal":
                        emit(
                            ScorerExecutionDraft(
                                component_id=component.component_id,
                                component_version=component.version,
                                scorer_id=component.component_id,
                                scorer_version=component.version,
                                status="skipped",
                                input_hash=input_hash,
                                output=output,
                                latency_ms=0,
                                usage=None,
                                findings=(),
                            )
                        )
                        continue
                else:
                    emit(
                        _component_error(
                            component=component,
                            bundle=scorer_bundle,
                            input_hash=input_hash,
                            error_code="harness_internal_error",
                            error_message="unsupported deterministic component",
                            latency_ms=0,
                        )
                    )
                    continue
                latency_ms = int(max(0.0, (time.perf_counter() - started) * 1000))
                emit(
                    ScorerExecutionDraft(
                        component_id=component.component_id,
                        component_version=component.version,
                        scorer_id=component.component_id,
                        scorer_version=component.version,
                        status="success",
                        input_hash=input_hash,
                        output=output,
                        latency_ms=latency_ms,
                        usage=None,
                        findings=findings,
                    )
                )
                continue

            if component.kind == "llm":
                started = time.perf_counter()
                try:
                    prompt = _build_prompt(
                        task=task,
                        candidate=candidate,
                        scorer_bundle=scorer_bundle,
                        dimensions=dimensions,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    emit(
                        _component_error(
                            component=component,
                            bundle=scorer_bundle,
                            input_hash=input_hash,
                            error_code="harness_internal_error",
                            error_message="scorer prompt build failed",
                            latency_ms=int(max(0.0, (time.perf_counter() - started) * 1000)),
                        )
                    )
                    continue
                try:
                    response = await self._invoke(prompt)
                except (asyncio.TimeoutError, TimeoutError):
                    emit(
                        _component_error(
                            component=component,
                            bundle=scorer_bundle,
                            input_hash=input_hash,
                            error_code="scorer_timeout",
                            error_message="scorer timed out",
                            latency_ms=int(max(0.0, (time.perf_counter() - started) * 1000)),
                        )
                    )
                    continue
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # Keep stable, credential-free diagnostics.  The original
                    # exception and traceback never enter the artifact.
                    emit(
                        _component_error(
                            component=component,
                            bundle=scorer_bundle,
                            input_hash=input_hash,
                            error_code="model_unavailable",
                            error_message="scorer model exception",
                            latency_ms=int(max(0.0, (time.perf_counter() - started) * 1000)),
                        )
                    )
                    continue
                try:
                    raw = getattr(response, "content", None)
                    parsed = parse_rubric_output(raw, dimensions)
                    usage = _extract_usage(response)
                    output = dict(parsed)
                    findings = tuple(parsed["findings"])
                    draft = ScorerExecutionDraft(
                        component_id=component.component_id,
                        component_version=component.version,
                        scorer_id=component.component_id,
                        scorer_version=component.version,
                        status="success",
                        input_hash=input_hash,
                        output=output,
                        latency_ms=int(max(0.0, (time.perf_counter() - started) * 1000)),
                        usage=usage,
                        findings=findings,
                    )
                except asyncio.CancelledError:
                    raise
                except RubricParseError:
                    draft = _component_error(
                        component=component,
                        bundle=scorer_bundle,
                        input_hash=input_hash,
                        error_code="scorer_parse_error",
                        error_message="scorer output malformed",
                        latency_ms=int(max(0.0, (time.perf_counter() - started) * 1000)),
                    )
                except Exception:
                    draft = _component_error(
                        component=component,
                        bundle=scorer_bundle,
                        input_hash=input_hash,
                        error_code="harness_internal_error",
                        error_message="scorer result processing failed",
                        latency_ms=int(max(0.0, (time.perf_counter() - started) * 1000)),
                    )
                emit(draft)
                continue

            emit(
                _component_error(
                    component=component,
                    bundle=scorer_bundle,
                    input_hash=input_hash,
                    error_code="harness_internal_error",
                    error_message="unsupported scorer component kind",
                    latency_ms=0,
                )
            )

        return derive_score_set(
            task=task,
            scorer_bundle=scorer_bundle,
            executions=tuple(emitted),
            input_hash=input_hash,
        )


def derive_score_set(
    task: TaskCase | CalibrationTaskInput,
    scorer_bundle: ScorerBundle,
    executions: Sequence[ScorerExecutionDraft] | None = None,
    *,
    drafts: Sequence[ScorerExecutionDraft] | None = None,
    input_hash: str,
) -> ScoreSetResultDraft:
    """Rebuild aggregate scores/verdict using only emitted execution drafts."""

    if executions is not None and drafts is not None:
        raise TypeError("provide executions or drafts, not both")
    emitted = tuple(executions if executions is not None else (drafts or ()))
    if not isinstance(scorer_bundle, ScorerBundle) or not isinstance(
        task, (TaskCase, CalibrationTaskInput)
    ) or not isinstance(input_hash, str) or not input_hash:
        return ScoreSetResultDraft(
            status="failed",
            verdict="inconclusive",
            aggregate_scores={},
            findings=(),
            executions=(),
            error_code="harness_internal_error",
            error_message="scoring input contract invalid",
            input_hash=input_hash,
        )
    if any(not isinstance(draft, ScorerExecutionDraft) for draft in emitted):
        return ScoreSetResultDraft(
            status="failed",
            verdict="inconclusive",
            aggregate_scores={},
            findings=(),
            executions=(),
            error_code="harness_internal_error",
            error_message="scorer execution contract invalid",
            input_hash=input_hash,
        )
    if not _valid_scorer_bundle_for_scoring(scorer_bundle):
        findings = [finding for draft in emitted for finding in draft.findings]
        return ScoreSetResultDraft(
            status="failed",
            verdict="inconclusive",
            aggregate_scores={},
            findings=findings,
            executions=emitted,
            error_code="harness_internal_error",
            error_message="scorer bundle contract invalid",
            input_hash=input_hash,
        )
    components = tuple(scorer_bundle.components)
    expected_ids = tuple(component.component_id for component in components)
    structural_error: str | None = None
    if len(expected_ids) != len(set(expected_ids)):
        structural_error = "scorer component identities are duplicated"
    elif len(emitted) != len(components):
        structural_error = "scorer execution count does not match components"
    elif len({draft.component_id for draft in emitted}) != len(emitted):
        structural_error = "scorer execution identities are duplicated"
    else:
        for draft, component in zip(emitted, components):
            if (
                draft.component_id != component.component_id
                or draft.component_version != component.version
                or draft.scorer_id != component.component_id
                or draft.scorer_version != component.version
            ):
                structural_error = "scorer execution identity does not match component"
                break
            if draft.input_hash != input_hash:
                structural_error = "scorer execution input hash mismatch"
                break
    if structural_error is not None:
        findings = [finding for draft in emitted for finding in draft.findings]
        return ScoreSetResultDraft(
            status="failed",
            verdict="inconclusive",
            aggregate_scores={},
            findings=findings,
            executions=emitted,
            error_code="harness_internal_error",
            error_message=structural_error,
            input_hash=input_hash,
        )
    required_dimensions = _task_dimensions(task, scorer_bundle)
    dimension_scores: dict[str, int] = {}
    findings: list[Mapping[str, Any]] = []
    failed: list[str] = []
    successful_or_skipped = 0
    successful = 0
    llm_component_id = "grounded-quality-rubric"
    for draft in emitted:
        findings.extend(draft.findings)
        if draft.status == "failed":
            failed.append(draft.component_id)
        elif draft.status == "success":
            successful += 1
            successful_or_skipped += 1
        elif draft.status == "skipped":
            successful_or_skipped += 1
        if (
            draft.status == "success"
            and draft.component_id == llm_component_id
            and isinstance(draft.output, Mapping)
        ):
            for dimension in required_dimensions:
                value = draft.output.get(dimension)
                if type(value) is int and 1 <= value <= 5:
                    dimension_scores[dimension] = value

    if not emitted:
        status = "failed"
        verdict = "not_evaluated"
    elif len(failed) == 0 and len(emitted) == len(expected_ids):
        status = "completed"
        verdict = "not_evaluated" if successful == 0 else None
    elif successful_or_skipped == 0:
        status = "failed"
        verdict = None
    else:
        status = "partial"
        verdict = None

    hard_codes = _hard_finding_codes(task=task, findings=findings)
    decision = derive_verdict(
        case_type=getattr(task, "case_type", ""),
        required_dimensions=required_dimensions,
        dimension_scores=dimension_scores,
        hard_findings=hard_codes,
        failed_scorers=failed,
        findings=findings,
        required_minimum=int(scorer_bundle.verdict_policy.get("required_minimum", 4)),
    )
    if verdict is None:
        verdict = decision.verdict
    error_code = None
    error_message = None
    for draft in emitted:
        if draft.status == "failed" and draft.error_code:
            error_code = draft.error_code
            error_message = draft.error_message
            break
    return ScoreSetResultDraft(
        status=status,
        verdict=verdict,
        aggregate_scores=dimension_scores,
        findings=findings,
        executions=emitted,
        error_code=error_code,
        error_message=error_message,
        input_hash=input_hash,
    )


# Explicit aliases keep the parser boundary discoverable to callers without
# introducing a second implementation or a permissive parser.
parse_llm_output = parse_rubric_output
parse_scorer_output = parse_rubric_output


__all__ = [
    "AsyncChatModel",
    "RubricParseError",
    "ScoringService",
    "VerdictDecision",
    "derive_score_set",
    "derive_verdict",
    "parse_llm_output",
    "parse_rubric_output",
    "parse_scorer_output",
]
