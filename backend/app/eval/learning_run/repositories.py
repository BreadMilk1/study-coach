"""Bounded persistence repositories for Learning Run evaluation artifacts.

The repositories deliberately expose only the lifecycle operations needed by
the harness.  They provide application-level append-only and checksum-verified
semantics; they are not a WORM or tamper-proof storage layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping
from uuid import uuid4

from sqlalchemy import and_, bindparam, insert, or_, select, text, update
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from app.db.models import EvalRun, EvalScoreSet, EvalScorerExecution

from .contracts import (
    CandidateArtifact,
    RunManifest,
    ScorerBundle,
    ScorerExecutionDraft,
    TaskCase,
    as_plain,
    canonical_hash,
    canonical_json_bytes,
    hash_without_field,
)
from .scoring import derive_score_set_from_stored


class LearningRunRepositoryError(RuntimeError):
    """Base error for bounded Learning Run persistence operations."""


class RepositoryConflictError(LearningRunRepositoryError):
    """A conditional transition lost a race or would violate append-only rules."""


class InvalidTransitionError(RepositoryConflictError):
    """A lifecycle transition is not valid for the row's current state."""


class ChecksumMismatchError(LearningRunRepositoryError):
    """Persisted or caller-supplied content does not match its checksum."""


class RepositoryNotFoundError(LearningRunRepositoryError):
    """The requested persistence row does not exist."""


class EvaluationBusyError(LearningRunRepositoryError):
    """The one local evaluation execution lease is held by a live entity."""

    def __init__(self, active_entity_id: str, active_kind: str):
        if active_kind not in {"run", "score_set"}:
            raise ValueError("active_kind must be run or score_set")
        self.active_entity_id = str(active_entity_id)
        self.active_kind = active_kind
        super().__init__("another evaluation is already running")


class EvaluationUnavailableError(LearningRunRepositoryError):
    """The evaluation store could not acquire its short-lived claim lock."""

    def __init__(self):
        super().__init__("evaluation storage is unavailable")


@dataclass(frozen=True)
class ReconciliationResult:
    """Typed startup repair counts from one short database transaction."""

    runs_reconciled: int
    score_sets_reconciled: int
    started_before: datetime


_RUN_OUTCOMES = {"system_failed", "timed_out", "budget_exceeded"}
_RUN_FINISHED_OUTCOMES = _RUN_OUTCOMES | {"success"}
_RUN_IDENTITY_FIELDS = (
    "experiment_id",
    "task_case_id",
    "task_case_version",
    "variant_id",
    "run_profile",
)
_SCORE_TERMINAL_STATUSES = {"completed", "partial", "failed", "cancelled"}
_QUALITY_VERDICTS = {"pass", "fail", "inconclusive", "not_evaluated"}
_EXECUTION_STATUSES = {"success", "failed", "skipped"}
_OPERATIONAL_ERROR_CODES = {
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
_SECRET_PATTERN = re.compile(
    r"(?i)(?:bearer\s+|authorization\s*[:=]\s*|api[_-]?key\s*[:=]\s*|sk-[A-Za-z0-9_-]+)[^\s,;]+"
)
_AUTHORIZATION_PATTERN = re.compile(
    r"""(?ix)[\"']?authorization(?:header)?[\"']?\s*(?:(?::|=)\s*)?[\"']?(?:(?:basic|bearer|digest|negotiate|ntlm|token)\s+)?[^\s,;}\]"']+[\"']?"""
)
_CREDENTIAL_ASSIGNMENT_PATTERN = re.compile(
    r"""(?ix)(?P<key>[\"']?[a-z][a-z0-9_-]*[\"']?)\s*(?::|=)\s*(?P<value>\"[^\"]*\"|'[^']*'|[^\s,;}\]"']+)"""
)
_SENSITIVE_ASSIGNMENT_KEYS = {
    "authorization",
    "authorizationheader",
    "password",
    "passwd",
    "apikey",
    "accesstoken",
    "refreshtoken",
    "clientsecret",
    "token",
    "secret",
    "cookie",
    "sessiontoken",
}


def _new_id(prefix: str | None = None) -> str:
    del prefix
    return str(uuid4())


def _plain_mapping(value: Any, *, label: str) -> dict[str, Any]:
    plain = as_plain(value)
    if not isinstance(plain, Mapping):
        raise TypeError(f"{label} must be a JSON mapping")
    return {str(key): item for key, item in plain.items()}


def _payload_and_hash(
    value: Any,
    *,
    expected_hash: str | None,
    label: str,
    hash_field: str,
) -> tuple[dict[str, Any], str]:
    supplied_hash = expected_hash
    if isinstance(value, RunManifest):
        payload = value.payload()
        plain = _plain_mapping(payload, label=label)
        actual_hash = canonical_hash(plain)
        embedded_hash = value.manifest_hash or None
        if embedded_hash is not None and embedded_hash != actual_hash:
            raise ChecksumMismatchError(f"{label} hash mismatch")
        if expected_hash is not None and str(expected_hash) != actual_hash:
            raise ChecksumMismatchError(f"{label} hash mismatch")
        return plain, actual_hash
    elif isinstance(value, CandidateArtifact):
        payload = value.to_dict()
    else:
        payload = _plain_mapping(value, label=label)
        if supplied_hash is None and payload.get(hash_field):
            supplied_hash = str(payload[hash_field])
        payload.pop(hash_field, None)

    plain = _plain_mapping(payload, label=label)
    actual_hash = canonical_hash(plain)
    if supplied_hash is not None and str(supplied_hash) != actual_hash:
        raise ChecksumMismatchError(f"{label} hash mismatch")
    return plain, actual_hash


def _sanitize_message(message: str | None) -> str | None:
    if message is None:
        return None
    # Keep operational diagnostics useful while excluding stack traces and
    # common credential forms from the persisted row.
    one_line = " ".join(str(message).split())
    one_line = re.split(r"(?i)\btraceback\b", one_line, maxsplit=1)[0].rstrip()

    sanitized = _AUTHORIZATION_PATTERN.sub("[redacted]", one_line)

    def redact_assignment(match: re.Match[str]) -> str:
        key = re.sub(r"[^a-z0-9]", "", match.group("key").lower())
        if key in _SENSITIVE_ASSIGNMENT_KEYS:
            return "[redacted]"
        return match.group(0)

    sanitized = _CREDENTIAL_ASSIGNMENT_PATTERN.sub(redact_assignment, sanitized)
    sanitized = _SECRET_PATTERN.sub("[redacted]", sanitized)
    return sanitized[:1000] or None


_MANIFEST_SENSITIVE_KEY = re.compile(
    r"(?:^|_)(?:api_key|apikey|authorization|access_token|refresh_token|"
    r"token|password|secret|credential|cookie|endpoint|base_url|baseurl)(?:_|$)"
)
_MANIFEST_URL = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
_MANIFEST_SENSITIVE_COMPACT_KEYS = {
    "apikey",
    "authorization",
    "authorizationheader",
    "accesstoken",
    "refreshtoken",
    "password",
    "passwd",
    "clientsecret",
    "token",
    "secret",
    "credential",
    "cookie",
    "sessiontoken",
    "endpoint",
    "baseurl",
}


def _validate_manifest_privacy(value: Any, *, path: str = "manifest") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(key))
            normalized_key = re.sub(r"[^a-z0-9]+", "_", key_text.lower()).strip("_")
            compact_key = re.sub(r"[^a-z0-9]", "", normalized_key)
            if (
                _MANIFEST_SENSITIVE_KEY.search(normalized_key)
                or compact_key in _MANIFEST_SENSITIVE_COMPACT_KEYS
            ):
                raise ValueError(f"manifest contains restricted key: {path}.{key}")
            _validate_manifest_privacy(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_manifest_privacy(item, path=f"{path}[{index}]")
        return
    if isinstance(value, str) and _MANIFEST_URL.search(value):
        raise ValueError(f"manifest contains endpoint URL: {path}")


def _require_identity_value(value: Any, *, label: str) -> str:
    text = "" if value is None else str(value)
    if not text:
        raise ValueError(f"{label} must be a non-empty string")
    return text


def _validate_run_identity_fields(
    manifest: Mapping[str, Any],
    identity: Mapping[str, Any],
    *,
    label: str,
) -> None:
    """Cross-check the five structured Run identity fields against the manifest.

    Both the frozen manifest and the structured row must carry every identity
    field as a non-empty value and agree exactly.  A missing field is rejected
    rather than silently passing as ``"" == ""``, so a partially written row can
    never validate against itself.
    """

    for field in _RUN_IDENTITY_FIELDS:
        manifest_value = _require_identity_value(
            manifest.get(field), label=f"{label} manifest.{field}"
        )
        identity_value = _require_identity_value(
            identity.get(field), label=f"{label}.{field}"
        )
        if manifest_value != identity_value:
            raise ValueError(f"{label}.{field} does not match manifest")


def _validate_score_set_relation(
    *,
    snapshot: Any,
    definition_hash: Any,
    scorer_id: Any,
    scorer_version: Any,
    artifact_input_hash: Any,
    parent_artifact_hash: Any,
    label: str,
) -> None:
    """Shared ScoreSet relationship check for both reads and suite import."""

    if not parent_artifact_hash or str(artifact_input_hash) != str(parent_artifact_hash):
        raise ChecksumMismatchError(f"score set artifact input hash mismatch: {label}")
    if not isinstance(snapshot, Mapping) or not snapshot:
        raise ChecksumMismatchError(f"score set scorer snapshot missing: {label}")
    if hash_without_field(snapshot, "definition_hash") != definition_hash:
        raise ChecksumMismatchError(f"score set scorer snapshot hash mismatch: {label}")
    if str(snapshot.get("scorer_id") or "") != str(scorer_id):
        raise ChecksumMismatchError(f"score set scorer id mismatch: {label}")
    if str(snapshot.get("version") or "") != str(scorer_version):
        raise ChecksumMismatchError(f"score set scorer version mismatch: {label}")


def _validate_execution_component(
    *,
    scorer_id: Any,
    scorer_version: Any,
    snapshot: Any,
    label: str,
) -> None:
    """Require an execution component to be declared by its frozen parent snapshot.

    Historical suites are never back-filled from the current Registry; only the
    components frozen into the parent ScoreSet's own snapshot are authoritative.
    """

    components = snapshot.get("components") if isinstance(snapshot, Mapping) else None
    if not isinstance(components, (list, tuple)):
        raise ValueError(f"{label} parent snapshot does not declare components")
    wanted = (str(scorer_id), str(scorer_version))
    for component in components:
        if not isinstance(component, Mapping):
            continue
        if (
            str(component.get("id") or ""),
            str(component.get("version") or ""),
        ) == wanted:
            return
    raise ValueError(f"{label} scorer component is not declared by the parent snapshot")


def _validate_score_domain(aggregate_scores: Any) -> None:
    """Reject an invalid aggregate-score payload before any terminal write.

    Only true integers on the frozen 1..5 rubric scale are accepted; ``bool``
    (``True`` is an ``int`` in Python), floats, ``NaN``/``Infinity`` and
    out-of-range values are all rejected.
    """

    if aggregate_scores is None:
        return
    if not isinstance(aggregate_scores, Mapping):
        raise ValueError("aggregate scores must be a mapping or null")
    for dimension, score in aggregate_scores.items():
        if not isinstance(dimension, str) or not dimension:
            raise ValueError("aggregate score dimension must be a non-empty string")
        if type(score) is not int or not 1 <= score <= 5:
            raise ValueError("aggregate score must be an integer between 1 and 5")


def _first_present(mapping: Mapping[str, Any], *keys: str) -> Any:
    """Return the first alias key that is present, ignoring value truthiness.

    Imported container fields (aggregate scores, findings, execution output and
    usage) may legitimately be an empty mapping or list.  Selecting by key
    presence keeps an explicit empty value and an explicit ``None`` instead of
    falling through to a non-empty alias or fabricating a new value.
    """

    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _validate_terminal_run_state(
    *,
    lifecycle: Any,
    outcome: Any,
    label: str,
) -> tuple[str, str | None]:
    """Validate and normalize a terminal Run lifecycle/outcome pair.

    Only terminal Runs are importable: ``finished`` requires an explicit, legal,
    non-empty outcome, while ``cancelled`` must not carry one.  Both suite import
    and the raw historical insert share this single rule so their semantics
    cannot drift.
    """

    lifecycle_text = "" if lifecycle is None else str(lifecycle)
    if lifecycle_text not in {"finished", "cancelled"}:
        raise ValueError(f"{label} run is not terminal")
    if lifecycle_text == "finished":
        outcome_text = "" if outcome is None else str(outcome)
        if outcome_text not in _RUN_FINISHED_OUTCOMES:
            raise ValueError(f"{label} finished run requires a terminal outcome")
        return lifecycle_text, outcome_text
    if outcome is not None:
        raise ValueError(f"{label} cancelled run must not carry an outcome")
    return lifecycle_text, None


def _validate_manifest_identity(
    manifest: RunManifest,
    *,
    experiment_id: str,
    task_case_id: str,
    task_case_version: str,
    variant_id: str,
    run_profile: str,
) -> None:
    identity = {
        "experiment_id": experiment_id,
        "task_case_id": task_case_id,
        "task_case_version": task_case_version,
        "variant_id": variant_id,
        "run_profile": run_profile,
    }
    _validate_run_identity_fields(manifest.payload(), identity, label="run")


def _error_json(
    error_code: str,
    sanitized_message: str | None,
    *,
    allow_cancelled: bool = False,
) -> dict[str, str]:
    code = str(error_code).strip()
    if not code or any(char.isspace() for char in code):
        raise ValueError("error_code must be a stable non-empty token")
    allowed_codes = _OPERATIONAL_ERROR_CODES | ({"cancelled"} if allow_cancelled else set())
    if code not in allowed_codes:
        raise ValueError(f"unsupported operational error code: {code}")
    payload = {"code": code}
    message = _sanitize_message(sanitized_message)
    if message is not None:
        payload["message"] = message
    return payload


def _load(session: Session, model: type[Any], row_id: str) -> Any:
    # populate_existing is intentional: direct SQL tampering tests and a
    # second worker must not receive an identity-map snapshot.
    row = session.execute(
        select(model)
        .where(model.id == row_id)
        .execution_options(populate_existing=True)
    ).scalar_one_or_none()
    if row is None:
        raise RepositoryNotFoundError(f"{model.__tablename__} row not found: {row_id}")
    return row


def _commit_update(
    session: Session,
    statement: Any,
    *,
    model: type[Any],
    row_id: str,
    conflict: str,
) -> None:
    try:
        result = session.execute(statement)
        if result.rowcount != 1:
            session.rollback()
            if session.get(model, row_id) is None:
                raise RepositoryNotFoundError(f"{model.__tablename__} row not found: {row_id}")
            raise RepositoryConflictError(conflict)
        session.commit()
    except RepositoryConflictError:
        raise
    except Exception:
        session.rollback()
        raise


def _resolve_scorer_identity(
    *,
    scorer_id: str | None,
    scorer_version: str | None,
    scorer_bundle_id: str | None = None,
    scorer_bundle_version: str | None = None,
    scorer_bundle: ScorerBundle | Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    if scorer_bundle is not None:
        if isinstance(scorer_bundle, ScorerBundle):
            scorer_id = scorer_id or scorer_bundle.scorer_id
            scorer_version = scorer_version or scorer_bundle.version
        else:
            bundle = _plain_mapping(scorer_bundle, label="scorer_bundle")
            scorer_id = scorer_id or str(bundle.get("scorer_id") or bundle.get("id") or "")
            scorer_version = scorer_version or str(
                bundle.get("version") or bundle.get("scorer_version") or ""
            )
    scorer_id = scorer_id or scorer_bundle_id
    scorer_version = scorer_version or scorer_bundle_version
    if not scorer_id or not scorer_version:
        raise ValueError("scorer id and version are required")
    return str(scorer_id), str(scorer_version)


# The two fields a complete scorer document adds on top of the nine-key
# `ScorerBundle.payload()` projection.  The projection of a frozen document is
# the payload the bundle executes, so removing these two keys is what the
# document-to-bundle check compares.
_SCORER_DOCUMENT_META_FIELDS = frozenset({"schema_version", "definition_hash"})


def freeze_scorer_document(
    document: Mapping[str, Any],
    *,
    bundle: ScorerBundle,
    expected_definition_hash: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Validate a complete frozen scorer document against its executing bundle.

    New live and re-score ScoreSets must carry the whole frozen scorer document,
    not a projection, so a later reader can audit the exact definition.  The
    document has to declare the existing fixed scorer schema as a non-empty
    string and its own ``definition_hash``; that hash must equal the executing
    bundle's declared hash, and a caller-supplied expected hash, when present,
    must agree as well.  The document must also name the bundle's id and version
    and project onto exactly the payload the bundle will execute.  Any mismatch
    is raised here, before a Run or ScoreSet row is written, with no silent
    fallback to a re-derived hash.
    """

    snapshot = as_plain(document)
    if not isinstance(snapshot, dict) or not snapshot:
        raise ValueError("scorer document must be a non-empty mapping")
    schema_version = snapshot.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version:
        raise ChecksumMismatchError("scorer document is missing schema_version")
    declared = snapshot.get("definition_hash")
    if not isinstance(declared, str) or not declared:
        raise ChecksumMismatchError("scorer document is missing definition_hash")
    if hash_without_field(snapshot, "definition_hash") != declared:
        raise ChecksumMismatchError("scorer document definition hash mismatch")
    bundle_hash = getattr(bundle, "definition_hash", None)
    if not isinstance(bundle_hash, str) or not bundle_hash:
        raise ChecksumMismatchError("scorer bundle is missing definition_hash")
    if declared != bundle_hash:
        raise ChecksumMismatchError(
            "scorer document definition hash does not match the executing bundle"
        )
    if expected_definition_hash is not None and str(expected_definition_hash) != declared:
        raise ChecksumMismatchError(
            "scorer document definition hash does not match the expected hash"
        )
    if str(snapshot.get("scorer_id") or "") != str(bundle.scorer_id):
        raise ChecksumMismatchError("scorer document id mismatch")
    if str(snapshot.get("version") or "") != str(bundle.version):
        raise ChecksumMismatchError("scorer document version mismatch")
    projection = {
        key: value
        for key, value in snapshot.items()
        if key not in _SCORER_DOCUMENT_META_FIELDS
    }
    if projection != bundle.payload():
        raise ChecksumMismatchError("scorer document projection mismatch")
    return snapshot, declared


def _freeze_scorer_audit(
    *,
    scorer_id: str,
    scorer_version: str,
    scorer_bundle: ScorerBundle | Mapping[str, Any] | None = None,
    scorer_snapshot: Mapping[str, Any] | None = None,
    scorer_definition_hash: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Freeze a ScoreSet's scorer document so historical re-score stays auditable."""

    if scorer_snapshot is not None:
        if isinstance(scorer_bundle, ScorerBundle):
            # New live/rescore rows must freeze the complete document and prove
            # it projects onto the bundle that actually executes.  A supplied
            # expected hash must agree; a conflicting expectation is rejected
            # rather than silently dropped.
            return freeze_scorer_document(
                scorer_snapshot,
                bundle=scorer_bundle,
                expected_definition_hash=scorer_definition_hash,
            )
        snapshot = as_plain(scorer_snapshot)
        if not isinstance(snapshot, dict) or not snapshot:
            raise ValueError("scorer_snapshot must be a non-empty mapping")
        expected = str(scorer_definition_hash or snapshot.get("definition_hash") or "")
        if not expected:
            raise ValueError("scorer_definition_hash is required")
        actual = hash_without_field(snapshot, "definition_hash")
        if actual != expected:
            raise ChecksumMismatchError("scorer snapshot definition hash mismatch")
        return snapshot, expected

    if isinstance(scorer_bundle, ScorerBundle):
        # A bundle alone cannot freeze a complete document.  Refuse to re-derive
        # a projection hash that would silently diverge from the stored
        # definition; callers must pass the frozen scorer document.
        raise ValueError(
            "a complete frozen scorer document is required to freeze a ScoreSet"
        )

    if isinstance(scorer_bundle, Mapping):
        snapshot = as_plain(scorer_bundle)
        expected = str(scorer_definition_hash or snapshot.get("definition_hash") or "")
        if expected:
            actual = hash_without_field(snapshot, "definition_hash")
            if actual != expected:
                raise ChecksumMismatchError("scorer snapshot definition hash mismatch")
            return snapshot, expected
        expected = canonical_hash(snapshot)
        return snapshot, expected

    snapshot = {"scorer_id": scorer_id, "version": scorer_version}
    return snapshot, canonical_hash(snapshot)


def _begin_immediate(session: Session) -> None:
    """Acquire the SQLite write gate on an otherwise clean eval session."""

    # A read through SQLAlchemy starts an implicit transaction.  The control
    # and claim repositories own their short transaction, so discard that
    # read-only boundary before issuing the explicit SQLite gate.
    session.rollback()
    try:
        session.execute(text("BEGIN IMMEDIATE"))
    except OperationalError:
        session.rollback()
        raise EvaluationUnavailableError() from None


class EvalRunRepository:
    """Create and atomically transition immutable CandidateArtifacts."""

    def __init__(self, session: Session):
        self.session = session

    def create(
        self,
        *,
        experiment_id: str,
        task_case_id: str,
        task_case_version: str,
        variant_id: str,
        run_profile: str,
        manifest: RunManifest,
        manifest_hash: str | None = None,
        id: str | None = None,
        suite_execution_id: str | None = None,
        created_at: datetime | None = None,
    ) -> EvalRun:
        if not isinstance(manifest, RunManifest):
            raise TypeError("manifest must be a RunManifest")
        _validate_manifest_identity(
            manifest,
            experiment_id=experiment_id,
            task_case_id=task_case_id,
            task_case_version=task_case_version,
            variant_id=variant_id,
            run_profile=run_profile,
        )
        manifest_json, verified_hash = _payload_and_hash(
            manifest,
            expected_hash=manifest_hash,
            label="manifest",
            hash_field="manifest_hash",
        )
        _validate_manifest_privacy(manifest_json)
        row = EvalRun(
            id=id or _new_id("run"),
            experiment_id=str(experiment_id),
            suite_execution_id=suite_execution_id,
            task_case_id=str(task_case_id),
            task_case_version=str(task_case_version),
            variant_id=str(variant_id),
            run_profile=str(run_profile),
            lifecycle="queued",
            outcome=None,
            operational_error_json=None,
            manifest_json=manifest_json,
            manifest_hash=verified_hash,
            candidate_artifact_json=None,
            artifact_hash=None,
            created_at=created_at or datetime.utcnow(),
        )
        try:
            self.session.add(row)
            self.session.commit()
        except Exception:
            self.session.rollback()
            raise
        return _load(self.session, EvalRun, row.id)

    def import_historical(
        self,
        *,
        row: EvalRun,
        auto_commit: bool = True,
    ) -> EvalRun:
        if not isinstance(row, EvalRun):
            raise TypeError("row must be an EvalRun")
        # A raw historical insert must satisfy the same manifest hash, identity
        # cross-check, artifact hash, privacy boundary and terminal lifecycle
        # rule as a suite import so it cannot create a row its own verified read
        # would reject or an active/non-terminal row that bypasses the import.
        self._verify_hashes(row)
        _validate_manifest_privacy(
            _plain_mapping(row.manifest_json, label="manifest"),
        )
        row.lifecycle, row.outcome = _validate_terminal_run_state(
            lifecycle=row.lifecycle,
            outcome=row.outcome,
            label=f"run {row.id}",
        )
        try:
            self.session.add(row)
            if auto_commit:
                self.session.commit()
            else:
                self.session.flush()
        except Exception:
            self.session.rollback()
            raise
        return _load(self.session, EvalRun, row.id)

    def get_verified(self, run_id: str) -> EvalRun:
        row = _load(self.session, EvalRun, run_id)
        self._verify_hashes(row)
        return row

    def get(self, run_id: str) -> EvalRun:
        return self.get_verified(run_id)

    def list_verified(self) -> list[EvalRun]:
        rows = self.session.execute(
            select(EvalRun)
            .order_by(EvalRun.created_at.desc(), EvalRun.id.desc())
            .execution_options(populate_existing=True)
        ).scalars().all()
        for row in rows:
            self._verify_hashes(row)
        return list(rows)

    def _verify_hashes(self, row: EvalRun) -> None:
        try:
            manifest_json = _plain_mapping(row.manifest_json, label="persisted manifest")
        except (TypeError, ValueError) as exc:
            raise ChecksumMismatchError(f"manifest payload malformed for run {row.id}") from exc
        if canonical_hash(manifest_json) != row.manifest_hash:
            raise ChecksumMismatchError(f"manifest hash mismatch for run {row.id}")
        try:
            _validate_run_identity_fields(
                manifest_json,
                {
                    "experiment_id": row.experiment_id,
                    "task_case_id": row.task_case_id,
                    "task_case_version": row.task_case_version,
                    "variant_id": row.variant_id,
                    "run_profile": row.run_profile,
                },
                label=f"run {row.id}",
            )
        except ValueError as exc:
            raise ChecksumMismatchError(str(exc)) from exc
        if row.candidate_artifact_json is None:
            if row.artifact_hash is not None:
                raise ChecksumMismatchError(f"artifact hash without artifact for run {row.id}")
            return
        try:
            artifact_json = _plain_mapping(
                row.candidate_artifact_json,
                label="persisted candidate artifact",
            )
        except (TypeError, ValueError) as exc:
            raise ChecksumMismatchError(f"artifact payload malformed for run {row.id}") from exc
        if row.artifact_hash is None or canonical_hash(artifact_json) != row.artifact_hash:
            raise ChecksumMismatchError(f"artifact hash mismatch for run {row.id}")

    def claim_running(self, run_id: str, *, expected_lifecycle: str = "queued") -> EvalRun:
        if expected_lifecycle != "queued":
            raise InvalidTransitionError("a Run can only claim running from queued")
        _commit_update(
            self.session,
            update(EvalRun)
            .where(
                EvalRun.id == run_id,
                EvalRun.lifecycle == expected_lifecycle,
                EvalRun.outcome.is_(None),
            )
            .values(lifecycle="running", started_at=datetime.utcnow()),
            model=EvalRun,
            row_id=run_id,
            conflict=f"run cannot transition to running: {run_id}",
        )
        return self.get_verified(run_id)

    start = claim_running

    def finalize_candidate(
        self,
        run_id: str,
        *,
        expected_lifecycle: str = "running",
        candidate_artifact: CandidateArtifact,
        artifact_hash: str | None = None,
    ) -> EvalRun:
        if expected_lifecycle != "running":
            raise InvalidTransitionError("candidate finalization requires a running Run")
        if not isinstance(candidate_artifact, CandidateArtifact):
            raise TypeError("candidate_artifact must be a CandidateArtifact")
        if candidate_artifact.usage != "unavailable" and not isinstance(
            candidate_artifact.usage, Mapping
        ):
            raise ValueError("candidate artifact usage must be a mapping or 'unavailable'")
        artifact_json, verified_hash = _payload_and_hash(
            candidate_artifact,
            expected_hash=artifact_hash,
            label="candidate artifact",
            hash_field="artifact_hash",
        )
        _commit_update(
            self.session,
            update(EvalRun)
            .where(
                EvalRun.id == run_id,
                EvalRun.lifecycle == expected_lifecycle,
                EvalRun.outcome.is_(None),
                # The scalar checksum is the authoritative CAS guard.  The
                # JSON predicate additionally rejects a present JSON value.
                EvalRun.artifact_hash.is_(None),
                EvalRun.candidate_artifact_json.is_(None),
            )
            .values(
                lifecycle="finished",
                outcome="success",
                operational_error_json=None,
                candidate_artifact_json=artifact_json,
                artifact_hash=verified_hash,
                finished_at=datetime.utcnow(),
            ),
            model=EvalRun,
            row_id=run_id,
            conflict=f"candidate artifact already finalized or run is not {expected_lifecycle}: {run_id}",
        )
        return self.get_verified(run_id)

    def freeze_candidate(
        self,
        run_id: str,
        candidate_artifact: CandidateArtifact,
        artifact_hash: str | None = None,
        expected_lifecycle: str = "running",
    ) -> EvalRun:
        """CAS-write an immutable Candidate while leaving the Run running."""

        if expected_lifecycle != "running":
            raise InvalidTransitionError("candidate freeze requires a running Run")
        if not isinstance(candidate_artifact, CandidateArtifact):
            raise TypeError("candidate_artifact must be a CandidateArtifact")
        if candidate_artifact.usage != "unavailable" and not isinstance(
            candidate_artifact.usage, Mapping
        ):
            raise ValueError("candidate artifact usage must be a mapping or 'unavailable'")
        artifact_json, verified_hash = _payload_and_hash(
            candidate_artifact,
            expected_hash=artifact_hash,
            label="candidate artifact",
            hash_field="artifact_hash",
        )
        _commit_update(
            self.session,
            update(EvalRun)
            .where(
                EvalRun.id == run_id,
                EvalRun.lifecycle == expected_lifecycle,
                EvalRun.outcome.is_(None),
                EvalRun.artifact_hash.is_(None),
                EvalRun.candidate_artifact_json.is_(None),
            )
            .values(
                candidate_artifact_json=artifact_json,
                artifact_hash=verified_hash,
            ),
            model=EvalRun,
            row_id=run_id,
            conflict=f"candidate artifact already frozen or run is not {expected_lifecycle}: {run_id}",
        )
        return self.get_verified(run_id)

    def finalize_success(
        self,
        run_id: str,
        expected_lifecycle: str = "running",
    ) -> EvalRun:
        """Terminally succeed a running Run only after Candidate freeze."""

        if expected_lifecycle != "running":
            raise InvalidTransitionError("success finalization requires a running Run")
        _commit_update(
            self.session,
            update(EvalRun)
            .where(
                EvalRun.id == run_id,
                EvalRun.lifecycle == expected_lifecycle,
                EvalRun.outcome.is_(None),
                EvalRun.artifact_hash.is_not(None),
                EvalRun.candidate_artifact_json.is_not(None),
            )
            .values(
                lifecycle="finished",
                outcome="success",
                operational_error_json=None,
                finished_at=datetime.utcnow(),
            ),
            model=EvalRun,
            row_id=run_id,
            conflict=f"run cannot finalize success without a frozen candidate: {run_id}",
        )
        return self.get_verified(run_id)

    def finalize_failure(
        self,
        run_id: str,
        *,
        outcome: str,
        error_code: str,
        sanitized_message: str | None = None,
        expected_lifecycle: str = "running",
        stage: str | None = None,
        retryable: bool | None = None,
        spent_budget: Mapping[str, Any] | None = None,
    ) -> EvalRun:
        if expected_lifecycle != "running":
            raise InvalidTransitionError("failure finalization requires a running Run")
        if outcome not in _RUN_OUTCOMES:
            raise ValueError(f"unsupported failure outcome: {outcome}")
        error_json = _error_json(error_code, sanitized_message)
        if stage is not None:
            if not isinstance(stage, str) or not stage or any(char.isspace() for char in stage):
                raise ValueError("stage must be a stable non-empty token")
            error_json["stage"] = stage
        if retryable is not None:
            if type(retryable) is not bool:
                raise TypeError("retryable must be a boolean or None")
            error_json["retryable"] = retryable
        if spent_budget is not None:
            plain_budget = as_plain(spent_budget)
            if not isinstance(plain_budget, Mapping):
                raise TypeError("spent_budget must be a mapping")
            try:
                _validate_manifest_privacy(plain_budget, path="spent_budget")
                # Canonicalization rejects NaN/Infinity and ensures nested
                # values are JSON-compatible before the row is written.
                canonical_json_bytes(plain_budget)
            except (TypeError, ValueError) as exc:
                raise ValueError("spent_budget must be typed and privacy-safe") from exc
            error_json["spent_budget"] = dict(plain_budget)
        _commit_update(
            self.session,
            update(EvalRun)
            .where(
                EvalRun.id == run_id,
                EvalRun.lifecycle == expected_lifecycle,
                EvalRun.outcome.is_(None),
            )
            .values(
                lifecycle="finished",
                outcome=outcome,
                operational_error_json=error_json,
                finished_at=datetime.utcnow(),
            ),
            model=EvalRun,
            row_id=run_id,
            conflict=f"run cannot finalize failure: {run_id}",
        )
        return self.get_verified(run_id)

    def cancel_once(
        self,
        run_id: str,
        *,
        expected_lifecycle: str = "running",
        error_code: str = "cancelled",
        sanitized_message: str | None = None,
    ) -> EvalRun:
        if expected_lifecycle != "running":
            raise InvalidTransitionError("a Run can only be cancelled from running")
        _commit_update(
            self.session,
            update(EvalRun)
            .where(
                EvalRun.id == run_id,
                EvalRun.lifecycle == expected_lifecycle,
                EvalRun.outcome.is_(None),
                EvalRun.artifact_hash.is_(None),
                EvalRun.candidate_artifact_json.is_(None),
            )
            .values(
                lifecycle="cancelled",
                operational_error_json=_error_json(
                    error_code,
                    sanitized_message,
                    allow_cancelled=True,
                ),
                finished_at=datetime.utcnow(),
            ),
            model=EvalRun,
            row_id=run_id,
            conflict=f"run cannot be cancelled: {run_id}",
        )
        return self.get_verified(run_id)

    cancel = cancel_once


class EvalExecutionClaimRepository:
    """DB source-of-truth single-flight claim for local evaluation execution.

    ``claim_run`` is deliberately typed around ``RunManifest``.  It does not
    expose a generic insert payload, so callers cannot bypass manifest hash,
    identity, or privacy validation performed by the bounded repositories.
    """

    def __init__(self, session: Session):
        self.session = session

    def claim_run(
        self,
        *,
        manifest: RunManifest,
        manifest_hash: str | None = None,
        suite_execution_id: str | None = None,
        created_at: datetime | None = None,
    ) -> EvalRun:
        if not isinstance(manifest, RunManifest):
            raise TypeError("manifest must be a RunManifest")
        _validate_manifest_identity(
            manifest,
            experiment_id=manifest.experiment_id,
            task_case_id=manifest.task_case_id,
            task_case_version=manifest.task_case_version,
            variant_id=manifest.variant_id,
            run_profile=manifest.run_profile,
        )
        manifest_json, verified_hash = _payload_and_hash(
            manifest,
            expected_hash=manifest_hash,
            label="manifest",
            hash_field="manifest_hash",
        )
        _validate_manifest_privacy(manifest_json)

        try:
            # This must be the first SQL statement issued by the clean eval
            # session.  It serializes both workers before either active query.
            _begin_immediate(self.session)
            active_run = self.session.execute(
                select(EvalRun.id)
                .where(EvalRun.lifecycle.in_(("queued", "running")))
                .order_by(EvalRun.created_at, EvalRun.id)
                .limit(1)
            ).scalar_one_or_none()
            if active_run is not None:
                self.session.rollback()
                raise EvaluationBusyError(active_run, "run")

            active_score_set = self.session.execute(
                select(EvalScoreSet.id)
                .where(EvalScoreSet.status.in_(("pending", "running")))
                .order_by(EvalScoreSet.created_at, EvalScoreSet.id)
                .limit(1)
            ).scalar_one_or_none()
            if active_score_set is not None:
                self.session.rollback()
                raise EvaluationBusyError(active_score_set, "score_set")

            started = created_at or datetime.utcnow()
            row = EvalRun(
                id=_new_id("run"),
                experiment_id=manifest.experiment_id,
                suite_execution_id=suite_execution_id,
                task_case_id=manifest.task_case_id,
                task_case_version=manifest.task_case_version,
                variant_id=manifest.variant_id,
                run_profile=manifest.run_profile,
                lifecycle="running",
                outcome=None,
                operational_error_json=None,
                manifest_json=manifest_json,
                manifest_hash=verified_hash,
                candidate_artifact_json=None,
                artifact_hash=None,
                created_at=started,
                started_at=started,
            )
            self.session.add(row)
            self.session.commit()
            return _load(self.session, EvalRun, row.id)
        except EvaluationBusyError:
            raise
        except EvaluationUnavailableError:
            raise
        except Exception:
            self.session.rollback()
            raise

    def claim_score_set(
        self,
        *,
        run_id: str,
        artifact_input_hash: str,
        scorer_id: str | None = None,
        scorer_version: str | None = None,
        scorer_bundle_id: str | None = None,
        scorer_bundle_version: str | None = None,
        scorer_bundle: ScorerBundle | Mapping[str, Any] | None = None,
        scorer_snapshot: Mapping[str, Any] | None = None,
        scorer_definition_hash: str | None = None,
        id: str | None = None,
        created_at: datetime | None = None,
    ) -> EvalScoreSet:
        """Atomically claim a ScoreSet while sharing the run single-flight gate.

        A re-score is only eligible for a finished Run with an immutable
        CandidateArtifact.  The explicit write gate checks active Runs first,
        then active ScoreSets, before inserting the new running row.
        """

        resolved_id, resolved_version = _resolve_scorer_identity(
            scorer_id=scorer_id,
            scorer_version=scorer_version,
            scorer_bundle_id=scorer_bundle_id,
            scorer_bundle_version=scorer_bundle_version,
            scorer_bundle=scorer_bundle,
        )
        snapshot, definition_hash = _freeze_scorer_audit(
            scorer_id=resolved_id,
            scorer_version=resolved_version,
            scorer_bundle=scorer_bundle,
            scorer_snapshot=scorer_snapshot,
            scorer_definition_hash=scorer_definition_hash,
        )
        try:
            _begin_immediate(self.session)
            active_run = self.session.execute(
                select(EvalRun.id)
                .where(EvalRun.lifecycle.in_(("queued", "running")))
                .order_by(EvalRun.created_at, EvalRun.id)
                .limit(1)
            ).scalar_one_or_none()
            if active_run is not None:
                self.session.rollback()
                raise EvaluationBusyError(active_run, "run")
            active_score_set = self.session.execute(
                select(EvalScoreSet.id)
                .where(EvalScoreSet.status.in_(("pending", "running")))
                .order_by(EvalScoreSet.created_at, EvalScoreSet.id)
                .limit(1)
            ).scalar_one_or_none()
            if active_score_set is not None:
                self.session.rollback()
                raise EvaluationBusyError(active_score_set, "score_set")

            run = self.session.execute(
                select(EvalRun)
                .where(EvalRun.id == run_id)
                .execution_options(populate_existing=True)
            ).scalar_one_or_none()
            if run is None:
                self.session.rollback()
                raise RepositoryNotFoundError(f"eval_runs row not found: {run_id}")
            # Verify the frozen parent while still holding the write gate.
            EvalRunRepository(self.session)._verify_hashes(run)
            if run.lifecycle != "finished" or run.outcome is None:
                self.session.rollback()
                raise RepositoryConflictError(
                    f"run is not a finished frozen candidate: {run_id}"
                )
            if run.artifact_hash is None or str(artifact_input_hash) != run.artifact_hash:
                self.session.rollback()
                raise ChecksumMismatchError("score set artifact input hash mismatch")

            started = created_at or datetime.utcnow()
            row = EvalScoreSet(
                id=id or _new_id("score-set"),
                run_id=run_id,
                scorer_id=resolved_id,
                scorer_version=resolved_version,
                scorer_snapshot_json=snapshot,
                scorer_definition_hash=definition_hash,
                artifact_input_hash=str(artifact_input_hash),
                status="running",
                quality_verdict="not_evaluated",
                created_at=started,
                started_at=started,
            )
            self.session.add(row)
            self.session.commit()
            return _load(self.session, EvalScoreSet, row.id)
        except (
            EvaluationBusyError,
            EvaluationUnavailableError,
            RepositoryNotFoundError,
            RepositoryConflictError,
            ChecksumMismatchError,
        ):
            self.session.rollback()
            raise
        except Exception:
            self.session.rollback()
            raise


class EvalScoreSetRepository:
    """Historical ScoreSet rows with one conditional terminal finalization."""

    def __init__(self, session: Session):
        self.session = session
        self.runs = EvalRunRepository(session)

    def create(
        self,
        *,
        run_id: str,
        artifact_input_hash: str,
        scorer_id: str | None = None,
        scorer_version: str | None = None,
        scorer_bundle_id: str | None = None,
        scorer_bundle_version: str | None = None,
        scorer_bundle: ScorerBundle | Mapping[str, Any] | None = None,
        scorer_snapshot: Mapping[str, Any] | None = None,
        scorer_definition_hash: str | None = None,
        id: str | None = None,
        created_at: datetime | None = None,
        auto_commit: bool = True,
    ) -> EvalScoreSet:
        run = self.runs.get_verified(run_id)
        if run.artifact_hash is None:
            raise RepositoryConflictError(f"run has no finalized candidate artifact: {run_id}")
        if str(artifact_input_hash) != run.artifact_hash:
            raise ChecksumMismatchError("score set artifact input hash mismatch")
        resolved_id, resolved_version = _resolve_scorer_identity(
            scorer_id=scorer_id,
            scorer_version=scorer_version,
            scorer_bundle_id=scorer_bundle_id,
            scorer_bundle_version=scorer_bundle_version,
            scorer_bundle=scorer_bundle,
        )
        snapshot, definition_hash = _freeze_scorer_audit(
            scorer_id=resolved_id,
            scorer_version=resolved_version,
            scorer_bundle=scorer_bundle,
            scorer_snapshot=scorer_snapshot,
            scorer_definition_hash=scorer_definition_hash,
        )
        row = EvalScoreSet(
            id=id or _new_id("score-set"),
            run_id=run_id,
            scorer_id=resolved_id,
            scorer_version=resolved_version,
            scorer_snapshot_json=snapshot,
            scorer_definition_hash=definition_hash,
            artifact_input_hash=str(artifact_input_hash),
            status="pending",
            quality_verdict="not_evaluated",
            created_at=created_at or datetime.utcnow(),
        )
        try:
            self.session.add(row)
            if auto_commit:
                self.session.commit()
            else:
                self.session.flush()
        except Exception:
            self.session.rollback()
            raise
        return _load(self.session, EvalScoreSet, row.id)

    def _verify_relation(self, row: EvalScoreSet) -> EvalScoreSet:
        run = self.runs.get_verified(row.run_id)
        _validate_score_set_relation(
            snapshot=row.scorer_snapshot_json,
            definition_hash=row.scorer_definition_hash,
            scorer_id=row.scorer_id,
            scorer_version=row.scorer_version,
            artifact_input_hash=row.artifact_input_hash,
            parent_artifact_hash=run.artifact_hash,
            label=row.id,
        )
        return row

    def get_verified(self, score_set_id: str) -> EvalScoreSet:
        return self._verify_relation(_load(self.session, EvalScoreSet, score_set_id))

    def get(self, score_set_id: str) -> EvalScoreSet:
        return self.get_verified(score_set_id)

    def get_for_compare(self, score_set_id: str) -> EvalScoreSet:
        return self.get_verified(score_set_id)

    def get_for_rescore(self, score_set_id: str) -> EvalScoreSet:
        return self.get_verified(score_set_id)

    load_for_compare = get_for_compare
    load_for_rescore = get_for_rescore

    def list_verified(self, run_id: str) -> list[EvalScoreSet]:
        self.runs.get_verified(run_id)
        rows = self.session.execute(
            select(EvalScoreSet)
            .where(EvalScoreSet.run_id == run_id)
            .order_by(EvalScoreSet.created_at, EvalScoreSet.id)
            .execution_options(populate_existing=True)
        ).scalars().all()
        return [self._verify_relation(row) for row in rows]

    list_for_run = list_verified

    def derive_from_executions(self, score_set_id: str) -> Any:
        """Rebuild the ScoreSet result strictly from persisted rows.

        The frozen task is read from the parent Run's manifest and the frozen
        bundle from the ScoreSet's own snapshot; neither the current Registry nor
        any in-memory draft list participates.
        """

        score_set = self.get_verified(score_set_id)
        run = self.runs.get_verified(score_set.run_id)
        task_snapshot = (run.manifest_json or {}).get("task_snapshot")
        if not isinstance(task_snapshot, Mapping) or not task_snapshot:
            raise ChecksumMismatchError(f"run has no frozen task snapshot: {run.id}")
        try:
            task = TaskCase.from_dict(task_snapshot)
        except Exception as exc:
            raise ChecksumMismatchError(
                f"run frozen task snapshot is invalid: {run.id}"
            ) from exc
        # The nested task identity must match the parent Run row; a snapshot that
        # describes a different task must not be scored under this Run's label.
        # This reads only the frozen manifest and row, never the current Registry.
        if (
            str(task.task_case_id) != str(run.task_case_id)
            or str(task.task_case_version) != str(run.task_case_version)
        ):
            raise ChecksumMismatchError(
                f"run frozen task snapshot does not match the run: {run.id}"
            )
        try:
            bundle = ScorerBundle.from_dict(score_set.scorer_snapshot_json)
        except Exception as exc:
            raise ChecksumMismatchError(
                f"score set frozen snapshot is invalid: {score_set.id}"
            ) from exc
        stored = [
            {
                "component_id": row.scorer_id,
                "component_version": row.scorer_version,
                "status": row.status,
                "input_hash": row.input_hash,
                "output_json": row.output_json,
                "error_code": row.operational_error_code,
                "error_message": row.operational_error_message,
                "latency_ms": row.latency_ms,
                "usage_json": row.usage_json,
            }
            for row in EvalScorerExecutionRepository(self.session).list_verified(
                score_set_id
            )
        ]
        return derive_score_set_from_stored(
            task=task,
            scorer_bundle=bundle,
            stored_executions=stored,
            input_hash=score_set.artifact_input_hash,
        )

    def claim_running(
        self,
        score_set_id: str,
        *,
        expected_status: str = "pending",
    ) -> EvalScoreSet:
        if expected_status != "pending":
            raise InvalidTransitionError("a ScoreSet can only claim running from pending")
        _commit_update(
            self.session,
            update(EvalScoreSet)
            .where(EvalScoreSet.id == score_set_id, EvalScoreSet.status == expected_status)
            .values(status="running", started_at=datetime.utcnow()),
            model=EvalScoreSet,
            row_id=score_set_id,
            conflict=f"score set cannot transition to running: {score_set_id}",
        )
        return self.get_verified(score_set_id)

    claim = claim_running

    def finalize_once(
        self,
        score_set_id: str,
        *,
        status: str,
        quality_verdict: str,
        aggregate_scores: Mapping[str, Any] | None = None,
        findings: Any = None,
        error_code: str | None = None,
        sanitized_message: str | None = None,
        expected_status: str = "running",
    ) -> EvalScoreSet:
        if expected_status != "running":
            raise InvalidTransitionError("ScoreSet finalization requires a running ScoreSet")
        if status not in _SCORE_TERMINAL_STATUSES - {"cancelled"}:
            raise ValueError(f"unsupported terminal ScoreSet status: {status}")
        if quality_verdict not in _QUALITY_VERDICTS:
            raise ValueError(f"unsupported quality verdict: {quality_verdict}")
        error_message = _sanitize_message(sanitized_message)
        if error_code is not None:
            error_code = _error_json(error_code, error_message)["code"]

        # Every terminal write must carry a legal aggregate-score payload.  The
        # narrow failure fallback is allowed to be scoreless (None or an empty
        # mapping), but a non-mapping or out-of-range value is rejected here
        # before either the derivation gate or the write.
        _validate_score_domain(aggregate_scores)

        # A terminal state that claims scoring must be reproducible from the
        # frozen task/bundle and the already-persisted executions.  This closes
        # the "write any score with an execution present" gap: a completed/pass
        # result, or any non-empty aggregate score, is rejected unless it equals
        # what the stored executions actually derive.
        requires_derivation = (
            status == "completed"
            or quality_verdict in {"pass", "fail"}
            or (isinstance(aggregate_scores, Mapping) and bool(aggregate_scores))
        )
        if requires_derivation:
            # Preserve the running/CAS contract: a row that is not running keeps
            # the existing conflict error and is never re-derived.
            current = _load(self.session, EvalScoreSet, score_set_id)
            if current.status != "running":
                raise RepositoryConflictError(
                    f"score set already terminal or missing: {score_set_id}"
                )
            derived = self.derive_from_executions(score_set_id)
            request_scores = (
                None if aggregate_scores is None else dict(as_plain(aggregate_scores))
            )
            request_findings = None if findings is None else list(as_plain(findings))
            if (
                status != derived.status
                or quality_verdict != derived.verdict
                or request_scores != dict(derived.aggregate_scores)
                or request_findings != list(as_plain(derived.findings))
                or (error_code or None) != (derived.error_code or None)
            ):
                raise ChecksumMismatchError(
                    "score set finalization does not match stored executions"
                )

        values: dict[str, Any] = {
            "status": status,
            "quality_verdict": quality_verdict,
            "aggregate_scores_json": as_plain(aggregate_scores)
            if aggregate_scores is not None
            else None,
            "findings_json": as_plain(findings) if findings is not None else None,
            "operational_error_code": error_code,
            "operational_error_message": error_message,
            "finished_at": datetime.utcnow(),
        }
        where = [
            EvalScoreSet.id == score_set_id,
            EvalScoreSet.status == "running",
        ]
        _commit_update(
            self.session,
            update(EvalScoreSet).where(*where).values(**values),
            model=EvalScoreSet,
            row_id=score_set_id,
            conflict=f"score set already terminal or missing: {score_set_id}",
        )
        return self.get_verified(score_set_id)

    finalize = finalize_once

    def cancel_once(
        self,
        score_set_id: str,
        *,
        error_code: str = "cancelled",
        sanitized_message: str | None = None,
        expected_status: str | None = None,
    ) -> EvalScoreSet:
        error = _error_json(
            error_code,
            sanitized_message,
            allow_cancelled=True,
        )
        where = [
            EvalScoreSet.id == score_set_id,
            EvalScoreSet.status.in_(("pending", "running")),
        ]
        if expected_status is not None:
            if expected_status not in {"pending", "running"}:
                raise InvalidTransitionError("ScoreSet is already terminal")
            where[1] = EvalScoreSet.status == expected_status
        try:
            _commit_update(
                self.session,
                update(EvalScoreSet)
                .where(*where)
                .values(
                    status="cancelled",
                    quality_verdict="not_evaluated",
                    operational_error_code=error["code"],
                    operational_error_message=error.get("message"),
                    finished_at=datetime.utcnow(),
                ),
                model=EvalScoreSet,
                row_id=score_set_id,
                conflict=f"score set already terminal or missing: {score_set_id}",
            )
        except RepositoryConflictError:
            row = self.get_verified(score_set_id)
            if row.status in {"cancelled", "completed", "partial", "failed"}:
                return row
            raise
        return self.get_verified(score_set_id)

    cancel = cancel_once


class EvalScorerExecutionRepository:
    """Append/read-only ScorerExecution repository."""

    def __init__(self, session: Session):
        self.session = session
        self.score_sets = EvalScoreSetRepository(session)

    def append(
        self,
        *,
        score_set_id: str,
        scorer_id: str | None = None,
        scorer_version: str | None = None,
        scorer_bundle_id: str | None = None,
        scorer_bundle_version: str | None = None,
        status: str,
        input_hash: str | None = None,
        output: Any = None,
        output_json: Any = None,
        error_code: str | None = None,
        sanitized_message: str | None = None,
        latency_ms: int | None = None,
        usage: Any = None,
        usage_json: Any = None,
        id: str | None = None,
        created_at: datetime | None = None,
    ) -> EvalScorerExecution:
        if status not in _EXECUTION_STATUSES:
            raise ValueError(f"unsupported scorer execution status: {status}")
        score_set = self.score_sets.get_verified(score_set_id)
        if score_set.status != "running":
            raise RepositoryConflictError(
                f"scorer execution requires a running score set: {score_set_id}"
            )
        if input_hash is None:
            input_hash = score_set.artifact_input_hash
        if str(input_hash) != score_set.artifact_input_hash:
            raise ChecksumMismatchError("scorer execution input hash mismatch")
        resolved_id, resolved_version = _resolve_scorer_identity(
            scorer_id=scorer_id,
            scorer_version=scorer_version,
            scorer_bundle_id=scorer_bundle_id,
            scorer_bundle_version=scorer_bundle_version,
        )
        # A new execution's component identity must be declared by the parent
        # ScoreSet's own frozen snapshot; the current Registry is never used to
        # guess a historical component.
        _validate_execution_component(
            scorer_id=resolved_id,
            scorer_version=resolved_version,
            snapshot=score_set.scorer_snapshot_json,
            label=f"score set {score_set_id}",
        )
        if output_json is not None and output is not None:
            raise ValueError("provide output or output_json, not both")
        if usage_json is not None and usage is not None:
            raise ValueError("provide usage or usage_json, not both")
        usage_value = usage_json if usage_json is not None else usage
        if usage_value is not None and not isinstance(usage_value, Mapping):
            raise ValueError("scorer execution usage must be a mapping or null")
        error_message = _sanitize_message(sanitized_message)
        if error_code is not None:
            error_code = _error_json(error_code, error_message)["code"]
        if latency_ms is not None and latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")
        execution_id = id or _new_id("scorer-execution")
        created_timestamp = created_at or datetime.utcnow()
        output_value = as_plain(output_json if output_json is not None else output)
        usage_value = as_plain(usage_value)
        execution_table = EvalScorerExecution.__table__
        score_set_table = EvalScoreSet.__table__
        source = (
            select(
                bindparam(
                    "new_execution_id",
                    execution_id,
                    type_=execution_table.c.id.type,
                ),
                bindparam(
                    "new_score_set_id",
                    score_set_id,
                    type_=execution_table.c.score_set_id.type,
                ),
                bindparam(
                    "new_scorer_id",
                    resolved_id,
                    type_=execution_table.c.scorer_id.type,
                ),
                bindparam(
                    "new_scorer_version",
                    resolved_version,
                    type_=execution_table.c.scorer_version.type,
                ),
                bindparam(
                    "new_status",
                    status,
                    type_=execution_table.c.status.type,
                ),
                bindparam(
                    "new_input_hash",
                    str(input_hash),
                    type_=execution_table.c.input_hash.type,
                ),
                bindparam(
                    "new_output_json",
                    output_value,
                    type_=execution_table.c.output_json.type,
                ),
                bindparam(
                    "new_error_code",
                    error_code,
                    type_=execution_table.c.operational_error_code.type,
                ),
                bindparam(
                    "new_error_message",
                    error_message,
                    type_=execution_table.c.operational_error_message.type,
                ),
                bindparam(
                    "new_latency_ms",
                    latency_ms,
                    type_=execution_table.c.latency_ms.type,
                ),
                bindparam(
                    "new_usage_json",
                    usage_value,
                    type_=execution_table.c.usage_json.type,
                ),
                bindparam(
                    "new_created_at",
                    created_timestamp,
                    type_=execution_table.c.created_at.type,
                ),
            )
            .select_from(score_set_table)
            .where(
                score_set_table.c.id
                == bindparam(
                    "parent_score_set_id",
                    score_set_id,
                    type_=score_set_table.c.id.type,
                ),
                score_set_table.c.status == "running",
                score_set_table.c.artifact_input_hash
                == bindparam(
                    "parent_artifact_input_hash",
                    str(input_hash),
                    type_=score_set_table.c.artifact_input_hash.type,
                ),
            )
        )
        statement = insert(execution_table).from_select(
            [
                "id",
                "score_set_id",
                "scorer_id",
                "scorer_version",
                "status",
                "input_hash",
                "output_json",
                "operational_error_code",
                "operational_error_message",
                "latency_ms",
                "usage_json",
                "created_at",
            ],
            source,
        )
        try:
            result = self.session.execute(statement)
            if result.rowcount != 1:
                self.session.rollback()
                fresh_score_sets = EvalScoreSetRepository(self.session)
                fresh_score_set = fresh_score_sets.get_verified(score_set_id)
                if fresh_score_set.artifact_input_hash != str(input_hash):
                    raise ChecksumMismatchError(
                        "scorer execution input hash mismatch"
                    )
                if fresh_score_set.status != "running":
                    raise RepositoryConflictError(
                        "scorer execution requires a running score set"
                    )
                raise RepositoryConflictError(
                    "scorer execution append lost its parent eligibility race"
                )
            self.session.commit()
        except IntegrityError as exc:
            self.session.rollback()
            raise RepositoryConflictError(
                "duplicate scorer id/version for score set"
            ) from exc
        except Exception:
            self.session.rollback()
            raise
        return self.get_verified(execution_id)

    def append_draft(
        self,
        score_set_id: str,
        draft: ScorerExecutionDraft,
    ) -> EvalScorerExecution:
        """Persist exactly one typed scorer draft as an auditable envelope."""

        if not isinstance(draft, ScorerExecutionDraft):
            raise TypeError("draft must be a ScorerExecutionDraft")
        output_json = None
        if draft.status in {"success", "skipped"}:
            output_json = {
                "result": as_plain(draft.output),
                "findings": as_plain(draft.findings),
            }
        return self.append(
            score_set_id=score_set_id,
            scorer_id=draft.scorer_id,
            scorer_version=draft.scorer_version,
            status=draft.status,
            input_hash=draft.input_hash,
            output_json=output_json,
            error_code=draft.error_code,
            sanitized_message=draft.error_message,
            latency_ms=draft.latency_ms,
            usage_json=as_plain(draft.usage),
        )

    def _verify_relation(self, row: EvalScorerExecution) -> EvalScorerExecution:
        score_set = self.score_sets.get_verified(row.score_set_id)
        if row.input_hash != score_set.artifact_input_hash:
            raise ChecksumMismatchError(
                f"scorer execution input hash mismatch: {row.id}"
            )
        try:
            _validate_execution_component(
                scorer_id=row.scorer_id,
                scorer_version=row.scorer_version,
                snapshot=score_set.scorer_snapshot_json,
                label=row.id,
            )
        except ValueError as exc:
            raise ChecksumMismatchError(str(exc)) from exc
        return row

    def get_verified(self, execution_id: str) -> EvalScorerExecution:
        return self._verify_relation(_load(self.session, EvalScorerExecution, execution_id))

    def get(self, execution_id: str) -> EvalScorerExecution:
        return self.get_verified(execution_id)

    def list_verified(self, score_set_id: str) -> list[EvalScorerExecution]:
        self.score_sets.get_verified(score_set_id)
        rows = self.session.execute(
            select(EvalScorerExecution)
            .where(EvalScorerExecution.score_set_id == score_set_id)
            .order_by(EvalScorerExecution.created_at, EvalScorerExecution.id)
            .execution_options(populate_existing=True)
        ).scalars().all()
        return [self._verify_relation(row) for row in rows]

    list_for_score_set = list_verified


class EvalExecutionControlRepository:
    """Durable cancellation and restart repair under one SQLite write gate."""

    def __init__(self, session: Session):
        self.session = session

    def _get_run(self, run_id: str) -> EvalRun:
        row = self.session.execute(
            select(EvalRun)
            .where(EvalRun.id == run_id)
            .execution_options(populate_existing=True)
        ).scalar_one_or_none()
        if row is None:
            raise RepositoryNotFoundError(f"eval_runs row not found: {run_id}")
        return row

    def is_cancelled(self, run_id: str) -> bool:
        """Read durable intent without retaining an implicit transaction."""

        try:
            value = self.session.execute(
                select(EvalRun.lifecycle).where(EvalRun.id == run_id)
            ).scalar_one_or_none()
        finally:
            self.session.rollback()
        if value is None:
            raise RepositoryNotFoundError(f"eval_runs row not found: {run_id}")
        return value == "cancelled"

    def cancel_run(
        self,
        run_id: str,
        *,
        error_code: str = "cancelled",
        sanitized_message: str | None = None,
    ) -> EvalRun:
        """Cancel active ScoreSets first, then CAS the parent Run.

        The operation is idempotent: a second request returns the same
        terminal row.  Finished Runs retain their outcome and artifact.
        """

        error = _error_json(error_code, sanitized_message, allow_cancelled=True)
        try:
            _begin_immediate(self.session)
            run = self._get_run(run_id)
            run_repository = EvalRunRepository(self.session)
            # Verify the parent before looking at or mutating any child.  A
            # tampered finished/cancelled row is still an integrity failure;
            # a valid terminal row is an idempotent no-op and must not cancel
            # an unrelated historical re-score.
            run_repository._verify_hashes(run)
            if run.lifecycle in {"finished", "cancelled"}:
                self.session.commit()
                return run_repository.get_verified(run_id)
            if run.lifecycle not in {"queued", "running"} or run.outcome is not None:
                self.session.rollback()
                raise RepositoryConflictError(f"run cannot be cancelled: {run_id}")

            active_score_sets = self.session.execute(
                select(EvalScoreSet)
                .where(
                    EvalScoreSet.run_id == run_id,
                    EvalScoreSet.status.in_(("pending", "running")),
                )
                .order_by(EvalScoreSet.created_at, EvalScoreSet.id)
                .execution_options(populate_existing=True)
            ).scalars().all()
            score_repository = EvalScoreSetRepository(self.session)
            # Verify every active child relation while the same write gate is
            # held.  No ScoreSet or Run update is issued before all checks pass.
            for score_set in active_score_sets:
                score_repository._verify_relation(score_set)

            self.session.execute(
                update(EvalScoreSet)
                .where(
                    EvalScoreSet.run_id == run_id,
                    EvalScoreSet.status.in_(("pending", "running")),
                )
                .values(
                    status="cancelled",
                    quality_verdict="not_evaluated",
                    operational_error_code=error["code"],
                    operational_error_message=error.get("message"),
                    finished_at=datetime.utcnow(),
                )
            )
            self.session.execute(
                update(EvalRun)
                .where(
                    EvalRun.id == run_id,
                    EvalRun.lifecycle.in_(("queued", "running")),
                    EvalRun.outcome.is_(None),
                )
                .values(
                    lifecycle="cancelled",
                    outcome=None,
                    operational_error_json=error,
                    finished_at=datetime.utcnow(),
                )
            )
            self.session.commit()
            return run_repository.get_verified(run_id)
        except (RepositoryNotFoundError, ChecksumMismatchError):
            self.session.rollback()
            raise
        except EvaluationUnavailableError:
            raise
        except Exception:
            self.session.rollback()
            raise

    cancel = cancel_run

    def reconcile(self, *, started_before: datetime) -> ReconciliationResult:
        """Terminalize only active rows whose start/create cutoff is older."""

        if not isinstance(started_before, datetime):
            raise TypeError("started_before must be a datetime")
        interrupted = {"code": "process_interrupted", "message": "process interrupted"}
        run_cutoff = or_(
            EvalRun.started_at < started_before,
            and_(EvalRun.started_at.is_(None), EvalRun.created_at < started_before),
        )
        score_cutoff = or_(
            EvalScoreSet.started_at < started_before,
            and_(
                EvalScoreSet.started_at.is_(None),
                EvalScoreSet.created_at < started_before,
            ),
        )
        try:
            _begin_immediate(self.session)
            stale_runs = self.session.execute(
                select(EvalRun)
                .where(
                    EvalRun.lifecycle.in_(('queued', 'running')),
                    run_cutoff,
                )
                .order_by(EvalRun.created_at, EvalRun.id)
                .execution_options(populate_existing=True)
            ).scalars().all()
            stale_score_sets = self.session.execute(
                select(EvalScoreSet)
                .where(
                    EvalScoreSet.status.in_(('pending', 'running')),
                    score_cutoff,
                )
                .order_by(EvalScoreSet.created_at, EvalScoreSet.id)
                .execution_options(populate_existing=True)
            ).scalars().all()

            # Validate every cutoff row and each ScoreSet/Run relation before
            # issuing either bulk update.  A single tampered row therefore
            # rolls back the whole repair transaction without partial repair.
            run_repository = EvalRunRepository(self.session)
            score_repository = EvalScoreSetRepository(self.session)
            for run in stale_runs:
                run_repository._verify_hashes(run)
            for score_set in stale_score_sets:
                score_repository._verify_relation(score_set)

            score_result = self.session.execute(
                update(EvalScoreSet)
                .where(
                    EvalScoreSet.status.in_(("pending", "running")),
                    score_cutoff,
                )
                .values(
                    status="failed",
                    quality_verdict="inconclusive",
                    operational_error_code=interrupted["code"],
                    operational_error_message=interrupted["message"],
                    finished_at=datetime.utcnow(),
                )
            )
            run_result = self.session.execute(
                update(EvalRun)
                .where(
                    EvalRun.lifecycle.in_(("queued", "running")),
                    run_cutoff,
                )
                .values(
                    lifecycle="finished",
                    outcome="system_failed",
                    operational_error_json=interrupted,
                    finished_at=datetime.utcnow(),
                )
            )
            result = ReconciliationResult(
                runs_reconciled=int(run_result.rowcount or 0),
                score_sets_reconciled=int(score_result.rowcount or 0),
                started_before=started_before,
            )
            self.session.commit()
            return result
        except EvaluationUnavailableError:
            raise
        except Exception:
            self.session.rollback()
            raise


class EvalSuiteImportRepository:
    """Parse a complete suite export, then write it in one transaction."""

    def __init__(self, session: Session, *, registry: Any):
        self.session = session
        self.registry = registry
        self.runs = EvalRunRepository(session)
        self.score_sets = EvalScoreSetRepository(session)

    def import_records(self, records: list[Mapping[str, Any]]) -> int:
        if not records:
            raise ValueError("suite import is empty")
        parsed = [self._validate_record(record, index) for index, record in enumerate(records)]
        run_ids = [item[0].id for item in parsed]
        if len(run_ids) != len(set(run_ids)):
            raise ChecksumMismatchError("suite import contains duplicate run ids")
        pairs = [(item[0].task_case_id, item[0].variant_id) for item in parsed]
        if len(pairs) != len(set(pairs)):
            raise ValueError("suite import contains duplicate case/variant pairs")
        score_set_ids = [score_set.id for item in parsed for score_set in item[1]]
        if len(score_set_ids) != len(set(score_set_ids)):
            raise ChecksumMismatchError("suite import contains duplicate score set ids")
        execution_ids = [execution.id for item in parsed for execution in item[2]]
        if len(execution_ids) != len(set(execution_ids)):
            raise ChecksumMismatchError("suite import contains duplicate execution ids")
        execution_identities = [
            (execution.score_set_id, execution.scorer_id, execution.scorer_version)
            for item in parsed
            for execution in item[2]
        ]
        if len(execution_identities) != len(set(execution_identities)):
            raise ValueError(
                "suite import contains duplicate scorer execution identities"
            )
        expected = {
            (case_id, variant_id)
            for case_id in self.registry.task_cases
            for variant_id in self.registry.experiment.variants
        }
        if set(pairs) != expected:
            raise ValueError("suite import does not cover the registry case/variant matrix")
        # Re-run the exact verified-read validators against the in-memory rows
        # before any write.  An imported row therefore cannot succeed a write
        # and then fail its own verified read.
        for run, score_sets, _executions in parsed:
            self.runs._verify_hashes(run)
            for score_set in score_sets:
                _validate_score_set_relation(
                    snapshot=score_set.scorer_snapshot_json,
                    definition_hash=score_set.scorer_definition_hash,
                    scorer_id=score_set.scorer_id,
                    scorer_version=score_set.scorer_version,
                    artifact_input_hash=score_set.artifact_input_hash,
                    parent_artifact_hash=run.artifact_hash,
                    label=f"import record {score_set.id}",
                )
        try:
            for run, score_sets, executions in parsed:
                self.session.add(run)
                for score_set in score_sets:
                    self.session.add(score_set)
                for execution in executions:
                    self.session.add(execution)
            self.session.commit()
        except Exception:
            self.session.rollback()
            raise
        return len(parsed)

    def _validate_record(
        self,
        record: Mapping[str, Any],
        index: int,
    ) -> tuple[EvalRun, list[EvalScoreSet], list[EvalScorerExecution]]:
        if not isinstance(record, Mapping):
            raise ValueError(f"import record {index} must be an object")
        raw_run = record.get("run")
        if not isinstance(raw_run, Mapping):
            raise ValueError(f"import record {index} is missing run")

        # A new suite import must carry the full, explicit Run identity.  The
        # five fields are read from the raw Mapping only; the Registry is never
        # used to back-fill a missing value.
        identity: dict[str, str] = {}
        for field in _RUN_IDENTITY_FIELDS:
            value = raw_run.get(field)
            text = "" if value is None else str(value)
            if not text:
                raise ValueError(f"import record {index} run is missing {field}")
            identity[field] = text

        if identity["experiment_id"] != self.registry.experiment.experiment_id:
            raise ValueError(f"unknown experiment in import record {index}")
        task_case_id = identity["task_case_id"]
        if task_case_id not in self.registry.task_cases:
            raise ValueError(f"unknown task case in import record {index}")
        expected_case_version = str(self.registry.task_cases[task_case_id].task_case_version)
        if identity["task_case_version"] != expected_case_version:
            raise ValueError(f"task case version mismatch in import record {index}")
        if identity["variant_id"] not in self.registry.experiment.variants:
            raise ValueError(f"unknown variant in import record {index}")
        if identity["run_profile"] != str(self.registry.experiment.run_profile):
            raise ValueError(f"run profile mismatch in import record {index}")

        manifest = raw_run.get("manifest") or raw_run.get("manifest_json")
        if not isinstance(manifest, Mapping):
            raise ValueError(f"import record {index} manifest is invalid")
        manifest_json = _plain_mapping(manifest, label=f"import record {index} manifest")
        _validate_run_identity_fields(
            manifest_json,
            identity,
            label=f"import record {index} run",
        )
        expected_manifest_hash = str(raw_run.get("manifest_hash") or "")
        if canonical_hash(manifest_json) != expected_manifest_hash:
            raise ChecksumMismatchError(f"import record {index} manifest hash mismatch")
        # New writes stay inside the same manifest privacy boundary as live
        # create/claim; the stored hash is still checked against the raw blob.
        _validate_manifest_privacy(manifest_json, path=f"import record {index} manifest")

        artifact = raw_run.get("candidate_artifact") or raw_run.get("candidate_artifact_json")
        artifact_hash = raw_run.get("artifact_hash")
        artifact_json: dict[str, Any] | None = None
        if artifact is not None:
            if not isinstance(artifact, Mapping):
                raise ValueError(f"import record {index} artifact is invalid")
            artifact_json = _plain_mapping(artifact, label=f"import record {index} artifact")
            artifact_hash_text = "" if artifact_hash is None else str(artifact_hash)
            if not artifact_hash_text or canonical_hash(artifact_json) != artifact_hash_text:
                raise ChecksumMismatchError(f"import record {index} artifact hash mismatch")
        elif artifact_hash:
            raise ChecksumMismatchError(f"import record {index} artifact hash mismatch")

        # Only terminal Runs are importable.  A finished Run needs an explicit,
        # legal outcome; a cancelled Run must not carry one.  Neither is ever
        # defaulted to success.  The rule is shared with import_historical.
        lifecycle, outcome_text = _validate_terminal_run_state(
            lifecycle=raw_run.get("lifecycle"),
            outcome=raw_run.get("outcome"),
            label=f"import record {index}",
        )

        run = EvalRun(
            id=str(raw_run.get("id") or _new_id("run")),
            experiment_id=identity["experiment_id"],
            suite_execution_id=raw_run.get("suite_execution_id"),
            task_case_id=identity["task_case_id"],
            task_case_version=identity["task_case_version"],
            variant_id=identity["variant_id"],
            run_profile=identity["run_profile"],
            lifecycle=lifecycle,
            outcome=outcome_text,
            operational_error_json=raw_run.get("operational_error_json"),
            manifest_json=manifest_json,
            manifest_hash=expected_manifest_hash,
            candidate_artifact_json=artifact_json,
            artifact_hash=str(artifact_hash) if artifact_hash else None,
        )

        score_sets: list[EvalScoreSet] = []
        score_set_snapshots: dict[str, Mapping[str, Any]] = {}
        for raw_score in record.get("score_sets") or ():
            if not isinstance(raw_score, Mapping):
                raise ValueError(f"import record {index} score set is invalid")
            snapshot = raw_score.get("scorer_snapshot") or raw_score.get("scorer_snapshot_json")
            definition_hash = raw_score.get("scorer_definition_hash")
            if not isinstance(snapshot, Mapping) or not snapshot:
                raise ValueError(f"import record {index} score set is missing scorer snapshot")
            snapshot_json = _plain_mapping(
                snapshot, label=f"import record {index} score snapshot"
            )
            definition_hash_text = "" if definition_hash is None else str(definition_hash)
            if not definition_hash_text:
                raise ValueError(
                    f"import record {index} score set is missing scorer definition hash"
                )
            if hash_without_field(snapshot_json, "definition_hash") != definition_hash_text:
                raise ChecksumMismatchError(f"import record {index} score set hash mismatch")
            raw_scorer_id = raw_score.get("scorer_id")
            raw_scorer_version = raw_score.get("scorer_version")
            scorer_id = "" if raw_scorer_id is None else str(raw_scorer_id)
            scorer_version = "" if raw_scorer_version is None else str(raw_scorer_version)
            if not scorer_id or not scorer_version:
                raise ValueError(f"import record {index} score set is missing scorer identity")
            if scorer_id != str(snapshot_json.get("scorer_id") or ""):
                raise ChecksumMismatchError(
                    f"import record {index} score set scorer id mismatch"
                )
            if scorer_version != str(snapshot_json.get("version") or ""):
                raise ChecksumMismatchError(
                    f"import record {index} score set scorer version mismatch"
                )
            known_scorers = getattr(
                self.registry, "scorers", {self.registry.scorer.version: self.registry.scorer}
            )
            if scorer_version not in known_scorers:
                raise ValueError(f"unknown scorer version in import record {index}")
            status = "" if raw_score.get("status") is None else str(raw_score.get("status"))
            if status not in _SCORE_TERMINAL_STATUSES:
                raise ValueError(f"import record {index} score set is not terminal")
            # A ScoreSet can only score a frozen artifact, and its declared input
            # hash must be explicit and equal to the parent Run's artifact hash.
            if artifact_json is None:
                raise ValueError(
                    f"import record {index} score set requires a frozen artifact"
                )
            raw_artifact_input_hash = raw_score.get("artifact_input_hash")
            artifact_input_hash = (
                "" if raw_artifact_input_hash is None else str(raw_artifact_input_hash)
            )
            if not artifact_input_hash or artifact_input_hash != run.artifact_hash:
                raise ChecksumMismatchError(
                    f"import record {index} score set artifact input hash mismatch"
                )
            score_set_id = str(raw_score.get("id") or _new_id("score-set"))
            score_sets.append(
                EvalScoreSet(
                    id=score_set_id,
                    run_id=run.id,
                    scorer_id=scorer_id,
                    scorer_version=scorer_version,
                    scorer_snapshot_json=snapshot_json,
                    scorer_definition_hash=definition_hash_text,
                    artifact_input_hash=artifact_input_hash,
                    status=status,
                    quality_verdict=str(raw_score.get("quality_verdict") or "not_evaluated"),
                    operational_error_code=raw_score.get("operational_error_code")
                    or raw_score.get("error_code"),
                    operational_error_message=raw_score.get("operational_error_message")
                    or raw_score.get("error_message"),
                    aggregate_scores_json=_first_present(
                        raw_score, "aggregate_scores", "aggregate_scores_json"
                    ),
                    findings_json=_first_present(raw_score, "findings", "findings_json"),
                )
            )
            score_set_snapshots[score_set_id] = snapshot_json

        executions: list[EvalScorerExecution] = []
        for raw_execution in record.get("executions") or ():
            if not isinstance(raw_execution, Mapping):
                raise ValueError(f"import record {index} execution is invalid")
            raw_parent_id = raw_execution.get("score_set_id")
            parent_id = "" if raw_parent_id is None else str(raw_parent_id)
            if not parent_id:
                raise ValueError(
                    f"import record {index} execution is missing score_set_id"
                )
            if parent_id not in score_set_snapshots:
                raise ValueError(
                    f"import record {index} execution parent is not in this record"
                )
            parent = next(
                score_set for score_set in score_sets if score_set.id == parent_id
            )
            status = (
                "" if raw_execution.get("status") is None else str(raw_execution.get("status"))
            )
            if status not in _EXECUTION_STATUSES:
                raise ValueError(f"import record {index} execution has an invalid status")
            raw_input_hash = raw_execution.get("input_hash")
            input_hash = "" if raw_input_hash is None else str(raw_input_hash)
            if not input_hash or input_hash != parent.artifact_input_hash:
                raise ChecksumMismatchError(
                    f"import record {index} execution input hash mismatch"
                )
            raw_execution_scorer_id = raw_execution.get("scorer_id")
            raw_execution_scorer_version = raw_execution.get("scorer_version")
            execution_scorer_id = (
                "" if raw_execution_scorer_id is None else str(raw_execution_scorer_id)
            )
            execution_scorer_version = (
                ""
                if raw_execution_scorer_version is None
                else str(raw_execution_scorer_version)
            )
            # The execution component must be declared by the parent snapshot;
            # the current Registry is never used to guess a historical component.
            _validate_execution_component(
                scorer_id=execution_scorer_id,
                scorer_version=execution_scorer_version,
                snapshot=score_set_snapshots[parent_id],
                label=f"import record {index} execution",
            )
            executions.append(
                EvalScorerExecution(
                    id=str(raw_execution.get("id") or _new_id("scorer")),
                    score_set_id=parent_id,
                    scorer_id=execution_scorer_id,
                    scorer_version=execution_scorer_version,
                    status=status,
                    input_hash=input_hash,
                    output_json=_first_present(
                        raw_execution, "output", "output_json"
                    ),
                    operational_error_code=raw_execution.get("operational_error_code")
                    or raw_execution.get("error_code"),
                    operational_error_message=raw_execution.get("operational_error_message")
                    or raw_execution.get("error_message"),
                    latency_ms=raw_execution.get("latency_ms"),
                    usage_json=_first_present(raw_execution, "usage", "usage_json"),
                )
            )
        return run, score_sets, executions


# Short aliases keep imports ergonomic while the explicit names mirror the
# table names and the approved Task 3 plan.
RunRepository = EvalRunRepository
ScoreSetRepository = EvalScoreSetRepository
ScorerExecutionRepository = EvalScorerExecutionRepository
ConflictError = RepositoryConflictError


__all__ = [
    "ChecksumMismatchError",
    "ConflictError",
    "EvalRunRepository",
    "EvalExecutionClaimRepository",
    "EvalExecutionControlRepository",
    "EvalScoreSetRepository",
    "EvalScorerExecutionRepository",
    "EvalSuiteImportRepository",
    "EvaluationBusyError",
    "EvaluationUnavailableError",
    "freeze_scorer_document",
    "InvalidTransitionError",
    "LearningRunRepositoryError",
    "RepositoryConflictError",
    "RepositoryNotFoundError",
    "ReconciliationResult",
    "RunRepository",
    "ScoreSetRepository",
    "ScorerExecutionRepository",
]
