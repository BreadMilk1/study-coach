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
    as_plain,
    canonical_hash,
    canonical_json_bytes,
)


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


def _validate_manifest_identity(
    manifest: RunManifest,
    *,
    experiment_id: str,
    task_case_id: str,
    task_case_version: str,
    variant_id: str,
    run_profile: str,
) -> None:
    expected = {
        "experiment_id": experiment_id,
        "task_case_id": task_case_id,
        "task_case_version": task_case_version,
        "variant_id": variant_id,
        "run_profile": run_profile,
    }
    for field, caller_value in expected.items():
        if str(caller_value) != str(getattr(manifest, field)):
            raise ValueError(f"{field} does not match manifest")


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
        id: str | None = None,
        created_at: datetime | None = None,
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
        row = EvalScoreSet(
            id=id or _new_id("score-set"),
            run_id=run_id,
            scorer_id=resolved_id,
            scorer_version=resolved_version,
            artifact_input_hash=str(artifact_input_hash),
            status="pending",
            quality_verdict="not_evaluated",
            created_at=created_at or datetime.utcnow(),
        )
        try:
            self.session.add(row)
            self.session.commit()
        except Exception:
            self.session.rollback()
            raise
        return _load(self.session, EvalScoreSet, row.id)

    def _verify_relation(self, row: EvalScoreSet) -> EvalScoreSet:
        run = self.runs.get_verified(row.run_id)
        if run.artifact_hash is None or row.artifact_input_hash != run.artifact_hash:
            raise ChecksumMismatchError(f"score set artifact input hash mismatch: {row.id}")
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
    "EvaluationBusyError",
    "EvaluationUnavailableError",
    "InvalidTransitionError",
    "LearningRunRepositoryError",
    "RepositoryConflictError",
    "RepositoryNotFoundError",
    "ReconciliationResult",
    "RunRepository",
    "ScoreSetRepository",
    "ScorerExecutionRepository",
]
