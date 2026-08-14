import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable, Iterator, Literal, NoReturn


logger = logging.getLogger(__name__)


class ResetInProgress(RuntimeError):
    pass


class ResetRecoveryRequired(ResetInProgress):
    def __init__(self, required_scope: "ResetScope") -> None:
        self.required_scope = required_scope
        super().__init__(f"Retry the incomplete {required_scope} reset")


class DataOperationInProgress(RuntimeError):
    pass


class ResetStageError(RuntimeError):
    stage: Literal["chroma", "sqlite"]
    retryable: bool = True

    def __init__(self, stage: Literal["chroma", "sqlite"]) -> None:
        self.stage = stage
        super().__init__(f"{stage} reset failed")


ResetScope = Literal["learning", "factory"]


_EMPTY_EVAL = {
    "runs": 0,
    "score_sets": 0,
    "scorer_executions": 0,
    "estimated_bytes": 0,
}


@dataclass(frozen=True)
class ResetResult:
    scope: ResetScope
    status: Literal["completed"]
    deleted: dict[str, int]
    deleted_eval: dict[str, int] = field(default_factory=lambda: dict(_EMPTY_EVAL))


class DataLifecycleGate:
    def __init__(self) -> None:
        self._lock = Lock()
        self._active_operations = 0
        self._reset_active = False
        self._recovery_scope: ResetScope | None = None

    @contextmanager
    def shared_operation(self) -> Iterator[None]:
        with self._lock:
            if self._reset_active:
                raise ResetInProgress("A data reset is in progress")
            if self._recovery_scope is not None:
                raise ResetRecoveryRequired(self._recovery_scope)
            self._active_operations += 1
        try:
            yield
        finally:
            with self._lock:
                self._active_operations -= 1

    @contextmanager
    def exclusive_reset(self, scope: ResetScope | None = None) -> Iterator[None]:
        with self._lock:
            if self._reset_active:
                raise ResetInProgress("A data reset is in progress")
            if self._active_operations:
                raise DataOperationInProgress("A data operation is in progress")
            if self._recovery_scope is not None and scope != self._recovery_scope:
                raise ResetRecoveryRequired(self._recovery_scope)
            self._reset_active = True
        try:
            yield
        finally:
            with self._lock:
                self._reset_active = False

    def mark_recovery_required(self, scope: ResetScope) -> None:
        with self._lock:
            self._recovery_scope = scope

    def complete_reset(self, scope: ResetScope) -> None:
        with self._lock:
            if self._recovery_scope == scope:
                self._recovery_scope = None


class ResetCoordinator:
    def __init__(
        self,
        *,
        gate: DataLifecycleGate,
        runtime: Any,
        repository: Any,
        session: Any,
        app_state: Any,
        checkpointer_factory: Callable[[], Any],
    ) -> None:
        self.gate = gate
        self.runtime = runtime
        self.repository = repository
        self.session = session
        self.app_state = app_state
        self.checkpointer_factory = checkpointer_factory

    def summary(self, *, reset_enabled: bool) -> dict[str, int | bool]:
        with self.gate.shared_operation():
            counts = dict(self.repository.count_all())
            eval_counts = dict(self.repository.count_eval()) if hasattr(self.repository, "count_eval") else dict(_EMPTY_EVAL)
            vectors = self.runtime.vector_count()
            has_learning_data = vectors > 0 or any(
                count > 0 for name, count in counts.items() if name != "users"
            )
            return {
                **counts,
                "vectors": vectors,
                "reset_enabled": reset_enabled,
                "has_learning_data": has_learning_data,
                "eval": eval_counts,
            }

    def _raise_sqlite_stage(self, exc: Exception) -> NoReturn:
        try:
            self.session.rollback()
        except Exception:
            logger.exception("SQLite rollback failed during data reset")
        raise ResetStageError("sqlite") from exc

    def _recover_app_bundle(self, *, checkpointer: Any) -> None:
        try:
            self.app_state.retriever = self.runtime.retriever
        except Exception:
            logger.exception("Failed to recover retriever after publish failure")
        try:
            self.app_state.checkpointer = checkpointer
        except Exception:
            logger.exception("Failed to recover checkpointer after publish failure")

    def reset(self, scope: ResetScope) -> ResetResult:
        with self.gate.exclusive_reset(scope):
            try:
                counts = dict(self.repository.count_all())
                eval_counts = (
                    dict(self.repository.count_eval())
                    if hasattr(self.repository, "count_eval")
                    else dict(_EMPTY_EVAL)
                )
            except Exception as exc:
                self._raise_sqlite_stage(exc)

            try:
                vectors = self.runtime.vector_count()
                new_checkpointer = self.checkpointer_factory()
                self.gate.mark_recovery_required(scope)
                self.runtime.reset_empty()
            except Exception as exc:
                raise ResetStageError("chroma") from exc

            try:
                self.app_state.retriever = self.runtime.retriever
                self.app_state.checkpointer = new_checkpointer
            except Exception as exc:
                self._recover_app_bundle(checkpointer=new_checkpointer)
                raise ResetStageError("chroma") from exc

            include_users = scope == "factory"
            try:
                if include_users and hasattr(self.repository, "delete_eval_data"):
                    self.repository.delete_eval_data()
                self.repository.delete_learning_data(include_users=include_users)
                self.session.commit()
            except Exception as exc:
                self._raise_sqlite_stage(exc)

            self.gate.complete_reset(scope)
            deleted = {**counts, "vectors": vectors}
            if not include_users:
                deleted["users"] = 0
            deleted_eval = dict(eval_counts) if include_users else dict(_EMPTY_EVAL)
            return ResetResult(
                scope=scope,
                status="completed",
                deleted=deleted,
                deleted_eval=deleted_eval,
            )
