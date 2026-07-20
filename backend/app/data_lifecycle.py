import logging
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable, Iterator, Literal, NoReturn


logger = logging.getLogger(__name__)


class ResetInProgress(RuntimeError):
    pass


class DataOperationInProgress(RuntimeError):
    pass


class ResetStageError(RuntimeError):
    stage: Literal["chroma", "sqlite"]
    retryable: bool = True

    def __init__(self, stage: Literal["chroma", "sqlite"]) -> None:
        self.stage = stage
        super().__init__(f"{stage} reset failed")


ResetScope = Literal["learning", "factory"]


@dataclass(frozen=True)
class ResetResult:
    scope: ResetScope
    status: Literal["completed"]
    deleted: dict[str, int]


class DataLifecycleGate:
    def __init__(self) -> None:
        self._lock = Lock()
        self._active_operations = 0
        self._reset_active = False

    @contextmanager
    def shared_operation(self) -> Iterator[None]:
        with self._lock:
            if self._reset_active:
                raise ResetInProgress("A data reset is in progress")
            self._active_operations += 1
        try:
            yield
        finally:
            with self._lock:
                self._active_operations -= 1

    @contextmanager
    def exclusive_reset(self) -> Iterator[None]:
        with self._lock:
            if self._reset_active:
                raise ResetInProgress("A data reset is in progress")
            if self._active_operations:
                raise DataOperationInProgress("A data operation is in progress")
            self._reset_active = True
        try:
            yield
        finally:
            with self._lock:
                self._reset_active = False


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
            vectors = self.runtime.vector_count()
            has_learning_data = vectors > 0 or any(
                count > 0 for name, count in counts.items() if name != "users"
            )
            return {
                **counts,
                "vectors": vectors,
                "reset_enabled": reset_enabled,
                "has_learning_data": has_learning_data,
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
        with self.gate.exclusive_reset():
            try:
                counts = dict(self.repository.count_all())
            except Exception as exc:
                self._raise_sqlite_stage(exc)

            try:
                vectors = self.runtime.vector_count()
                new_checkpointer = self.checkpointer_factory()
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
                self.repository.delete_learning_data(include_users=include_users)
                self.session.commit()
            except Exception as exc:
                self._raise_sqlite_stage(exc)

            deleted = {**counts, "vectors": vectors}
            if not include_users:
                deleted["users"] = 0
            return ResetResult(scope=scope, status="completed", deleted=deleted)
