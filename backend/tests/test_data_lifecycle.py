from threading import Event, Thread

import pytest
from chromadb.errors import NotFoundError

from app import data_lifecycle as data_lifecycle_module
from app.data_lifecycle import (
    DataLifecycleGate,
    DataOperationInProgress,
    ResetCoordinator,
    ResetInProgress,
    ResetResult,
    ResetStageError,
)
from app.rag.runtime import RetrieverRuntime


def test_active_shared_operation_rejects_reset_without_waiting():
    gate = DataLifecycleGate()

    with gate.shared_operation():
        with pytest.raises(DataOperationInProgress):
            with gate.exclusive_reset():
                pass


def test_active_reset_rejects_shared_operation_and_second_reset():
    gate = DataLifecycleGate()

    with gate.exclusive_reset():
        with pytest.raises(ResetInProgress):
            with gate.shared_operation():
                pass
        with pytest.raises(ResetInProgress):
            with gate.exclusive_reset():
                pass


def test_multiple_shared_operations_can_overlap():
    gate = DataLifecycleGate()

    with gate.shared_operation():
        with gate.shared_operation():
            pass


@pytest.mark.parametrize("lease_name", ["shared_operation", "exclusive_reset"])
def test_gate_releases_lease_when_operation_raises(lease_name):
    gate = DataLifecycleGate()

    with pytest.raises(RuntimeError, match="operation failed"):
        with getattr(gate, lease_name)():
            raise RuntimeError("operation failed")

    with gate.shared_operation():
        pass
    with gate.exclusive_reset():
        pass


def test_active_shared_operation_rejects_reset_across_threads_without_waiting():
    gate = DataLifecycleGate()
    shared_entered = Event()
    release_shared = Event()
    reset_finished = Event()
    reset_errors = []

    def hold_shared():
        with gate.shared_operation():
            shared_entered.set()
            release_shared.wait()

    def try_reset():
        try:
            with gate.exclusive_reset():
                pass
        except Exception as exc:
            reset_errors.append(exc)
        finally:
            reset_finished.set()

    holder = Thread(target=hold_shared)
    contender = Thread(target=try_reset)
    holder.start()
    contender_started = False
    try:
        assert shared_entered.wait(1)
        contender.start()
        contender_started = True
        assert reset_finished.wait(1)
        assert len(reset_errors) == 1
        assert isinstance(reset_errors[0], DataOperationInProgress)
    finally:
        release_shared.set()
        holder.join(1)
        if contender_started:
            contender.join(1)
        assert not holder.is_alive()
        assert not contender_started or not contender.is_alive()


def test_active_reset_rejects_shared_and_second_reset_across_threads_without_waiting():
    gate = DataLifecycleGate()
    reset_entered = Event()
    release_reset = Event()
    results = {}

    def hold_reset():
        with gate.exclusive_reset():
            reset_entered.set()
            release_reset.wait()

    def try_lease(name, lease):
        try:
            with lease():
                results[name] = "entered"
        except Exception as exc:
            results[name] = exc

    holder = Thread(target=hold_reset)
    shared = Thread(target=try_lease, args=("shared", gate.shared_operation))
    second_reset = Thread(target=try_lease, args=("reset", gate.exclusive_reset))
    holder.start()
    contenders_started = False
    try:
        assert reset_entered.wait(1)
        shared.start()
        second_reset.start()
        contenders_started = True
        shared.join(1)
        second_reset.join(1)
        assert not shared.is_alive()
        assert not second_reset.is_alive()
        assert isinstance(results["shared"], ResetInProgress)
        assert isinstance(results["reset"], ResetInProgress)
    finally:
        release_reset.set()
        holder.join(1)
        if contenders_started:
            shared.join(1)
            second_reset.join(1)
        assert not holder.is_alive()
        assert not contenders_started or not shared.is_alive()
        assert not contenders_started or not second_reset.is_alive()


class FakeRepository:
    def __init__(self, counts):
        self.counts = counts

    def count_all(self):
        return self.counts


class FakeRuntime:
    def __init__(self, vector_count):
        self._vector_count = vector_count

    def vector_count(self):
        return self._vector_count


@pytest.mark.parametrize(
    ("counts", "vectors", "reset_enabled", "has_learning_data"),
    [
        ({"users": 0, "documents": 0}, 0, False, False),
        ({"users": 2, "documents": 0}, 0, True, False),
        ({"users": 0, "documents": 0}, 3, True, True),
        ({"users": 1, "documents": 2}, 0, False, True),
    ],
)
def test_summary_reports_counts_and_ignores_users_for_learning_data(
    counts,
    vectors,
    reset_enabled,
    has_learning_data,
):
    repository = FakeRepository(counts)
    coordinator = ResetCoordinator(
        gate=DataLifecycleGate(),
        runtime=FakeRuntime(vectors),
        repository=repository,
        session=None,
        app_state=None,
        checkpointer_factory=object,
    )

    result = coordinator.summary(reset_enabled=reset_enabled)

    assert result == {
        **counts,
        "vectors": vectors,
        "reset_enabled": reset_enabled,
        "has_learning_data": has_learning_data,
    }
    assert repository.counts == counts
    assert result is not repository.counts


def test_summary_holds_shared_gate_across_both_counts_to_prevent_mixed_snapshot():
    gate = DataLifecycleGate()
    sql_counted = Event()
    allow_vector_count = Event()
    summary_result = {}

    class BlockingRepository:
        counts = {"users": 1, "documents": 2}

        def count_all(self):
            sql_counted.set()
            return dict(self.counts)

    class BlockingRuntime:
        vectors = 3

        def vector_count(self):
            allow_vector_count.wait()
            return self.vectors

    repository = BlockingRepository()
    runtime = BlockingRuntime()
    coordinator = ResetCoordinator(
        gate=gate,
        runtime=runtime,
        repository=repository,
        session=None,
        app_state=None,
        checkpointer_factory=object,
    )

    thread = Thread(
        target=lambda: summary_result.update(coordinator.summary(reset_enabled=True))
    )
    thread.start()
    try:
        assert sql_counted.wait(1)
        with pytest.raises(DataOperationInProgress):
            with gate.exclusive_reset():
                repository.counts["documents"] = 0
                runtime.vectors = 0
        allow_vector_count.set()
        thread.join(1)
        assert not thread.is_alive()
        assert summary_result["documents"] == 2
        assert summary_result["vectors"] == 3
    finally:
        allow_vector_count.set()
        thread.join(1)
        assert not thread.is_alive()


def test_summary_is_rejected_while_reset_is_active():
    gate = DataLifecycleGate()
    coordinator = ResetCoordinator(
        gate=gate,
        runtime=FakeRuntime(0),
        repository=FakeRepository({"users": 0}),
        session=None,
        app_state=None,
        checkpointer_factory=object,
    )

    with gate.exclusive_reset():
        with pytest.raises(ResetInProgress):
            coordinator.summary(reset_enabled=True)


RESET_COUNTS = {
    "users": 2,
    "documents": 3,
    "messages": 4,
    "source_chunks": 5,
}


class EventRepository:
    def __init__(self, events):
        self.events = events
        self.counts = dict(RESET_COUNTS)

    def count_all(self):
        self.events.append("count")
        return dict(self.counts)

    def delete_learning_data(self, *, include_users):
        self.events.append(f"sqlite:{include_users}")
        for name in self.counts:
            if name != "users" or include_users:
                self.counts[name] = 0


class EventRuntime:
    def __init__(self, events):
        self.events = events
        self.vectors = 7
        self.retriever = object()

    def vector_count(self):
        self.events.append("vectors")
        return self.vectors

    def reset_empty(self):
        self.events.append("chroma")
        self.vectors = 0
        self.retriever = object()
        return self.retriever


class EventAppState:
    def __init__(self, events):
        self.events = events
        self._retriever = object()
        self._checkpointer = object()

    @property
    def retriever(self):
        return self._retriever

    @retriever.setter
    def retriever(self, value):
        self.events.append("app_retriever")
        self._retriever = value

    @property
    def checkpointer(self):
        return self._checkpointer

    @checkpointer.setter
    def checkpointer(self, value):
        self.events.append("app_checkpointer")
        self._checkpointer = value


class EventSession:
    def __init__(self, events):
        self.events = events

    def commit(self):
        self.events.append("commit")

    def rollback(self):
        self.events.append("rollback")


def build_event_coordinator():
    events = []
    repository = EventRepository(events)
    runtime = EventRuntime(events)
    app_state = EventAppState(events)
    session = EventSession(events)
    checkpointer = object()

    def checkpointer_factory():
        events.append("checkpointer_factory")
        return checkpointer

    coordinator = ResetCoordinator(
        gate=DataLifecycleGate(),
        runtime=runtime,
        repository=repository,
        session=session,
        app_state=app_state,
        checkpointer_factory=checkpointer_factory,
    )
    return coordinator, events, repository, runtime, app_state, checkpointer


@pytest.mark.parametrize(
    ("scope", "include_users", "deleted_users"),
    [("learning", False, 0), ("factory", True, RESET_COUNTS["users"])],
)
def test_reset_uses_fixed_stage_order_and_replaces_memory_references(
    scope,
    include_users,
    deleted_users,
):
    coordinator, events, _, runtime, app_state, checkpointer = build_event_coordinator()
    previous_retriever = app_state.retriever
    previous_checkpointer = app_state.checkpointer

    result = coordinator.reset(scope)

    assert events == [
        "count",
        "vectors",
        "checkpointer_factory",
        "chroma",
        "app_retriever",
        "app_checkpointer",
        f"sqlite:{include_users}",
        "commit",
    ]
    assert app_state.retriever is runtime.retriever
    assert app_state.retriever is not previous_retriever
    assert app_state.checkpointer is checkpointer
    assert app_state.checkpointer is not previous_checkpointer
    assert result == ResetResult(
        scope=scope,
        status="completed",
        deleted={**RESET_COUNTS, "users": deleted_users, "vectors": 7},
    )


def test_repeated_reset_is_idempotent():
    coordinator, _, _, _, _, _ = build_event_coordinator()

    first = coordinator.reset("factory")
    second = coordinator.reset("factory")

    assert first.deleted == {**RESET_COUNTS, "vectors": 7}
    assert second.deleted == {key: 0 for key in (*RESET_COUNTS, "vectors")}


class ChromaFailureRuntime(EventRuntime):
    def __init__(self, events, failure_at):
        super().__init__(events)
        self.failure_at = failure_at

    def vector_count(self):
        if self.failure_at == "vector_count":
            raise RuntimeError("vector count failed")
        return super().vector_count()

    def reset_empty(self):
        if self.failure_at == "reset":
            raise RuntimeError("chroma reset failed")
        return super().reset_empty()


class ChromaFailureAppState(EventAppState):
    def __init__(self, events, failure_at):
        super().__init__(events)
        self.failure_at = failure_at
        self.failure_consumed = False

    @EventAppState.retriever.setter
    def retriever(self, value):
        if self.failure_at == "retriever_assignment" and not self.failure_consumed:
            self.failure_consumed = True
            raise RuntimeError("retriever assignment failed")
        self.events.append("app_retriever")
        self._retriever = value

    @EventAppState.checkpointer.setter
    def checkpointer(self, value):
        if self.failure_at == "checkpointer_assignment" and not self.failure_consumed:
            self.failure_consumed = True
            raise RuntimeError("checkpointer assignment failed")
        self.events.append("app_checkpointer")
        self._checkpointer = value


@pytest.mark.parametrize(
    "failure_at",
    [
        "vector_count",
        "reset",
        "retriever_assignment",
        "checkpointer_factory",
        "checkpointer_assignment",
    ],
)
def test_chroma_or_memory_failure_is_wrapped_without_touching_sql_and_releases_gate(
    failure_at,
):
    events = []
    gate = DataLifecycleGate()
    repository = EventRepository(events)
    runtime = ChromaFailureRuntime(events, failure_at)
    app_state = ChromaFailureAppState(events, failure_at)
    session = EventSession(events)
    previous_runtime_retriever = runtime.retriever
    previous_vectors = runtime.vectors
    previous_app_retriever = app_state.retriever
    previous_app_checkpointer = app_state.checkpointer
    built_checkpointers = []

    def checkpointer_factory():
        events.append("checkpointer_factory")
        if failure_at == "checkpointer_factory":
            raise RuntimeError("checkpointer factory failed")
        checkpointer = object()
        built_checkpointers.append(checkpointer)
        return checkpointer

    coordinator = ResetCoordinator(
        gate=gate,
        runtime=runtime,
        repository=repository,
        session=session,
        app_state=app_state,
        checkpointer_factory=checkpointer_factory,
    )

    with pytest.raises(ResetStageError) as caught:
        coordinator.reset("learning")

    assert caught.value.stage == "chroma"
    assert caught.value.retryable is True
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert not any(event.startswith("sqlite:") for event in events)
    assert "commit" not in events
    assert "rollback" not in events
    if failure_at in {"retriever_assignment", "checkpointer_assignment"}:
        assert app_state.retriever is runtime.retriever
        assert app_state.checkpointer is built_checkpointers[-1]

        result = coordinator.reset("learning")

        assert result.status == "completed"
        assert app_state.retriever is runtime.retriever
        assert app_state.checkpointer is built_checkpointers[-1]
        assert events.count("sqlite:False") == 1
        assert events.count("commit") == 1
    else:
        assert app_state.retriever is previous_app_retriever
        assert app_state.checkpointer is previous_app_checkpointer
    if failure_at == "checkpointer_factory":
        assert runtime.retriever is previous_runtime_retriever
        assert runtime.vectors == previous_vectors
        assert events == ["count", "vectors", "checkpointer_factory"]
    with gate.exclusive_reset():
        pass


def test_permanent_publish_recovery_failure_keeps_original_cause_and_skips_sql(monkeypatch):
    original = RuntimeError("initial checkpointer publish failed")
    events = []

    class PermanentFailureAppState(EventAppState):
        def __init__(self, events):
            super().__init__(events)
            self.publish_attempts = 0

        @EventAppState.checkpointer.setter
        def checkpointer(self, value):
            self.publish_attempts += 1
            if self.publish_attempts == 1:
                raise original
            raise RuntimeError("checkpointer recovery failed")

    runtime = EventRuntime(events)
    repository = EventRepository(events)
    session = EventSession(events)
    app_state = PermanentFailureAppState(events)
    previous_checkpointer = app_state.checkpointer
    logged_errors = []
    coordinator = ResetCoordinator(
        gate=DataLifecycleGate(),
        runtime=runtime,
        repository=repository,
        session=session,
        app_state=app_state,
        checkpointer_factory=object,
    )
    monkeypatch.setattr(
        data_lifecycle_module.logger,
        "exception",
        logged_errors.append,
    )

    with pytest.raises(ResetStageError) as caught:
        coordinator.reset("learning")

    assert caught.value.__cause__ is original
    assert app_state.retriever is runtime.retriever
    assert app_state.checkpointer is previous_checkpointer
    assert app_state.publish_attempts == 2
    assert logged_errors == ["Failed to recover checkpointer after publish failure"]
    assert not any(event.startswith("sqlite:") for event in events)
    assert "commit" not in events


class TransactionalRepository(EventRepository):
    def __init__(self, events, failure_at):
        super().__init__(events)
        self.failure_at = failure_at
        self.failed_once = False
        self.pending_include_users = None

    def delete_learning_data(self, *, include_users):
        self.events.append(f"sqlite:{include_users}")
        if self.failure_at == "delete" and not self.failed_once:
            self.failed_once = True
            raise RuntimeError("sqlite delete failed")
        self.pending_include_users = include_users

    def apply_pending_delete(self):
        if self.pending_include_users is None:
            return
        for name in self.counts:
            if name != "users" or self.pending_include_users:
                self.counts[name] = 0
        self.pending_include_users = None


class TransactionalSession(EventSession):
    def __init__(self, events, repository, failure_at):
        super().__init__(events)
        self.repository = repository
        self.failure_at = failure_at
        self.failed_once = False

    def commit(self):
        self.events.append("commit")
        if self.failure_at == "commit" and not self.failed_once:
            self.failed_once = True
            raise RuntimeError("sqlite commit failed")
        self.repository.apply_pending_delete()

    def rollback(self):
        self.events.append("rollback")
        self.repository.pending_include_users = None


@pytest.mark.parametrize("failure_at", ["delete", "commit"])
def test_sqlite_failure_rolls_back_and_same_coordinator_retry_completes(failure_at):
    events = []
    gate = DataLifecycleGate()
    repository = TransactionalRepository(events, failure_at)
    runtime = EventRuntime(events)
    app_state = EventAppState(events)
    session = TransactionalSession(events, repository, failure_at)
    coordinator = ResetCoordinator(
        gate=gate,
        runtime=runtime,
        repository=repository,
        session=session,
        app_state=app_state,
        checkpointer_factory=object,
    )

    with pytest.raises(ResetStageError) as caught:
        coordinator.reset("factory")

    assert caught.value.stage == "sqlite"
    assert caught.value.retryable is True
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert events[-1] == "rollback"

    result = coordinator.reset("factory")

    assert result.status == "completed"
    assert result.deleted == {**RESET_COUNTS, "vectors": 0}
    assert repository.counts == {key: 0 for key in RESET_COUNTS}
    assert events.count("rollback") == 1
    with gate.exclusive_reset():
        pass


@pytest.mark.parametrize("rollback_fails", [False, True])
def test_sql_count_failure_is_sqlite_stage_and_rollback_cannot_mask_it(rollback_fails):
    original = RuntimeError("sql count failed")
    gate = DataLifecycleGate()

    class CountFailureRepository:
        delete_calls = 0

        def count_all(self):
            raise original

        def delete_learning_data(self, *, include_users):
            self.delete_calls += 1

    class CountFailureSession:
        rollback_calls = 0

        def commit(self):
            raise AssertionError("commit must not run")

        def rollback(self):
            self.rollback_calls += 1
            if rollback_fails:
                raise RuntimeError("rollback exploded")

    repository = CountFailureRepository()
    session = CountFailureSession()
    coordinator = ResetCoordinator(
        gate=gate,
        runtime=EventRuntime([]),
        repository=repository,
        session=session,
        app_state=EventAppState([]),
        checkpointer_factory=object,
    )

    with pytest.raises(ResetStageError) as caught:
        coordinator.reset("factory")

    assert caught.value.stage == "sqlite"
    assert caught.value.__cause__ is original
    assert "rollback exploded" not in str(caught.value)
    assert session.rollback_calls == 1
    assert repository.delete_calls == 0
    with gate.exclusive_reset():
        pass


@pytest.mark.parametrize("failure_at", ["delete", "commit"])
def test_rollback_failure_does_not_mask_original_sqlite_write_error(failure_at):
    original = RuntimeError(f"{failure_at} failed")
    gate = DataLifecycleGate()

    class WriteFailureRepository(EventRepository):
        def delete_learning_data(self, *, include_users):
            self.events.append(f"sqlite:{include_users}")
            if failure_at == "delete":
                raise original

    class WriteFailureSession(EventSession):
        def commit(self):
            self.events.append("commit")
            if failure_at == "commit":
                raise original

        def rollback(self):
            self.events.append("rollback")
            raise RuntimeError("rollback exploded")

    events = []
    coordinator = ResetCoordinator(
        gate=gate,
        runtime=EventRuntime(events),
        repository=WriteFailureRepository(events),
        session=WriteFailureSession(events),
        app_state=EventAppState(events),
        checkpointer_factory=object,
    )

    with pytest.raises(ResetStageError) as caught:
        coordinator.reset("factory")

    assert caught.value.stage == "sqlite"
    assert caught.value.__cause__ is original
    assert "rollback exploded" not in str(caught.value)
    assert events[-1] == "rollback"
    with gate.exclusive_reset():
        pass


def test_runtime_builder_failure_keeps_public_refs_and_retry_publishes_consistently():
    class Collection:
        def __init__(self, count):
            self._count = count
            self.valid = True

        def count(self):
            if not self.valid:
                raise NotFoundError("collection handle is invalid")
            return self._count

    class RecreatingClient:
        def __init__(self):
            self.collection = None
            self.created = 0

        def get_or_create_collection(self, name):
            if self.collection is None:
                count = 3 if self.created == 0 else 0
                self.collection = Collection(count)
                self.created += 1
            return self.collection

        def delete_collection(self, name):
            self.collection.valid = False
            self.collection = None

    client = RecreatingClient()
    build_calls = 0

    def builder(collection):
        nonlocal build_calls
        build_calls += 1
        if build_calls == 2:
            raise RuntimeError("builder unavailable")
        return object()

    runtime = RetrieverRuntime(
        client=client,
        collection_name="study_coach_chunks",
        builder=builder,
    )
    events = []
    repository = EventRepository(events)
    session = EventSession(events)
    app_state = EventAppState(events)
    app_state._retriever = runtime.retriever
    previous_collection = runtime.collection
    previous_runtime_retriever = runtime.retriever
    previous_app_checkpointer = app_state.checkpointer
    built_checkpointers = []

    def checkpointer_factory():
        checkpointer = object()
        built_checkpointers.append(checkpointer)
        return checkpointer

    coordinator = ResetCoordinator(
        gate=DataLifecycleGate(),
        runtime=runtime,
        repository=repository,
        session=session,
        app_state=app_state,
        checkpointer_factory=checkpointer_factory,
    )

    with pytest.raises(ResetStageError) as caught:
        coordinator.reset("factory")

    assert caught.value.stage == "chroma"
    assert previous_collection.valid is False
    assert runtime.collection is client.collection
    assert runtime.collection is not previous_collection
    assert runtime.vector_count() == 0
    assert runtime.retriever is previous_runtime_retriever
    assert app_state.retriever is previous_runtime_retriever
    assert app_state.checkpointer is previous_app_checkpointer
    assert len(built_checkpointers) == 1
    assert not any(event.startswith("sqlite:") for event in events)
    assert "commit" not in events
    assert "rollback" not in events

    result = coordinator.reset("factory")

    assert result.status == "completed"
    assert runtime.collection is client.collection
    assert app_state.retriever is runtime.retriever
    assert app_state.checkpointer is built_checkpointers[-1]
    assert len(built_checkpointers) == 2
    assert events.count("sqlite:True") == 1
    assert events.count("commit") == 1
    assert "rollback" not in events


def test_runtime_recreate_failure_is_retryable_through_same_coordinator():
    class Collection:
        def __init__(self, count):
            self._count = count
            self.deleted = False

        def count(self):
            if self.deleted:
                raise NotFoundError("collection was deleted")
            return self._count

    class RecreateFailsOnceClient:
        def __init__(self):
            self.collection = None
            self.initialized = False
            self.recreate_failures = 1
            self.delete_calls = 0

        def get_or_create_collection(self, name):
            if not self.initialized:
                self.initialized = True
                self.collection = Collection(3)
                return self.collection
            if self.collection is None and self.recreate_failures:
                self.recreate_failures -= 1
                raise RuntimeError("recreate temporarily unavailable")
            if self.collection is None:
                self.collection = Collection(0)
            return self.collection

        def delete_collection(self, name):
            self.delete_calls += 1
            if self.collection is None:
                raise NotFoundError("collection is already missing")
            self.collection.deleted = True
            self.collection = None

    client = RecreateFailsOnceClient()
    runtime = RetrieverRuntime(
        client=client,
        collection_name="study_coach_chunks",
        builder=lambda collection: object(),
    )
    events = []
    repository = EventRepository(events)
    session = EventSession(events)
    app_state = EventAppState(events)
    app_state._retriever = runtime.retriever
    previous_collection = runtime.collection
    previous_retriever = runtime.retriever
    previous_checkpointer = app_state.checkpointer
    built_checkpointers = []

    def checkpointer_factory():
        checkpointer = object()
        built_checkpointers.append(checkpointer)
        return checkpointer

    coordinator = ResetCoordinator(
        gate=DataLifecycleGate(),
        runtime=runtime,
        repository=repository,
        session=session,
        app_state=app_state,
        checkpointer_factory=checkpointer_factory,
    )

    with pytest.raises(ResetStageError) as caught:
        coordinator.reset("factory")

    assert caught.value.stage == "chroma"
    assert str(caught.value.__cause__) == "recreate temporarily unavailable"
    with pytest.raises(NotFoundError):
        previous_collection.count()
    assert runtime.collection is previous_collection
    assert runtime.vector_count() == 0
    assert runtime.retriever is previous_retriever
    assert app_state.retriever is previous_retriever
    assert app_state.checkpointer is previous_checkpointer
    assert len(built_checkpointers) == 1
    assert not any(event.startswith("sqlite:") for event in events)
    assert "commit" not in events
    assert "rollback" not in events

    result = coordinator.reset("factory")

    assert result.status == "completed"
    assert client.delete_calls == 2
    assert runtime.collection is client.collection
    assert runtime.vector_count() == 0
    assert app_state.retriever is runtime.retriever
    assert app_state.checkpointer is built_checkpointers[-1]
    assert len(built_checkpointers) == 2
    assert events.count("sqlite:True") == 1
    assert events.count("commit") == 1
    assert "rollback" not in events


def test_create_app_has_data_lifecycle_gate_in_test_mode(monkeypatch):
    from app import main
    from app.db import session as session_module

    monkeypatch.setenv("STUDY_COACH_TEST_MODE", "1")
    monkeypatch.setattr(session_module, "migrate_to_head", lambda: None)

    app = main.create_app()

    assert isinstance(app.state.data_lifecycle_gate, DataLifecycleGate)
