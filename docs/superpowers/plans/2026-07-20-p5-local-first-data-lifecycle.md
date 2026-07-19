# P5 Local-first Data Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship an honest local-first Study Coach experience in which each tab must explicitly continue or clear existing learning data, and local users can safely retry either a learning-data reset or a full factory reset.

**Architecture:** The backend owns one instance-wide lifecycle boundary: a shared/exclusive in-process gate coordinates learning routes with a reset coordinator that clears Chroma, replaces retriever and LangGraph process state, then deletes SQLite rows child-first. The frontend treats `/api/data/summary` as capability negotiation, keeps startup/reset decisions in a small lifecycle store, invalidates other tabs through `BroadcastChannel`, and clears browser state only after confirmed backend success.

**Tech Stack:** FastAPI 0.136, SQLAlchemy 2, Chroma 1.5, LangGraph `InMemorySaver`, Pytest, Vue 3, Pinia, TypeScript, native `<dialog>`, `BroadcastChannel`, Vitest, Docker Compose.

---

## Resolved Defaults

These decisions are fixed for P5 implementation:

1. `reset_enabled=false` skips the startup gate entirely and hides Start fresh and Settings Danger Zone. It must not show a Continue-only gate.
2. The destructive API is protected by layered deployment defaults: `STUDY_COACH_LOCAL_MODE` defaults to `0`, Docker Compose explicitly enables it and binds the host port to `127.0.0.1`, and `fly.toml` explicitly keeps it disabled. P5 does not promise request-IP enforcement.
3. SQLite deletion order is derived from the current foreign keys and locked by a test with `PRAGMA foreign_keys=ON`. In particular, `plan_events` and `plan_milestones` are deleted before `topics` because milestones reference topics.
4. Summary and success payloads report every deleted table plus SQL source chunks and Chroma vectors. No deleted table is silently omitted from counts.
5. Chroma and SQLite failure injection is an automated-test requirement. Manual acceptance covers stable success paths and one real retry only when it can be reproduced without test hooks.
6. P5 supports one backend worker. A process-local gate is not presented as a multi-worker safety mechanism.

## File and Responsibility Map

### Backend

- `backend/app/data_lifecycle.py`: lifecycle counts, shared/exclusive gate, reset coordinator, stable reset exceptions.
- `backend/app/rag/runtime.py`: own the Chroma client/collection and build or atomically replace the complete dense + BM25 + reranking stack.
- `backend/app/api/data_routes.py`: strict summary/reset request and response contract; no `default-user` fallback.
- `backend/app/api/deps.py`: strict signed-token dependency and app-state lifecycle dependencies.
- `backend/app/api/routes.py`: attach a request-scoped shared lease to all learning-data routes and make upload temp files unique and self-cleaning.
- `backend/app/db/repositories.py`: instance-wide count and child-first bulk-delete repository; no commit inside the repository.
- `backend/app/main.py`: construct and expose runtime, retriever, lifecycle gate, and checkpointer on `app.state`; the request-scoped coordinator is built with the current SQLAlchemy session.
- `backend/tests/db/test_data_lifecycle_repository.py`: foreign-key-on count/delete integration tests.
- `backend/tests/rag/test_runtime.py`: Chroma collection recreation and complete retriever replacement tests.
- `backend/tests/test_data_lifecycle.py`: gate, ordering, idempotency, and cross-store retry tests.
- `backend/tests/api/test_data_routes.py`: auth, capability, response, conflict, and factory-retry API tests.
- `backend/tests/api/test_routes.py`: learning-route lease and upload temp-file regression tests.

### Frontend

- `frontend/src/lib/api.ts`: strict awaited-token lifecycle API and complete response types.
- `frontend/src/lib/dataLifecycle.ts`: pure startup decision and browser-key clearing policy.
- `frontend/src/lib/dataLifecycleChannel.ts`: typed `BroadcastChannel` adapter and invalidation messages.
- `frontend/src/lib/resetClientState.ts`: clear/refetch all data-backed Pinia stores without copying their state.
- `frontend/src/stores/dataLifecycle.ts`: startup inspection, required choice, reset progress, retry, and external-reset acknowledgement state machine.
- `frontend/src/stores/notifications.ts`: five-second dismissible toast queue.
- `frontend/src/components/StartupDataGate.vue`: non-dismissible startup choice and persistent inspection error.
- `frontend/src/components/ResetConfirmDialog.vue`: cancelable learning/factory confirmation, non-cancelable progress, and factory restart success state.
- `frontend/src/components/ToastHost.vue`: one app-level accessible live region.
- `frontend/src/views/Settings.vue`: remove Google UI and add capability-aware Danger Zone.
- `frontend/src/App.vue`: initialize lifecycle orchestration and host top-layer dialogs/toasts.
- `frontend/src/test/memoryStorage.ts`: deterministic `Storage` test double.
- `frontend/src/**/*.test.ts`: DOM-light Vitest coverage for policy, orchestration, channel, and notifications.

### Product and deployment

- `frontend/index.html`: remove Google Identity Services runtime.
- `docker-compose.yml`: enable local mode, bind backend to loopback, and use the application-supported Chroma variable.
- `.env.example`: document disabled-by-default local mode and `CHROMA_PATH`.
- `fly.toml`: explicitly disable global reset in cloud hosting.
- `README.md`, `docs/ARCHITECTURE.md`, `docs/DEMO.md`, `docs/ROADMAP.md`: keep product claims and verified behavior synchronized with each shipped slice.

---

## P5.0 — Honest local-first product boundary

### Task 1: Remove the incomplete Google product surface

**Files:**
- Modify: `frontend/index.html`
- Modify: `frontend/src/stores/settings.ts`
- Modify: `frontend/src/views/Settings.vue`
- Modify: `frontend/src/App.vue`
- Modify: `frontend/src/locales/en.json`
- Modify: `frontend/src/locales/zh-CN.json`
- Modify: `README.md`
- Modify: `docs/ROADMAP.md`

- [ ] **Step 1: Add a static product-boundary regression check**

Create a temporary verification command in the shell; do not add a new test framework in this task:

```bash
rg -n "accounts.google.com/gsi/client|googleLogin|googleSignOut|Google One Tap|Google Sign-In|GOOGLE_CLIENT_ID" frontend/index.html frontend/src README.md
```

Expected before implementation: at least one match in `frontend/index.html`, `frontend/src/stores/settings.ts`, `frontend/src/views/Settings.vue`, and `README.md`.

- [ ] **Step 2: Remove the GIS runtime and unreachable frontend auth helpers**

Apply these exact removals:

```diff
--- a/frontend/index.html
+++ b/frontend/index.html
@@
-    <script src="https://accounts.google.com/gsi/client" async defer></script>
```

```diff
--- a/frontend/src/stores/settings.ts
+++ b/frontend/src/stores/settings.ts
@@
-export async function googleLogin(credential: string): Promise<void> {
-  const resp = await fetch('/api/auth/google', {
-    method: 'POST',
-    headers: { 'Content-Type': 'application/json' },
-    body: JSON.stringify({ credential }),
-  })
-  if (!resp.ok) throw new Error('Google login failed')
-  const { access_token, tier } = await resp.json()
-  const existingRaw = localStorage.getItem(STORAGE_KEY)
-  const existing = existingRaw ? JSON.parse(existingRaw) : {}
-  existing.accessToken = access_token
-  existing.tier = tier || 'member'
-  localStorage.setItem(STORAGE_KEY, JSON.stringify(existing))
-  _tokenPromise = null
-}
```

In `Settings.vue`, delete GIS imports, global `google` declarations, polling/lifecycle hooks, login/sign-out handlers, and the Account card. Do not delete `accessToken`, anonymous provisioning, backend auth routes, JWT fields, or migrations.

- [ ] **Step 3: Replace product claims with the local-first boundary**

Use this copy consistently in the README and locale files:

```text
Study Coach is a local-first AI learning workspace. No registration is required. Learning data belongs to this Study Coach instance and can be cleared from the startup gate or Settings when local reset is enabled.
```

Update the app footer to:

```vue
<div class="mt-auto text-xs text-fg-dim px-2">P5 · local-first</div>
```

Mark Google OAuth as experimental/deferred in README, and mark only P5.0 complete in `docs/ROADMAP.md` after the build and static check pass.

- [ ] **Step 4: Verify the product surface and build**

Run:

```bash
rg -n "accounts.google.com/gsi/client|googleLogin|googleSignOut|Google One Tap|Google Sign-In|GOOGLE_CLIENT_ID" frontend/index.html frontend/src README.md
cd frontend
pnpm build
```

Expected: `rg` exits 1 with no matches; `pnpm build` exits 0. Backend `/api/auth/google` and `/api/auth/upgrade` remain unchanged.

- [ ] **Step 5: Commit P5.0**

```bash
git add frontend/index.html frontend/src/stores/settings.ts frontend/src/views/Settings.vue frontend/src/App.vue frontend/src/locales/en.json frontend/src/locales/zh-CN.json README.md docs/ROADMAP.md
git commit -m "feat: align study coach with local-first product boundary"
```

---

## P5.1 — Backend data lifecycle

### Task 2: Add complete SQLite lifecycle counts and child-first deletion

**Files:**
- Modify: `backend/app/db/repositories.py`
- Create: `backend/tests/db/test_data_lifecycle_repository.py`

- [ ] **Step 1: Write the foreign-key-on failing integration test**

Add a fixture that creates an isolated SQLite engine and enables foreign keys on every connection:

```python
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import Session

from app.db.models import (
    Base, ChatSession, Citation, Document, Goal, Mastery, Message, Mistake,
    Plan, PlanEvent, PlanMilestone, Question, Topic, User,
)
from app.db.repositories import DataLifecycleRepository


def _session(tmp_path) -> Session:
    engine = create_engine(f"sqlite:///{tmp_path / 'lifecycle.db'}")

    @event.listens_for(engine, "connect")
    def _foreign_keys_on(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    return Session(engine)


def test_delete_learning_data_is_child_first_and_preserves_users(tmp_path):
    session = _session(tmp_path)
    user = User(id="u1", fingerprint="fp1")
    goal = Goal(id="g1", user_id="u1", title="Exam", status="active")
    topic = Topic(id="t1", goal_id="g1", name="ChatGPT", source_chunks=[])
    plan = Plan(id="p1", goal_id="g1", milestones_json=[])
    milestone = PlanMilestone(
        id="pm1", plan_id="p1", topic_id="t1", title="Review",
        done=False, sort_order=0, source="manual",
    )
    plan_event = PlanEvent(
        id="pe1", plan_id="p1", milestone_id="pm1", actor="user",
        action="created", before_json=None, after_json={}, reason=None,
    )
    question = Question(
        id="q1", topic_id="t1", prompt="Prompt?", options_json=["A", "B", "C", "D"],
        answer="A", explanation="Grounded",
    )
    chat = ChatSession(id="s1", user_id="u1")
    message = Message(id="m1", session_id="s1", role="assistant", content="Answer")
    citation = Citation(
        id="c1", message_id="m1", chunk_id="chunk-1", page=1,
        span_start=0, span_end=6,
    )
    document = Document(
        id="d1", user_id="u1", filename="owned.pdf", hash="hash-1", chunks_count=3,
    )
    session.add_all([user, goal, topic, plan, milestone, plan_event, question, chat, message, citation, document])
    session.commit()

    repo = DataLifecycleRepository(session)
    before = repo.count_all()
    repo.delete_learning_data(include_users=False)
    session.commit()

    assert before["plan_events"] == 1
    assert before["plan_milestones"] == 1
    assert before["topics"] == 1
    assert before["source_chunks"] == 3
    assert session.scalar(select(func.count()).select_from(User)) == 1
    for model in (Citation, Message, ChatSession, PlanEvent, PlanMilestone, Plan,
                  Mistake, Mastery, Question, Topic, Goal, Document):
        assert session.scalar(select(func.count()).select_from(model)) == 0
```

- [ ] **Step 2: Run the test and confirm the missing repository failure**

Run:

```bash
cd backend
uv run pytest tests/db/test_data_lifecycle_repository.py -q
```

Expected: collection fails with `ImportError: cannot import name 'DataLifecycleRepository'`.

- [ ] **Step 3: Implement complete counts and exact delete order**

Add this repository at the end of `backend/app/db/repositories.py`; it deliberately does not commit:

```python
class DataLifecycleRepository:
    COUNT_MODELS = {
        "users": User,
        "documents": Document,
        "chat_sessions": ChatSession,
        "messages": Message,
        "citations": Citation,
        "goals": Goal,
        "topics": Topic,
        "plans": Plan,
        "plan_milestones": PlanMilestone,
        "plan_events": PlanEvent,
        "questions": Question,
        "mastery": Mastery,
        "mistakes": Mistake,
    }

    DELETE_ORDER = (
        Citation, Message, ChatSession,
        PlanEvent, PlanMilestone, Plan,
        Mistake, Mastery,
        Question, Topic, Goal,
        Document,
    )

    def __init__(self, session: Session):
        self.session = session

    def count_all(self) -> dict[str, int]:
        counts = {
            name: int(self.session.scalar(select(func.count()).select_from(model)) or 0)
            for name, model in self.COUNT_MODELS.items()
        }
        counts["source_chunks"] = int(
            self.session.scalar(select(func.coalesce(func.sum(Document.chunks_count), 0))) or 0
        )
        return counts

    def delete_learning_data(self, *, include_users: bool) -> None:
        for model in self.DELETE_ORDER:
            self.session.execute(delete(model))
        if include_users:
            self.session.execute(delete(User))
```

Add `delete`, `func`, every referenced model, and `select` to the file imports using the existing import style.

- [ ] **Step 4: Add factory and idempotency assertions**

Add:

```python
def test_factory_delete_removes_users_and_is_idempotent(tmp_path):
    session = _session(tmp_path)
    session.add(User(id="u1", fingerprint="fp1"))
    session.commit()
    repo = DataLifecycleRepository(session)

    repo.delete_learning_data(include_users=True)
    session.commit()
    repo.delete_learning_data(include_users=True)
    session.commit()

    assert repo.count_all() == {
        "users": 0, "documents": 0, "chat_sessions": 0, "messages": 0,
        "citations": 0, "goals": 0, "topics": 0, "plans": 0,
        "plan_milestones": 0, "plan_events": 0, "questions": 0,
        "mastery": 0, "mistakes": 0, "source_chunks": 0,
    }
```

- [ ] **Step 5: Run repository tests and commit**

Run:

```bash
uv run pytest tests/db/test_data_lifecycle_repository.py -q
```

Expected: all lifecycle repository tests pass with foreign keys enabled.

```bash
git add backend/app/db/repositories.py backend/tests/db/test_data_lifecycle_repository.py
git commit -m "feat: add instance data lifecycle repository"
```

### Task 3: Own Chroma and the complete retriever stack in a runtime

**Files:**
- Create: `backend/app/rag/runtime.py`
- Modify: `backend/app/main.py`
- Create: `backend/tests/rag/test_runtime.py`

- [ ] **Step 1: Write the failing runtime tests**

Use a fake Chroma client so the test does not need embeddings or FastEmbed:

```python
from app.rag.runtime import RetrieverRuntime


class FakeCollection:
    def __init__(self, count: int):
        self._count = count

    def count(self) -> int:
        return self._count


class FakeClient:
    def __init__(self):
        self.collection = FakeCollection(4)
        self.deleted = 0

    def get_or_create_collection(self, _name: str):
        return self.collection

    def delete_collection(self, _name: str):
        self.deleted += 1
        self.collection = FakeCollection(0)


def test_reset_empty_recreates_collection_and_replaces_retriever():
    client = FakeClient()
    built = []

    def builder(collection):
        retriever = object()
        built.append((collection, retriever))
        return retriever

    runtime = RetrieverRuntime(client=client, collection_name="study_coach_chunks", builder=builder)
    original = runtime.retriever
    replacement = runtime.reset_empty()

    assert runtime.vector_count() == 0
    assert client.deleted == 1
    assert replacement is runtime.retriever
    assert replacement is not original
    assert built[-1][0] is client.collection
```

- [ ] **Step 2: Run and confirm the missing module failure**

Run:

```bash
cd backend
uv run pytest tests/rag/test_runtime.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'app.rag.runtime'`.

- [ ] **Step 3: Implement the runtime and preserve the existing factory import**

Create the runtime with this public interface:

```python
from collections.abc import Callable
from typing import Any

from chromadb.errors import NotFoundError


class RetrieverRuntime:
    def __init__(
        self,
        *,
        client: Any,
        collection_name: str,
        builder: Callable[[Any], Any],
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.builder = builder
        self.collection = self.client.get_or_create_collection(self.collection_name)
        self.retriever = self.builder(self.collection)

    def vector_count(self) -> int:
        return int(self.collection.count())

    def reset_empty(self):
        try:
            self.client.delete_collection(self.collection_name)
        except NotFoundError:
            pass
        self.collection = self.client.get_or_create_collection(self.collection_name)
        self.retriever = self.builder(self.collection)
        return self.retriever
```

Add a second test in which `delete_collection()` raises `NotFoundError("missing")` once and assert `reset_empty()` still recreates an empty collection. This pins retry behavior when a previous attempt deleted Chroma but failed before recreation.

Move the existing embedder, dense retriever, BM25 hydration, hybrid retriever, and reranker construction into `_build_retriever(collection)` in `runtime.py`. Add `build_default_runtime()` using `CHROMA_PATH` and the fixed collection name. Keep this compatibility wrapper in `main.py` because evaluation modules import it:

```python
def _build_default_retriever() -> RerankingRetriever:
    return build_default_runtime().retriever
```

In `create_app()` construct one runtime and expose both references:

```python
runtime = build_default_runtime()
app.state.retriever_runtime = runtime
app.state.retriever = runtime.retriever
```

- [ ] **Step 4: Verify runtime behavior and main imports**

Run:

```bash
uv run pytest tests/rag/test_runtime.py -q
STUDY_COACH_TEST_MODE=1 uv run python -c "from app.main import _build_default_retriever, create_app"
```

Expected: runtime tests pass and the compatibility factory plus app factory import without constructing local Chroma.

- [ ] **Step 5: Commit**

```bash
git add backend/app/rag/runtime.py backend/app/main.py backend/tests/rag/test_runtime.py
git commit -m "refactor: own retriever lifecycle in runtime"
```

### Task 4: Add the lifecycle gate and reset coordinator

**Files:**
- Create: `backend/app/data_lifecycle.py`
- Modify: `backend/app/main.py`
- Create: `backend/tests/test_data_lifecycle.py`

- [ ] **Step 1: Write gate conflict tests**

```python
import pytest

from app.data_lifecycle import (
    DataLifecycleGate, DataOperationInProgress, ResetInProgress,
)


def test_active_data_operation_rejects_reset():
    gate = DataLifecycleGate()
    with gate.shared_operation():
        with pytest.raises(DataOperationInProgress):
            with gate.exclusive_reset():
                raise AssertionError("exclusive reset must not start")


def test_active_reset_rejects_new_data_operation_and_second_reset():
    gate = DataLifecycleGate()
    with gate.exclusive_reset():
        with pytest.raises(ResetInProgress):
            with gate.shared_operation():
                raise AssertionError("shared operation must not start")
        with pytest.raises(ResetInProgress):
            with gate.exclusive_reset():
                raise AssertionError("second reset must not start")
```

- [ ] **Step 2: Run and confirm missing lifecycle types**

Run:

```bash
cd backend
uv run pytest tests/test_data_lifecycle.py -q
```

Expected: collection fails because `app.data_lifecycle` does not exist.

- [ ] **Step 3: Implement the non-waiting shared/exclusive gate**

```python
from contextlib import contextmanager
from threading import Lock


class ResetInProgress(RuntimeError):
    pass


class DataOperationInProgress(RuntimeError):
    pass


class DataLifecycleGate:
    def __init__(self) -> None:
        self._lock = Lock()
        self._active_operations = 0
        self._reset_active = False

    @contextmanager
    def shared_operation(self):
        with self._lock:
            if self._reset_active:
                raise ResetInProgress
            self._active_operations += 1
        try:
            yield
        finally:
            with self._lock:
                self._active_operations -= 1

    @contextmanager
    def exclusive_reset(self):
        with self._lock:
            if self._reset_active:
                raise ResetInProgress
            if self._active_operations:
                raise DataOperationInProgress
            self._reset_active = True
        try:
            yield
        finally:
            with self._lock:
                self._reset_active = False
```

- [ ] **Step 4: Write coordinator ordering and retry tests**

Use fakes that append stage names and can fail one stage:

```python
from langgraph.checkpoint.memory import InMemorySaver

from app.data_lifecycle import ResetCoordinator


def test_reset_orders_chroma_memory_then_sql_and_replaces_app_state():
    stages = []
    runtime = FakeRuntime(stages)
    repository = FakeRepository(stages)
    state = FakeState(retriever=object(), checkpointer=object())
    session = FakeSession(stages)
    coordinator = ResetCoordinator(
        gate=DataLifecycleGate(), runtime=runtime, repository=repository,
        session=session, app_state=state, checkpointer_factory=InMemorySaver,
    )

    result = coordinator.reset("learning")

    assert stages == ["count", "chroma", "sqlite", "commit"]
    assert state.retriever is runtime.retriever
    assert isinstance(state.checkpointer, InMemorySaver)
    assert result.status == "completed"


def test_sql_failure_after_chroma_is_retryable_and_second_call_completes():
    stages = []
    runtime = FakeRuntime(stages)
    repository = FakeRepository(stages, fail_sql_once=True)
    state = FakeState(retriever=object(), checkpointer=object())
    session = FakeSession(stages)
    coordinator = ResetCoordinator(
        gate=DataLifecycleGate(), runtime=runtime, repository=repository,
        session=session, app_state=state, checkpointer_factory=InMemorySaver,
    )

    with pytest.raises(ResetStageError) as failed:
        coordinator.reset("learning")
    result = coordinator.reset("learning")

    assert failed.value.stage == "sqlite"
    assert failed.value.retryable is True
    assert runtime.reset_calls == 2
    assert result.status == "completed"
```

Define `FakeRuntime`, `FakeRepository`, `FakeSession`, and `FakeState` in the test file. `FakeRepository` exposes `count_all()` and `delete_learning_data()`; `FakeSession` exposes `commit()` and `rollback()`; `FakeRuntime` exposes `vector_count()`, `reset_empty()`, and `retriever`.

- [ ] **Step 5: Implement stable coordinator results and failures**

Implement these public types and order:

```python
from dataclasses import dataclass
from typing import Literal

ResetScope = Literal["learning", "factory"]


@dataclass(frozen=True)
class ResetResult:
    scope: ResetScope
    status: Literal["completed"]
    deleted: dict[str, int]


class ResetStageError(RuntimeError):
    def __init__(self, stage: Literal["chroma", "sqlite"]):
        super().__init__(stage)
        self.stage = stage
        self.retryable = True


class ResetCoordinator:
    def __init__(self, *, gate, runtime, repository, session, app_state, checkpointer_factory):
        self.gate = gate
        self.runtime = runtime
        self.repository = repository
        self.session = session
        self.app_state = app_state
        self.checkpointer_factory = checkpointer_factory

    def summary(self, *, reset_enabled: bool) -> dict[str, int | bool]:
        counts = self.repository.count_all()
        counts["vectors"] = self.runtime.vector_count()
        learning_keys = tuple(key for key in counts if key != "users")
        return {
            "reset_enabled": reset_enabled,
            "has_learning_data": any(int(counts[key]) > 0 for key in learning_keys),
            **counts,
        }

    def reset(self, scope: ResetScope) -> ResetResult:
        with self.gate.exclusive_reset():
            counts = self.repository.count_all()
            counts["vectors"] = self.runtime.vector_count()
            if scope == "learning":
                counts["users"] = 0
            try:
                self.runtime.reset_empty()
                self.app_state.retriever = self.runtime.retriever
                self.app_state.checkpointer = self.checkpointer_factory()
            except Exception as exc:
                raise ResetStageError("chroma") from exc
            try:
                self.repository.delete_learning_data(include_users=scope == "factory")
                self.session.commit()
            except Exception as exc:
                self.session.rollback()
                raise ResetStageError("sqlite") from exc
            return ResetResult(scope=scope, status="completed", deleted=counts)
```

Wire `DataLifecycleGate` into `app.state` in `create_app()`. The API task will construct the request-scoped repository/coordinator so it uses the request SQLAlchemy session.

- [ ] **Step 6: Run lifecycle tests and commit**

Run:

```bash
uv run pytest tests/test_data_lifecycle.py -q
```

Expected: conflict, ordering, Chroma-failure-no-SQL, SQL-retry, idempotency, and scope tests pass.

```bash
git add backend/app/data_lifecycle.py backend/app/main.py backend/tests/test_data_lifecycle.py
git commit -m "feat: coordinate instance-wide data reset"
```

### Task 5: Expose strict capability-aware summary and reset APIs

**Files:**
- Modify: `backend/app/api/deps.py`
- Create: `backend/app/api/data_routes.py`
- Modify: `backend/app/main.py`
- Create: `backend/tests/api/test_data_routes.py`

- [ ] **Step 1: Write strict-auth and disabled-by-default API tests**

```python
def test_summary_rejects_missing_bearer(client):
    response = client.get("/api/data/summary")
    assert response.status_code == 401


def test_summary_reports_reset_disabled_by_default(client, signed_headers):
    response = client.get("/api/data/summary", headers=signed_headers)
    assert response.status_code == 200
    assert response.json()["reset_enabled"] is False


def test_reset_is_forbidden_when_local_mode_is_disabled(client, signed_headers):
    response = client.post(
        "/api/data/reset", headers=signed_headers,
        json={"scope": "learning", "confirmation": "CLEAR_LEARNING_DATA"},
    )
    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "reset_disabled"
```

Use the existing API fixtures and token factory from backend auth tests. Override runtime and session dependencies with isolated fakes; do not touch the developer's local SQLite or Chroma directories.

- [ ] **Step 2: Run and confirm routes are absent**

Run:

```bash
cd backend
uv run pytest tests/api/test_data_routes.py -q
```

Expected: tests fail with 404 for `/api/data/summary` and `/api/data/reset`.

- [ ] **Step 3: Add strict signed-user and app-state dependencies**

Add to `backend/app/api/deps.py`:

```python
async def require_signed_user(
    authorization: str | None = Header(None),
) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, detail="signed bearer token required")
    try:
        return decode_token(authorization[len("Bearer "):]).user_id
    except ValueError as exc:
        raise HTTPException(401, detail=str(exc)) from exc


def get_lifecycle_gate(request: Request):
    return request.app.state.data_lifecycle_gate


def get_retriever_runtime(request: Request):
    return request.app.state.retriever_runtime
```

`require_signed_user` must not call `UserRepository`, because a still-valid token must retry a factory reset after the users table is already empty.

- [ ] **Step 4: Implement the exact API contract**

Create `data_routes.py` with Pydantic models containing all counts:

```python
data_router = APIRouter(prefix="/api/data")

COUNT_FIELDS = (
    "users", "documents", "source_chunks", "vectors", "chat_sessions",
    "messages", "citations", "goals", "topics", "plans",
    "plan_milestones", "plan_events", "questions", "mastery", "mistakes",
)


class DataCounts(BaseModel):
    users: int
    documents: int
    source_chunks: int
    vectors: int
    chat_sessions: int
    messages: int
    citations: int
    goals: int
    topics: int
    plans: int
    plan_milestones: int
    plan_events: int
    questions: int
    mastery: int
    mistakes: int


class DataSummary(DataCounts):
    reset_enabled: bool
    has_learning_data: bool


class ResetRequest(BaseModel):
    scope: Literal["learning", "factory"]
    confirmation: str


class ResetResponse(BaseModel):
    scope: Literal["learning", "factory"]
    status: Literal["completed"]
    deleted: DataCounts
```

Construct `DataLifecycleRepository(session)` and `ResetCoordinator(session=session, app_state=request.app.state, gate=gate, runtime=runtime, checkpointer_factory=InMemorySaver)` inside a dependency for these routes. Include `data_router` once from `create_app()`.

Use this local-mode helper and confirmation map:

```python
def reset_enabled() -> bool:
    return os.environ.get("STUDY_COACH_LOCAL_MODE", "0") == "1"


CONFIRMATIONS = {
    "learning": "CLEAR_LEARNING_DATA",
    "factory": "FACTORY_RESET",
}
```

Both routes depend on `require_signed_user`. Summary remains readable with reset disabled. Reset returns `403 {code: reset_disabled}`, `422 {code: invalid_confirmation}`, `409 {code: reset_in_progress}`, `409 {code: data_operation_in_progress}`, or `500 {code: reset_failed, failed_stage, retryable: true}` using safe fixed text. Include `data_router` in `create_app()`.

- [ ] **Step 5: Add populated, complete-count, conflict, and factory-retry tests**

Add assertions that:

```python
assert set(response.json()["deleted"]) == {
    "users", "documents", "source_chunks", "vectors", "chat_sessions",
    "messages", "citations", "goals", "topics", "plans",
    "plan_milestones", "plan_events", "questions", "mastery", "mistakes",
}
```

Also test that a reset endpoint called twice with the same signed token succeeds after the first factory call deleted users; neither call may use the `default-user` fallback.

- [ ] **Step 6: Run API and regression tests, then commit**

Run:

```bash
uv run pytest tests/api/test_data_routes.py -q
uv run pytest tests/db/test_user_auth.py tests/api/test_auth_routes.py -q
```

Expected: lifecycle API tests pass and existing auth behavior remains green.

```bash
git add backend/app/api/deps.py backend/app/api/data_routes.py backend/app/main.py backend/tests/api/test_data_routes.py
git commit -m "feat: expose strict local data lifecycle API"
```

### Task 6: Lease every learning route and clean exact upload temp files

**Files:**
- Modify: `backend/app/api/deps.py`
- Modify: `backend/app/api/routes.py`
- Modify: `backend/tests/api/test_routes.py`

- [ ] **Step 1: Write request-lifetime lease tests**

Add tests that hold a streaming Chat response open and assert reset receives `data_operation_in_progress`, then hold the exclusive reset lease and assert upload and a representative read route receive `reset_in_progress`:

```python
def test_learning_route_is_rejected_during_reset(client, app, auth_headers):
    gate = app.state.data_lifecycle_gate
    with gate.exclusive_reset():
        upload = client.post(
            "/api/documents", headers=auth_headers,
            files={"file": ("owned.pdf", b"pdf", "application/pdf")},
        )
        documents = client.get("/api/documents", headers=auth_headers)
    assert upload.status_code == 409
    assert documents.status_code == 409
    assert upload.json()["detail"]["code"] == "reset_in_progress"
```

Parameterize the read-side check over current chat, chat messages, goals/plans/events/reorder, documents, mistakes/review, mastery, and stats route families so a new unleased learning endpoint is visible in one test table.

- [ ] **Step 2: Run and confirm routes currently ignore the gate**

Run:

```bash
cd backend
uv run pytest tests/api/test_routes.py -k "lifecycle or reset" -q
```

Expected: the routes execute instead of returning 409.

- [ ] **Step 3: Implement a request-scoped shared dependency and learning router**

Add this generator dependency:

```python
def data_operation_lease(
    gate: Annotated[object, Depends(get_lifecycle_gate)],
):
    try:
        with gate.shared_operation():
            yield
    except ResetInProgress as exc:
        raise HTTPException(
            409, detail={"code": "reset_in_progress", "message": "Data reset is in progress."},
        ) from exc
```

In `routes.py`, keep the existing public `router` for health/model endpoints and create:

```python
learning_router = APIRouter(
    dependencies=[Depends(data_operation_lease, scope="request")],
)
```

Move every learning-data decorator from `@router` to `@learning_router`, leaving health, model ping, and tool check on `router`. Include `learning_router` into `router` after all route declarations:

```python
router.include_router(learning_router)
```

The `scope="request"` dependency must remain alive until `StreamingResponse` finishes, so Chat does not release the lease after returning the response object.

- [ ] **Step 4: Write unique temp-file success, failure, and concurrency tests**

Capture the path passed to `document_processor.process_pdf` and assert:

```python
assert first_path != second_path
assert first_path.name.startswith("sc_")
assert first_path.suffix == ".pdf"
assert not first_path.exists()
assert not second_path.exists()
```

Make `process_pdf` raise `RuntimeError("parse failed")` in a second test and assert the captured exact path no longer exists after the response. Do not scan or delete other `/tmp/sc_*` files.

- [ ] **Step 5: Replace the hash-derived file with exact try/finally cleanup**

Use:

```python
content = await file.read()
file_hash = hashlib.sha256(content).hexdigest()
tmp_path: Path | None = None
try:
    with tempfile.NamedTemporaryFile(prefix="sc_", suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    chunks = document_processor.process_pdf(tmp_path)
    for chunk in chunks:
        chunk["source"] = file.filename or chunk.get("source", "uploaded.pdf")
    if chunks:
        retriever.add_chunks(chunks)
    doc = DocumentRepository(session).create(
        user_id=user_id,
        filename=file.filename or "uploaded.pdf",
        hash_=file_hash,
        chunks_count=len(chunks),
    )
    return {
        "document_id": doc.id,
        "filename": doc.filename,
        "chunks_count": doc.chunks_count,
    }
finally:
    if tmp_path is not None:
        tmp_path.unlink(missing_ok=True)
```

- [ ] **Step 6: Run route tests and commit**

Run:

```bash
uv run pytest tests/api/test_routes.py -q
uv run pytest tests/api/test_data_routes.py -q
```

Expected: all learning routes participate in the gate, Streaming Chat holds its lease, and temp files disappear on success and failure.

```bash
git add backend/app/api/deps.py backend/app/api/routes.py backend/tests/api/test_routes.py
git commit -m "feat: coordinate learning routes with data reset"
```

### Task 7: Lock down local deployment boundaries

**Files:**
- Modify: `docker-compose.yml`
- Modify: `.env.example`
- Modify: `fly.toml`
- Create: `backend/tests/test_deployment_config.py`

- [ ] **Step 1: Write a config regression test**

```python
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[2]


def test_local_reset_is_loopback_only_and_cloud_disabled():
    compose = (ROOT / "docker-compose.yml").read_text()
    assert '"127.0.0.1:8000:8000"' in compose
    assert "STUDY_COACH_LOCAL_MODE: \"1\"" in compose
    assert "CHROMA_PATH: /app/data/chroma" in compose
    assert "CHROMA_PERSIST_DIR" not in compose

    fly = tomllib.loads((ROOT / "fly.toml").read_text())
    assert fly["env"]["STUDY_COACH_LOCAL_MODE"] == "0"
```

- [ ] **Step 2: Run and confirm current configuration fails**

Run:

```bash
cd backend
uv run pytest tests/test_deployment_config.py -q
```

Expected: failure because Compose exposes `8000:8000`, uses `CHROMA_PERSIST_DIR`, or omits local mode.

- [ ] **Step 3: Apply the fixed boundary**

Set the backend Compose entries to:

```yaml
ports:
  - "127.0.0.1:8000:8000"
environment:
  CHROMA_PATH: /app/data/chroma
  STUDY_COACH_LOCAL_MODE: "1"
```

Keep the container's internal server bind at `0.0.0.0`; the host loopback mapping is the delivery boundary. Add to `.env.example`:

```dotenv
STUDY_COACH_LOCAL_MODE=0
CHROMA_PATH=./chroma_data
```

Add under `[env]` in `fly.toml`:

```toml
STUDY_COACH_LOCAL_MODE = "0"
```

- [ ] **Step 4: Verify parsed and rendered Compose configuration**

Run:

```bash
uv run pytest tests/test_deployment_config.py -q
cd ..
docker compose config
```

Expected: test passes; rendered backend port binds `127.0.0.1:8000`; local mode is `1`; Chroma uses `/app/data/chroma`.

- [ ] **Step 5: Commit**

```bash
git add docker-compose.yml .env.example fly.toml backend/tests/test_deployment_config.py
git commit -m "chore: constrain local reset deployment boundary"
```

---

## P5.2 — Startup decision and client lifecycle

### Task 8: Establish Vitest and strict lifecycle API policy

**Files:**
- Modify: `frontend/package.json`
- Modify: `frontend/pnpm-lock.yaml`
- Create: `frontend/vitest.config.ts`
- Modify: `frontend/tsconfig.node.json`
- Modify: `frontend/src/lib/quiz.test.ts`
- Create: `frontend/src/test/memoryStorage.ts`
- Create: `frontend/src/lib/dataLifecycle.test.ts`
- Create: `frontend/src/lib/dataLifecycle.ts`
- Modify: `frontend/src/lib/api.ts`
- Create: `frontend/src/lib/api.lifecycle.test.ts`

- [ ] **Step 1: Add only Vitest and convert the existing test**

Run:

```bash
cd frontend
pnpm add -D vitest
mkdir -p src/test
```

Add the script:

```json
"test": "vitest"
```

Create:

```typescript
import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts'],
  },
})
```

Add `vitest.config.ts` to `tsconfig.node.json`. Replace `node:test` and `node:assert/strict` imports in `quiz.test.ts` with:

```typescript
import { describe, expect, it } from 'vitest'
```

Convert assertions to `expect(value).toEqual(expected)` without changing quiz behavior.

- [ ] **Step 2: Verify the test runner before adding lifecycle code**

Run:

```bash
pnpm test --run
```

Expected: the converted quiz tests pass. No jsdom, Vue Test Utils, Playwright, or notification dependency is added.

- [ ] **Step 3: Write failing startup-policy tests**

Create `memoryStorage.ts` with a Map-backed implementation of all `Storage` methods. Then add:

```typescript
import { describe, expect, it } from 'vitest'
import { memoryStorage } from '../test/memoryStorage'
import {
  clearFactoryBrowserState, clearLearningBrowserState,
  resolveStartupDecision, STARTUP_CHOICE_KEY,
} from './dataLifecycle'


describe('resolveStartupDecision', () => {
  it('skips the gate when reset is disabled even if learning data exists', () => {
    const storage = memoryStorage()
    expect(resolveStartupDecision({ resetEnabled: false, hasLearningData: true }, storage)).toBe('ready')
  })

  it('requires a choice once per tab when reset is enabled and data exists', () => {
    const storage = memoryStorage()
    expect(resolveStartupDecision({ resetEnabled: true, hasLearningData: true }, storage)).toBe('choice_required')
    storage.setItem(STARTUP_CHOICE_KEY, 'continue')
    expect(resolveStartupDecision({ resetEnabled: true, hasLearningData: true }, storage)).toBe('ready')
  })
})
```

Also test that learning clear removes only `study-coach:current-chat-session-id`; factory clear removes every `study-coach:*` key from both storages while preserving unrelated keys.

- [ ] **Step 4: Implement pure storage and startup policy**

```typescript
export const STARTUP_CHOICE_KEY = 'study-coach:startup-choice-made'
export const CHAT_SESSION_KEY = 'study-coach:current-chat-session-id'
const APP_PREFIX = 'study-coach:'

export type StartupDecision = 'ready' | 'choice_required'

export function resolveStartupDecision(
  summary: { resetEnabled: boolean; hasLearningData: boolean },
  session: Storage,
): StartupDecision {
  if (!summary.resetEnabled || !summary.hasLearningData) return 'ready'
  return session.getItem(STARTUP_CHOICE_KEY) ? 'ready' : 'choice_required'
}

export function clearLearningBrowserState(local: Storage): void {
  local.removeItem(CHAT_SESSION_KEY)
}

export function markStartupChoice(session: Storage): void {
  session.setItem(STARTUP_CHOICE_KEY, 'continue')
}

export function clearStartupChoice(session: Storage): void {
  session.removeItem(STARTUP_CHOICE_KEY)
}

export function clearFactoryBrowserState(local: Storage, session: Storage): void {
  for (const storage of [local, session]) {
    const keys = Array.from({ length: storage.length }, (_, index) => storage.key(index))
    for (const key of keys) {
      if (key?.startsWith(APP_PREFIX)) storage.removeItem(key)
    }
  }
}
```

- [ ] **Step 5: Write failing strict API tests**

Mock `fetch` and token provisioning, then assert the lifecycle request waits for the token and sends a bearer header:

```typescript
it('awaits a signed token before summary', async () => {
  vi.stubGlobal('localStorage', memoryStorage({
    'study-coach:settings': '{"accessToken":"signed-token"}',
  }))
  const fetchMock = vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({
      reset_enabled: false,
      has_learning_data: false,
      users: 0,
      documents: 0,
      source_chunks: 0,
      vectors: 0,
      chat_sessions: 0,
      messages: 0,
      citations: 0,
      goals: 0,
      topics: 0,
      plans: 0,
      plan_milestones: 0,
      plan_events: 0,
      questions: 0,
      mastery: 0,
      mistakes: 0,
    }),
  })
  vi.stubGlobal('fetch', fetchMock)
  await getDataSummary()
  expect(fetchMock).toHaveBeenCalledWith('/api/data/summary', {
    headers: { Authorization: 'Bearer signed-token' },
  })
})
```

- [ ] **Step 6: Add complete API types and awaited-token calls**

Export these types from `api.ts`:

```typescript
export type ResetScope = 'learning' | 'factory'

export interface DataCounts {
  users: number
  documents: number
  source_chunks: number
  vectors: number
  chat_sessions: number
  messages: number
  citations: number
  goals: number
  topics: number
  plans: number
  plan_milestones: number
  plan_events: number
  questions: number
  mastery: number
  mistakes: number
}

export interface DataSummaryDto extends DataCounts {
  reset_enabled: boolean
  has_learning_data: boolean
}

export interface ResetResultDto {
  scope: ResetScope
  status: 'completed'
  deleted: DataCounts
}
```

Add the complete error type before the API functions:

```typescript
export class DataLifecycleApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly code: string,
    public readonly failedStage: string | null,
    public readonly retryable: boolean,
    detail: string,
  ) {
    super(detail)
  }

  static async fromResponse(response: Response): Promise<DataLifecycleApiError> {
    const body = await response.json().catch(() => ({})) as Record<string, unknown>
    const raw = (body.detail ?? body) as Record<string, unknown>
    return new DataLifecycleApiError(
      response.status,
      String(raw.code ?? 'data_lifecycle_failed'),
      typeof raw.failed_stage === 'string' ? raw.failed_stage : null,
      raw.retryable === true,
      String(raw.message ?? raw.detail ?? 'Data lifecycle request failed.'),
    )
  }
}
```

Implement lifecycle calls with an awaited token, independent of the module-load provisioning side effect:

```typescript
async function strictAuthHeaders(): Promise<Record<string, string>> {
  const token = await getAccessToken()
  return { Authorization: `Bearer ${token}` }
}

export async function getDataSummary(): Promise<DataSummaryDto> {
  const resp = await fetch('/api/data/summary', { headers: await strictAuthHeaders() })
  if (!resp.ok) throw await DataLifecycleApiError.fromResponse(resp)
  return resp.json() as Promise<DataSummaryDto>
}

const RESET_CONFIRMATION: Record<ResetScope, string> = {
  learning: 'CLEAR_LEARNING_DATA',
  factory: 'FACTORY_RESET',
}

export async function resetData(scope: ResetScope): Promise<ResetResultDto> {
  const resp = await fetch('/api/data/reset', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...(await strictAuthHeaders()) },
    body: JSON.stringify({ scope, confirmation: RESET_CONFIRMATION[scope] }),
  })
  if (!resp.ok) throw await DataLifecycleApiError.fromResponse(resp)
  return resp.json() as Promise<ResetResultDto>
}
```

`DataLifecycleApiError` stores `status`, `code`, `failedStage`, `retryable`, and safe `detail` parsed from the backend body.

- [ ] **Step 7: Run frontend tests/build and commit**

Run:

```bash
pnpm test --run
pnpm build
```

Expected: policy/API/converted quiz tests pass and production build succeeds.

```bash
git add frontend/package.json frontend/pnpm-lock.yaml frontend/vitest.config.ts frontend/tsconfig.node.json frontend/src/lib/quiz.test.ts frontend/src/test/memoryStorage.ts frontend/src/lib/dataLifecycle.ts frontend/src/lib/dataLifecycle.test.ts frontend/src/lib/api.ts frontend/src/lib/api.lifecycle.test.ts
git commit -m "test: establish frontend lifecycle policy harness"
```

### Task 9: Reset all client learning stores from one orchestrator

**Files:**
- Modify: `frontend/src/lib/api.ts`
- Modify: `frontend/src/stores/chat.ts`
- Modify: `frontend/src/stores/quiz.ts`
- Modify: `frontend/src/stores/documents.ts`
- Modify: `frontend/src/stores/plan.ts`
- Modify: `frontend/src/stores/mistakes.ts`
- Modify: `frontend/src/stores/mastery.ts`
- Create: `frontend/src/lib/resetClientState.ts`
- Create: `frontend/src/lib/resetClientState.test.ts`

- [ ] **Step 1: Write the failing orchestration test**

```typescript
import { describe, expect, it, vi } from 'vitest'
import { resetClientLearningState } from './resetClientState'


it('clears chat and quiz then refetches every server-backed store', async () => {
  const calls: string[] = []
  const dependencies = {
    clearChatSession: () => calls.push('chat-key'),
    chat: { resetAfterDataClear: () => calls.push('chat') },
    quiz: { reset: () => calls.push('quiz') },
    documents: { fetch: async () => { calls.push('documents') } },
    plan: { fetch: async () => { calls.push('plan') } },
    mistakes: { fetch: async () => { calls.push('mistakes') } },
    mastery: { fetch: async () => { calls.push('mastery') } },
  }

  await resetClientLearningState(dependencies)

  expect(calls).toEqual(['chat-key', 'chat', 'quiz', 'documents', 'plan', 'mistakes', 'mastery'])
})
```

- [ ] **Step 2: Run and confirm the orchestrator is missing**

Run:

```bash
cd frontend
pnpm test --run src/lib/resetClientState.test.ts
```

Expected: module resolution fails for `./resetClientState`.

- [ ] **Step 3: Add owned reset actions and exact chat-key helper**

Move `CHAT_SESSION_KEY` ownership to `dataLifecycle.ts` and export:

```typescript
export function clearStoredChatSessionId(): void {
  try { localStorage.removeItem(CHAT_SESSION_KEY) }
  catch { /* storage unavailable */ }
}
```

Add this Chat action using its existing state property names:

```typescript
resetAfterDataClear() {
  this.messages = []
  this.streaming = false
  this.trace = []
  this.sessionId = ''
  this.restoring = false
}
```

Keep the existing Quiz `reset()` action. Documents, Plan, Mistakes, and Mastery continue to own their current `fetch()` actions; do not duplicate their state in the lifecycle store.

- [ ] **Step 4: Implement dependency-injected orchestration**

```typescript
export interface ClientLearningStores {
  clearChatSession: () => void
  chat: { resetAfterDataClear: () => void }
  quiz: { reset: () => void }
  documents: { fetch: () => Promise<unknown> }
  plan: { fetch: () => Promise<unknown> }
  mistakes: { fetch: () => Promise<unknown> }
  mastery: { fetch: () => Promise<unknown> }
}

export async function resetClientLearningState(stores: ClientLearningStores): Promise<void> {
  stores.clearChatSession()
  stores.chat.resetAfterDataClear()
  stores.quiz.reset()
  await Promise.all([
    stores.documents.fetch(),
    stores.plan.fetch(),
    stores.mistakes.fetch(),
    stores.mastery.fetch(),
  ])
}
```

The Overview page derives its statistics from these stores and therefore receives no second reset source of truth.

- [ ] **Step 5: Run store tests/build and commit**

Run:

```bash
pnpm test --run src/lib/resetClientState.test.ts
pnpm build
```

Expected: the orchestrator calls every owned reset/refetch path and type-checking passes.

```bash
git add frontend/src/lib/api.ts frontend/src/stores/chat.ts frontend/src/stores/quiz.ts frontend/src/stores/documents.ts frontend/src/stores/plan.ts frontend/src/stores/mistakes.ts frontend/src/stores/mastery.ts frontend/src/lib/resetClientState.ts frontend/src/lib/resetClientState.test.ts
git commit -m "feat: reset all client learning state"
```

### Task 10: Add the lifecycle state machine and cross-tab invalidation

**Files:**
- Create: `frontend/src/stores/dataLifecycle.ts`
- Create: `frontend/src/stores/dataLifecycle.test.ts`
- Create: `frontend/src/lib/dataLifecycleChannel.ts`
- Create: `frontend/src/lib/dataLifecycleChannel.test.ts`

- [ ] **Step 1: Write startup, failure, and success state tests**

```typescript
import { createPinia, setActivePinia } from 'pinia'
import { expect, it, vi } from 'vitest'
import { memoryStorage } from '../test/memoryStorage'
import { useDataLifecycle } from './dataLifecycle'

it('keeps an inspection failure blocking but allows explicit continue', async () => {
  vi.stubGlobal('sessionStorage', memoryStorage())
  setActivePinia(createPinia())
  const store = useDataLifecycle()
  store.initialize({
    summary: async () => { throw new Error('offline') },
    reset: async () => { throw new Error('not called') },
    resetClient: async () => undefined,
    markChoice: () => undefined,
    clearChoice: () => undefined,
    clearFactory: () => undefined,
    broadcast: () => undefined,
    reload: () => undefined,
    pause: async () => undefined,
  })
  await store.inspect()
  expect(store.phase).toBe('inspection_error')
  expect(store.canContinueWithoutClearing).toBe(true)
  expect(store.canStartFresh).toBe(false)
})

it('learning reset unlocks only after backend success and client invalidation', async () => {
  vi.stubGlobal('sessionStorage', memoryStorage())
  const calls: string[] = []
  setActivePinia(createPinia())
  const store = useDataLifecycle()
  store.initialize({
    summary: async () => ({
      reset_enabled: true,
      has_learning_data: true,
      users: 1,
      documents: 1,
      source_chunks: 3,
      vectors: 3,
      chat_sessions: 0,
      messages: 0,
      citations: 0,
      goals: 0,
      topics: 0,
      plans: 0,
      plan_milestones: 0,
      plan_events: 0,
      questions: 0,
      mastery: 0,
      mistakes: 0,
    }),
    reset: async () => ({
      scope: 'learning',
      status: 'completed',
      deleted: {
        users: 0,
        documents: 1,
        source_chunks: 3,
        vectors: 3,
        chat_sessions: 0,
        messages: 0,
        citations: 0,
        goals: 0,
        topics: 0,
        plans: 0,
        plan_milestones: 0,
        plan_events: 0,
        questions: 0,
        mastery: 0,
        mistakes: 0,
      },
    }),
    resetClient: async () => { calls.push('client') },
    markChoice: () => { calls.push('choice') },
    clearChoice: () => undefined,
    clearFactory: () => undefined,
    broadcast: () => { calls.push('broadcast') },
    reload: () => undefined,
    pause: async () => undefined,
  })
  await store.inspect()
  store.requestLearningReset()
  await store.confirmLearningReset()
  expect(calls).toEqual(['client', 'choice', 'broadcast'])
  expect(store.phase).toBe('ready')
})
```

Also test `reset_enabled=false` goes directly to `ready`, backend reset failure remains blocking with retry data, and browser keys are not cleared on failure.

- [ ] **Step 2: Run and confirm the lifecycle store is missing**

Run:

```bash
cd frontend
pnpm test --run src/stores/dataLifecycle.test.ts
```

Expected: module resolution fails for `./dataLifecycle`.

- [ ] **Step 3: Implement the explicit lifecycle phases**

Use this phase union and state boundary:

```typescript
export type LifecyclePhase =
  | 'checking'
  | 'ready'
  | 'choice_required'
  | 'inspection_error'
  | 'confirming_learning'
  | 'confirming_factory'
  | 'resetting'
  | 'reset_error'
  | 'external_reset'
  | 'factory_restarting'
```

The Pinia store owns `phase`, `summary`, `lastResult`, and `error`, and exposes `inspect`, `continueExisting`, `continueWithoutClearing`, `requestLearningReset`, `requestFactoryReset`, `cancelReset`, `confirmLearningReset`, `confirmFactoryReset`, `retryReset`, `handleExternalReset`, and `acknowledgeExternalReset`.

Use one dependency interface so tests and `App.vue` share the same method names:

```typescript
export interface LifecycleDependencies {
  summary: () => Promise<DataSummaryDto>
  reset: (scope: ResetScope) => Promise<ResetResultDto>
  resetClient: () => Promise<void>
  markChoice: () => void
  clearChoice: () => void
  clearFactory: () => void
  broadcast: (scope: ResetScope) => void
  reload: () => void
  pause: (milliseconds: number) => Promise<void>
}
```

Use one module-local dependency slot with an options-style Pinia store so action assignments such as `this.phase = 'ready'` stay type-safe:

```typescript
let lifecycleDependencies: LifecycleDependencies | null = null

function dependencies(): LifecycleDependencies {
  if (lifecycleDependencies === null) throw new Error('Lifecycle store is not initialized')
  return lifecycleDependencies
}

export const useDataLifecycle = defineStore('dataLifecycle', {
  state: () => ({
    phase: 'checking' as LifecyclePhase,
    summary: null as DataSummaryDto | null,
    lastResult: null as ResetResultDto | null,
    error: null as DataLifecycleApiError | Error | null,
    pendingScope: null as ResetScope | null,
    returnPhase: 'ready' as 'ready' | 'choice_required',
  }),
  getters: {
    canContinueWithoutClearing: state => state.phase === 'inspection_error',
    canStartFresh: state => state.phase === 'choice_required' && state.summary?.reset_enabled === true,
  },
  actions: {
    initialize(value: LifecycleDependencies) {
      lifecycleDependencies = value
    },

    async inspect() {
      this.phase = 'checking'
      this.error = null
      try {
        this.summary = await dependencies().summary()
        const decision = resolveStartupDecision(
          {
            resetEnabled: this.summary.reset_enabled,
            hasLearningData: this.summary.has_learning_data,
          },
          sessionStorage,
        )
        this.phase = decision === 'ready' ? 'ready' : 'choice_required'
      } catch (error) {
        this.error = error instanceof Error ? error : new Error('Local data inspection failed.')
        this.phase = 'inspection_error'
      }
    },

    continueExisting() {
      dependencies().markChoice()
      this.phase = 'ready'
    },

    continueWithoutClearing() {
      dependencies().markChoice()
      this.phase = 'ready'
    },

    requestLearningReset() {
      this.returnPhase = this.phase === 'choice_required' ? 'choice_required' : 'ready'
      this.pendingScope = 'learning'
      this.error = null
      this.phase = 'confirming_learning'
    },

    requestFactoryReset() {
      this.returnPhase = 'ready'
      this.pendingScope = 'factory'
      this.error = null
      this.phase = 'confirming_factory'
    },

    cancelReset() {
      if (this.phase === 'resetting' || this.phase === 'factory_restarting') return
      this.pendingScope = null
      this.error = null
      this.phase = this.returnPhase
    },

    async confirmLearningReset() {
      this.phase = 'resetting'
      this.error = null
      try {
        this.lastResult = await dependencies().reset('learning')
        await dependencies().resetClient()
        dependencies().markChoice()
        dependencies().broadcast('learning')
        this.summary = await dependencies().summary()
        this.pendingScope = null
        this.phase = 'ready'
      } catch (error) {
        this.error = error instanceof Error ? error : new Error('Learning reset failed.')
        this.pendingScope = 'learning'
        this.phase = 'reset_error'
      }
    },

    async confirmFactoryReset() {
      this.phase = 'resetting'
      this.error = null
      try {
        this.lastResult = await dependencies().reset('factory')
        dependencies().broadcast('factory')
        this.phase = 'factory_restarting'
        await dependencies().pause(750)
        dependencies().clearFactory()
        dependencies().reload()
      } catch (error) {
        this.error = error instanceof Error ? error : new Error('Factory reset failed.')
        this.pendingScope = 'factory'
        this.phase = 'reset_error'
      }
    },

    async retryReset() {
      if (this.pendingScope === 'factory') await this.confirmFactoryReset()
      else await this.confirmLearningReset()
    },

    async handleExternalReset(scope: ResetScope) {
      if (scope === 'factory') {
        dependencies().clearFactory()
        dependencies().reload()
        return
      }
      dependencies().clearChoice()
      await dependencies().resetClient()
      this.phase = 'external_reset'
    },

    acknowledgeExternalReset() {
      dependencies().markChoice()
      this.phase = 'ready'
    },
  },
})
```

`App.vue` initializes the store before calling `inspect()`.

`inspect()` maps API snake_case into the pure helper shape:

```typescript
const decision = resolveStartupDecision(
  { resetEnabled: summary.reset_enabled, hasLearningData: summary.has_learning_data },
  sessionStorage,
)
this.phase = decision === 'ready' ? 'ready' : 'choice_required'
```

Remote learning reset must set `phase='external_reset'` even when the backend summary is empty. This dedicated one-button acknowledgement prevents stale UI from being silently usable after another tab deleted the data.

- [ ] **Step 4: Write and implement the typed channel adapter**

Test a fake channel factory, then create:

```typescript
export const DATA_LIFECYCLE_CHANNEL = 'study-coach:data-lifecycle'

export interface ResetBroadcast {
  type: 'reset-completed'
  scope: 'learning' | 'factory'
  epoch: number
}

export function createDataLifecycleChannel(
  onReset: (message: ResetBroadcast) => void,
  factory: (name: string) => BroadcastChannel = name => new BroadcastChannel(name),
) {
  const channel = factory(DATA_LIFECYCLE_CHANNEL)
  channel.onmessage = event => {
    const value = event.data as ResetBroadcast
    if (value?.type === 'reset-completed') onReset(value)
  }
  return {
    publish(scope: ResetBroadcast['scope']) {
      channel.postMessage({ type: 'reset-completed', scope, epoch: Date.now() } satisfies ResetBroadcast)
    },
    close() { channel.close() },
  }
}
```

On remote learning reset, clear/refetch client state and enter `external_reset`. On remote factory reset, clear every app browser key and reload. Do not transmit tokens or deleted records through the channel.

- [ ] **Step 5: Run lifecycle/channel tests and commit**

Run:

```bash
pnpm test --run src/stores/dataLifecycle.test.ts src/lib/dataLifecycleChannel.test.ts
pnpm build
```

Expected: startup, retry, external acknowledgement, learning broadcast, and factory reload tests pass.

```bash
git add frontend/src/stores/dataLifecycle.ts frontend/src/stores/dataLifecycle.test.ts frontend/src/lib/dataLifecycleChannel.ts frontend/src/lib/dataLifecycleChannel.test.ts
git commit -m "feat: coordinate local data lifecycle across tabs"
```

---

## P5.3 — Notifications and destructive UI

### Task 11: Add the notification store and accessible toast host

**Files:**
- Create: `frontend/src/stores/notifications.ts`
- Create: `frontend/src/stores/notifications.test.ts`
- Create: `frontend/src/components/ToastHost.vue`
- Modify: `frontend/src/App.vue`

- [ ] **Step 1: Write fake-timer queue tests**

```typescript
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useNotifications } from './notifications'

describe('notifications', () => {
  beforeEach(() => vi.useFakeTimers())

  it('expires a toast after five seconds and supports manual dismissal', () => {
    setActivePinia(createPinia())
    const store = useNotifications()
    const first = store.push({ kind: 'success', message: 'Learning data cleared.' })
    const second = store.push({ kind: 'info', message: 'Settings saved.' })
    store.dismiss(second)
    expect(store.items.map(item => item.id)).toEqual([first])
    vi.advanceTimersByTime(5000)
    expect(store.items).toEqual([])
  })
})
```

- [ ] **Step 2: Run and confirm the store is missing**

Run:

```bash
cd frontend
pnpm test --run src/stores/notifications.test.ts
```

Expected: module resolution fails for `./notifications`.

- [ ] **Step 3: Implement a minimal typed queue**

```typescript
import { defineStore } from 'pinia'
import { ref } from 'vue'

export type NotificationKind = 'success' | 'info'

export interface NotificationItem {
  id: number
  kind: NotificationKind
  message: string
}

export const useNotifications = defineStore('notifications', () => {
  const items = ref<NotificationItem[]>([])
  let nextId = 1

  function dismiss(id: number): void {
    items.value = items.value.filter(item => item.id !== id)
  }

  function push(input: Omit<NotificationItem, 'id'>): number {
    const id = nextId++
    items.value.push({ id, ...input })
    globalThis.setTimeout(() => dismiss(id), 5000)
    return id
  }

  return { items, push, dismiss }
})
```

- [ ] **Step 4: Add one app-level live region**

`ToastHost.vue` renders current items with icon plus text, an explicit dismiss button, `aria-live="polite"`, `aria-atomic="false"`, and existing semantic tokens. Use Vue transition classes that are disabled under `prefers-reduced-motion: reduce`. Mount exactly once in `App.vue`:

```vue
<ToastHost />
```

Persistent reset and connection failures stay inline; do not put error retries into expiring toasts.

- [ ] **Step 5: Run tests/build and commit**

Run:

```bash
pnpm test --run src/stores/notifications.test.ts
pnpm build
```

Expected: timer/dismiss tests pass and Vue type-check/build succeeds.

```bash
git add frontend/src/stores/notifications.ts frontend/src/stores/notifications.test.ts frontend/src/components/ToastHost.vue frontend/src/App.vue
git commit -m "feat: add accessible lifecycle notifications"
```

### Task 12: Add the required startup gate and reset confirmation dialog

**Files:**
- Create: `frontend/src/components/StartupDataGate.vue`
- Create: `frontend/src/components/ResetConfirmDialog.vue`
- Modify: `frontend/src/App.vue`
- Modify: `frontend/src/locales/en.json`
- Modify: `frontend/src/locales/zh-CN.json`

- [ ] **Step 1: Connect lifecycle bootstrap at the app root**

In `App.vue`, instantiate all store dependencies, build the channel once, call lifecycle `inspect()` on mount, and close the channel on unmount. Render the shell only when interaction is allowed, but leave it mounted behind the modal so `<dialog>.showModal()` makes it inert:

```vue
<StartupDataGate
  :phase="lifecycle.phase"
  :summary="lifecycle.summary"
  :error="lifecycle.error"
  @continue="lifecycle.continueExisting()"
  @continue-without-clearing="lifecycle.continueWithoutClearing()"
  @start-fresh="lifecycle.requestLearningReset()"
  @retry="lifecycle.inspect()"
  @acknowledge-external="lifecycle.acknowledgeExternalReset()"
/>
<ResetConfirmDialog
  :phase="lifecycle.phase"
  :scope="lifecycle.pendingScope"
  :summary="lifecycle.summary"
  :error="lifecycle.error"
  @cancel="lifecycle.cancelReset()"
  @confirm-learning="lifecycle.confirmLearningReset()"
  @confirm-factory="lifecycle.confirmFactoryReset()"
  @retry="lifecycle.retryReset()"
/>
```

- [ ] **Step 2: Implement non-dismissible startup dialog behavior**

`StartupDataGate.vue` uses `<Teleport to="body">` and a native dialog ref. Whenever phase is `checking`, `choice_required`, `inspection_error`, or `external_reset`, call `showModal()` if not already open. The checking state shows a non-interactive “Inspecting local data…” status so the app cannot be used before capability and data presence are known. Prevent cancellation:

```typescript
function preventCancel(event: Event): void {
  event.preventDefault()
}
```

```vue
<dialog ref="dialog" @cancel="preventCancel" @click.self.prevent>
```

Behavior by phase:

- `choice_required`: autofocus Continue; show Start fresh only when `summary.reset_enabled` is true.
- `inspection_error`: show Retry and Continue without clearing; never show Start fresh with unknown counts.
- `external_reset`: explain another tab cleared learning data and show one Continue button.
- `reset_enabled=false`: lifecycle is already `ready`, so this component does not open.

- [ ] **Step 3: Implement cancelable confirmation and locked progress**

`ResetConfirmDialog.vue` is cancelable by button and Esc only during `confirming_learning` or `confirming_factory`. During `resetting`, prevent `cancel`, disable actions, and show progress. Factory confirmation requires exact `RESET` input before enabling submit:

```typescript
const factoryText = ref('')
const factoryConfirmed = computed(() => factoryText.value === 'RESET')
```

After learning success, close and show a success toast with counts. After factory success, render `Reset complete, restarting…` for about 750 ms, then clear browser keys and reload. Do not promise a factory-reset toast because reload removes it.

- [ ] **Step 4: Add precise bilingual copy**

The dialog must state:

```text
Clear learning data deletes imported document records and embeddings, Chat history, Quiz and review data, goals, plans, milestones, mistakes, mastery, and current in-memory learning state. Model, provider, API, language, and interface settings remain.

Factory reset deletes all learning data, local model and interface settings, cached model capability, local identity, and backend local user rows. This cannot be undone.
```

Use equivalent Simplified Chinese copy in `zh-CN.json`; preserve English product/API names.

- [ ] **Step 5: Build and perform the modal behavior check**

Run:

```bash
cd frontend
pnpm test --run
pnpm build
```

Expected: tests and build pass. In a browser with populated data: background is inert, Esc/backdrop cannot bypass startup, Continue receives initial focus, inspection failure offers Retry and Continue without clearing, and deletion progress cannot be canceled.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/StartupDataGate.vue frontend/src/components/ResetConfirmDialog.vue frontend/src/App.vue frontend/src/locales/en.json frontend/src/locales/zh-CN.json
git commit -m "feat: require a startup data lifecycle choice"
```

### Task 13: Add the capability-aware Settings Danger Zone

**Files:**
- Modify: `frontend/src/views/Settings.vue`
- Modify: `frontend/src/App.vue`
- Modify: `frontend/src/stores/dataLifecycle.ts`
- Modify: `frontend/src/locales/en.json`
- Modify: `frontend/src/locales/zh-CN.json`

- [ ] **Step 1: Reuse the root confirmation state rather than creating page-local reset state**

The Settings buttons call lifecycle-store actions:

```vue
<button type="button" @click="lifecycle.requestLearningReset()">
  {{ $t('settings.danger.clearLearning') }}
</button>
<button type="button" @click="lifecycle.requestFactoryReset()">
  {{ $t('settings.danger.factoryReset') }}
</button>
```

The root `ResetConfirmDialog` remains the sole confirmation/progress surface, so navigation cannot duplicate or lose reset state.

- [ ] **Step 2: Hide the entire section when capability is disabled**

```vue
<section v-if="lifecycle.summary?.reset_enabled" class="danger-zone">
```

On Settings mount, refresh summary so counts are current. Display documents, SQL chunks, actual vectors, Chat sessions/messages, questions, mistakes, mastery, plans/milestones/events, and local users. Label chunk/vector differences rather than combining them.

- [ ] **Step 3: Verify success and failure semantics**

Learning reset success must:

```typescript
await resetClientLearningState(clientStores)
notifications.push({ kind: 'success', message: 'Learning data cleared.' })
channel.publish('learning')
```

Factory reset must clear browser keys only after `resetData('factory')` resolves, publish `factory`, show the 750 ms restart state, then reload. Reset errors stay inline with failed stage and Retry. All buttons are disabled while `phase === 'resetting'`.

- [ ] **Step 4: Run tests/build and commit**

Run:

```bash
cd frontend
pnpm test --run
pnpm build
```

Expected: reset-disabled UI is absent; enabled UI shows complete counts; both scopes use the shared confirmation state; build passes.

```bash
git add frontend/src/views/Settings.vue frontend/src/App.vue frontend/src/stores/dataLifecycle.ts frontend/src/locales/en.json frontend/src/locales/zh-CN.json
git commit -m "feat: add local data danger zone"
```

---

## P5.4 — Closure, evidence, and documentation

### Task 14: Synchronize architecture, demo, roadmap, and full verification

**Files:**
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/DEMO.md`
- Modify: `docs/ROADMAP.md`
- Modify: `README.md`

- [ ] **Step 1: Update architecture with the shipped boundary**

Add an ADR covering:

```text
P5 is a single-user, local-first instance boundary. Summary is readable with a strict signed bearer token; global reset is disabled unless STUDY_COACH_LOCAL_MODE=1. Docker Compose enables it only behind 127.0.0.1 host binding, while Fly keeps it disabled. Reset order is Chroma, complete in-memory retriever/checkpointer replacement, then one child-first SQLite transaction. This is idempotent recovery, not a cross-store transaction. P5 supports one backend worker and does not enforce request source IP.
```

Document the complete count fields, both scopes, stable 403/409/500 error codes, strict no-fallback auth, and the future requirement to replace global reset when per-user Chroma ownership is introduced.

- [ ] **Step 2: Replace the demo path with stable manual acceptance**

Document this exact manual path:

1. Start Ollama, backend, and frontend through the supported local configuration.
2. Import two user-owned PDFs and create Chat, Quiz, Plan, mistake, and mastery state.
3. Open a new tab; verify startup blocks every other interaction.
4. Choose Continue; verify all existing data remains.
5. Refresh that tab; verify the gate does not reopen in the same tab session.
6. Open another new tab; choose Start fresh and confirm learning reset.
7. Verify Library, Chat, Quiz, Plan, milestones/events, mistakes, mastery, Chroma vectors, retriever cache, and checkpoint state are empty.
8. Verify provider, model, Base URL, API key, language, and debug settings remain.
9. Re-import a user-owned PDF, run Factory reset, observe the restart state, and verify a new anonymous first-run state.
10. If a naturally reproducible transient failure occurs, retry the same reset and verify completion; do not add a production failure-injection control.

Modal focus, Esc, backdrop, keyboard reachability, inert background, cross-tab learning acknowledgement, and cross-tab factory reload are explicit browser checks.

- [ ] **Step 3: Run focused backend verification**

Run:

```bash
cd backend
uv run pytest tests/db/test_data_lifecycle_repository.py -q
uv run pytest tests/rag/test_runtime.py -q
uv run pytest tests/test_data_lifecycle.py -q
uv run pytest tests/api/test_data_routes.py -q
uv run pytest tests/api/test_routes.py -q
```

Expected: all focused backend tests pass with no local developer data used.

- [ ] **Step 4: Run full automated verification**

Run:

```bash
cd backend
uv run pytest -q
cd ../frontend
pnpm test --run
pnpm build
cd ..
docker compose config
```

Expected: full backend suite passes; all Vitest files pass; production frontend build succeeds; Compose renders loopback binding, local mode enabled, and correct Chroma path. Record exact current counts and the accepted existing Vite chunk warning in `docs/ROADMAP.md` and `docs/DEMO.md`.

- [ ] **Step 5: Run static safety and cleanliness checks**

Run:

```bash
rg -n "accounts.google.com/gsi/client|googleLogin|googleSignOut|Google One Tap|Google Sign-In|GOOGLE_CLIENT_ID" frontend/index.html frontend/src README.md
rg -n "STUDY_COACH_LOCAL_MODE|127.0.0.1:8000:8000|CHROMA_PATH|CHROMA_PERSIST_DIR" docker-compose.yml fly.toml .env.example
git diff --check
git status --short
```

Expected: no visible Google runtime/product matches; local mode is explicit in all environments; Compose contains loopback and `CHROMA_PATH` but not `CHROMA_PERSIST_DIR`; diff check is clean; only intentional P5 files are modified.

- [ ] **Step 6: Complete manual acceptance and record evidence**

Run the documented browser path with screenshots and short text evidence. Automated tests are the required proof for injected Chroma failure, SQLite failure, lock conflict, and idempotent retry; manual failure injection is optional.

Mark P5.1 through P5.4 complete only after both automated and manual evidence pass. If manual acceptance exposes a defect, leave the relevant roadmap checkbox open and fix it through a new red-green-refactor step before closure.

- [ ] **Step 7: Commit verified closure**

```bash
git add docs/ARCHITECTURE.md docs/DEMO.md docs/ROADMAP.md README.md
git commit -m "docs: close p5 local-first data lifecycle"
```

---

## Final Release Gate

Before pushing or merging, verify all of the following:

- Missing or invalid bearer tokens never reach summary/reset and never fall back to `default-user`.
- `STUDY_COACH_LOCAL_MODE` defaults off; reset-disabled startup skips the gate and hides Danger Zone.
- Docker Compose is loopback-only for backend host traffic; Fly explicitly disables reset.
- Summary and reset report all 15 count fields, including topics, mastery, citations, milestones, and events.
- Foreign-key-on tests prove `plan_events` and `plan_milestones` are deleted before topics.
- Active learning operations and reset are mutually exclusive for the full response lifetime.
- Chroma is cleared before SQLite, complete retriever/checkpointer references are replaced, and retry finishes partial reset.
- Start fresh always uses learning scope; factory browser keys survive backend failure and clear only after success.
- Every other tab is invalidated; remote learning reset requires acknowledgement and remote factory reset reloads.
- The startup dialog cannot be bypassed with Esc or backdrop and has keyboard-reachable choices.
- Google OAuth remains frozen in backend code but absent from the shipped frontend product surface.
- Upload cleanup removes only the exact per-request temp file on success and failure.
- Full Pytest, Vitest, frontend build, Compose render, browser acceptance, and `git diff --check` pass.
