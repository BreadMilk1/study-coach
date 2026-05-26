# Plan Milestone Progression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add stable, user-progressable Plan milestones with complete/reopen actions, event history, and mastery-linked quick quiz prompts without letting Plan completion directly mutate Mastery.

**Architecture:** Introduce normalized `plan_milestones` and `plan_events` tables while keeping `plans.milestones_json` as a short-term compatibility cache for existing Planner/eval code. Backend reads prefer normalized rows, manual updates write both normalized rows and the JSON cache, and Planner check-in continues to use the same tool contract while its output is synchronized into normalized rows. Frontend Plan rows become clickable, show mastery validation hints, and bridge low-mastery completed milestones to `/quiz?topic=...`.

**Tech Stack:** Python 3.11, FastAPI, SQLAlchemy 2.x, Alembic, Pydantic 2, pytest, Vue 3, TypeScript, Pinia, Vite.

---

## Scope Decision for This Implementation

This plan implements the first production-safe slice:

- Stable milestone IDs.
- Manual complete/reopen.
- Plan event logging.
- Low-mastery quick quiz prompt.
- Planner compatibility with normalized milestones.

This plan does **not** implement pending AI change-set confirmation UI. To avoid unsafe auto-application without confirmation UI, the check-in prompt is narrowed in this slice: AI check-in may mark explicit user-reported completion/reopen and postpone overdue unfinished milestones, but it must not add, delete, reorder, or substantially retitle milestones.

`plans.milestones_json` remains as a compatibility cache and is kept in sync until all Planner/eval code has moved to normalized reads.

---

## File Structure

Backend:

- Modify `backend/app/db/models.py`: add `PlanMilestone` and `PlanEvent` ORM models.
- Create `backend/alembic/versions/8b7d2c4f9a31_p4_plan_milestone_progression.py`: create new tables and migrate existing JSON milestones into rows.
- Modify `backend/app/db/repositories.py`: add normalized milestone/event methods on `PlanRepository`.
- Modify `backend/app/agent/tools/schemas.py`: make `Milestone.id` optional and preserve existing tool compatibility.
- Modify `backend/app/agent/tools/plan.py`: keep `update_study_plan` calling `PlanRepository.update_milestones`.
- Modify `backend/app/agent/progress.py`: support plans that expose normalized `milestones` as well as legacy `milestones_json`.
- Modify `backend/app/agent/prompts/planner_check_in.txt`: narrow AI check-in to low-risk changes until confirmation UI exists.
- Modify `backend/app/api/routes.py`: extend plan DTOs and add milestone PATCH + events GET.
- Modify `backend/tests/db/test_alembic.py`: verify new tables.
- Create `backend/tests/db/test_plan_milestone_repository.py`: repository tests.
- Modify `backend/tests/api/test_routes_p3_plans.py`: API tests for IDs, toggle, events, validation hint.
- Modify `backend/tests/agent/test_progress.py`: normalized progress test.
- Modify existing planner tests only if assertions need updated schemas.

Frontend:

- Modify `frontend/src/lib/api.ts`: extend Plan DTOs and add `patchMilestoneDone()` / `getPlanEvents()`.
- Modify `frontend/src/stores/plan.ts`: add event state and toggle action.
- Modify `frontend/src/components/MilestoneList.vue`: clickable rows, completed timestamp, mastery chip, quick quiz CTA.
- Modify `frontend/src/views/PlanTimeline.vue`: wire toggle/events/quiz CTA.
- Modify `frontend/src/views/QuizAdaptive.vue`: support `?topic=` query.
- Modify `frontend/src/components/PlanProgressCard.vue`: surface low-mastery completed count.
- Modify `frontend/src/components/RadarChart.vue`: no behavioral change expected; verify Plan progress still derives from `done`.

Docs:

- Modify `docs/ARCHITECTURE.md`: document normalized Plan milestone/event tables and the Plan/Mastery boundary.
- Modify `docs/ROADMAP.md`: mark the milestone progression backlog item as implemented when complete.

---

## Task 1: Normalized Plan Tables and Migration

**Files:**
- Modify: `backend/app/db/models.py`
- Create: `backend/alembic/versions/8b7d2c4f9a31_p4_plan_milestone_progression.py`
- Modify: `backend/tests/db/test_alembic.py`

- [ ] **Step 1: Write failing Alembic table test**

Add this test to `backend/tests/db/test_alembic.py`:

```python
def test_alembic_upgrade_head_creates_plan_milestone_progression_tables(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'p4_plan.db'}"
    cfg = _alembic_config(db_url)
    command.upgrade(cfg, "head")

    engine = create_engine(db_url)
    tables = set(inspect(engine).get_table_names())
    assert {"plan_milestones", "plan_events"} <= tables

    milestone_cols = {c["name"] for c in inspect(engine).get_columns("plan_milestones")}
    assert {
        "id",
        "plan_id",
        "topic_id",
        "topic_name",
        "title",
        "due_at",
        "done",
        "completed_at",
        "sort_order",
        "source",
        "created_at",
        "updated_at",
    } <= milestone_cols

    event_cols = {c["name"] for c in inspect(engine).get_columns("plan_events")}
    assert {
        "id",
        "plan_id",
        "milestone_id",
        "actor",
        "action",
        "before_json",
        "after_json",
        "reason",
        "created_at",
    } <= event_cols
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
cd backend
uv run pytest tests/db/test_alembic.py::test_alembic_upgrade_head_creates_plan_milestone_progression_tables -q
```

Expected: FAIL because `plan_milestones` and `plan_events` do not exist.

- [ ] **Step 3: Add ORM models**

In `backend/app/db/models.py`, add `Boolean` import and the two models after `Plan`:

```python
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
```

```python
class PlanMilestone(Base):
    __tablename__ = "plan_milestones"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    plan_id: Mapped[str] = mapped_column(ForeignKey("plans.id"))
    topic_id: Mapped[str | None] = mapped_column(ForeignKey("topics.id"), nullable=True)
    topic_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    title: Mapped[str] = mapped_column(Text)
    due_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    done: Mapped[bool] = mapped_column(Boolean, default=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
    source: Mapped[str] = mapped_column(String(20), default="ai")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class PlanEvent(Base):
    __tablename__ = "plan_events"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    plan_id: Mapped[str] = mapped_column(ForeignKey("plans.id"))
    milestone_id: Mapped[str | None] = mapped_column(ForeignKey("plan_milestones.id"), nullable=True)
    actor: Mapped[str] = mapped_column(String(20))
    action: Mapped[str] = mapped_column(String(40))
    before_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    after_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
```

- [ ] **Step 4: Add Alembic migration**

Create `backend/alembic/versions/8b7d2c4f9a31_p4_plan_milestone_progression.py`:

```python
"""p4_plan_milestone_progression

Revision ID: 8b7d2c4f9a31
Revises: cae9687d6295
Create Date: 2026-05-26 00:00:00.000000
"""
from __future__ import annotations

from datetime import datetime
from typing import Sequence, Union
import uuid

from alembic import op
import sqlalchemy as sa


revision: str = "8b7d2c4f9a31"
down_revision: Union[str, Sequence[str], None] = "cae9687d6295"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _uuid() -> str:
    return str(uuid.uuid4())


def _parse_due_at(value):
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def upgrade() -> None:
    op.create_table(
        "plan_milestones",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("plan_id", sa.String(length=36), nullable=False),
        sa.Column("topic_id", sa.String(length=36), nullable=True),
        sa.Column("topic_name", sa.String(length=200), nullable=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("due_at", sa.DateTime(), nullable=True),
        sa.Column("done", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("sort_order", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("source", sa.String(length=20), nullable=False, server_default="migrated"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["plan_id"], ["plans.id"]),
        sa.ForeignKeyConstraint(["topic_id"], ["topics.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "plan_events",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("plan_id", sa.String(length=36), nullable=False),
        sa.Column("milestone_id", sa.String(length=36), nullable=True),
        sa.Column("actor", sa.String(length=20), nullable=False),
        sa.Column("action", sa.String(length=40), nullable=False),
        sa.Column("before_json", sa.JSON(), nullable=True),
        sa.Column("after_json", sa.JSON(), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["plan_id"], ["plans.id"]),
        sa.ForeignKeyConstraint(["milestone_id"], ["plan_milestones.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    bind = op.get_bind()
    plans = sa.table(
        "plans",
        sa.column("id", sa.String),
        sa.column("milestones_json", sa.JSON),
        sa.column("updated_at", sa.DateTime),
    )
    milestones = sa.table(
        "plan_milestones",
        sa.column("id", sa.String),
        sa.column("plan_id", sa.String),
        sa.column("topic_id", sa.String),
        sa.column("topic_name", sa.String),
        sa.column("title", sa.Text),
        sa.column("due_at", sa.DateTime),
        sa.column("done", sa.Boolean),
        sa.column("completed_at", sa.DateTime),
        sa.column("sort_order", sa.Integer),
        sa.column("source", sa.String),
        sa.column("created_at", sa.DateTime),
        sa.column("updated_at", sa.DateTime),
    )
    now = datetime.utcnow()
    for plan in bind.execute(sa.select(plans.c.id, plans.c.milestones_json, plans.c.updated_at)):
        for idx, raw in enumerate(plan.milestones_json or []):
            title = str(raw.get("title") or "").strip()
            if not title:
                continue
            done = bool(raw.get("done", False))
            created = plan.updated_at or now
            bind.execute(
                milestones.insert().values(
                    id=raw.get("id") or _uuid(),
                    plan_id=plan.id,
                    topic_id=raw.get("topic_id"),
                    topic_name=raw.get("topic") or raw.get("topic_name"),
                    title=title,
                    due_at=_parse_due_at(raw.get("due_at")),
                    done=done,
                    completed_at=created if done else None,
                    sort_order=idx,
                    source="migrated",
                    created_at=created,
                    updated_at=created,
                )
            )


def downgrade() -> None:
    op.drop_table("plan_events")
    op.drop_table("plan_milestones")
```

- [ ] **Step 5: Run Alembic tests**

Run:

```bash
cd backend
uv run pytest tests/db/test_alembic.py -q
```

Expected: all tests in `test_alembic.py` pass.

- [ ] **Step 6: Commit**

```bash
git add backend/app/db/models.py backend/alembic/versions/8b7d2c4f9a31_p4_plan_milestone_progression.py backend/tests/db/test_alembic.py
git commit -m "feat: add plan milestone tables"
```

---

## Task 2: Plan Repository Normalized Milestone Operations

**Files:**
- Modify: `backend/app/db/repositories.py`
- Create: `backend/tests/db/test_plan_milestone_repository.py`

- [ ] **Step 1: Write repository tests**

Create `backend/tests/db/test_plan_milestone_repository.py`:

```python
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.db.models import Base
from app.db.repositories import GoalRepository, MasteryRepository, PlanRepository, TopicRepository, UserRepository


@pytest.fixture
def session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,
    )
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def _seed_plan(session):
    user = UserRepository(session).get_or_create("fp-plan-milestones")
    goal = GoalRepository(session).create(user_id=user.id, title="Master HyDE")
    topic = TopicRepository(session).create(goal_id=goal.id, name="HyDE")
    repo = PlanRepository(session)
    plan = repo.update_milestones(
        goal_id=goal.id,
        milestones=[
            {"title": "Read HyDE", "due_at": "2026-05-30", "done": False, "topic": "HyDE", "topic_id": topic.id},
            {"title": "Quiz HyDE", "due_at": None, "done": False, "topic": "HyDE", "topic_id": topic.id},
        ],
    )
    return user, goal, topic, plan, repo


def test_update_milestones_creates_normalized_rows_and_json_cache(session):
    _, _, topic, plan, repo = _seed_plan(session)

    rows = repo.list_milestones(plan.id)

    assert len(rows) == 2
    assert rows[0].id
    assert rows[0].title == "Read HyDE"
    assert rows[0].topic_id == topic.id
    assert rows[0].topic_name == "HyDE"
    assert rows[0].sort_order == 0
    assert plan.milestones_json[0]["id"] == rows[0].id
    assert plan.milestones_json[0]["topic"] == "HyDE"


def test_set_milestone_done_completes_and_reopens_with_events(session):
    _, _, _, plan, repo = _seed_plan(session)
    milestone = repo.list_milestones(plan.id)[0]

    updated = repo.set_milestone_done(
        plan_id=plan.id,
        milestone_id=milestone.id,
        done=True,
        actor="user",
        reason="User marked milestone complete",
    )

    assert updated.done is True
    assert updated.completed_at is not None
    assert repo.get_by_goal(plan.goal_id).milestones_json[0]["done"] is True
    assert repo.list_events(plan.id)[0].action == "completed"

    reopened = repo.set_milestone_done(
        plan_id=plan.id,
        milestone_id=milestone.id,
        done=False,
        actor="user",
        reason="User reopened milestone",
    )

    assert reopened.done is False
    assert reopened.completed_at is None
    events = repo.list_events(plan.id)
    assert [e.action for e in events[:2]] == ["reopened", "completed"]


def test_plan_repository_returns_mastery_score_by_topic_name(session):
    user, _, topic, plan, repo = _seed_plan(session)
    MasteryRepository(session).upsert(user_id=user.id, topic_id=topic.id, score=0.25)

    dto_rows = repo.list_milestone_dicts(plan.id, user_id=user.id)

    assert dto_rows[0]["mastery_score"] == pytest.approx(0.25)
    assert dto_rows[0]["validation_recommended"] is True
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
cd backend
uv run pytest tests/db/test_plan_milestone_repository.py -q
```

Expected: FAIL because methods and models are not wired yet.

- [ ] **Step 3: Implement repository helpers**

In `backend/app/db/repositories.py`, import models and helpers:

```python
from datetime import datetime
from app.db.models import Plan, PlanEvent, PlanMilestone, Topic, Mastery
```

Add these private helpers near `PlanRepository`:

```python
def _parse_due_at(value):
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _due_to_json(value):
    return value.isoformat() if value is not None else None
```

Replace/extend `PlanRepository` with these methods while preserving `create()` and `get_by_goal()` public behavior:

```python
    def list_milestones(self, plan_id: str) -> list[PlanMilestone]:
        stmt = (
            select(PlanMilestone)
            .where(PlanMilestone.plan_id == plan_id)
            .order_by(PlanMilestone.sort_order.asc(), PlanMilestone.created_at.asc())
        )
        return list(self.session.execute(stmt).scalars())

    def _milestone_json(self, row: PlanMilestone) -> dict:
        return {
            "id": row.id,
            "title": row.title,
            "due_at": _due_to_json(row.due_at),
            "done": row.done,
            "topic": row.topic_name,
            "topic_id": row.topic_id,
        }

    def _sync_milestones_json(self, plan: Plan) -> None:
        plan.milestones_json = [self._milestone_json(m) for m in self.list_milestones(plan.id)]
        plan.updated_at = datetime.utcnow()

    def _find_topic_id(self, *, plan: Plan, raw: dict) -> str | None:
        if raw.get("topic_id"):
            return raw["topic_id"]
        name = raw.get("topic") or raw.get("topic_name")
        if not name:
            return None
        stmt = (
            select(Topic.id)
            .where(Topic.goal_id == plan.goal_id, Topic.name == name)
            .limit(1)
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def _log_event(
        self,
        *,
        plan_id: str,
        milestone_id: str | None,
        actor: str,
        action: str,
        before: dict | None = None,
        after: dict | None = None,
        reason: str | None = None,
    ) -> PlanEvent:
        event = PlanEvent(
            id=_uuid(),
            plan_id=plan_id,
            milestone_id=milestone_id,
            actor=actor,
            action=action,
            before_json=before,
            after_json=after,
            reason=reason,
            created_at=datetime.utcnow(),
        )
        self.session.add(event)
        return event

    def list_events(self, plan_id: str, *, limit: int = 20) -> list[PlanEvent]:
        stmt = (
            select(PlanEvent)
            .where(PlanEvent.plan_id == plan_id)
            .order_by(PlanEvent.created_at.desc())
            .limit(limit)
        )
        return list(self.session.execute(stmt).scalars())

    def update_milestones(self, *, goal_id: str, milestones: list) -> Plan:
        existing = self.get_by_goal(goal_id)
        plan = existing if existing is not None else self.create(goal_id=goal_id, milestones_json=[])

        existing_rows = self.list_milestones(plan.id)
        by_id = {m.id: m for m in existing_rows}
        by_key = {(m.title, m.topic_name): m for m in existing_rows}
        seen_ids: set[str] = set()

        for idx, raw in enumerate(milestones):
            title = str(raw.get("title") or "").strip()
            if not title:
                continue
            topic_name = raw.get("topic") or raw.get("topic_name")
            row = None
            if raw.get("id"):
                row = by_id.get(raw["id"])
            if row is None:
                row = by_key.get((title, topic_name))
            before = self._milestone_json(row) if row is not None else None
            if row is None:
                row = PlanMilestone(
                    id=raw.get("id") or _uuid(),
                    plan_id=plan.id,
                    created_at=datetime.utcnow(),
                    source=raw.get("source") or "ai",
                )
                self.session.add(row)
            row.topic_id = self._find_topic_id(plan=plan, raw=raw)
            row.topic_name = topic_name
            row.title = title
            row.due_at = _parse_due_at(raw.get("due_at"))
            row.done = bool(raw.get("done", False))
            row.completed_at = datetime.utcnow() if row.done and row.completed_at is None else (row.completed_at if row.done else None)
            row.sort_order = idx
            row.updated_at = datetime.utcnow()
            seen_ids.add(row.id)
            after = self._milestone_json(row)
            self._log_event(
                plan_id=plan.id,
                milestone_id=row.id,
                actor="ai",
                action="created" if before is None else "applied",
                before=before,
                after=after,
                reason="Planner updated study plan",
            )

        for row in existing_rows:
            if row.id not in seen_ids:
                self.session.delete(row)

        self.session.flush()
        self._sync_milestones_json(plan)
        self.session.commit()
        self.session.refresh(plan)
        return plan

    def get_milestone(self, *, plan_id: str, milestone_id: str) -> PlanMilestone | None:
        stmt = select(PlanMilestone).where(
            PlanMilestone.plan_id == plan_id,
            PlanMilestone.id == milestone_id,
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def set_milestone_done(
        self,
        *,
        plan_id: str,
        milestone_id: str,
        done: bool,
        actor: str,
        reason: str,
    ) -> PlanMilestone:
        row = self.get_milestone(plan_id=plan_id, milestone_id=milestone_id)
        if row is None:
            raise ValueError(f"milestone {milestone_id} not found")
        plan = self.session.get(Plan, plan_id)
        if plan is None:
            raise ValueError(f"plan {plan_id} not found")
        before = self._milestone_json(row)
        row.done = done
        row.completed_at = datetime.utcnow() if done else None
        row.updated_at = datetime.utcnow()
        after = self._milestone_json(row)
        self._log_event(
            plan_id=plan_id,
            milestone_id=milestone_id,
            actor=actor,
            action="completed" if done else "reopened",
            before=before,
            after=after,
            reason=reason,
        )
        self.session.flush()
        self._sync_milestones_json(plan)
        self.session.commit()
        self.session.refresh(row)
        return row

    def list_milestone_dicts(self, plan_id: str, *, user_id: str | None = None) -> list[dict]:
        rows = self.list_milestones(plan_id)
        mastery_by_topic_id: dict[str, float] = {}
        if user_id:
            stmt = select(Mastery.topic_id, Mastery.score).where(Mastery.user_id == user_id)
            mastery_by_topic_id = {topic_id: score for topic_id, score in self.session.execute(stmt)}
        out = []
        for row in rows:
            item = self._milestone_json(row)
            item["completed_at"] = row.completed_at.isoformat() if row.completed_at else None
            item["sort_order"] = row.sort_order
            item["source"] = row.source
            item["mastery_score"] = mastery_by_topic_id.get(row.topic_id) if row.topic_id else None
            item["validation_recommended"] = (
                row.done and (item["mastery_score"] is None or item["mastery_score"] < 0.5)
            )
            out.append(item)
        return out
```

- [ ] **Step 4: Run repository tests**

Run:

```bash
cd backend
uv run pytest tests/db/test_plan_milestone_repository.py tests/db/test_repositories_p2_1_5.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add backend/app/db/repositories.py backend/tests/db/test_plan_milestone_repository.py
git commit -m "feat: normalize plan milestone repository"
```

---

## Task 3: Backend API for Milestone IDs, Toggle, and Events

**Files:**
- Modify: `backend/app/api/routes.py`
- Modify: `backend/tests/api/test_routes_p3_plans.py`

- [ ] **Step 1: Extend API tests**

Append these tests to `backend/tests/api/test_routes_p3_plans.py`:

```python
def test_get_plans_current_returns_milestone_ids_and_mastery_hint(client):
    from app.db.session import session_scope
    from app.db.repositories import UserRepository, GoalRepository, PlanRepository, TopicRepository, MasteryRepository

    with session_scope() as s:
        user = UserRepository(s).get_or_create("fp-plan-id")
        goal = GoalRepository(s).create(user_id=user.id, title="Master HyDE", exam_date=None)
        topic = TopicRepository(s).create(goal_id=goal.id, name="HyDE")
        MasteryRepository(s).upsert(user_id=user.id, topic_id=topic.id, score=0.25)
        PlanRepository(s).update_milestones(
            goal_id=goal.id,
            milestones=[{"title": "Read HyDE", "done": True, "topic": "HyDE", "topic_id": topic.id}],
        )

    resp = client.get("/api/plans/current", headers={"x-fingerprint": "fp-plan-id"})

    assert resp.status_code == 200
    milestone = resp.json()["milestones"][0]
    assert milestone["id"]
    assert milestone["topic_id"] == topic.id
    assert milestone["topic"] == "HyDE"
    assert milestone["mastery_score"] == 0.25
    assert milestone["validation_recommended"] is True


def test_patch_milestone_done_toggles_state_without_changing_mastery(client):
    from app.db.session import session_scope
    from app.db.repositories import UserRepository, GoalRepository, PlanRepository, TopicRepository, MasteryRepository

    with session_scope() as s:
        user = UserRepository(s).get_or_create("fp-toggle")
        goal = GoalRepository(s).create(user_id=user.id, title="Master HyDE", exam_date=None)
        topic = TopicRepository(s).create(goal_id=goal.id, name="HyDE")
        MasteryRepository(s).upsert(user_id=user.id, topic_id=topic.id, score=0.25)
        plan = PlanRepository(s).update_milestones(
            goal_id=goal.id,
            milestones=[{"title": "Read HyDE", "done": False, "topic": "HyDE", "topic_id": topic.id}],
        )
        milestone_id = PlanRepository(s).list_milestones(plan.id)[0].id

    resp = client.patch(
        f"/api/plans/{plan.id}/milestones/{milestone_id}",
        headers={"x-fingerprint": "fp-toggle"},
        json={"done": True},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["plan"]["milestones"][0]["done"] is True
    assert body["event"]["action"] == "completed"
    assert body["validation_hint"]["show_quick_quiz"] is True

    mastery = client.get("/api/mastery", headers={"x-fingerprint": "fp-toggle"}).json()
    assert mastery["scores"][0]["score"] == 0.25


def test_get_plan_events_returns_recent_changes(client):
    from app.db.session import session_scope
    from app.db.repositories import UserRepository, GoalRepository, PlanRepository

    with session_scope() as s:
        user = UserRepository(s).get_or_create("fp-events")
        goal = GoalRepository(s).create(user_id=user.id, title="Master HyDE", exam_date=None)
        plan = PlanRepository(s).update_milestones(
            goal_id=goal.id,
            milestones=[{"title": "Read HyDE", "done": False, "topic": "HyDE"}],
        )
        milestone_id = PlanRepository(s).list_milestones(plan.id)[0].id
        PlanRepository(s).set_milestone_done(
            plan_id=plan.id,
            milestone_id=milestone_id,
            done=True,
            actor="user",
            reason="User marked milestone complete",
        )

    resp = client.get(f"/api/plans/{plan.id}/events", headers={"x-fingerprint": "fp-events"})

    assert resp.status_code == 200
    assert resp.json()[0]["action"] == "completed"
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
cd backend
uv run pytest tests/api/test_routes_p3_plans.py -q
```

Expected: new tests fail because DTO fields and endpoints do not exist.

- [ ] **Step 3: Implement route models and helpers**

In `backend/app/api/routes.py`, extend models:

```python
class MilestoneOut(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str | None = None
    title: str
    due_at: str | None = None
    done: bool = False
    completed_at: str | None = None
    topic_id: str | None = None
    topic: str | None = None
    mastery_score: float | None = None
    validation_recommended: bool = False
    sort_order: int | None = None
    source: str | None = None


class MilestonePatchIn(BaseModel):
    done: bool


class PlanEventOut(BaseModel):
    id: str
    plan_id: str
    milestone_id: str | None = None
    actor: str
    action: str
    before_json: dict | None = None
    after_json: dict | None = None
    reason: str | None = None
    created_at: str


class ValidationHintOut(BaseModel):
    show_quick_quiz: bool
    topic: str | None = None
    reason: str | None = None


class MilestonePatchOut(BaseModel):
    plan: PlanCurrentOut
    event: PlanEventOut
    validation_hint: ValidationHintOut
```

Add helper functions below DTO definitions:

```python
def _plan_belongs_to_user(session: Session, *, user_id: str, plan_id: str):
    goals = GoalRepository(session).list_active_for_user(user_id)
    if not goals:
        raise HTTPException(status_code=404, detail="no active plan for user")
    goal = goals[0]
    plan = PlanRepository(session).get_by_goal(goal.id)
    if plan is None or plan.id != plan_id:
        raise HTTPException(status_code=404, detail="no active plan for user")
    return goal, plan


def _plan_current_out(session: Session, *, user_id: str, goal, plan) -> PlanCurrentOut:
    repo = PlanRepository(session)
    milestone_dicts = repo.list_milestone_dicts(plan.id, user_id=user_id)
    if not milestone_dicts:
        milestone_dicts = [dict(m) for m in plan.milestones_json]
    return PlanCurrentOut(
        plan_id=plan.id,
        goal_id=goal.id,
        goal_title=goal.title,
        milestones=[MilestoneOut(**m) for m in milestone_dicts],
        updated_at=plan.updated_at.isoformat(),
    )
```

- [ ] **Step 4: Update current plan route**

Replace the return body in `get_plans_current()`:

```python
    return _plan_current_out(session, user_id=user_id, goal=goal, plan=plan)
```

- [ ] **Step 5: Add PATCH and events routes**

Add below `get_plans_current()`:

```python
@router.patch("/plans/{plan_id}/milestones/{milestone_id}", response_model=MilestonePatchOut)
def patch_plan_milestone(
    plan_id: str,
    milestone_id: str,
    body: MilestonePatchIn,
    user_id: Annotated[str, Depends(get_user_id)],
    session: Annotated[Session, Depends(get_session)],
):
    goal, plan = _plan_belongs_to_user(session, user_id=user_id, plan_id=plan_id)
    repo = PlanRepository(session)
    updated = repo.set_milestone_done(
        plan_id=plan.id,
        milestone_id=milestone_id,
        done=body.done,
        actor="user",
        reason="User marked milestone complete" if body.done else "User reopened milestone",
    )
    event = repo.list_events(plan.id, limit=1)[0]
    refreshed = repo.get_by_goal(goal.id)
    current = _plan_current_out(session, user_id=user_id, goal=goal, plan=refreshed)
    milestone_out = next((m for m in current.milestones if m.id == updated.id), None)
    show_quiz = bool(body.done and milestone_out and milestone_out.validation_recommended)
    return MilestonePatchOut(
        plan=current,
        event=PlanEventOut(
            id=event.id,
            plan_id=event.plan_id,
            milestone_id=event.milestone_id,
            actor=event.actor,
            action=event.action,
            before_json=event.before_json,
            after_json=event.after_json,
            reason=event.reason,
            created_at=event.created_at.isoformat(),
        ),
        validation_hint=ValidationHintOut(
            show_quick_quiz=show_quiz,
            topic=milestone_out.topic if milestone_out else None,
            reason="Topic mastery is still below the validation threshold" if show_quiz else None,
        ),
    )


@router.get("/plans/{plan_id}/events", response_model=list[PlanEventOut])
def get_plan_events(
    plan_id: str,
    user_id: Annotated[str, Depends(get_user_id)],
    session: Annotated[Session, Depends(get_session)],
    limit: int = 20,
):
    _goal, plan = _plan_belongs_to_user(session, user_id=user_id, plan_id=plan_id)
    return [
        PlanEventOut(
            id=e.id,
            plan_id=e.plan_id,
            milestone_id=e.milestone_id,
            actor=e.actor,
            action=e.action,
            before_json=e.before_json,
            after_json=e.after_json,
            reason=e.reason,
            created_at=e.created_at.isoformat(),
        )
        for e in PlanRepository(session).list_events(plan.id, limit=limit)
    ]
```

- [ ] **Step 6: Run API tests**

Run:

```bash
cd backend
uv run pytest tests/api/test_routes_p3_plans.py tests/api/test_routes_p3_mastery.py -q
```

Expected: selected API tests pass and mastery score remains unchanged after milestone toggle.

- [ ] **Step 7: Commit**

```bash
git add backend/app/api/routes.py backend/tests/api/test_routes_p3_plans.py
git commit -m "feat: add milestone progress API"
```

---

## Task 4: Planner and Progress Compatibility

**Files:**
- Modify: `backend/app/agent/tools/schemas.py`
- Modify: `backend/app/agent/progress.py`
- Modify: `backend/app/agent/prompts/planner_check_in.txt`
- Modify: `backend/tests/agent/test_progress.py`
- Modify: `backend/tests/agent/test_planner.py`
- Modify: `backend/tests/agent/test_planner_agent_tools.py`

- [ ] **Step 1: Add normalized progress test**

Append to `backend/tests/agent/test_progress.py`:

```python
def test_compute_progress_accepts_normalized_milestones():
    from types import SimpleNamespace

    plan = SimpleNamespace(
        milestones=[
            {"title": "Done", "done": True, "due_at": "2026-05-01"},
            {"title": "Late", "done": False, "due_at": "2026-05-01"},
        ],
        milestones_json=[],
    )

    summary = compute_progress(plan, {"HyDE": 0.2}, [], now=datetime(2026, 5, 22))

    assert summary.done_count == 1
    assert summary.total_count == 2
    assert [m["title"] for m in summary.overdue] == ["Late"]
    assert summary.weak_topics == ["HyDE"]
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
cd backend
uv run pytest tests/agent/test_progress.py::test_compute_progress_accepts_normalized_milestones -q
```

Expected: FAIL because `compute_progress()` only reads `milestones_json`.

- [ ] **Step 3: Update `Milestone` schema**

In `backend/app/agent/tools/schemas.py`, change `Milestone`:

```python
class Milestone(BaseModel):
    id: str | None = None
    title: str
    due_at: str | None = None      # ISO date "YYYY-MM-DD" or None
    done: bool = False
    topic: str | None = None
    topic_id: str | None = None
```

- [ ] **Step 4: Update progress helper**

In `backend/app/agent/progress.py`, add:

```python
def _milestone_dicts(plan) -> list[dict]:
    normalized = getattr(plan, "milestones", None)
    if normalized:
        return list(normalized)
    return list(getattr(plan, "milestones_json", []) or [])
```

Replace:

```python
milestones = list(getattr(plan, "milestones_json", []) or [])
```

with:

```python
milestones = _milestone_dicts(plan)
```

- [ ] **Step 5: Narrow check-in prompt**

Replace `backend/app/agent/prompts/planner_check_in.txt` with:

```text
You are a study-plan adjuster for an exam coach.

Current milestones (JSON):
{current_milestones}

Progress summary:
- Done: {done_count} / {total_count}
- Overdue (titles): {overdue_titles}
- Weak topics (mastery < 0.4): {weak_topics}
- Recent mistakes: {recent_mistake_count}

User's check-in message: {user_msg}

Adjust the milestones conservatively. You MAY:
- Mark an item done only if the user explicitly said that specific item is complete.
- Reopen an item only if the user explicitly said that specific item is not complete.
- Postpone overdue unfinished items by pushing due_at later.
- Keep id fields unchanged when present.

You may NOT:
- Add new milestones.
- Delete milestones.
- Reorder milestones.
- Rewrite titles beyond minor typo fixes.
- Change mastery or quiz state.

Output ONLY the updated JSON array, no prose, no fences. Schema same as input.
```

- [ ] **Step 6: Run planner/progress tests**

Run:

```bash
cd backend
uv run pytest tests/agent/test_progress.py tests/agent/test_planner.py tests/agent/test_planner_agent_tools.py -q
```

Expected: selected tests pass. If `test_planner_check_in_adjusts_existing_plan` expects added milestones, update the assertion to verify conservative postponement/completion behavior instead, because v1 intentionally defers AI-added milestones until confirmation UI exists.

- [ ] **Step 7: Commit**

```bash
git add backend/app/agent/tools/schemas.py backend/app/agent/progress.py backend/app/agent/prompts/planner_check_in.txt backend/tests/agent/test_progress.py backend/tests/agent/test_planner.py backend/tests/agent/test_planner_agent_tools.py
git commit -m "feat: keep planner compatible with milestone ids"
```

---

## Task 5: Frontend Plan Store and API Client

**Files:**
- Modify: `frontend/src/lib/api.ts`
- Modify: `frontend/src/stores/plan.ts`

- [ ] **Step 1: Update API types and functions**

In `frontend/src/lib/api.ts`, replace `MilestoneDto` and add event/toggle APIs:

```ts
export interface MilestoneDto {
  id: string | null
  title: string
  due_at: string | null
  done: boolean
  completed_at: string | null
  topic_id: string | null
  topic: string | null
  mastery_score: number | null
  validation_recommended: boolean
  sort_order: number | null
  source: string | null
}

export interface PlanEventDto {
  id: string
  plan_id: string
  milestone_id: string | null
  actor: string
  action: string
  before_json: Record<string, unknown> | null
  after_json: Record<string, unknown> | null
  reason: string | null
  created_at: string
}

export interface ValidationHintDto {
  show_quick_quiz: boolean
  topic: string | null
  reason: string | null
}

export interface MilestonePatchDto {
  plan: PlanCurrentDto
  event: PlanEventDto
  validation_hint: ValidationHintDto
}
```

Add:

```ts
export async function patchMilestoneDone(
  planId: string,
  milestoneId: string,
  done: boolean,
): Promise<MilestonePatchDto> {
  const resp = await fetch(`/api/plans/${planId}/milestones/${milestoneId}`, {
    method: 'PATCH',
    headers: {
      'Content-Type': 'application/json',
      'x-fingerprint': getFingerprint(),
    },
    body: JSON.stringify({ done }),
  })
  if (!resp.ok) throw new Error(`milestone update failed: ${resp.status}`)
  return resp.json() as Promise<MilestonePatchDto>
}

export function getPlanEvents(planId: string, limit = 20): Promise<PlanEventDto[]> {
  return getJSON<PlanEventDto[]>(`/api/plans/${planId}/events?limit=${limit}`)
}
```

- [ ] **Step 2: Update Pinia store**

In `frontend/src/stores/plan.ts`, update imports:

```ts
import {
  getCurrentPlan,
  getPlanEvents,
  patchMilestoneDone,
  type PlanCurrentDto,
  type PlanEventDto,
  type ValidationHintDto,
} from '../lib/api'
```

Update state:

```ts
interface PlanState {
  plan: PlanCurrentDto | null
  events: PlanEventDto[]
  lastValidationHint: ValidationHintDto | null
  loading: boolean
  updatingMilestoneId: string | null
  error: string | null
  noActive: boolean
  mindmapMermaid: string | null
}
```

Update `state()`:

```ts
events: [],
lastValidationHint: null,
updatingMilestoneId: null,
```

Add actions:

```ts
async fetchEvents() {
  if (!this.plan) {
    this.events = []
    return
  }
  this.events = await getPlanEvents(this.plan.plan_id)
},
async toggleMilestone(milestoneId: string, done: boolean) {
  if (!this.plan) return
  this.updatingMilestoneId = milestoneId
  this.error = null
  try {
    const result = await patchMilestoneDone(this.plan.plan_id, milestoneId, done)
    this.plan = result.plan
    this.lastValidationHint = result.validation_hint
    this.events = [result.event, ...this.events.filter(e => e.id !== result.event.id)].slice(0, 20)
  } catch (e: any) {
    this.error = e?.message ?? 'failed'
  } finally {
    this.updatingMilestoneId = null
  }
},
```

In `fetch()`, after `this.plan = await getCurrentPlan()`, call `await this.fetchEvents()`.

- [ ] **Step 3: Typecheck frontend**

Run:

```bash
cd frontend
pnpm build
```

Expected: fails until components are updated in the next task. If it passes here, continue.

Do not commit yet if the frontend build fails.

---

## Task 6: Frontend Plan Interaction and Quiz Bridge

**Files:**
- Modify: `frontend/src/components/MilestoneList.vue`
- Modify: `frontend/src/views/PlanTimeline.vue`
- Modify: `frontend/src/views/QuizAdaptive.vue`
- Modify: `frontend/src/components/PlanProgressCard.vue`

- [ ] **Step 1: Update `MilestoneList.vue` props and emits**

Replace the script with:

```vue
<script setup lang="ts">
import { computed } from 'vue'
import { CheckCircle2, AlertCircle, AlertTriangle, Circle } from 'lucide-vue-next'
import type { MilestoneDto } from '../lib/api'

const props = defineProps<{
  milestones: MilestoneDto[]
  updatingMilestoneId?: string | null
}>()

const emit = defineEmits<{
  toggle: [milestone: MilestoneDto]
  validate: [milestone: MilestoneDto]
}>()

function statusOf(m: MilestoneDto): 'success' | 'warning' | 'danger' | 'neutral' {
  if (m.done) return 'success'
  if (!m.due_at) return 'neutral'
  const due = new Date(m.due_at).getTime()
  const now = Date.now()
  const dayMs = 86_400_000
  if (due < now - dayMs) return 'danger'
  if (due < now + dayMs) return 'warning'
  return 'neutral'
}

function masteryLabel(m: MilestoneDto): string | null {
  if (m.mastery_score === null || m.mastery_score === undefined) return null
  return `${Math.round(m.mastery_score * 100)}% mastery`
}

const rows = computed(() =>
  props.milestones.map(m => ({ m, status: statusOf(m), mastery: masteryLabel(m) })),
)

const iconFor = { success: CheckCircle2, warning: AlertCircle, danger: AlertTriangle, neutral: Circle }
const colorFor = {
  success: 'text-success',
  warning: 'text-warning',
  danger: 'text-danger',
  neutral: 'text-fg-muted',
}
</script>
```

Update the row button in the template:

```vue
<button type="button"
        :disabled="!r.m.id || props.updatingMilestoneId === r.m.id"
        class="mt-0.5 shrink-0 disabled:opacity-40"
        :aria-label="r.m.done ? 'Reopen milestone' : 'Complete milestone'"
        @click="emit('toggle', r.m)">
  <component :is="iconFor[r.status]" :class="['w-5 h-5', colorFor[r.status]]" />
</button>
```

Add mastery and validation UI inside the metadata row:

```vue
<span v-if="r.mastery" class="font-mono px-2 py-0.5 rounded-md bg-bg text-fg-muted">
  {{ r.mastery }}
</span>
<button v-if="r.m.validation_recommended"
        type="button"
        class="text-xs text-warning underline"
        @click="emit('validate', r.m)">
  Validate with quiz
</button>
```

Keep existing title/due rendering.

- [ ] **Step 2: Wire Plan page actions**

In `frontend/src/views/PlanTimeline.vue`, import router:

```ts
import { useRouter } from 'vue-router'
```

Add:

```ts
const router = useRouter()
```

Add methods:

```ts
async function toggleMilestone(milestone: any) {
  if (!milestone.id) return
  await planStore.toggleMilestone(milestone.id, !milestone.done)
}

function validateMilestone(milestone: any) {
  if (!milestone.topic) return
  router.push({ path: '/quiz', query: { topic: milestone.topic } })
}
```

Replace:

```vue
<MilestoneList :milestones="planStore.plan.milestones" />
```

with:

```vue
<MilestoneList
  :milestones="planStore.plan.milestones"
  :updating-milestone-id="planStore.updatingMilestoneId"
  @toggle="toggleMilestone"
  @validate="validateMilestone"
/>
```

Add a recent changes panel below the list:

```vue
<section v-if="planStore.events.length" class="mt-6 rounded-lg border border-border bg-surface p-4">
  <h2 class="text-sm font-semibold text-fg-muted uppercase tracking-wider">Recent changes</h2>
  <ul class="mt-3 flex flex-col gap-2 text-xs text-fg-muted">
    <li v-for="event in planStore.events.slice(0, 5)" :key="event.id">
      <span class="font-mono text-fg">{{ event.action }}</span>
      <span v-if="event.reason"> — {{ event.reason }}</span>
    </li>
  </ul>
</section>
```

- [ ] **Step 3: Make Quiz page respect `?topic=`**

In `frontend/src/views/QuizAdaptive.vue`, update `onMounted`:

```ts
onMounted(async () => {
  await Promise.all([mistakes.fetch(), docs.fetch()])
  if (route.query.topic) {
    topicHint.value = String(route.query.topic)
    return
  }
  if (route.query.mistake_id) {
    const m = mistakes.items.find(d => d.mistake_id === route.query.mistake_id)
    topicHint.value = m?.topic_name ?? ''
  }
})
```

Update watcher:

```ts
watch(() => [route.query.mistake_id, route.query.topic], ([mistakeId, topic]) => {
  if (topic) {
    topicHint.value = String(topic)
    quiz.reset()
    return
  }
  if (!mistakeId) { topicHint.value = ''; quiz.reset(); return }
  const m = mistakes.items.find(d => d.mistake_id === mistakeId)
  topicHint.value = m?.topic_name ?? ''
  quiz.reset()
})
```

Add topic hint copy above the generate button:

```vue
<p v-if="topicHint" class="mb-3 text-sm text-fg-muted">
  Topic: <span class="font-mono text-fg">{{ topicHint }}</span>
</p>
```

- [ ] **Step 4: Update Plan progress card**

In `frontend/src/components/PlanProgressCard.vue`, add:

```ts
const lowMasteryCompleted = computed(() =>
  plan.plan?.milestones.filter(m => m.done && m.validation_recommended).length ?? 0
)
```

Add below `Next:`:

```vue
<div v-if="lowMasteryCompleted" class="mt-2 text-xs text-warning">
  {{ lowMasteryCompleted }} completed milestone{{ lowMasteryCompleted === 1 ? '' : 's' }} still need mastery validation.
</div>
```

- [ ] **Step 5: Build frontend**

Run:

```bash
cd frontend
pnpm build
```

Expected: typecheck and production build pass.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/lib/api.ts frontend/src/stores/plan.ts frontend/src/components/MilestoneList.vue frontend/src/views/PlanTimeline.vue frontend/src/views/QuizAdaptive.vue frontend/src/components/PlanProgressCard.vue
git commit -m "feat: add milestone progress UI"
```

---

## Task 7: Backend Full Regression and Integration

**Files:**
- Modify tests only if prior cuts reveal legitimate contract shifts.

- [ ] **Step 1: Run focused backend tests**

Run:

```bash
cd backend
uv run pytest \
  tests/db/test_alembic.py \
  tests/db/test_plan_milestone_repository.py \
  tests/db/test_repositories_p2_1_5.py \
  tests/api/test_routes_p3_plans.py \
  tests/api/test_routes_p3_mastery.py \
  tests/agent/test_progress.py \
  tests/agent/test_planner.py \
  tests/agent/test_planner_agent_tools.py \
  -q
```

Expected: selected tests pass.

- [ ] **Step 2: Run full backend suite**

Run:

```bash
cd backend
uv run pytest -q
```

Expected: full backend suite passes. If eval output fixtures fail because `milestones_json` now includes optional `id` fields, update only assertions that inspect the exact JSON shape and preserve behavioral assertions.

- [ ] **Step 3: Run frontend build**

Run:

```bash
cd frontend
pnpm build
```

Expected: build passes.

- [ ] **Step 4: Commit test/contract fixes if needed**

If this task required legitimate test updates:

```bash
git add backend/tests frontend/src
git commit -m "test: update plan milestone progression coverage"
```

If no files changed, skip commit.

---

## Task 8: Documentation Sync

**Files:**
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/ROADMAP.md`

- [ ] **Step 1: Update architecture docs**

In `docs/ARCHITECTURE.md`, update the Plan model section to mention:

```markdown
Plan milestones are normalized in `plan_milestones` for stable user progression.
`plans.milestones_json` is retained as a compatibility cache for Planner/eval paths during the P4 transition.

Milestone completion is self-reported plan progress. It never directly mutates `mastery`.
Mastery remains quiz/mistake evidence, and Plan uses mastery as an input for weak-topic intervention and validation prompts.

Plan events are stored in `plan_events` and power the Recent changes panel on `/plan`.
```

- [ ] **Step 2: Update roadmap**

In `docs/ROADMAP.md`, move or annotate the P4 backlog item:

```markdown
- [x] Plan milestone progression: stable milestone ids, manual complete/reopen, plan events, and mastery-linked quick quiz prompt. Spec: `docs/superpowers/specs/2026-05-26-plan-milestone-progression-design.md`
```

- [ ] **Step 3: Verify docs mention no stale read-only Plan claim**

Run:

```bash
rg -n "read-only|currently read-only|Drag-reorder milestones" docs README.md
```

Expected: any stale statement that says Plan is simply read-only is updated or clearly scoped to drag-reorder only.

- [ ] **Step 4: Commit docs**

```bash
git add docs/ARCHITECTURE.md docs/ROADMAP.md
git commit -m "docs: document plan milestone progression"
```

---

## Final Verification

- [ ] **Step 1: Confirm clean branch state before final manual run**

```bash
git status --short
```

Expected: empty.

- [ ] **Step 2: Run backend**

```bash
cd backend
uv run uvicorn app.main:app --reload --port 8000
```

Expected: backend starts and Alembic migrates the local SQLite DB to head.

- [ ] **Step 3: Run frontend**

```bash
cd frontend
pnpm dev
```

Expected: frontend starts on `http://localhost:5173`.

- [ ] **Step 4: Manual incognito smoke**

Use an incognito browser window:

1. Upload a PDF.
2. In Chat, ask `帮我做学习计划 on HyDE`.
3. Open Plan and confirm each row has a clickable completion icon.
4. Complete one milestone.
5. Confirm Plan progress increases and Recent changes shows `completed`.
6. Confirm Mastery card does not change from the milestone click alone.
7. Reopen the milestone.
8. Confirm Plan progress decreases and Recent changes shows `reopened`.
9. Complete a low-mastery topic milestone.
10. Click `Validate with quiz`.
11. Confirm Quiz opens with the milestone topic and waits for user to click Generate.

- [ ] **Step 5: Final status**

```bash
git status --short
git log --oneline -5
```

Expected: clean tree and recent task commits visible.

---

## Plan Self-Review

Spec coverage:

- Stable IDs: Task 1 and Task 2.
- Manual complete/reopen: Task 2, Task 3, Task 6.
- Plan events: Task 2, Task 3, Task 6.
- Mastery boundary: Task 3 tests, Task 7 regression, Task 8 docs.
- Low-mastery quick quiz: Task 3 validation hint, Task 6 UI bridge.
- Planner compatibility: Task 2 JSON cache, Task 4 schema/progress/prompt.
- Migration risk: Task 1 migration and Final Verification.

Intentional deferrals:

- Pending AI change-set confirmation UI is deferred.
- Drag reorder, Gantt, subtasks, dependencies, collaboration, and auto-start quiz are out of scope.

Red-flag scan:

- No red-flag tokens or unspecified edge-case steps are intentionally left in this plan.
