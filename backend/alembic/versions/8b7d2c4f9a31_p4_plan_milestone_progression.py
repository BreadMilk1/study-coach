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


def _parse_done(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "done"}
    return bool(value)


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
        sa.Column("source", sa.String(length=20), nullable=False, server_default="ai"),
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
            if not isinstance(raw, dict):
                continue
            title = str(raw.get("title") or "").strip()
            if not title:
                continue
            done = _parse_done(raw.get("done", False))
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
