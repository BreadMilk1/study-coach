"""persist independently auditable ScoreSet scorer snapshots

Revision ID: c8e1f0a2b4d3
Revises: f4c9a1d2e7b6
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c8e1f0a2b4d3"
down_revision: Union[str, Sequence[str], None] = "f4c9a1d2e7b6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "eval_score_sets",
        sa.Column("scorer_snapshot_json", sa.JSON(), nullable=False, server_default="{}"),
    )
    op.add_column(
        "eval_score_sets",
        sa.Column(
            "scorer_definition_hash",
            sa.String(length=64),
            nullable=False,
            server_default="",
        ),
    )


def downgrade() -> None:
    op.drop_column("eval_score_sets", "scorer_definition_hash")
    op.drop_column("eval_score_sets", "scorer_snapshot_json")
