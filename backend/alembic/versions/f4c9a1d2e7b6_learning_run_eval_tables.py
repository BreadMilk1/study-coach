"""learning run harness bounded persistence tables

Revision ID: f4c9a1d2e7b6
Revises: 7a52fe598fd1
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "f4c9a1d2e7b6"
down_revision: Union[str, Sequence[str], None] = "7a52fe598fd1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "eval_runs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("experiment_id", sa.String(length=128), nullable=False),
        sa.Column("suite_execution_id", sa.String(length=36), nullable=True),
        sa.Column("task_case_id", sa.String(length=128), nullable=False),
        sa.Column("task_case_version", sa.String(length=64), nullable=False),
        sa.Column("variant_id", sa.String(length=128), nullable=False),
        sa.Column("run_profile", sa.String(length=32), nullable=False),
        sa.Column("lifecycle", sa.String(length=16), nullable=False),
        sa.Column("outcome", sa.String(length=32), nullable=True),
        sa.Column(
            "operational_error_json",
            sa.JSON(none_as_null=True),
            nullable=True,
        ),
        sa.Column("manifest_json", sa.JSON(), nullable=False),
        sa.Column("manifest_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "candidate_artifact_json",
            sa.JSON(none_as_null=True),
            nullable=True,
        ),
        sa.Column("artifact_hash", sa.String(length=64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.CheckConstraint(
            "lifecycle IN ('queued', 'running', 'finished', 'cancelled')",
            name="ck_eval_runs_lifecycle",
        ),
        sa.CheckConstraint(
            "outcome IS NULL OR outcome IN ('success', 'system_failed', 'timed_out', 'budget_exceeded')",
            name="ck_eval_runs_outcome",
        ),
        sa.CheckConstraint(
            "(lifecycle = 'finished' AND outcome IS NOT NULL) OR "
            "(lifecycle <> 'finished' AND outcome IS NULL)",
            name="ck_eval_runs_finished_outcome",
        ),
    )
    op.create_index(
        "ix_eval_runs_experiment_case",
        "eval_runs",
        ["experiment_id", "task_case_id"],
    )
    op.create_index(
        "ix_eval_runs_lifecycle_created_at",
        "eval_runs",
        ["lifecycle", "created_at"],
    )

    op.create_table(
        "eval_score_sets",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("scorer_id", sa.String(length=128), nullable=False),
        sa.Column("scorer_version", sa.String(length=64), nullable=False),
        sa.Column("artifact_input_hash", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("quality_verdict", sa.String(length=16), nullable=False),
        sa.Column("operational_error_code", sa.String(length=80), nullable=True),
        sa.Column("operational_error_message", sa.Text(), nullable=True),
        sa.Column(
            "aggregate_scores_json",
            sa.JSON(none_as_null=True),
            nullable=True,
        ),
        sa.Column("findings_json", sa.JSON(none_as_null=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["eval_runs.id"],
            name="fk_eval_score_sets_run_id_eval_runs",
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'partial', 'failed', 'cancelled')",
            name="ck_eval_score_sets_status",
        ),
        sa.CheckConstraint(
            "quality_verdict IN ('pass', 'fail', 'inconclusive', 'not_evaluated')",
            name="ck_eval_score_sets_quality_verdict",
        ),
    )
    op.create_index(
        "ix_eval_score_sets_run_created_at",
        "eval_score_sets",
        ["run_id", "created_at"],
    )
    op.create_index(
        "ix_eval_score_sets_status_created_at",
        "eval_score_sets",
        ["status", "created_at"],
    )

    op.create_table(
        "eval_scorer_executions",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("score_set_id", sa.String(length=36), nullable=False),
        sa.Column("scorer_id", sa.String(length=128), nullable=False),
        sa.Column("scorer_version", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("input_hash", sa.String(length=64), nullable=False),
        sa.Column("output_json", sa.JSON(none_as_null=True), nullable=True),
        sa.Column("operational_error_code", sa.String(length=80), nullable=True),
        sa.Column("operational_error_message", sa.Text(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("usage_json", sa.JSON(none_as_null=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.ForeignKeyConstraint(
            ["score_set_id"],
            ["eval_score_sets.id"],
            name="fk_eval_scorer_executions_score_set_id_eval_score_sets",
        ),
        sa.CheckConstraint(
            "status IN ('success', 'failed', 'skipped')",
            name="ck_eval_scorer_executions_status",
        ),
        sa.UniqueConstraint(
            "score_set_id",
            "scorer_id",
            "scorer_version",
            name="uq_eval_scorer_execution_identity",
        ),
    )
    op.create_index(
        "ix_eval_scorer_executions_score_set_created_at",
        "eval_scorer_executions",
        ["score_set_id", "created_at"],
    )


def downgrade() -> None:
    # Explicit child-first order preserves the database's FK deletion guard.
    op.drop_index(
        "ix_eval_scorer_executions_score_set_created_at",
        table_name="eval_scorer_executions",
    )
    op.drop_table("eval_scorer_executions")
    op.drop_index("ix_eval_score_sets_status_created_at", table_name="eval_score_sets")
    op.drop_index("ix_eval_score_sets_run_created_at", table_name="eval_score_sets")
    op.drop_table("eval_score_sets")
    op.drop_index("ix_eval_runs_lifecycle_created_at", table_name="eval_runs")
    op.drop_index("ix_eval_runs_experiment_case", table_name="eval_runs")
    op.drop_table("eval_runs")
