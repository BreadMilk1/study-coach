from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    fingerprint: Mapped[str] = mapped_column(String(64), unique=True)
    google_id: Mapped[str | None] = mapped_column(String(128), unique=True, nullable=True)
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("user_id", "hash", name="uq_user_doc_hash"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    filename: Mapped[str] = mapped_column(String(255))
    hash: Mapped[str] = mapped_column(String(64))
    chunks_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Goal(Base):
    __tablename__ = "goals"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    title: Mapped[str] = mapped_column(String(200))
    exam_date: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="active")


class Topic(Base):
    __tablename__ = "topics"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    goal_id: Mapped[str] = mapped_column(ForeignKey("goals.id"))
    name: Mapped[str] = mapped_column(String(200))
    source_chunks: Mapped[list] = mapped_column(JSON, default=list)


class Plan(Base):
    __tablename__ = "plans"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    goal_id: Mapped[str] = mapped_column(ForeignKey("goals.id"))
    milestones_json: Mapped[list] = mapped_column(JSON, default=list)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


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


class Question(Base):
    __tablename__ = "questions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    topic_id: Mapped[str] = mapped_column(ForeignKey("topics.id"))
    prompt: Mapped[str] = mapped_column(Text)
    options_json: Mapped[list] = mapped_column(JSON, default=list)
    answer: Mapped[str] = mapped_column(String(10))
    explanation: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Mastery(Base):
    __tablename__ = "mastery"
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), primary_key=True)
    topic_id: Mapped[str] = mapped_column(ForeignKey("topics.id"), primary_key=True)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    last_reviewed: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Mistake(Base):
    __tablename__ = "mistakes"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    question_id: Mapped[str] = mapped_column(ForeignKey("questions.id"))
    user_answer: Mapped[str] = mapped_column(String(10))
    srs_due_at: Mapped[datetime] = mapped_column(DateTime)
    srs_interval_days: Mapped[int] = mapped_column(Integer, default=1)
    srs_ease: Mapped[float] = mapped_column(Float, default=2.5)


class ChatSession(Base):
    """Chat session row. Named ChatSession to avoid clashing with `sqlalchemy.orm.Session`."""

    __tablename__ = "sessions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)


class Message(Base):
    __tablename__ = "messages"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"))
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    tool_calls_json: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Citation(Base):
    __tablename__ = "citations"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    message_id: Mapped[str] = mapped_column(ForeignKey("messages.id"))
    chunk_id: Mapped[str] = mapped_column(String(64))
    page: Mapped[int] = mapped_column(Integer)
    span_start: Mapped[int] = mapped_column(Integer)
    span_end: Mapped[int] = mapped_column(Integer)


class EvalRun(Base):
    """A local-instance-global, checksum-verified Learning Run."""

    __tablename__ = "eval_runs"
    __table_args__ = (
        CheckConstraint(
            "lifecycle IN ('queued', 'running', 'finished', 'cancelled')",
            name="ck_eval_runs_lifecycle",
        ),
        CheckConstraint(
            "outcome IS NULL OR outcome IN ('success', 'system_failed', 'timed_out', 'budget_exceeded')",
            name="ck_eval_runs_outcome",
        ),
        CheckConstraint(
            "(lifecycle = 'finished' AND outcome IS NOT NULL) OR "
            "(lifecycle <> 'finished' AND outcome IS NULL)",
            name="ck_eval_runs_finished_outcome",
        ),
        Index("ix_eval_runs_experiment_case", "experiment_id", "task_case_id"),
        Index("ix_eval_runs_lifecycle_created_at", "lifecycle", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    experiment_id: Mapped[str] = mapped_column(String(128), nullable=False)
    suite_execution_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    task_case_id: Mapped[str] = mapped_column(String(128), nullable=False)
    task_case_version: Mapped[str] = mapped_column(String(64), nullable=False)
    variant_id: Mapped[str] = mapped_column(String(128), nullable=False)
    run_profile: Mapped[str] = mapped_column(String(32), nullable=False)
    lifecycle: Mapped[str] = mapped_column(String(16), nullable=False, default="queued")
    outcome: Mapped[str | None] = mapped_column(String(32), nullable=True)
    operational_error_json: Mapped[dict | None] = mapped_column(
        JSON(none_as_null=True), nullable=True
    )
    manifest_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    manifest_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    candidate_artifact_json: Mapped[dict | None] = mapped_column(
        JSON(none_as_null=True), nullable=True
    )
    artifact_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class EvalScoreSet(Base):
    """One historical scorer-bundle evaluation for a frozen CandidateArtifact."""

    __tablename__ = "eval_score_sets"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'partial', 'failed', 'cancelled')",
            name="ck_eval_score_sets_status",
        ),
        CheckConstraint(
            "quality_verdict IN ('pass', 'fail', 'inconclusive', 'not_evaluated')",
            name="ck_eval_score_sets_quality_verdict",
        ),
        Index("ix_eval_score_sets_run_created_at", "run_id", "created_at"),
        Index("ix_eval_score_sets_status_created_at", "status", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("eval_runs.id", name="fk_eval_score_sets_run_id_eval_runs"),
        nullable=False,
    )
    scorer_id: Mapped[str] = mapped_column(String(128), nullable=False)
    scorer_version: Mapped[str] = mapped_column(String(64), nullable=False)
    artifact_input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    quality_verdict: Mapped[str] = mapped_column(
        String(16), nullable=False, default="not_evaluated"
    )
    operational_error_code: Mapped[str | None] = mapped_column(
        String(80), nullable=True
    )
    operational_error_message: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )
    aggregate_scores_json: Mapped[dict | None] = mapped_column(
        JSON(none_as_null=True), nullable=True
    )
    findings_json: Mapped[list | dict | None] = mapped_column(
        JSON(none_as_null=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class EvalScorerExecution(Base):
    """Append-only execution result for one scorer identity within a ScoreSet."""

    __tablename__ = "eval_scorer_executions"
    __table_args__ = (
        CheckConstraint(
            "status IN ('success', 'failed', 'skipped')",
            name="ck_eval_scorer_executions_status",
        ),
        UniqueConstraint(
            "score_set_id",
            "scorer_id",
            "scorer_version",
            name="uq_eval_scorer_execution_identity",
        ),
        Index(
            "ix_eval_scorer_executions_score_set_created_at",
            "score_set_id",
            "created_at",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    score_set_id: Mapped[str] = mapped_column(
        ForeignKey(
            "eval_score_sets.id",
            name="fk_eval_scorer_executions_score_set_id_eval_score_sets",
        ),
        nullable=False,
    )
    scorer_id: Mapped[str] = mapped_column(String(128), nullable=False)
    scorer_version: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    output_json: Mapped[dict | list | None] = mapped_column(
        JSON(none_as_null=True), nullable=True
    )
    operational_error_code: Mapped[str | None] = mapped_column(
        String(80), nullable=True
    )
    operational_error_message: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    usage_json: Mapped[dict | None] = mapped_column(
        JSON(none_as_null=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
