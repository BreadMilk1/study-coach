from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
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
