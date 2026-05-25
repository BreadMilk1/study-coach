import uuid
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import (
    ChatSession,
    Citation,
    Document,
    Goal,
    Mastery,
    Message,
    Mistake,
    Plan,
    Question,
    Topic,
    User,
)


def _uuid() -> str:
    return str(uuid.uuid4())


class UserRepository:
    def __init__(self, session: Session):
        self.session = session

    def get_or_create(self, fingerprint: str) -> User:
        stmt = select(User).where(User.fingerprint == fingerprint)
        existing = self.session.execute(stmt).scalar_one_or_none()
        if existing:
            return existing
        user = User(id=_uuid(), fingerprint=fingerprint)
        self.session.add(user)
        self.session.commit()
        self.session.refresh(user)
        return user


class DocumentRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(self, *, user_id: str, filename: str, hash_: str, chunks_count: int) -> Document:
        stmt = select(Document).where(
            Document.user_id == user_id,
            Document.hash == hash_,
        )
        existing = self.session.execute(stmt).scalar_one_or_none()
        if existing:
            return existing
        doc = Document(
            id=_uuid(),
            user_id=user_id,
            filename=filename,
            hash=hash_,
            chunks_count=chunks_count,
        )
        self.session.add(doc)
        self.session.commit()
        self.session.refresh(doc)
        return doc

    def list_for_user(self, user_id: str) -> list[Document]:
        stmt = select(Document).where(Document.user_id == user_id)
        return list(self.session.execute(stmt).scalars())


class GoalRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(
        self,
        *,
        user_id: str,
        title: str,
        exam_date: datetime | None = None,
    ) -> Goal:
        goal = Goal(
            id=_uuid(),
            user_id=user_id,
            title=title,
            exam_date=exam_date,
            status="active",
        )
        self.session.add(goal)
        self.session.commit()
        self.session.refresh(goal)
        return goal

    def list_active_for_user(self, user_id: str) -> list[Goal]:
        stmt = select(Goal).where(Goal.user_id == user_id, Goal.status == "active")
        return list(self.session.execute(stmt).scalars())


class TopicRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(
        self,
        *,
        goal_id: str,
        name: str,
        source_chunks: list[str] | None = None,
    ) -> Topic:
        topic = Topic(
            id=_uuid(),
            goal_id=goal_id,
            name=name,
            source_chunks=list(source_chunks or []),
        )
        self.session.add(topic)
        self.session.commit()
        self.session.refresh(topic)
        return topic

    def get_by_name(self, *, goal_id: str, name: str) -> Topic | None:
        stmt = select(Topic).where(Topic.goal_id == goal_id, Topic.name == name)
        return self.session.execute(stmt).scalar_one_or_none()

    def set_source_chunks(self, *, topic_id: str, chunk_ids: list[str]) -> None:
        """Refresh Topic.source_chunks (overwrite, not append) so the field
        reflects the latest quiz grounding."""
        topic = self.session.get(Topic, topic_id)
        if topic is None:
            return
        topic.source_chunks = list(chunk_ids)
        self.session.commit()


class PlanRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(self, *, goal_id: str, milestones_json: list) -> Plan:
        plan = Plan(
            id=_uuid(),
            goal_id=goal_id,
            milestones_json=list(milestones_json),
            updated_at=datetime.utcnow(),
        )
        self.session.add(plan)
        self.session.commit()
        self.session.refresh(plan)
        return plan

    def get_by_goal(self, goal_id: str) -> Plan | None:
        stmt = select(Plan).where(Plan.goal_id == goal_id)
        return self.session.execute(stmt).scalar_one_or_none()

    def update_milestones(self, *, goal_id: str, milestones: list) -> Plan:
        """Upsert: overwrite milestones_json on the goal's plan, else create."""
        existing = self.get_by_goal(goal_id)
        if existing is None:
            return self.create(goal_id=goal_id, milestones_json=milestones)
        existing.milestones_json = list(milestones)
        existing.updated_at = datetime.utcnow()
        self.session.commit()
        self.session.refresh(existing)
        return existing


class QuestionRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(
        self,
        *,
        topic_id: str,
        prompt: str,
        options_json: list[str],
        answer: str,
        explanation: str,
    ) -> Question:
        question = Question(
            id=_uuid(),
            topic_id=topic_id,
            prompt=prompt,
            options_json=list(options_json),
            answer=answer,
            explanation=explanation,
        )
        self.session.add(question)
        self.session.commit()
        self.session.refresh(question)
        return question

    def get_by_id(self, question_id: str) -> Question | None:
        return self.session.get(Question, question_id)


class MasteryRepository:
    def __init__(self, session: Session):
        self.session = session

    def upsert(self, *, user_id: str, topic_id: str, score: float) -> Mastery:
        existing = self.session.get(Mastery, (user_id, topic_id))
        if existing is not None:
            existing.score = score
            existing.last_reviewed = datetime.utcnow()
            self.session.commit()
            self.session.refresh(existing)
            return existing
        row = Mastery(
            user_id=user_id,
            topic_id=topic_id,
            score=score,
            last_reviewed=datetime.utcnow(),
        )
        self.session.add(row)
        self.session.commit()
        self.session.refresh(row)
        return row

    def apply_delta(self, *, user_id: str, topic_id: str, delta: float) -> float:
        existing = self.session.get(Mastery, (user_id, topic_id))
        current = existing.score if existing is not None else 0.0
        new_score = current + delta
        self.upsert(user_id=user_id, topic_id=topic_id, score=new_score)
        return new_score

    def get_for_user(self, user_id: str) -> dict[str, float]:
        stmt = (
            select(Topic.name, Mastery.score)
            .join(Mastery, Mastery.topic_id == Topic.id)
            .where(Mastery.user_id == user_id)
        )
        return {name: score for name, score in self.session.execute(stmt)}

    def list_for_user_detailed(self, user_id: str) -> list[tuple]:
        """Return (Topic, Mastery) pairs for the user.

        Used by P3 GET /api/mastery — it needs Topic.name + Mastery.score + last_reviewed.
        """
        stmt = (
            select(Topic, Mastery)
            .join(Mastery, Mastery.topic_id == Topic.id)
            .where(Mastery.user_id == user_id)
        )
        return [(topic, mastery) for topic, mastery in self.session.execute(stmt).all()]


class MistakeRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(
        self,
        *,
        user_id: str,
        question_id: str,
        user_answer: str,
        srs_due_at: datetime,
        srs_interval_days: int = 1,
        srs_ease: float = 2.5,
    ) -> Mistake:
        row = Mistake(
            id=_uuid(),
            user_id=user_id,
            question_id=question_id,
            user_answer=user_answer,
            srs_due_at=srs_due_at,
            srs_interval_days=srs_interval_days,
            srs_ease=srs_ease,
        )
        self.session.add(row)
        self.session.commit()
        self.session.refresh(row)
        return row

    def get_due_for_user(
        self,
        user_id: str,
        *,
        now: datetime | None = None,
        limit: int = 10,
    ) -> list[str]:
        cutoff = now or datetime.utcnow()
        stmt = (
            select(Mistake.id)
            .where(Mistake.user_id == user_id, Mistake.srs_due_at <= cutoff)
            .order_by(Mistake.srs_due_at.asc())
            .limit(limit)
        )
        return [row for row in self.session.execute(stmt).scalars()]

    def list_due_with_details(
        self,
        user_id: str,
        *,
        now: datetime | None = None,
        include_future: bool = False,
        limit: int = 20,
    ) -> list[tuple]:
        """Return (Mistake, Question, Topic) tuples for due mistakes, ordered by due ascending.

        Used by P3 GET /api/mistakes/due — the route needs question.prompt / .options /
        .answer / .explanation + topic.name in a single payload, but there is no
        SQLAlchemy relationship() declared, so we do the joins here explicitly.
        """
        cutoff = now or datetime.utcnow()
        stmt = (
            select(Mistake, Question, Topic)
            .join(Question, Question.id == Mistake.question_id)
            .join(Topic, Topic.id == Question.topic_id)
        )
        if include_future:
            stmt = stmt.where(Mistake.user_id == user_id)
        else:
            stmt = stmt.where(Mistake.user_id == user_id, Mistake.srs_due_at <= cutoff)
        stmt = stmt.order_by(Mistake.srs_due_at.asc()).limit(limit)
        return [(m, q, t) for m, q, t in self.session.execute(stmt).all()]


class ChatSessionRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(self, *, user_id: str, summary: str | None = None) -> ChatSession:
        chat = ChatSession(
            id=_uuid(),
            user_id=user_id,
            started_at=datetime.utcnow(),
            summary=summary,
        )
        self.session.add(chat)
        self.session.commit()
        self.session.refresh(chat)
        return chat

    def get_by_id(self, chat_id: str) -> ChatSession | None:
        return self.session.get(ChatSession, chat_id)


class MessageRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        tool_calls_json: list | None = None,
    ) -> Message:
        msg = Message(
            id=_uuid(),
            session_id=session_id,
            role=role,
            content=content,
            tool_calls_json=tool_calls_json,
            created_at=datetime.utcnow(),
        )
        self.session.add(msg)
        self.session.commit()
        self.session.refresh(msg)
        return msg

    def list_by_session(self, session_id: str) -> list[Message]:
        stmt = (
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at.asc())
        )
        return list(self.session.execute(stmt).scalars())


class CitationRepository:
    def __init__(self, session: Session):
        self.session = session

    def bulk_create_for_message(
        self,
        *,
        message_id: str,
        citations: list[dict],
    ) -> list[Citation]:
        rows = [
            Citation(
                id=_uuid(),
                message_id=message_id,
                chunk_id=c["chunk_id"],
                page=c["page"],
                span_start=c["span_start"],
                span_end=c["span_end"],
            )
            for c in citations
        ]
        self.session.add_all(rows)
        self.session.commit()
        for r in rows:
            self.session.refresh(r)
        return rows
