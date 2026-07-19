import uuid
from datetime import datetime

from sqlalchemy import delete, func, select
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
    PlanEvent,
    PlanMilestone,
    Question,
    Topic,
    User,
)


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


def _due_to_json(value):
    if value is None:
        return None
    if isinstance(value, datetime) and value.time() == datetime.min.time():
        return value.date().isoformat()
    return value.isoformat()


class DataLifecycleRepository:
    def __init__(self, session: Session):
        self.session = session

    def count_all(self) -> dict[str, int]:
        def count(model) -> int:
            stmt = select(func.count()).select_from(model)
            return int(self.session.execute(stmt).scalar_one())

        source_chunks = self.session.execute(
            select(func.coalesce(func.sum(Document.chunks_count), 0))
        ).scalar_one()
        return {
            "users": count(User),
            "documents": count(Document),
            "chat_sessions": count(ChatSession),
            "messages": count(Message),
            "citations": count(Citation),
            "goals": count(Goal),
            "topics": count(Topic),
            "plans": count(Plan),
            "plan_milestones": count(PlanMilestone),
            "plan_events": count(PlanEvent),
            "questions": count(Question),
            "mastery": count(Mastery),
            "mistakes": count(Mistake),
            "source_chunks": int(source_chunks),
        }

    def delete_learning_data(self, *, include_users: bool) -> None:
        for model in (
            Citation,
            Message,
            ChatSession,
            PlanEvent,
            PlanMilestone,
            Plan,
            Mistake,
            Mastery,
            Question,
            Topic,
            Goal,
            Document,
        ):
            self.session.execute(delete(model))
        if include_users:
            self.session.execute(delete(User))


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

    def get_or_create_google(self, *, google_id: str, email: str) -> User:
        stmt = select(User).where(User.google_id == google_id)
        existing = self.session.execute(stmt).scalar_one_or_none()
        if existing:
            if existing.email != email:
                existing.email = email
                self.session.commit()
            return existing
        user = User(
            id=_uuid(),
            fingerprint=f"google-{google_id}",
            google_id=google_id,
            email=email,
        )
        self.session.add(user)
        self.session.commit()
        self.session.refresh(user)
        return user

    def upgrade_anonymous(self, *, fingerprint: str, google_id: str, email: str) -> User:
        stmt = select(User).where(User.fingerprint == fingerprint, User.google_id.is_(None))
        existing = self.session.execute(stmt).scalar_one_or_none()
        if existing is None:
            return self.get_or_create_google(google_id=google_id, email=email)
        existing.google_id = google_id
        existing.email = email
        self.session.commit()
        self.session.refresh(existing)
        return existing

    def get_by_id(self, user_id: str) -> User | None:
        return self.session.get(User, user_id)


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
        stmt = select(Topic.id).where(Topic.goal_id == plan.goal_id, Topic.name == name).limit(1)
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
            .order_by(PlanEvent.created_at.desc(), PlanEvent.id.desc())
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
        event_entries: list[dict] = []

        for idx, raw in enumerate(milestones):
            title = str(raw.get("title") or "").strip()
            if not title:
                continue
            topic_name = raw.get("topic") or raw.get("topic_name")
            row = by_id.get(raw.get("id")) if raw.get("id") else None
            if row is None:
                row = by_key.get((title, topic_name))
            before = self._milestone_json(row) if row is not None else None
            if row is None:
                row = PlanMilestone(
                    id=raw.get("id") or _uuid(),
                    plan_id=plan.id,
                    topic_id=self._find_topic_id(plan=plan, raw=raw),
                    topic_name=topic_name,
                    title=title,
                    due_at=_parse_due_at(raw.get("due_at")),
                    done=bool(raw.get("done", False)),
                    completed_at=None,
                    sort_order=idx,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
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
            event_entries.append(
                {
                    "plan_id": plan.id,
                    "milestone_id": row.id,
                    "actor": "ai",
                    "action": "created" if before is None else "applied",
                    "before": before,
                    "after": self._milestone_json(row),
                    "reason": "Planner updated study plan",
                }
            )

        # Ensure milestone rows exist before FK-backed events reference them.
        self.session.flush()

        removed_rows = [row for row in existing_rows if row.id not in seen_ids]
        for row in removed_rows:
            stmt = select(PlanEvent).where(PlanEvent.milestone_id == row.id)
            for event in self.session.execute(stmt).scalars():
                event.milestone_id = None
        self.session.flush()

        for row in removed_rows:
            self.session.delete(row)
        self.session.flush()

        for entry in event_entries:
            self._log_event(**entry)
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
        row, _event = self.set_milestone_done_with_event(
            plan_id=plan_id,
            milestone_id=milestone_id,
            done=done,
            actor=actor,
            reason=reason,
        )
        return row

    def set_milestone_done_with_event(
        self,
        *,
        plan_id: str,
        milestone_id: str,
        done: bool,
        actor: str,
        reason: str,
    ) -> tuple[PlanMilestone, PlanEvent]:
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
        event = self._log_event(
            plan_id=plan_id,
            milestone_id=milestone_id,
            actor=actor,
            action="completed" if done else "reopened",
            before=before,
            after=self._milestone_json(row),
            reason=reason,
        )
        self.session.flush()
        self._sync_milestones_json(plan)
        self.session.commit()
        self.session.refresh(row)
        self.session.refresh(event)
        return row, event

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
            item["validation_recommended"] = row.done and (
                item["mastery_score"] is None or item["mastery_score"] < 0.5
            )
            out.append(item)
        return out

    def reorder_milestones(self, plan_id: str, milestone_ids: list[str]) -> None:
        for idx, mid in enumerate(milestone_ids):
            row = self.get_milestone(plan_id=plan_id, milestone_id=mid)
            if row is not None:
                row.sort_order = idx
                row.updated_at = datetime.utcnow()
        plan = self.session.get(Plan, plan_id)
        if plan:
            self._sync_milestones_json(plan)
        self.session.commit()


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
        options = list(options_json)
        if len(options) != 4 or not all(isinstance(o, str) for o in options):
            raise ValueError(
                f"options_json must be 4 strings (e.g. 'A) ...'), "
                f"got len={len(options)}: {str(options)[:120]!r}"
            )
        question = Question(
            id=_uuid(),
            topic_id=topic_id,
            prompt=prompt,
            options_json=options,
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

    def get_by_id(self, mistake_id: str) -> Mistake | None:
        return self.session.get(Mistake, mistake_id)

    def update_srs(
        self,
        mistake_id: str,
        *,
        interval_days: int,
        ease: float,
        due_at: datetime,
    ) -> Mistake | None:
        row = self.session.get(Mistake, mistake_id)
        if row is None:
            return None
        row.srs_interval_days = interval_days
        row.srs_ease = ease
        row.srs_due_at = due_at
        self.session.commit()
        self.session.refresh(row)
        return row

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

    def create(
        self,
        *,
        user_id: str,
        summary: str | None = None,
        chat_id: str | None = None,
    ) -> ChatSession:
        chat = ChatSession(
            id=chat_id or _uuid(),
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

    def get_for_user(self, *, chat_id: str, user_id: str) -> ChatSession | None:
        stmt = select(ChatSession).where(
            ChatSession.id == chat_id,
            ChatSession.user_id == user_id,
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def latest_for_user(self, user_id: str) -> ChatSession | None:
        stmt = (
            select(ChatSession)
            .where(ChatSession.user_id == user_id)
            .order_by(ChatSession.started_at.desc())
            .limit(1)
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def count_for_user(self, user_id: str) -> int:
        stmt = select(func.count()).select_from(ChatSession).where(
            ChatSession.user_id == user_id,
        )
        return int(self.session.execute(stmt).scalar_one())

    def count_active_days(self, user_id: str, *, since_days: int = 30) -> int:
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(days=since_days)
        stmt = (
            select(ChatSession.started_at)
            .where(ChatSession.user_id == user_id, ChatSession.started_at >= cutoff)
            .order_by(ChatSession.started_at.desc())
        )
        dates = {row.date() for row in self.session.execute(stmt).scalars()}
        if not dates:
            return 0
        today = datetime.utcnow().date()
        streak = 0
        check = today
        for d in sorted(dates, reverse=True):
            if d == check:
                streak += 1
                check = check - timedelta(days=1)
            elif d == check - timedelta(days=1):
                streak += 1
                check = d
            elif d < check:
                break
        return streak

    def activity_daily(self, user_id: str, *, days: int = 30) -> list[dict]:
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(days=days)
        stmt = (
            select(ChatSession.started_at)
            .where(ChatSession.user_id == user_id, ChatSession.started_at >= cutoff)
        )
        counts: dict[str, int] = {}
        for row in self.session.execute(stmt).scalars():
            d = row.date().isoformat()
            counts[d] = counts.get(d, 0) + 1
        result = []
        for i in range(days):
            d = (datetime.utcnow() - timedelta(days=days - 1 - i)).date().isoformat()
            result.append({"date": d, "count": counts.get(d, 0)})
        return result


class MessageRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        tool_calls_json: dict | list | None = None,
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

    def list_by_message_ids(self, message_ids: list[str]) -> list[Citation]:
        if not message_ids:
            return []
        stmt = select(Citation).where(Citation.message_id.in_(message_ids))
        return list(self.session.execute(stmt).scalars())
