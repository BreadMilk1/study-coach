from datetime import datetime

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.db.models import (
    Base,
    ChatSession,
    Citation,
    Document,
    EvalRun,
    EvalScoreSet,
    EvalScorerExecution,
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
from app.db.repositories import DataLifecycleRepository


EXPECTED_COUNTS = {
    "users": 1,
    "documents": 2,
    "chat_sessions": 1,
    "messages": 1,
    "citations": 1,
    "goals": 1,
    "topics": 1,
    "plans": 1,
    "plan_milestones": 1,
    "plan_events": 1,
    "questions": 1,
    "mastery": 1,
    "mistakes": 1,
    "source_chunks": 5,
}


@pytest.fixture
def session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,
    )

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def _seed_complete_graph(session):
    now = datetime(2026, 7, 20, 12, 0, 0)
    user = User(id="user-1", fingerprint="lifecycle-fingerprint", created_at=now)
    goal = Goal(id="goal-1", user_id=user.id, title="Pass the exam", status="active")
    topic = Topic(id="topic-1", goal_id=goal.id, name="SQL", source_chunks=["chunk-1"])
    plan = Plan(id="plan-1", goal_id=goal.id, milestones_json=[], updated_at=now)
    milestone = PlanMilestone(
        id="milestone-1",
        plan_id=plan.id,
        topic_id=topic.id,
        topic_name=topic.name,
        title="Review joins",
        sort_order=0,
        source="ai",
        created_at=now,
        updated_at=now,
    )
    plan_event = PlanEvent(
        id="plan-event-1",
        plan_id=plan.id,
        milestone_id=milestone.id,
        actor="ai",
        action="created",
        created_at=now,
    )
    question = Question(
        id="question-1",
        topic_id=topic.id,
        prompt="What does INNER JOIN return?",
        options_json=["matching rows"],
        answer="A",
        explanation="Rows with matching keys.",
        created_at=now,
    )
    mastery = Mastery(user_id=user.id, topic_id=topic.id, score=0.5, last_reviewed=now)
    mistake = Mistake(
        id="mistake-1",
        user_id=user.id,
        question_id=question.id,
        user_answer="B",
        srs_due_at=now,
        srs_interval_days=1,
        srs_ease=2.5,
    )
    document_one = Document(
        id="document-1",
        user_id=user.id,
        filename="notes.pdf",
        hash="document-hash-1",
        chunks_count=3,
        created_at=now,
    )
    document_two = Document(
        id="document-2",
        user_id=user.id,
        filename="slides.pdf",
        hash="document-hash-2",
        chunks_count=2,
        created_at=now,
    )
    chat_session = ChatSession(id="chat-session-1", user_id=user.id, started_at=now)
    message = Message(
        id="message-1",
        session_id=chat_session.id,
        role="assistant",
        content="Study SQL joins.",
        created_at=now,
    )
    citation = Citation(
        id="citation-1",
        message_id=message.id,
        chunk_id="chunk-1",
        page=1,
        span_start=0,
        span_end=10,
    )
    session.add(user)
    session.flush()
    session.add_all([document_one, document_two, goal, chat_session])
    session.flush()
    session.add_all([topic, plan])
    session.flush()
    session.add_all([milestone, question, mastery, message])
    session.flush()
    session.add_all([plan_event, mistake, citation])
    session.commit()


def test_learning_delete_preserves_users_and_can_be_rolled_back(session):
    _seed_complete_graph(session)
    repo = DataLifecycleRepository(session)

    assert repo.count_all() == EXPECTED_COUNTS

    repo.delete_learning_data(include_users=False)

    assert repo.count_all() == {**EXPECTED_COUNTS, "users": 1, **{
        key: 0 for key in EXPECTED_COUNTS if key != "users"
    }}

    session.rollback()
    assert repo.count_all() == EXPECTED_COUNTS

    repo.delete_learning_data(include_users=False)
    session.commit()
    repo.delete_learning_data(include_users=False)
    session.commit()

    assert repo.count_all() == {key: (1 if key == "users" else 0) for key in EXPECTED_COUNTS}


def test_factory_delete_removes_full_foreign_key_graph_and_is_idempotent(session):
    _seed_complete_graph(session)
    repo = DataLifecycleRepository(session)

    repo.delete_learning_data(include_users=True)
    session.commit()

    assert repo.count_all() == {key: 0 for key in EXPECTED_COUNTS}

    repo.delete_learning_data(include_users=True)
    session.commit()

    assert repo.count_all() == {key: 0 for key in EXPECTED_COUNTS}


def _seed_eval_rows(session):
    session.add(
        EvalRun(
            id="eval-run-1",
            experiment_id="tutor-prompt-regression-v1",
            task_case_id="tgqa-001",
            task_case_version="1",
            variant_id="tutor-v2",
            run_profile="evaluation",
            lifecycle="finished",
            outcome="success",
            manifest_json={"task_case_id": "tgqa-001"},
            manifest_hash="a" * 64,
            candidate_artifact_json={"answer": "x"},
            artifact_hash="b" * 64,
        )
    )
    session.add(
        EvalScoreSet(
            id="eval-score-1",
            run_id="eval-run-1",
            scorer_id="hybrid",
            scorer_version="hybrid-v1",
            scorer_snapshot_json={"scorer_id": "hybrid", "version": "hybrid-v1"},
            scorer_definition_hash="c" * 64,
            artifact_input_hash="b" * 64,
            status="completed",
            quality_verdict="pass",
            aggregate_scores_json={"groundedness": 4},
        )
    )
    session.add(
        EvalScorerExecution(
            id="eval-exec-1",
            score_set_id="eval-score-1",
            scorer_id="retrieval-integrity",
            scorer_version="v1",
            status="success",
            input_hash="b" * 64,
            output_json={"result": True},
        )
    )
    session.commit()


def test_eval_counts_are_independent_and_learning_delete_preserves_them(session):
    _seed_complete_graph(session)
    _seed_eval_rows(session)
    repo = DataLifecycleRepository(session)
    eval_before = repo.count_eval()
    assert eval_before["runs"] == 1
    assert eval_before["score_sets"] == 1
    assert eval_before["scorer_executions"] == 1
    assert eval_before["estimated_bytes"] > 0

    repo.delete_learning_data(include_users=False)
    session.commit()
    assert repo.count_eval() == eval_before


def test_factory_eval_delete_is_child_first_and_idempotent(session):
    _seed_eval_rows(session)
    repo = DataLifecycleRepository(session)
    repo.delete_eval_data()
    session.commit()
    assert repo.count_eval() == {
        "runs": 0,
        "score_sets": 0,
        "scorer_executions": 0,
        "estimated_bytes": 0,
    }
    repo.delete_eval_data()
    session.commit()
    assert repo.count_eval()["runs"] == 0
