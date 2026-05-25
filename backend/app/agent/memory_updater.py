"""Memory Updater nodes for the Study Coach LangGraph.

Two factory functions returning sync state-update callables:

  - build_memory_hydrator → graph ENTRY node.
    Loads mastery_scores (dict[topic_name, float]) + recent_mistakes (list[mistake_id])
    from the DB into state, so downstream nodes (Tutor / QuizMaster / Planner) can
    condition responses on the learner's history.

  - build_memory_writer → graph EXIT node (after Judge Guard).
    Drains state.pending_mastery_delta + state.pending_mistake into the DB and
    returns cleared values so they don't get re-written on the next turn.

Both no-op when state lacks user_id (anonymous / unit-test paths). This keeps the
P1/P2.1-② test fixtures usable without DB plumbing.

Wiring into the graph is Cut ④'s job; this module exposes pure builders.
"""
from datetime import datetime, timedelta
from typing import Callable

from app.db.repositories import MasteryRepository, MistakeRepository

from .state import CoachState


def build_memory_hydrator(
    *,
    mastery_repo: MasteryRepository,
    mistake_repo: MistakeRepository,
    now_fn: Callable[[], datetime] = datetime.utcnow,
    recent_mistakes_limit: int = 10,
):
    def memory_hydrator(state: CoachState) -> dict:
        user_id = state.get("user_id")
        if not user_id:
            return {}
        return {
            "mastery_scores": mastery_repo.get_for_user(user_id),
            "recent_mistakes": mistake_repo.get_due_for_user(
                user_id,
                now=now_fn(),
                limit=recent_mistakes_limit,
            ),
        }

    return memory_hydrator


def build_memory_writer(
    *,
    mastery_repo: MasteryRepository,
    mistake_repo: MistakeRepository,
    now_fn: Callable[[], datetime] = datetime.utcnow,
):
    def memory_writer(state: CoachState) -> dict:
        user_id = state.get("user_id")
        if not user_id:
            return {}

        pending_delta = state.get("pending_mastery_delta") or {}
        for topic_id, delta in pending_delta.items():
            mastery_repo.apply_delta(user_id=user_id, topic_id=topic_id, delta=delta)

        pending_mistake = state.get("pending_mistake")
        if pending_mistake:
            now = now_fn()
            mistake_repo.create(
                user_id=user_id,
                question_id=pending_mistake["question_id"],
                user_answer=pending_mistake["user_answer"],
                srs_due_at=pending_mistake.get(
                    "srs_due_at", now + timedelta(days=1)
                ),
                srs_interval_days=pending_mistake.get("srs_interval_days", 1),
                srs_ease=pending_mistake.get("srs_ease", 2.5),
            )

        return {"pending_mastery_delta": {}, "pending_mistake": None}

    return memory_writer
