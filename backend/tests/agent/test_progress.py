"""Cut ⑤a — compute_progress pure function tests.

Inputs:
- plan: object with `.milestones_json: list[dict]` (each dict has `title`, optional
  `due_at: str|None` ISO date, `done: bool`, optional `topic: str`).
- mastery_scores: dict[topic_name, float 0..1]
- recent_mistakes: list[str] of mistake_ids (count only, not contents)
- now: datetime injection seam

Output: ProgressSummary dataclass with done_count / total_count / overdue / weak_topics
/ recent_mistake_count.
"""
from datetime import datetime
from types import SimpleNamespace

from app.agent.progress import ProgressSummary, compute_progress


def _plan(milestones):
    return SimpleNamespace(milestones_json=milestones)


def test_empty_plan_yields_zeros():
    summary = compute_progress(_plan([]), {}, [], now=datetime(2026, 5, 22))
    assert summary == ProgressSummary(
        done_count=0,
        total_count=0,
        overdue=[],
        weak_topics=[],
        recent_mistake_count=0,
    )


def test_all_done_no_overdue():
    milestones = [
        {"title": "M1", "due_at": "2026-05-01", "done": True},
        {"title": "M2", "due_at": "2026-05-02", "done": True},
    ]
    summary = compute_progress(_plan(milestones), {}, [], now=datetime(2026, 5, 22))
    assert summary.done_count == 2
    assert summary.total_count == 2
    assert summary.overdue == []


def test_overdue_detects_past_due_at_not_done():
    milestones = [
        {"title": "Past undone", "due_at": "2026-05-10", "done": False},
        {"title": "Past done",   "due_at": "2026-05-10", "done": True},
        {"title": "Future",      "due_at": "2026-06-10", "done": False},
        {"title": "No due_at",   "due_at": None,         "done": False},
    ]
    summary = compute_progress(_plan(milestones), {}, [], now=datetime(2026, 5, 22))
    assert [m["title"] for m in summary.overdue] == ["Past undone"]


def test_weak_topics_lists_mastery_below_threshold():
    summary = compute_progress(
        _plan([]),
        {"HyDE": 0.2, "BM25": 0.7, "RAG": 0.39},
        [],
        now=datetime(2026, 5, 22),
    )
    assert sorted(summary.weak_topics) == ["HyDE", "RAG"]


def test_recent_mistake_count_passthrough():
    summary = compute_progress(
        _plan([]),
        {},
        ["m1", "m2", "m3"],
        now=datetime(2026, 5, 22),
    )
    assert summary.recent_mistake_count == 3
