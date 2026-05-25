"""Pure progress summary for Plan CHECK-IN path (P2.1-⑤).

No LLM, no DB. Inputs are arbitrary objects with a `.milestones_json` attribute
(typed loosely so SQLAlchemy `Plan` rows and ad-hoc dicts both work).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

_WEAK_MASTERY_THRESHOLD = 0.4


@dataclass(frozen=True)
class ProgressSummary:
    done_count: int
    total_count: int
    overdue: list[dict]
    weak_topics: list[str] = field(default_factory=list)
    recent_mistake_count: int = 0


def _parse_due(due_at: Any) -> datetime | None:
    if not due_at:
        return None
    if isinstance(due_at, datetime):
        return due_at
    # Tolerate "YYYY-MM-DD" or full ISO; fall back to None on garbage.
    try:
        return datetime.fromisoformat(str(due_at))
    except ValueError:
        return None


def compute_progress(
    plan,
    mastery_scores: dict[str, float],
    recent_mistakes: list[str],
    *,
    now: datetime | None = None,
) -> ProgressSummary:
    now = now or datetime.utcnow()
    milestones = list(getattr(plan, "milestones_json", []) or [])
    done = [m for m in milestones if m.get("done")]
    overdue = []
    for m in milestones:
        if m.get("done"):
            continue
        due = _parse_due(m.get("due_at"))
        if due is not None and due < now:
            overdue.append(m)
    weak_topics = [name for name, score in mastery_scores.items() if score < _WEAK_MASTERY_THRESHOLD]
    return ProgressSummary(
        done_count=len(done),
        total_count=len(milestones),
        overdue=overdue,
        weak_topics=weak_topics,
        recent_mistake_count=len(recent_mistakes),
    )
