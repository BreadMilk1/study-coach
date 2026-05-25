"""Router: keyword-based intent classification for the multi-node LangGraph.

First cut of P2.1-①. LLM structured-output upgrade is deferred until real
Quiz / Plan nodes land in P2.1-④ / ⑤.

Conflict rule (locked with user): `quiz` > `plan` > `tutor`.
"""

from typing import Literal

Intent = Literal["tutor", "quiz", "plan"]

QUIZ_KEYWORDS: tuple[str, ...] = (
    "quiz",
    "测我",
    "考我",
    "出题",
    "practice",
    "test me",
)

PLAN_KEYWORDS: tuple[str, ...] = (
    "plan",
    "计划",
    "目标",
    "复习计划",
    "schedule",
    "学习计划",
    # P2.1-⑤ — check-in / edit phrasing so router catches second-turn messages
    # even without an `active_plan_id` (defense-in-depth; state-aware override
    # handles the in-flight case).
    "进度",
    "check in",
    "check-in",
    "调整",
)


def route_intent(message: str) -> Intent:
    lowered = message.lower()
    if any(kw in lowered for kw in QUIZ_KEYWORDS):
        return "quiz"
    if any(kw in lowered for kw in PLAN_KEYWORDS):
        return "plan"
    return "tutor"
