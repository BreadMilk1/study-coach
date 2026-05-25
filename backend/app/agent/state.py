from typing import Annotated, Literal, NotRequired, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class Citation(TypedDict):
    chunk_id: str
    source: str
    page: int
    span_start: int
    span_end: int


Intent = Literal["tutor", "quiz", "plan"]


class CoachState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    intent: NotRequired[Intent]
    citations: NotRequired[list[Citation]]

    # P2.1-② Judge Guard
    judge_score: NotRequired[float]
    retry_count: NotRequired[int]
    weak_dims: NotRequired[list[str]]
    judge_reasoning: NotRequired[str]
    degraded: NotRequired[bool]
    last_context: NotRequired[str]

    # P2.1-③ Memory Updater
    user_id: NotRequired[str]
    # Set by memory_hydrator at graph entry, consumed by tutor/quiz/plan nodes.
    mastery_scores: NotRequired[dict[str, float]]   # topic_name -> 0..1
    recent_mistakes: NotRequired[list[str]]          # mistake_ids due soon
    # Set by upstream nodes (P2.1-④ quiz_master etc.), drained by memory_writer at exit.
    pending_mastery_delta: NotRequired[dict[str, float]]  # topic_id -> delta
    pending_mistake: NotRequired[dict | None]             # {question_id, user_answer, srs_due_at?, ...}

    # P2.1-④ Quiz
    active_quiz_question_id: NotRequired[str | None]
    # Cut ④g: phase signal so Judge can skip the deterministic grade path.
    quiz_action: NotRequired[Literal["generate", "grade"]]

    # P2.1-⑤ Plan
    active_plan_id: NotRequired[str | None]
    plan_action: NotRequired[Literal["generate", "check_in"]]

    # P2.2 — agent loop instrumentation. Populated only when the agent_loop
    # planner mode is active; deterministic path leaves this field absent.
    agent_trace: NotRequired[dict]
