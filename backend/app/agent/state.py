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
    # judge_score None means "no usable score this turn" (production
    # unavailable/skipped); a float is a verified payload score.
    judge_score: NotRequired[float | None]
    # D3 production judge status: valid (payload legal — not provider
    # completion evidence), unavailable (kept content + fixed warning), or
    # skipped (deterministic GRADE / CHECK-IN turns).
    judge_status: NotRequired[Literal["valid", "unavailable", "skipped"] | None]
    # Fixed safe failure classification for unavailable turns (already
    # normalize_llm_error-projected or a fixed sentence; never raw detail).
    judge_error: NotRequired[str | None]
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
    # None (a cleared prior-turn phase) is a valid resting value: production
    # turns reset it at the router and the consumer rewrites it.
    quiz_action: NotRequired[Literal["generate", "grade"] | None]

    # P2.1-⑤ Plan
    active_plan_id: NotRequired[str | None]
    plan_action: NotRequired[Literal["generate", "check_in"] | None]

    # P2.2 — agent loop instrumentation. Populated only when the agent_loop
    # planner mode is active; deterministic path leaves this field absent.
    agent_trace: NotRequired[dict]
