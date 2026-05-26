import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Goal, Plan
from app.db.repositories import DocumentRepository, GoalRepository, PlanRepository
from app.db.session import get_session
from app.llm.provider import LLMConfig

from .deps import (
    get_document_processor,
    get_graph,
    get_judge_dependencies,
    get_llm_config,
    get_memory_hydrator,
    get_memory_writer,
    get_planner,
    get_planner_agent,
    get_planner_mode,
    get_quiz_master,
    get_quiz_master_agent,
    get_quiz_mode,
    get_retriever,
    get_current_user,
)

router = APIRouter(prefix="/api")

_SAME_MODEL_WARNING = (
    "⚠️ Self-check note: the judge is using the same model as the generator — "
    "self-preference bias possible. Set the x-judge-model header to a different "
    "model to mitigate.\n\n"
)


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class MilestoneOut(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str | None = None
    title: str
    due_at: str | None = None
    done: bool = False
    completed_at: str | None = None
    topic_id: str | None = None
    topic: str | None = None
    mastery_score: float | None = None
    validation_recommended: bool = False
    sort_order: int | None = None
    source: str | None = None


class PlanCurrentOut(BaseModel):
    plan_id: str
    goal_id: str
    goal_title: str
    milestones: list[MilestoneOut]
    updated_at: str


class MilestonePatchIn(BaseModel):
    done: bool


class PlanEventOut(BaseModel):
    id: str
    plan_id: str
    milestone_id: str | None = None
    actor: str
    action: str
    before_json: dict | None = None
    after_json: dict | None = None
    reason: str | None = None
    created_at: str


class ValidationHintOut(BaseModel):
    show_quick_quiz: bool
    topic: str | None = None
    reason: str | None = None


class MilestonePatchOut(BaseModel):
    plan: PlanCurrentOut
    event: PlanEventOut
    validation_hint: ValidationHintOut


class DocumentOut(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    filename: str
    chunks_count: int


class MistakeQuestionOut(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    prompt: str
    options: list[str]
    answer: str
    explanation: str


class MistakeDueOut(BaseModel):
    model_config = ConfigDict(extra="ignore")
    mistake_id: str
    question: MistakeQuestionOut
    due_at: str
    srs_interval_days: int
    srs_ease: float
    topic_name: str


class MasteryScoreOut(BaseModel):
    model_config = ConfigDict(extra="ignore")
    topic_id: str
    topic_name: str
    score: float
    last_reviewed: str


class MasteryOut(BaseModel):
    model_config = ConfigDict(extra="ignore")
    scores: list[MasteryScoreOut]
    weak_topics: list[str]
    overdue_milestones_count: int


def _plan_belongs_to_user(session: Session, *, user_id: str, plan_id: str):
    stmt = (
        select(Goal, Plan)
        .join(Plan, Plan.goal_id == Goal.id)
        .where(Plan.id == plan_id, Goal.user_id == user_id, Goal.status == "active")
    )
    row = session.execute(stmt).one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="no active plan for user")
    return row


def _plan_current_out(session: Session, *, user_id: str, goal, plan) -> PlanCurrentOut:
    repo = PlanRepository(session)
    milestone_dicts = repo.list_milestone_dicts(plan.id, user_id=user_id)
    if not milestone_dicts:
        milestone_dicts = [dict(m) for m in plan.milestones_json]
    return PlanCurrentOut(
        plan_id=plan.id,
        goal_id=goal.id,
        goal_title=goal.title,
        milestones=[MilestoneOut(**m) for m in milestone_dicts],
        updated_at=plan.updated_at.isoformat(),
    )


@router.get("/health")
def health(request: Request) -> dict:
    ollama_enabled = os.environ.get("OLLAMA_ENABLED", "true").lower() == "true"
    return {"status": "ok", "ollama_enabled": ollama_enabled}


class ToolCapableOut(BaseModel):
    tool_capable: bool
    model: str
    note: str


@router.get("/models/tool-check", response_model=ToolCapableOut)
async def tool_check(
    llm_config: Annotated[LLMConfig, Depends(get_llm_config)],
):
    """Ping the configured model with a dummy tool to detect tool-call support.

    Local models like gemma3:4b don't support tool calling at all —
    llm.bind_tools() returns a Runnable that silently ignores the tools,
    and the LLM never returns tool_calls. Cloud providers (OpenAI, Anthropic,
    Gemini) reliably support it.
    """
    from langchain_core.messages import HumanMessage
    from langchain_core.tools import tool

    from app.llm.provider import get_chat_model

    # cloud-adapt: cloud BYOK will always return True here; we could
    # short-circuit by provider instead of calling the LLM.

    @tool
    def ping() -> str:
        """Respond with 'pong' when called."""
        return "pong"

    try:
        llm = get_chat_model(llm_config)
        llm_with_tools = llm.bind_tools([ping])
        response = await llm_with_tools.ainvoke(
            [HumanMessage(content="Call the ping tool")]
        )
        tool_calls = getattr(response, "tool_calls", None) or []
        capable = len(tool_calls) > 0
        note = (
            "Model supports tool calling"
            if capable
            else "Model did not return any tool_calls — agent_loop mode unavailable"
        )
    except Exception as exc:
        capable = False
        note = f"Tool check failed: {type(exc).__name__}"

    return ToolCapableOut(
        tool_capable=capable,
        model=llm_config.model,
        note=note,
    )


class PingOut(BaseModel):
    ok: bool
    model: str
    latency_ms: float
    note: str


@router.get("/models/ping", response_model=PingOut)
async def ping_model(
    llm_config: Annotated[LLMConfig, Depends(get_llm_config)],
):
    """Lightweight connectivity test — single ping message, no tools."""
    import time

    from langchain_core.messages import HumanMessage

    from app.llm.provider import get_chat_model

    t0 = time.monotonic()
    try:
        llm = get_chat_model(llm_config)
        await llm.ainvoke([HumanMessage(content="ping")])
        latency_ms = round((time.monotonic() - t0) * 1000)
        return PingOut(
            ok=True,
            model=llm_config.model,
            latency_ms=latency_ms,
            note=f"Connected — responded in {latency_ms}ms",
        )
    except Exception as exc:
        latency_ms = round((time.monotonic() - t0) * 1000)
        return PingOut(
            ok=False,
            model=llm_config.model,
            latency_ms=latency_ms,
            note=f"Failed: {type(exc).__name__} — {exc}",
        )


@router.post("/documents")
async def upload_document(
    file: Annotated[UploadFile, File()],
    user_id: Annotated[str, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_session)],
    document_processor: Annotated[object, Depends(get_document_processor)],
    retriever: Annotated[object, Depends(get_retriever)],
):
    content = await file.read()
    file_hash = hashlib.sha256(content).hexdigest()
    tmp_path = Path(tempfile.gettempdir()) / f"sc_{file_hash}.pdf"
    tmp_path.write_bytes(content)

    chunks = document_processor.process_pdf(tmp_path)
    for c in chunks:
        c["source"] = file.filename or c.get("source", "uploaded.pdf")
    if chunks:
        retriever.add_chunks(chunks)

    doc = DocumentRepository(session).create(
        user_id=user_id,
        filename=file.filename or "uploaded.pdf",
        hash_=file_hash,
        chunks_count=len(chunks),
    )
    return {
        "document_id": doc.id,
        "filename": doc.filename,
        "chunks_count": doc.chunks_count,
    }


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@router.post("/chat")
async def chat(
    body: ChatRequest,
    user_id: Annotated[str, Depends(get_current_user)],
    graph: Annotated[object, Depends(get_graph)],
    judge: Annotated[dict, Depends(get_judge_dependencies)],
    quiz_master: Annotated[object, Depends(get_quiz_master)],
    planner: Annotated[object, Depends(get_planner)],
    planner_agent: Annotated[object, Depends(get_planner_agent)],
    planner_mode: Annotated[str, Depends(get_planner_mode)],
    quiz_master_agent: Annotated[object, Depends(get_quiz_master_agent)],
    quiz_mode: Annotated[str, Depends(get_quiz_mode)],
    memory_hydrator: Annotated[object, Depends(get_memory_hydrator)],
    memory_writer: Annotated[object, Depends(get_memory_writer)],
):
    # thread_id keys LangGraph's checkpointer; fall back to user_id when the
    # client doesn't track a session (single-conversation-per-user UX).
    thread_id = body.session_id or user_id

    async def event_stream():
        input_state = {
            "messages": [HumanMessage(content=body.message)],
            "user_id": user_id,
        }
        config = {
            "configurable": {
                "thread_id": thread_id,
                "judge_llm": judge["llm"],
                "quiz_master": quiz_master,
                "planner": planner,
                "planner_agent": planner_agent,
                "planner_mode": planner_mode,
                "quiz_master_agent": quiz_master_agent,
                "quiz_mode": quiz_mode,
                "memory_hydrator": memory_hydrator,
                "memory_writer": memory_writer,
            }
        }
        warning_yielded = False
        async for chunk in graph.astream(input_state, stream_mode="custom", config=config):
            yield _sse(chunk)
            # After the first non-empty citations event (Tutor path only),
            # surface the same-model bias warning once before tokens start.
            if (
                not warning_yielded
                and judge["same_model"]
                and chunk.get("type") == "citations"
                and chunk.get("citations")
            ):
                # Inline-emit the bias warning as a token so the frontend
                # (which renders all `token` events) surfaces it without
                # needing a new event-type case.
                yield _sse({"type": "token", "text": _SAME_MODEL_WARNING})
                warning_yielded = True
        yield _sse({"type": "done"})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/plans/current", response_model=PlanCurrentOut)
def get_plans_current(
    user_id: Annotated[str, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_session)],
):
    goals = GoalRepository(session).list_active_for_user(user_id)
    if not goals:
        raise HTTPException(status_code=404, detail="no active plan for user")
    goal = goals[0]  # one active goal per user (P2.1-③ invariant)
    plan = PlanRepository(session).get_by_goal(goal.id)
    if plan is None:
        raise HTTPException(status_code=404, detail="no active plan for user")
    return _plan_current_out(session, user_id=user_id, goal=goal, plan=plan)


@router.patch("/plans/{plan_id}/milestones/{milestone_id}", response_model=MilestonePatchOut)
def patch_plan_milestone(
    plan_id: str,
    milestone_id: str,
    body: MilestonePatchIn,
    user_id: Annotated[str, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_session)],
):
    goal, plan = _plan_belongs_to_user(session, user_id=user_id, plan_id=plan_id)
    repo = PlanRepository(session)
    try:
        updated, event = repo.set_milestone_done_with_event(
            plan_id=plan.id,
            milestone_id=milestone_id,
            done=body.done,
            actor="user",
            reason="User marked milestone complete" if body.done else "User reopened milestone",
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    refreshed = repo.get_by_goal(goal.id)
    current = _plan_current_out(session, user_id=user_id, goal=goal, plan=refreshed)
    milestone_out = next((m for m in current.milestones if m.id == updated.id), None)
    show_quiz = bool(body.done and milestone_out and milestone_out.validation_recommended)
    return MilestonePatchOut(
        plan=current,
        event=PlanEventOut(
            id=event.id,
            plan_id=event.plan_id,
            milestone_id=event.milestone_id,
            actor=event.actor,
            action=event.action,
            before_json=event.before_json,
            after_json=event.after_json,
            reason=event.reason,
            created_at=event.created_at.isoformat(),
        ),
        validation_hint=ValidationHintOut(
            show_quick_quiz=show_quiz,
            topic=milestone_out.topic if milestone_out else None,
            reason="Topic mastery is still below the validation threshold" if show_quiz else None,
        ),
    )


@router.get("/plans/{plan_id}/events", response_model=list[PlanEventOut])
def get_plan_events(
    plan_id: str,
    user_id: Annotated[str, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_session)],
    limit: int = 20,
):
    _goal, plan = _plan_belongs_to_user(session, user_id=user_id, plan_id=plan_id)
    return [
        PlanEventOut(
            id=e.id,
            plan_id=e.plan_id,
            milestone_id=e.milestone_id,
            actor=e.actor,
            action=e.action,
            before_json=e.before_json,
            after_json=e.after_json,
            reason=e.reason,
            created_at=e.created_at.isoformat(),
        )
        for e in PlanRepository(session).list_events(plan.id, limit=limit)
    ]


@router.get("/documents", response_model=list[DocumentOut])
def get_documents(
    user_id: Annotated[str, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_session)],
):
    docs = DocumentRepository(session).list_for_user(user_id)
    return [
        DocumentOut(id=d.id, filename=d.filename, chunks_count=d.chunks_count)
        for d in docs
    ]


@router.get("/mistakes/due", response_model=list[MistakeDueOut])
def get_mistakes_due(
    user_id: Annotated[str, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_session)],
    limit: int = 20,
    include_future: bool = False,
):
    from app.db.repositories import MistakeRepository

    rows = MistakeRepository(session).list_due_with_details(
        user_id,
        limit=limit,
        include_future=include_future,
    )
    return [
        MistakeDueOut(
            mistake_id=m.id,
            question=MistakeQuestionOut(
                id=q.id,
                prompt=q.prompt,
                options=q.options_json,
                answer=q.answer,
                explanation=q.explanation,
            ),
            due_at=m.srs_due_at.isoformat(),
            srs_interval_days=m.srs_interval_days,
            srs_ease=m.srs_ease,
            topic_name=t.name,
        )
        for m, q, t in rows
    ]


class MistakeReviewIn(BaseModel):
    answer: str  # "A" / "B" / "C" / "D"


class MistakeReviewOut(BaseModel):
    correct: bool
    correct_answer: str
    explanation: str
    new_interval_days: int
    next_due_at: str


@router.post("/mistakes/{mistake_id}/review", response_model=MistakeReviewOut)
def review_mistake(
    mistake_id: str,
    body: MistakeReviewIn,
    user_id: Annotated[str, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_session)],
):
    from app.db.repositories import MasteryRepository, MistakeRepository, QuestionRepository
    from app.srs.sm2 import next_schedule

    mistake = MistakeRepository(session).get_by_id(mistake_id)
    if mistake is None or mistake.user_id != user_id:
        raise HTTPException(status_code=404, detail="mistake not found")

    question = QuestionRepository(session).get_by_id(mistake.question_id)
    if question is None:
        raise HTTPException(status_code=404, detail="question not found")

    normalized = body.answer.strip().upper()
    correct = normalized == question.answer.strip().upper()
    quality = 4 if correct else 2

    sched = next_schedule(
        quality=quality,
        previous_interval_days=mistake.srs_interval_days,
        previous_ease=mistake.srs_ease,
    )
    MistakeRepository(session).update_srs(
        mistake_id=mistake.id,
        interval_days=sched.interval_days,
        ease=sched.ease,
        due_at=sched.due_at,
    )

    delta = 0.1 if correct else -0.05
    MasteryRepository(session).apply_delta(
        user_id=user_id,
        topic_id=question.topic_id,
        delta=delta,
    )

    return MistakeReviewOut(
        correct=correct,
        correct_answer=question.answer,
        explanation=question.explanation,
        new_interval_days=sched.interval_days,
        next_due_at=sched.due_at.isoformat(),
    )


@router.get("/mastery", response_model=MasteryOut)
def get_mastery(
    user_id: Annotated[str, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_session)],
):
    from datetime import datetime

    from app.db.repositories import MasteryRepository

    rows = MasteryRepository(session).list_for_user_detailed(user_id)
    scores = [
        MasteryScoreOut(
            topic_id=topic.id,
            topic_name=topic.name,
            score=m.score,
            last_reviewed=m.last_reviewed.isoformat(),
        )
        for topic, m in rows
    ]
    weak = sorted([s for s in scores if s.score < 0.5], key=lambda s: s.score)[:5]
    weak_names = [s.topic_name for s in weak]

    goals = GoalRepository(session).list_active_for_user(user_id)
    overdue = 0
    if goals:
        plan = PlanRepository(session).get_by_goal(goals[0].id)
        if plan is not None:
            now = datetime.utcnow().isoformat()
            for m in plan.milestones_json:
                if not m.get("done") and m.get("due_at") and m["due_at"] < now:
                    overdue += 1
    return MasteryOut(scores=scores, weak_topics=weak_names, overdue_milestones_count=overdue)
