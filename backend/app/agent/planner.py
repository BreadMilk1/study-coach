"""Planner — deterministic node for P2.1-⑤ Plan chain.

State machine (mirrors P2.1-④ QuizMaster):
  - active_plan_id absent OR points to a missing plan → GENERATE
  - active_plan_id present + plan in DB                → CHECK-IN

GENERATE: extract topic, resolve/create goal, RAG-ground via retriever, ask LLM
for milestones JSON, persist via `update_study_plan`, optionally call
`generate_mindmap` when the user message contains a mindmap keyword.

CHECK-IN: read existing plan, compute deterministic progress summary, ask LLM
to adjust milestones, validate schema, persist if valid, format progress card.

Why deterministic (not LLM tool-calling) — same reason as QuizMaster: stable
baseline on small Ollama models; LLM tool-calling variant deferred to P2.2/P3.
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Callable

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.config import get_stream_writer
from pydantic import ValidationError

from app.db.repositories import (
    GoalRepository,
    MasteryRepository,
    MistakeRepository,
    PlanRepository,
)

from .progress import ProgressSummary, compute_progress
from .state import CoachState
from .tools.plan import generate_mindmap, update_study_plan
from .tools.schemas import Milestone


_DEFAULT_GOAL_TITLE = "Default Study Goal"

_TOPIC_TRAILING_RE = re.compile(
    r"\s*(?:画脑图|思维导图|mindmap|mind\s+map|脑图)\s*$",
    re.IGNORECASE,
)
_TOPIC_PUNCT_RE = re.compile(r"[?!？.,。！]+$")

_TOPIC_PATTERNS = [
    re.compile(r"学习计划.*on\s*(.+)", re.IGNORECASE),
    re.compile(r"plan.*on\s+(.+)", re.IGNORECASE),
    re.compile(r"plan\s+(?:for|for\s+studying)\s+(.+)", re.IGNORECASE),
    re.compile(r"复习计划.*on\s*(.+)", re.IGNORECASE),
]
_MINDMAP_KEYWORDS = ("脑图", "mindmap", "mind map", "思维导图")
_ARRAY_FENCE_RE = re.compile(r"```(?:json)?\s*(\[.*?\])\s*```", re.DOTALL)
_ARRAY_BARE_RE = re.compile(r"\[.*\]", re.DOTALL)

_CREATE_PLAN_KEYWORDS = (
    "帮我做",
    "帮我制定",
    "make me a plan",
    "make a plan",
    "create a plan",
    "重新做",
    "重做学习计划",
)


def _has_create_plan_keyword(text: str) -> bool:
    lowered = text.lower()
    return any(kw.lower() in lowered for kw in _CREATE_PLAN_KEYWORDS)

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _safe_writer():
    try:
        return get_stream_writer()
    except RuntimeError:
        return lambda _payload: None


def _extract_topic(text: str) -> str:
    for pattern in _TOPIC_PATTERNS:
        m = pattern.search(text)
        if m:
            raw = m.group(1).strip()
            raw = _TOPIC_TRAILING_RE.sub("", raw).strip()
            raw = _TOPIC_PUNCT_RE.sub("", raw).strip()
            return raw
    return text.strip()


def _has_mindmap_keyword(text: str) -> bool:
    lowered = text.lower()
    return any(kw in lowered for kw in _MINDMAP_KEYWORDS)


def _parse_milestones_json(raw: str) -> list[Milestone]:
    fence = _ARRAY_FENCE_RE.search(raw)
    if fence:
        candidate = fence.group(1)
    else:
        bare = _ARRAY_BARE_RE.search(raw)
        if not bare:
            return []
        candidate = bare.group(0)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    out: list[Milestone] = []
    for item in parsed:
        try:
            out.append(Milestone.model_validate(item))
        except ValidationError:
            continue
    return out


def _format_chunks(chunks: list[dict]) -> str:
    if not chunks:
        return "(no source chunks)"
    return "\n".join(f"[{i + 1}] {c.get('content', '')}" for i, c in enumerate(chunks))


def _format_generate_prompt(
    *,
    topic: str,
    chunks: list[dict],
    weak_topics: list[str],
    exam_date,
    today: date,
) -> str:
    template = (_PROMPTS_DIR / "planner_generate.txt").read_text(encoding="utf-8")
    context_section = ""
    if chunks:
        context_section = "Source chunks:\n" + _format_chunks(chunks) + "\n"
    return template.format(
        topic=topic,
        context_section=context_section,
        weak_topics=", ".join(weak_topics) if weak_topics else "(none)",
        exam_date=exam_date.isoformat() if exam_date else "(none)",
        today=today.isoformat(),
    )


def _format_check_in_prompt(
    *,
    current_milestones: list,
    progress: ProgressSummary,
    user_msg: str,
) -> str:
    template = (_PROMPTS_DIR / "planner_check_in.txt").read_text(encoding="utf-8")
    return template.format(
        current_milestones=json.dumps(current_milestones, ensure_ascii=False),
        done_count=progress.done_count,
        total_count=progress.total_count,
        overdue_titles=", ".join(m.get("title", "") for m in progress.overdue) or "(none)",
        weak_topics=", ".join(progress.weak_topics) if progress.weak_topics else "(none)",
        recent_mistake_count=progress.recent_mistake_count,
        user_msg=user_msg,
    )


def _format_milestones_md(milestones: list[Milestone]) -> str:
    lines = []
    for i, m in enumerate(milestones, start=1):
        check = "[x]" if m.done else "[ ]"
        due = f" — due {m.due_at}" if m.due_at else ""
        topic = f" *({m.topic})*" if m.topic else ""
        lines.append(f"{i}. {check} {m.title}{due}{topic}")
    return "\n".join(lines)


def _format_plan_output(milestones: list[Milestone], mindmap=None) -> str:
    parts = ["📋 **Study Plan**", "", _format_milestones_md(milestones)]
    if mindmap and mindmap.mermaid_src:
        parts += ["", "```mermaid", mindmap.mermaid_src, "```"]
    if mindmap and mindmap.markdown_outline:
        parts += ["", "**Outline**", mindmap.markdown_outline]
    return "\n".join(parts)


def _format_check_in_output(milestones: list[Milestone], progress: ProgressSummary) -> str:
    parts = [
        "📊 **Progress Check-in**",
        f"- Done: {progress.done_count} / {progress.total_count}",
    ]
    if progress.overdue:
        titles = ", ".join(m.get("title", "") for m in progress.overdue)
        parts.append(f"- Overdue: {titles}")
    if progress.weak_topics:
        parts.append(f"- Weak topics: {', '.join(progress.weak_topics)}")
    if progress.recent_mistake_count:
        parts.append(f"- Recent mistakes: {progress.recent_mistake_count}")
    parts += ["", "**Updated Plan**", _format_milestones_md(milestones)]
    return "\n".join(parts)


def build_planner(
    *,
    llm,
    plan_repo: PlanRepository,
    goal_repo: GoalRepository,
    mastery_repo: MasteryRepository,
    mistake_repo: MistakeRepository,
    retriever=None,
    now_fn: Callable[[], datetime] = datetime.utcnow,
):
    async def planner_node(state: CoachState) -> dict:
        writer = _safe_writer()
        user_id = state.get("user_id")
        user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        user_msg = user_msgs[-1].content if user_msgs else ""

        if not user_id:
            err = "Sign in (provide x-fingerprint header) to use the planner."
            writer({"type": "citations", "citations": []})
            writer({"type": "token", "text": err})
            return {"messages": [AIMessage(content=err)], "citations": []}

        active_plan_id = state.get("active_plan_id")
        existing_plan = None
        if active_plan_id:
            active_goals = goal_repo.list_active_for_user(user_id)
            if active_goals:
                existing_plan = plan_repo.get_by_goal(active_goals[0].id)
                # cloud-adapt: a richer plan ownership check could disambiguate multiple goals.

        # Cut ⑤i — if user explicitly asks for a NEW plan (creation keywords),
        # force GENERATE even when a plan already exists. `update_study_plan` is
        # upsert so the old plan gets overwritten in-place.
        force_generate = _has_create_plan_keyword(user_msg)

        # ---------- CHECK-IN ----------
        if existing_plan is not None and not force_generate:
            mastery_scores = state.get("mastery_scores", {}) or {}
            recent_mistakes = state.get("recent_mistakes", []) or []
            progress = compute_progress(
                existing_plan, mastery_scores, recent_mistakes, now=now_fn(),
            )
            prompt = _format_check_in_prompt(
                current_milestones=existing_plan.milestones_json,
                progress=progress,
                user_msg=user_msg,
            )
            try:
                response = await llm.ainvoke([HumanMessage(content=prompt)])
                raw = getattr(response, "content", "") or ""
            except Exception:
                raw = ""
            adjusted = _parse_milestones_json(raw)
            skip_note = ""
            if adjusted:
                patch = update_study_plan(
                    goal_id=existing_plan.goal_id,
                    milestones=adjusted,
                    plan_repo=plan_repo,
                )
                plan_id = patch.plan_id
                final_milestones = adjusted
            else:
                # cloud-adapt: cloud LLMs rarely fail schema; gemma3:4b is the reason we keep this branch.
                plan_id = existing_plan.id
                final_milestones = []
                for raw_m in existing_plan.milestones_json or []:
                    try:
                        final_milestones.append(Milestone.model_validate(raw_m))
                    except ValidationError:
                        continue
                skip_note = "\n\n⚠️ Auto-adjust skipped: model output didn't match plan schema."

            # Cut ⑤i — recompute progress against the post-update milestones so the
            # displayed counts match the displayed list.
            final_progress = compute_progress(
                type("FinalPlan", (), {"milestones_json": [m.model_dump() for m in final_milestones]})(),
                state.get("mastery_scores", {}) or {},
                state.get("recent_mistakes", []) or [],
                now=now_fn(),
            )
            text = _format_check_in_output(final_milestones, final_progress) + skip_note
            writer({"type": "citations", "citations": []})
            writer({"type": "token", "text": text})
            return {
                "messages": [AIMessage(content=text)],
                "citations": [],
                "active_plan_id": plan_id,
                "plan_action": "check_in",
                "last_context": "",
            }

        # ---------- GENERATE (or recover from stale active_plan_id) ----------
        topic = _extract_topic(user_msg)
        active = goal_repo.list_active_for_user(user_id)
        goal = active[0] if active else goal_repo.create(
            user_id=user_id, title=_DEFAULT_GOAL_TITLE,
        )

        chunks: list[dict] = []
        if retriever is not None:
            chunks = retriever.search(topic, top_k=5) or []

        mastery_scores = state.get("mastery_scores", {}) or {}
        weak_topics = [name for name, score in mastery_scores.items() if score < 0.4]
        prompt = _format_generate_prompt(
            topic=topic,
            chunks=chunks,
            weak_topics=weak_topics,
            exam_date=goal.exam_date,
            today=now_fn().date(),
        )
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        raw = getattr(response, "content", "") or ""
        milestones = _parse_milestones_json(raw)
        if not milestones:
            err = f"Couldn't draft a plan on '{topic}'. Try a clearer goal."
            writer({"type": "citations", "citations": []})
            writer({"type": "token", "text": err})
            return {"messages": [AIMessage(content=err)], "citations": []}

        patch = update_study_plan(
            goal_id=goal.id, milestones=milestones, plan_repo=plan_repo,
        )

        mindmap = None
        # cloud-adapt: when provider=cloud, can pass mindmap_default=True via dep injection.
        if _has_mindmap_keyword(user_msg):
            try:
                mindmap = await generate_mindmap(
                    topic=topic, milestones=milestones, llm=llm,
                )
            except Exception:
                mindmap = None

        text = _format_plan_output(milestones, mindmap)
        writer({"type": "citations", "citations": []})
        writer({"type": "token", "text": text})
        return {
            "messages": [AIMessage(content=text)],
            "citations": [],
            "active_plan_id": patch.plan_id,
            "plan_action": "generate",
            "last_context": _format_chunks(chunks),
        }

    return planner_node
