"""Plan tool chain (ARCHITECTURE.md §3, rows `update_study_plan` and `generate_mindmap`).

Two plain functions: `update_study_plan` (sync DB upsert) and `generate_mindmap`
(async LLM call). Stateless; explicit repo / llm args. No LangGraph imports.
"""
import re

from langchain_core.messages import HumanMessage

from app.db.repositories import PlanRepository

from .schemas import MindmapOut, PlanPatchOut, Milestone


_MERMAID_FENCE_RE = re.compile(r"```(?:mermaid)?\s*(mindmap.*?)```", re.DOTALL | re.IGNORECASE)
_MERMAID_BARE_RE = re.compile(r"(mindmap[\s\S]+?)(?:\n\n|\Z)", re.IGNORECASE)

_MINDMAP_PROMPT = """You are a study-plan mindmap generator.

Topic: {topic}

Milestones:
{milestone_lines}

Output two parts:

1. A valid mermaid `mindmap` block wrapped in ```mermaid fences, root labelled with the topic, branches for each milestone.
2. After the fence, a markdown bullet outline of the same structure.

Do NOT add any other commentary.
"""


def _milestone_lines(milestones: list[Milestone]) -> str:
    return "\n".join(f"- {m.title}" for m in milestones) or "- (no milestones)"


def _fallback_outline(topic: str, milestones: list[Milestone]) -> str:
    bullets = "\n".join(f"  - {m.title}" for m in milestones) or "  - (no milestones)"
    return f"- {topic}\n{bullets}"


def update_study_plan(
    *,
    goal_id: str,
    milestones: list[Milestone],
    plan_repo: PlanRepository,
) -> PlanPatchOut:
    milestone_dicts = [m.model_dump() for m in milestones]
    plan = plan_repo.update_milestones(goal_id=goal_id, milestones=milestone_dicts)
    return PlanPatchOut(plan_id=plan.id, updated_at=plan.updated_at)


async def generate_mindmap(
    *,
    topic: str,
    milestones: list[Milestone],
    llm,
) -> MindmapOut:
    prompt = _MINDMAP_PROMPT.format(
        topic=topic,
        milestone_lines=_milestone_lines(milestones),
    )
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        raw = getattr(response, "content", "") or ""
    except Exception:
        # cloud-adapt: cloud LLMs rarely crash here; the fallback keeps gemma3:4b stable.
        return MindmapOut(mermaid_src="", markdown_outline=_fallback_outline(topic, milestones))

    mermaid_src = ""
    fence = _MERMAID_FENCE_RE.search(raw)
    if fence:
        mermaid_src = fence.group(1).strip()
    else:
        bare = _MERMAID_BARE_RE.search(raw)
        if bare:
            mermaid_src = bare.group(1).strip()

    # Outline = whatever the LLM emitted minus the mermaid block; on failure, derive from milestones.
    outline = re.sub(r"```mermaid[\s\S]*?```", "", raw, flags=re.IGNORECASE).strip()
    if not outline:
        outline = _fallback_outline(topic, milestones)

    return MindmapOut(mermaid_src=mermaid_src, markdown_outline=outline)
