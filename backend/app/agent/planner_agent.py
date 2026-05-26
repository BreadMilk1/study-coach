"""LLM tool-calling Planner agent — P2.2 ablation variant.

Parallels `planner.py` (the deterministic baseline). Same LangGraph node
contract — async (state) -> dict update — so `plan_node` can dispatch to
either based on a per-request mode flag. Same SSE contract — citations
event, single token event with the final markdown, done event.

The module exposes:
  - `_make_planner_tools(...)`: closure factory producing 5 LangChain @tool
    wrappers (retriever_search / get_existing_plan / update_study_plan /
    generate_mindmap / compute_progress). Cut P2.2-①a — implemented here.
  - `AgentTrace`: instrumentation dataclass for the eval matrix. Cut P2.2-①b.
  - `build_planner_agent(...)`: top-level factory returning the async node
    callable. Cut P2.2-①c onward.

`_make_planner_tools` is INTENTIONALLY a private name — the loop is the only
caller; downstream code reaches the agent via the factory.

Business logic is NOT reimplemented here. Three tools delegate to existing
pure functions (`update_study_plan_fn`, `generate_mindmap_fn`,
`compute_progress_fn`); two tools (`retriever_search`, `get_existing_plan`)
are direct three-line wrappers — abstracting them would violate YAGNI.
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime
from typing import Callable, Literal

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.config import get_stream_writer
from pydantic import ValidationError

from app.agent.agent_trace import AgentTrace
from app.agent.progress import compute_progress as compute_progress_fn
from app.agent.state import CoachState
from app.agent.tools.plan import generate_mindmap as generate_mindmap_fn
from app.agent.tools.plan import update_study_plan as update_study_plan_fn
from app.agent.tools.schemas import Milestone
from app.db.repositories import (
    GoalRepository,
    MasteryRepository,
    MistakeRepository,
    PlanRepository,
)


_DEFAULT_GOAL_TITLE = "Default Study Goal"


def _make_planner_tools(
    *,
    user_id: str,
    llm,
    retriever,
    plan_repo: PlanRepository | None,
    goal_repo: GoalRepository | None,
    mastery_scores: dict[str, float],
    recent_mistakes: list[str],
    now_fn: Callable[[], datetime] = datetime.utcnow,
) -> list:
    """Build a per-request tool set with user/session context baked in.

    The model sees only the public args (the @tool decorator strips closure
    variables from the generated JSON schema). user_id is NEVER an LLM-visible
    arg — identity is not a behavior input.
    """

    @tool
    def retriever_search(query: str, top_k: int = 5) -> str:
        """Search the user's uploaded PDF corpus for chunks relevant to a topic.
        Use BEFORE drafting a study plan to ground milestones in real sources.
        Returns a JSON list: [{"chunk_id","content","page"}, ...].
        """
        if retriever is None:
            return "[]"
        chunks = retriever.search(query, top_k=top_k) or []
        return json.dumps(chunks, ensure_ascii=False)

    @tool
    def get_existing_plan() -> str:
        """Return the user's currently active study plan, if any.
        Use on CHECK-IN turns to see what plan exists before adjusting.
        Returns JSON {"plan_id","milestones","updated_at"} or the literal "null".
        """
        if goal_repo is None or plan_repo is None:
            return "null"
        active = goal_repo.list_active_for_user(user_id)
        if not active:
            return "null"
        plan = plan_repo.get_by_goal(active[0].id)
        if plan is None:
            return "null"
        return json.dumps({
            "plan_id": plan.id,
            "milestones": plan.milestones_json,
            "updated_at": plan.updated_at.isoformat(),
        }, ensure_ascii=False)

    @tool
    def update_study_plan(milestones: list[dict]) -> str:
        """Persist a list of milestones as the user's study plan (upsert).
        Each milestone: {id?:str, title:str, due_at:str|null, done:bool,
        topic:str|null, topic_id?:str|null}. On CHECK-IN, preserve id/topic_id
        from the existing plan when present.
        Call AFTER you've decided on the final milestone list.
        Returns JSON {"plan_id","milestones_count","updated_at"}.
        """
        if goal_repo is None or plan_repo is None:
            return json.dumps({"error": "repository not available"})
        active = goal_repo.list_active_for_user(user_id)
        goal = active[0] if active else goal_repo.create(
            user_id=user_id, title=_DEFAULT_GOAL_TITLE,
        )
        try:
            validated = [Milestone.model_validate(m) for m in milestones]
        except ValidationError as exc:
            err = exc.errors()[0] if exc.errors() else {"loc": [], "msg": str(exc)}
            loc = ".".join(str(x) for x in err.get("loc", []))
            return json.dumps({"error": f"invalid milestone at {loc}: {err.get('msg', '')}"})
        out = update_study_plan_fn(
            goal_id=goal.id, milestones=validated, plan_repo=plan_repo,
        )
        return json.dumps({
            "plan_id": out.plan_id,
            "milestones_count": len(validated),
            "updated_at": out.updated_at.isoformat(),
        }, ensure_ascii=False)

    @tool
    async def generate_mindmap(topic: str, milestones: list[dict]) -> str:
        """Generate a mermaid mindmap + markdown outline for a study plan.
        Call ONLY when the user asks for a mindmap / 脑图 / 思维导图.
        Returns JSON {"mermaid_src","markdown_outline"}.
        """
        try:
            validated = [Milestone.model_validate(m) for m in milestones]
        except ValidationError as exc:
            err = exc.errors()[0] if exc.errors() else {"loc": [], "msg": str(exc)}
            loc = ".".join(str(x) for x in err.get("loc", []))
            return json.dumps({"error": f"invalid milestone at {loc}: {err.get('msg', '')}"})
        out = await generate_mindmap_fn(topic=topic, milestones=validated, llm=llm)
        return json.dumps({
            "mermaid_src": out.mermaid_src,
            "markdown_outline": out.markdown_outline,
        }, ensure_ascii=False)

    @tool
    def compute_progress() -> str:
        """Compute deterministic progress summary for the user's active plan.
        Use on CHECK-IN turns to see what's done/overdue before adjusting.
        Returns JSON {"done_count","total_count","overdue","weak_topics","recent_mistake_count"}.
        """
        if goal_repo is None or plan_repo is None:
            return json.dumps({"error": "repository not available"})
        active = goal_repo.list_active_for_user(user_id)
        if not active:
            return json.dumps({"error": "No active goal"})
        plan = plan_repo.get_by_goal(active[0].id)
        if plan is None:
            return json.dumps({"error": "No active plan"})
        progress = compute_progress_fn(
            plan, mastery_scores, recent_mistakes, now=now_fn(),
        )
        return json.dumps({
            "done_count": progress.done_count,
            "total_count": progress.total_count,
            "overdue": [m.get("title", "") for m in progress.overdue],
            "weak_topics": progress.weak_topics,
            "recent_mistake_count": progress.recent_mistake_count,
        }, ensure_ascii=False)

    return [
        retriever_search,
        get_existing_plan,
        update_study_plan,
        generate_mindmap,
        compute_progress,
    ]


_AGENT_SYSTEM_PROMPT = """You are a study coach planner agent.

The user wants either a new study plan or a check-in on an existing plan.

Your job:
1. Read the user's message to understand what topic they care about.
2. If you don't know what plan (if any) they already have, call `get_existing_plan` first.
3. If they want a NEW plan or use explicit re-plan keywords (帮我做 / make a plan / 重做):
   - Call `retriever_search` with the topic to ground in their source materials.
   - Call `update_study_plan` with 3-7 specific, dated milestones.
4. If they want a CHECK-IN (existing plan + 进度 / check-in / 调整 / etc):
   - Call `compute_progress` to see what's done/overdue.
   - Call `update_study_plan` with the adjusted milestone list.
   - Preserve milestone id/topic_id values from the existing plan.
   - Do not add, delete, or reorder milestones during check-in.
5. If they mention mindmap / 脑图 / mind map / 思维导图: call `generate_mindmap`.
6. When done, write a short markdown summary for the user with the milestones (and mindmap if generated). Do NOT call more tools after the summary.

Today is {today}. Be concise. Call tools to act, prose to summarize."""

# cloud-adapt: tool descriptions can be terser for cloud models; the long-form
# "When to use" guidance above is necessary for small Ollama models only.

_LLM_FAILED_MSG = "⚠️ Could not reach the planner model. Please try again."
_BUDGET_EXHAUSTED_MSG = (
    "⚠️ Agent exceeded reasoning budget (10 turns). The last partial plan was not persisted."
)

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


def _extract_topic_for_agent_prompt(text: str) -> str:
    """Mirror of planner._extract_topic — kept here so the agent prompt has a
    sensible topic snippet if needed in future variants. Currently used by
    tests as a regression anchor for the P2.1-⑤i char-set-vs-word-suffix fix.
    """
    for pattern in _TOPIC_PATTERNS:
        m = pattern.search(text)
        if m:
            raw = m.group(1).strip()
            raw = _TOPIC_TRAILING_RE.sub("", raw).strip()
            raw = _TOPIC_PUNCT_RE.sub("", raw).strip()
            return raw
    return text.strip()


def _safe_writer():
    """get_stream_writer() with a no-op fallback for direct unit-test calls."""
    try:
        return get_stream_writer()
    except RuntimeError:
        return lambda _payload: None


def _last_human_msg(state: CoachState) -> str:
    user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    return user_msgs[-1].content if user_msgs else ""


def _infer_plan_action(trace: AgentTrace) -> Literal["generate", "check_in"]:
    """Agent doesn't use a done() sentinel by design. Infer from trace.

    Rule: if the model called get_existing_plan AND that call returned a
    non-null plan blob, treat the turn as a CHECK-IN. Otherwise the model
    was drafting a fresh plan (or no tools at all → fallback to generate so
    the judge still routes through the rubric path).
    """
    if trace.get_existing_plan_returned_nonnull():
        return "check_in"
    return "generate"


async def _safe_invoke_tool(tool_map, tc, trace: AgentTrace) -> str:
    """Dispatch one tool call. Tool errors are RECOVERABLE — they go back to
    the model as a ToolMessage so it can self-correct. Only LLM-level errors
    in the parent loop short-circuit to degrade.
    """
    name = tc.get("name", "")
    args = tc.get("args", {}) or {}
    handler = tool_map.get(name)
    if handler is None:
        msg = f"Error: unknown tool '{name}'. Available: {sorted(tool_map.keys())}"
        trace.record_tool_call(name, args, msg, error=True)
        return msg
    try:
        output = await handler.ainvoke(args)
    except Exception as exc:
        output = f"Error calling {name}: {exc}. Check arg types and retry."
        trace.record_tool_call(name, args, output, error=True)
        return output
    # Tools also signal recoverable failures via {"error": ...} JSON payloads
    # (e.g. update_study_plan's ValidationError branch). Detect those so the
    # trace records the failure and the model can self-correct on the next
    # iteration.
    output_str = str(output)
    is_tool_error = False
    if output_str.startswith("{"):
        try:
            parsed = json.loads(output_str)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict) and "error" in parsed:
            is_tool_error = True
    trace.record_tool_call(name, args, output_str, error=is_tool_error)
    return output_str


def _format_final_output(writer, trace: AgentTrace, last_response) -> dict:
    plan_action = _infer_plan_action(trace)
    plan_id = trace.last_persisted_plan_id()
    final_text = getattr(last_response, "content", "") or ""
    if not isinstance(final_text, str):
        # AIMessage.content can be a list of content blocks for some providers
        final_text = "".join(
            (b.get("text", "") if isinstance(b, dict) else str(b))
            for b in final_text
        )

    writer({"type": "citations", "citations": []})
    writer({"type": "token", "text": final_text})

    return {
        "messages": [AIMessage(content=final_text)],
        "citations": [],
        "active_plan_id": plan_id,
        "plan_action": plan_action,
        "last_context": trace.aggregated_retriever_context(),
        "agent_trace": trace.serialize(),
    }


def _format_degrade_output(writer, trace: AgentTrace, reason: str) -> dict:
    if reason == "llm_call_failed":
        text = _LLM_FAILED_MSG
    elif reason == "budget_exhausted":
        text = _BUDGET_EXHAUSTED_MSG
    else:
        text = "⚠️ Planner agent stopped unexpectedly."

    writer({"type": "citations", "citations": []})
    writer({"type": "token", "text": text})

    return {
        "messages": [AIMessage(content=text)],
        "citations": [],
        # Do NOT set active_plan_id — the loop didn't reach a confirmed persist
        "plan_action": _infer_plan_action(trace),
        "last_context": trace.aggregated_retriever_context(),
        "agent_trace": trace.serialize(),
        "degraded": True,
    }


def build_planner_agent(
    *,
    llm,
    plan_repo: PlanRepository,
    goal_repo: GoalRepository,
    mastery_repo: MasteryRepository,
    mistake_repo: MistakeRepository,
    retriever=None,
    now_fn: Callable[[], datetime] = datetime.utcnow,
    max_iter: int = 10,
    system_prompt: str = _AGENT_SYSTEM_PROMPT,
):
    """Factory returning an async LangGraph node that runs an LLM tool-calling
    agent loop. Mirror of `build_planner` (deterministic) in shape — same
    state→dict contract, same SSE emit pattern, same factory kwargs surface
    plus max_iter / system_prompt for experimentation.
    """
    # cloud-adapt: cloud BYOK provider can raise max_iter to 20-30 here.

    async def planner_agent_node(state: CoachState) -> dict:
        writer = _safe_writer()
        user_id = state.get("user_id")
        user_msg = _last_human_msg(state)

        if not user_id:
            err = "Sign in (provide x-fingerprint header) to use the planner."
            writer({"type": "citations", "citations": []})
            writer({"type": "token", "text": err})
            return {"messages": [AIMessage(content=err)], "citations": []}

        tools = _make_planner_tools(
            user_id=user_id,
            llm=llm,
            retriever=retriever,
            plan_repo=plan_repo,
            goal_repo=goal_repo,
            mastery_scores=state.get("mastery_scores", {}) or {},
            recent_mistakes=state.get("recent_mistakes", []) or [],
            now_fn=now_fn,
        )
        tool_map = {t.name: t for t in tools}
        llm_with_tools = llm.bind_tools(tools)

        today = now_fn().date().isoformat()
        messages: list[BaseMessage] = [
            SystemMessage(content=system_prompt.format(today=today)),
            HumanMessage(content=user_msg),
        ]
        trace = AgentTrace(t_start=time.monotonic())

        for iteration in range(max_iter):
            try:
                response = await llm_with_tools.ainvoke(messages)
            except Exception as exc:
                trace.record_llm_error(f"{type(exc).__name__}: {exc}")
                return _format_degrade_output(writer, trace, "llm_call_failed")

            messages.append(response)
            trace.record_iteration(response, iteration)

            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                trace.exit_reason = "natural_stop"
                return _format_final_output(writer, trace, response)

            for tc in tool_calls:
                output = await _safe_invoke_tool(tool_map, tc, trace)
                messages.append(ToolMessage(content=str(output), tool_call_id=tc.get("id", "")))

        trace.record_budget_exhaustion(max_iter)
        return _format_degrade_output(writer, trace, "budget_exhausted")

    return planner_agent_node
