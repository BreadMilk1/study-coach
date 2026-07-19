"""LLM tool-calling Quiz GENERATE agent — P2.3 ablation variant.

Parallels `quiz_master.py` (the deterministic baseline) in shape — same
LangGraph node contract — async (state) -> dict update — so `quiz_node` can
dispatch to either based on a per-request mode flag. Same SSE contract —
citations event, single token event with the final markdown, done event.

The module exposes:
  - `_make_quiz_tools(...)`: closure factory producing 2 LangChain @tool
    wrappers (retriever_search / persist_quiz_question). Cut P2.3-①b — implemented here.
  - `build_quiz_master_agent(...)`: top-level factory returning the async node
    callable. Cut P2.3-①c onward.

`_make_quiz_tools` is INTENTIONALLY a private name — the loop is the only
caller; downstream code reaches the agent via the factory.

Business logic is NOT reimplemented here. `persist_quiz_question` directly
uses repository calls (goal_repo.list_active_for_user, topic_repo.get_by_name
/ create, question_repo.create) — abstracting them to a separate function
would be a one-line wrapper, violating YAGNI. retriever_search is a one-liner
over retriever.search.
"""
from __future__ import annotations

import json
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
from app.agent.state import CoachState
from app.agent.tools.schemas import QuizQuestionPersist
from app.db.repositories import (
    GoalRepository,
    QuestionRepository,
    TopicRepository,
)


_DEFAULT_GOAL_TITLE = "Default Study Goal"


def _make_quiz_tools(
    *,
    user_id: str,
    retriever,
    topic_repo: TopicRepository | None,
    question_repo: QuestionRepository | None,
    goal_repo: GoalRepository | None,
) -> list:
    """Build a per-request tool set with user/session context baked in.

    The model sees only the public args (the @tool decorator strips closure
    variables from the generated JSON schema). user_id is NEVER an LLM-visible
    arg — identity is not a behavior input.
    """

    @tool
    def retriever_search(query: str, top_k: int = 5) -> str:
        """Search the user's PDF corpus for chunks relevant to a quiz topic.
        Call BEFORE drafting the question to ground in real source material.
        Returns JSON list: [{"chunk_id","content","page"}, ...].
        """
        if retriever is None:
            return "[]"
        chunks = retriever.search(query, top_k=top_k) or []
        return json.dumps(chunks, ensure_ascii=False)

    @tool
    def persist_quiz_question(
        topic: str,
        prompt: str,
        options: list[str],
        answer: str,
        explanation: str,
    ) -> str:
        """Save the multiple-choice quiz question to the database.
        Call AFTER you've written:
        - a clear topic name
        - a question prompt
        - exactly 4 options, each prefixed "A) ", "B) ", "C) ", "D) "
        - the correct answer letter (A/B/C/D)
        - a 1-2 sentence explanation of why the answer is correct

        Returns JSON {"question_id","topic_id","persisted":true} on success
        or {"error": "..."} on schema validation failure (retry with valid args).
        """
        if goal_repo is None or topic_repo is None or question_repo is None:
            return json.dumps({"error": "repository not available"})
        try:
            validated = QuizQuestionPersist(
                topic=topic, prompt=prompt, options=options,
                answer=answer, explanation=explanation,
            )
        except ValidationError as exc:
            err = exc.errors()[0] if exc.errors() else {"loc": [], "msg": str(exc)}
            loc = ".".join(str(x) for x in err.get("loc", []))
            return json.dumps({"error": f"invalid at {loc}: {err.get('msg', '')}"})

        active = goal_repo.list_active_for_user(user_id)
        goal = active[0] if active else goal_repo.create(
            user_id=user_id, title=_DEFAULT_GOAL_TITLE,
        )
        topic_row = (
            topic_repo.get_by_name(goal_id=goal.id, name=validated.topic)
            or topic_repo.create(goal_id=goal.id, name=validated.topic)
        )
        question = question_repo.create(
            topic_id=topic_row.id,
            prompt=validated.prompt,
            options_json=list(validated.options),
            answer=validated.answer,
            explanation=validated.explanation,
        )
        return json.dumps({
            "question_id": question.id,
            "topic_id": topic_row.id,
            "persisted": True,
        }, ensure_ascii=False)

    return [retriever_search, persist_quiz_question]


_AGENT_SYSTEM_PROMPT = """You are a study coach quiz generator.

The user wants a multiple-choice quiz question on a topic.

Your job:
1. Read the user's message to extract the topic.
2. Call `retriever_search` with the topic to ground in the user's PDF source material.
3. Write a single multiple-choice question:
   - One clear, unambiguous question prompt
   - Exactly 4 options, each prefixed "A) ", "B) ", "C) ", "D) "
   - Distractors should be plausible (not obviously wrong)
   - One correct answer letter
   - A 1-2 sentence explanation grounded in the retrieved source chunks
4. Call `persist_quiz_question` with the question, options, answer, and explanation.
5. After persistence succeeds, write a short markdown reply to the user showing
   the question. Do NOT call more tools.

Today is {today}. Difficulty: medium. Ground strictly in retrieved chunks; do
not invent facts."""

# cloud-adapt: cloud BYOK models can use a terser prompt (3-line bullet form)
# cloud-adapt: cloud models with stronger reasoning may not need step-by-step instruction

_LLM_FAILED_MSG = "⚠️ Could not reach the quiz model. Please try again."
_BUDGET_EXHAUSTED_MSG = (
    "⚠️ Quiz agent exceeded reasoning budget (6 turns). Try a different topic."
)
_QUIZ_PERSIST_FAILED_MSG = (
    "⚠️ I couldn't save a gradeable quiz question. Please try again, "
    "or switch Quiz mode to deterministic for a faster fallback."
)


def _safe_writer():
    """get_stream_writer() with a no-op fallback for direct unit-test calls."""
    try:
        return get_stream_writer()
    except RuntimeError:
        return lambda _payload: None


def _emit_agent_run(writer, trace: AgentTrace) -> dict:
    run = trace.serialize_public(node="quiz")
    writer({"type": "agent_run", "run": run})
    return run


def _format_unpersisted_quiz_output(writer, trace: AgentTrace) -> dict:
    trace.exit_reason = "quiz_persist_failed"
    text = _QUIZ_PERSIST_FAILED_MSG
    writer({"type": "citations", "citations": []})
    writer({"type": "token", "text": text})
    _emit_agent_run(writer, trace)
    return {
        "messages": [AIMessage(content=text)],
        "citations": [],
        "active_quiz_question_id": None,
        "quiz_action": _infer_quiz_action(trace),
        "last_context": trace.aggregated_retriever_context(),
        "agent_trace": trace.serialize(),
        "degraded": True,
    }


def _last_human_msg(state: CoachState) -> str:
    user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    return user_msgs[-1].content if user_msgs else ""


def _infer_quiz_action(trace: AgentTrace) -> Literal["generate", "grade"]:
    """Agent never sees GRADE turns (dispatcher in graph.py routes GRADE to
    deterministic quiz_master regardless of mode). Always returns 'generate'.

    Exists for state-contract uniformity with deterministic quiz_master path
    (which sets quiz_action='generate' on GENERATE turns) — judge_node reads
    quiz_action to decide rubric application vs skip.
    """
    return "generate"


async def _safe_invoke_tool(tool_map, tc, trace: AgentTrace) -> str:
    """Dispatch one tool call. Tool errors are RECOVERABLE — they go back to
    the model as a ToolMessage so it can self-correct. Only LLM-level errors
    in the parent loop short-circuit to degrade."""
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


def _format_final_output(
    writer,
    trace: AgentTrace,
    question_repo: QuestionRepository,
) -> dict:
    persisted_question_id = trace.last_persisted_question_id()
    if persisted_question_id is None:
        return _format_unpersisted_quiz_output(writer, trace)

    question = question_repo.get_by_id(persisted_question_id)
    if question is None:
        return _format_unpersisted_quiz_output(writer, trace)

    final_text = (
        "📝 Quiz:\n\n"
        f"{question.prompt}\n\n"
        + "\n".join(question.options_json)
        + "\n\nReply with A, B, C, or D."
    )

    writer({"type": "citations", "citations": []})
    writer({"type": "token", "text": final_text})
    _emit_agent_run(writer, trace)

    return {
        "messages": [AIMessage(content=final_text)],
        "citations": [],
        "active_quiz_question_id": persisted_question_id,
        "quiz_action": _infer_quiz_action(trace),
        "last_context": trace.aggregated_retriever_context(),
        "agent_trace": trace.serialize(),
    }


def _format_degrade_output(writer, trace: AgentTrace, reason: str) -> dict:
    if reason == "llm_call_failed":
        text = _LLM_FAILED_MSG
    elif reason == "budget_exhausted":
        text = _BUDGET_EXHAUSTED_MSG
    else:
        text = "⚠️ Quiz agent stopped unexpectedly."

    writer({"type": "citations", "citations": []})
    writer({"type": "token", "text": text})
    _emit_agent_run(writer, trace)

    return {
        "messages": [AIMessage(content=text)],
        "citations": [],
        "quiz_action": _infer_quiz_action(trace),
        "last_context": trace.aggregated_retriever_context(),
        "agent_trace": trace.serialize(),
        "degraded": True,
    }


def build_quiz_master_agent(
    *,
    llm,
    topic_repo: TopicRepository,
    question_repo: QuestionRepository,
    goal_repo: GoalRepository,
    retriever=None,
    now_fn: Callable[[], datetime] = datetime.utcnow,
    max_iter: int = 6,
    system_prompt: str = _AGENT_SYSTEM_PROMPT,
):
    """Factory returning an async LangGraph node that runs an LLM tool-calling
    agent loop for Quiz GENERATE. Mirror of `build_planner_agent` shape — same
    state→dict contract, same SSE emit pattern.

    GRADE turn handling: this factory is NEVER invoked on GRADE turns. The
    quiz_node dispatcher in graph.py routes GRADE (active_quiz_question_id
    truthy) to deterministic quiz_master regardless of configured mode.
    """
    # cloud-adapt: cloud BYOK provider can raise max_iter from 6 to 12 here

    async def quiz_master_agent_node(state: CoachState) -> dict:
        writer = _safe_writer()
        user_id = state.get("user_id")
        user_msg = _last_human_msg(state)

        if not user_id:
            err = "Sign in (provide x-fingerprint header) to start a quiz session."
            writer({"type": "citations", "citations": []})
            writer({"type": "token", "text": err})
            return {"messages": [AIMessage(content=err)], "citations": []}

        tools = _make_quiz_tools(
            user_id=user_id,
            retriever=retriever,
            topic_repo=topic_repo,
            question_repo=question_repo,
            goal_repo=goal_repo,
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
                return _format_final_output(writer, trace, question_repo)

            for tc in tool_calls:
                output = await _safe_invoke_tool(tool_map, tc, trace)
                messages.append(ToolMessage(
                    content=str(output), tool_call_id=tc.get("id", ""),
                ))

        trace.record_budget_exhaustion(max_iter)
        return _format_degrade_output(writer, trace, "budget_exhausted")

    return quiz_master_agent_node
