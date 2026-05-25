"""Multi-node LangGraph for Study Coach.

P2.1-①: Router → {Tutor | QuizStub | PlanStub} → END
P2.1-②: Tutor → Judge Guard → (pass: END | retry: Tutor | degrade: END)
P2.1-③/④: + Memory Hydrator (entry) + Memory Writer (exit)
    START → memory_hydrator → router → {tutor → judge | quiz → judge | plan_stub}
                                                      ↘ retry (tutor only): tutor
          → memory_writer → END

P2.1-④: Quiz branch becomes real (`quiz_node`) when a `quiz_master` callable
    is supplied via RunnableConfig.configurable; otherwise falls back to the
    P2.1-① stub message (so legacy tests stay green). Quiz path uses the quiz
    rubric in Judge; weak verdict on quiz degrades immediately (no retry —
    re-running deterministic grade would loop; re-generating questions is left
    to a future ablation cut).

Both memory nodes pull their callable from RunnableConfig.configurable
({"memory_hydrator": ..., "memory_writer": ...}); when missing they
no-op so legacy P2.0/P2.1-①/② tests keep passing without changes.

Tutor node:
- emits `citations` then per-token `{type:"token","text":...}` via get_stream_writer()
- on retry (state.retry_count > 0), prepends a hint to the prompt with
  previous score + weak_dims so the LLM can self-correct (PDCA "Act")

Judge node (Guard):
- pulls judge_llm from RunnableConfig.configurable so build_graph stays small
- when no judge_llm configured, short-circuits to pass (baseline-friendly)
- rubric selected from state.intent (quiz vs tutor); quiz weak → degrade direct
- tutor weak + retry_count < 2 → Command(goto="tutor", update={retry_count+1,...})
- tutor weak + retry_count == 2 → degrade: append disclaimer, route to memory_writer
- pass → memory_writer
"""

from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from .judge import (
    PLAN_DIMENSIONS,
    QUIZ_DIMENSIONS,
    judge_response,
    load_plan_rubric,
    load_quiz_rubric,
    load_tutor_rubric,
)
from .prompt import build_citations, build_prompt, format_context
from .router import route_intent
from .state import CoachState

_QUIZ_STUB_MESSAGE = (
    "[P2.1-④ Quiz feature: configure a `quiz_master` callable via "
    "RunnableConfig.configurable to enable real quiz generation. "
    "This is the fallback stub.]"
)
_PLAN_STUB_MESSAGE = (
    "[P2.1-⑤ Plan feature coming soon — this is a stub. "
    "Goal-driven study planning, milestone tracking, and review will land in P2.1-⑤.]"
)
_MAX_RETRIES = 2


def _retry_hint(previous_score: float, weak_dims: list[str], reasoning: str) -> str:
    dims_str = ", ".join(weak_dims) if weak_dims else "none"
    return (
        f"\n\n---\nPrevious attempt rating: previous_score={previous_score:.2f} "
        f"(weak dimensions: {dims_str}). Judge reason: {reasoning or 'unspecified'}. "
        "Re-read the sources carefully and produce a more grounded, well-paced answer "
        "for an exam-prep student. Address the weak dimensions specifically."
    )


def _degrade_disclaimer(score: float, weak_dims: list[str], retry_count: int = 0) -> str:
    dims_str = ", ".join(weak_dims) if weak_dims else "groundedness"
    after_clause = f" after {retry_count} retries" if retry_count > 0 else ""
    return (
        f"\n\n---\n⚠️ Self-check note: this answer scored low ({score:.2f}; "
        f"weak on {dims_str}){after_clause}. "
        "Consider rephrasing the question or uploading more relevant sources."
    )


def build_graph(*, retriever, llm, checkpointer=None):
    def memory_hydrator_node(state: CoachState, config) -> dict:
        configurable = (config or {}).get("configurable", {}) or {}
        hydrator = configurable.get("memory_hydrator")
        if hydrator is None:
            return {}
        return hydrator(state)

    def memory_writer_node(state: CoachState, config) -> dict:
        configurable = (config or {}).get("configurable", {}) or {}
        writer = configurable.get("memory_writer")
        if writer is None:
            return {}
        return writer(state)

    def router_node(state: CoachState) -> dict:
        # State-aware override: if a quiz turn is in flight, route to quiz
        # regardless of message content (the user is answering, not asking).
        if state.get("active_quiz_question_id"):
            return {"intent": "quiz"}
        last_user = state["messages"][-1].content
        base_intent = route_intent(last_user)
        # P2.1-⑤ — plan stickiness: keep user in plan chain ONLY when the
        # message doesn't carry an explicit tutor/quiz signal. This lets
        # users ask questions or take quizzes mid-plan without switching session.
        if state.get("active_plan_id") and base_intent == "plan":
            return {"intent": "plan"}
        if state.get("active_plan_id") and base_intent == "tutor":
            # Plain tutor question with active plan in flight → respect tutor intent.
            return {"intent": "tutor"}
        # quiz override is already handled by active_quiz_question_id above.
        return {"intent": base_intent}

    async def tutor_node(state: CoachState) -> dict:
        writer = get_stream_writer()
        user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        last_user = user_msgs[-1].content if user_msgs else state["messages"][-1].content

        chunks = retriever.search(last_user, top_k=5)
        citations = build_citations(chunks)
        writer({"type": "citations", "citations": citations})

        prompt = build_prompt(last_user, chunks)
        retry_count = state.get("retry_count", 0)
        if retry_count > 0:
            prompt = prompt + _retry_hint(
                previous_score=state.get("judge_score", 0.0),
                weak_dims=list(state.get("weak_dims", [])),
                reasoning=state.get("judge_reasoning", ""),
            )

        parts: list[str] = []
        async for chunk in llm.astream([HumanMessage(content=prompt)]):
            text = getattr(chunk, "content", "") or ""
            if text:
                writer({"type": "token", "text": text})
                parts.append(text)

        return {
            "messages": [AIMessage(content="".join(parts))],
            "citations": citations,
            "last_context": format_context(chunks),
        }

    def quiz_stub_node(_state: CoachState) -> dict:
        writer = get_stream_writer()
        writer({"type": "citations", "citations": []})
        writer({"type": "token", "text": _QUIZ_STUB_MESSAGE})
        return {
            "messages": [AIMessage(content=_QUIZ_STUB_MESSAGE)],
            "citations": [],
        }

    async def quiz_node(state: CoachState, config) -> dict:
        """State-aware + mode-aware Quiz dispatcher.

        Routing precedence:
        1. GRADE turn (``active_quiz_question_id`` truthy) → ALWAYS deterministic
           ``quiz_master``, regardless of configured mode. P2.3 ``agent_loop``
           never sees GRADE turns by design; the deterministic GRADE branch is
           the single source of truth for mastery updates.
        2. GENERATE turn → mode-aware: ``agent_loop`` → ``quiz_master_agent``;
           ``deterministic`` (default) → ``quiz_master``.
        Missing dependencies → fall back to ``quiz_stub_node``.
        """
        configurable = (config or {}).get("configurable", {}) or {}

        # State-aware: GRADE turn always deterministic
        if state.get("active_quiz_question_id"):
            quiz_master = configurable.get("quiz_master")
            if quiz_master is None:
                return quiz_stub_node(state)
            return await quiz_master(state)

        # GENERATE turn: mode-aware dispatch
        mode = configurable.get("quiz_mode", "deterministic")
        if mode == "agent_loop":
            agent = configurable.get("quiz_master_agent")
            if agent is None:
                return quiz_stub_node(state)
            return await agent(state)
        # Default / "deterministic" — current production path.
        quiz_master = configurable.get("quiz_master")
        if quiz_master is None:
            return quiz_stub_node(state)
        return await quiz_master(state)

    def plan_stub_node(_state: CoachState) -> dict:
        writer = get_stream_writer()
        writer({"type": "citations", "citations": []})
        writer({"type": "token", "text": _PLAN_STUB_MESSAGE})
        return {
            "messages": [AIMessage(content=_PLAN_STUB_MESSAGE)],
            "citations": [],
        }

    async def plan_node(state: CoachState, config) -> dict:
        configurable = (config or {}).get("configurable", {}) or {}
        mode = configurable.get("planner_mode", "deterministic")
        if mode == "agent_loop":
            agent = configurable.get("planner_agent")
            if agent is None:
                return plan_stub_node(state)
            return await agent(state)
        # Default / "deterministic" — current production path.
        # cloud-adapt: cloud BYOK provider may default to agent_loop here based
        # on llm_config.provider rather than the header — leave threading to P3.
        planner = configurable.get("planner")
        if planner is None:
            return plan_stub_node(state)
        return await planner(state)

    async def judge_node(
        state: CoachState,
        config,
    ) -> Command[Literal["tutor", "memory_writer"]]:
        writer = get_stream_writer()
        configurable = (config or {}).get("configurable", {}) or {}
        judge_llm = configurable.get("judge_llm")

        # No judge configured -> baseline pass-through (keeps test_graph tests green).
        if judge_llm is None:
            return Command(goto="memory_writer", update={"judge_score": 1.0})

        intent = state.get("intent", "tutor")
        # Cut ④g + ⑤f: deterministic outputs short-circuit judge.
        if intent == "quiz" and state.get("quiz_action") == "grade":
            return Command(goto="memory_writer", update={"judge_score": 1.0})
        if intent == "plan" and state.get("plan_action") == "check_in":
            return Command(goto="memory_writer", update={"judge_score": 1.0})

        user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        question = user_msgs[-1].content if user_msgs else ""
        ai_msgs = [m for m in state["messages"] if isinstance(m, AIMessage)]
        answer = ai_msgs[-1].content if ai_msgs else ""
        context = state.get("last_context", "")

        is_tutor = intent == "tutor"
        if intent == "quiz":
            rubric, dimensions = load_quiz_rubric(), QUIZ_DIMENSIONS
        elif intent == "plan":
            rubric, dimensions = load_plan_rubric(), PLAN_DIMENSIONS
        else:
            rubric, dimensions = load_tutor_rubric(), None

        result = await judge_response(
            question=question,
            answer=answer,
            context=context,
            rubric=rubric,
            judge_llm=judge_llm,
            dimensions=dimensions,
        )
        print(f"[JUDGE/{intent}] score={result['score']:.2f}, verdict={result['verdict']}, "
        f"weak_dims={result['weak_dims']}, retry_count={state.get('retry_count', 0)}", flush=True)

        retry_count = state.get("retry_count", 0)

        if result["verdict"] == "pass":
            return Command(
                goto="memory_writer",
                update={
                    "judge_score": result["score"],
                    "weak_dims": result["weak_dims"],
                    "judge_reasoning": result["reasoning"],
                },
            )

        # Weak — tutor retries up to budget; quiz/plan degrade immediately
        # (re-running deterministic quiz grade would loop; re-generating
        # questions is a deferred ablation, not in scope here).
        if is_tutor and retry_count < _MAX_RETRIES:
            return Command(
                goto="tutor",
                update={
                    "judge_score": result["score"],
                    "weak_dims": result["weak_dims"],
                    "judge_reasoning": result["reasoning"],
                    "retry_count": retry_count + 1,
                },
            )

        # Retry budget exhausted (or non-tutor weak) -> degrade.
        disclaimer = _degrade_disclaimer(result["score"], result["weak_dims"], retry_count)
        writer({"type": "token", "text": disclaimer})
        return Command(
            goto="memory_writer",
            update={
                "judge_score": result["score"],
                "weak_dims": result["weak_dims"],
                "judge_reasoning": result["reasoning"],
                "degraded": True,
                "messages": [AIMessage(content=answer + disclaimer)],
            },
        )

    def _select_branch(state: CoachState) -> str:
        return state["intent"]

    g = StateGraph(CoachState)
    g.add_node("memory_hydrator", memory_hydrator_node)
    g.add_node("router", router_node)
    g.add_node("tutor", tutor_node)
    g.add_node("judge", judge_node)
    g.add_node("quiz", quiz_node)
    g.add_node("plan", plan_node)
    g.add_node("memory_writer", memory_writer_node)
    g.add_edge(START, "memory_hydrator")
    g.add_edge("memory_hydrator", "router")
    g.add_conditional_edges(
        "router",
        _select_branch,
        {"tutor": "tutor", "quiz": "quiz", "plan": "plan"},
    )
    g.add_edge("tutor", "judge")
    g.add_edge("quiz", "judge")
    g.add_edge("plan", "judge")
    g.add_edge("memory_writer", END)
    return g.compile(checkpointer=checkpointer)
