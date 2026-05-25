# P2.2 Agent Loop Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship an LLM tool-calling agent-loop variant of the Planner that lives in parallel with the deterministic `planner.py` shipped in P2.1-⑤, then run a head-to-head ablation matrix (4 models × 2 modes × 12 queries × 3 runs + 1 thinking on/off appendix) and produce `EVAL.md` + a portfolio blog post.

**Architecture:** A closure factory `_make_planner_tools(...)` builds 5 LangChain `@tool` wrappers with user/session context baked in (`retriever_search` / `get_existing_plan` / `update_study_plan` / `generate_mindmap` / `compute_progress`). A hand-written async `while`-loop (`max_iter=10`, exit on empty `response.tool_calls`, tool errors fed back as `ToolMessage` for self-correction, LLM errors → degrade) drives the model. The loop is embedded in the existing LangGraph `plan_node` via a mode-aware dispatcher keyed off `config.configurable.planner_mode`. A new HTTP header `x-planner-mode: deterministic|agent_loop` selects mode per request, defaulting to `deterministic` so production stays on the P2.1-⑤ baseline. Existing memory_hydrator / judge / memory_writer infrastructure is **untouched** — that is the experimental fairness guarantee.

**Tech Stack:** Python 3.11, LangChain 0.3 + langchain-ollama 1.1 (`bind_tools` + `@tool` decorator), LangGraph 0.2, FastAPI, SQLAlchemy, pytest-asyncio (`asyncio_mode = "auto"`).

**Spec:** `study-coach/docs/superpowers/specs/2026-05-22-p2-2-agent-loop-ablation-design.md`

**Discipline reminders for the implementer:**
- This repo is **not** a git repo — never run `git init / commit / push`. Use the project's existing `# checkpoint` convention (verify full test suite green) instead.
- Every `# cloud-adapt:` comment in `planner_agent.py` must be marker-only — do NOT implement cloud branches. See spec §11 for the canonical grep anchor list.
- TDD: every cut writes the failing test(s) first, runs them to confirm RED, then writes the minimal implementation, then runs the full suite. No exceptions.
- Project working directory is `study-coach/backend/` for all `uv run pytest` commands. Use `cd backend && uv run pytest -q ...`.
- Baseline before P2.2 starts: **157 backend tests passing** (P2.1-⑤i ship-state). Target after Cut ②a: **181 tests passing**.
- Do not modify any non-P2.2 file unless the cut says so — if you find an unrelated bug, log it as a follow-up and stop.

---

## File Structure

### Files to create

| Path | Responsibility |
|---|---|
| `backend/app/agent/planner_agent.py` | Closure-factory tool wrappers + `AgentTrace` dataclass + while-loop body + `build_planner_agent` factory. ~350 lines total. |
| `backend/tests/agent/test_planner_agent_tools.py` | Cut ①a — 5 unit tests for the 5 `@tool` wrappers (closure injection + JSON schema + JSON return shape only; business logic already covered by `test_progress.py` / `test_plan_tools.py`). |
| `backend/tests/agent/test_agent_trace.py` | Cut ①b — 3 unit tests for the `AgentTrace` dataclass (record/serialize/exit-reason transitions). |
| `backend/tests/agent/test_planner_agent_loop.py` | Cut ①c — 8 unit tests for the loop body: natural_stop / budget_exhausted / llm_error / tool_error_self_correction / plan_action inference (generate, check_in, no-tools fallback) / `_extract_topic` regression. |
| `backend/tests/agent/test_graph_plan_agent_e2e.py` | Cut ①d — 3 graph-level e2e tests with stub LLM: mode switching, `agent_trace` lands in state, judge still works on agent_loop output. |
| `backend/tests/api/test_routes_plan_agent.py` | Cut ①e — 2 route-level integration tests: `x-planner-mode` header routing + SSE contract identical to deterministic. |
| `backend/app/eval/p2_2_agent_ablation/__init__.py` | Cut ②a — package marker. |
| `backend/app/eval/p2_2_agent_ablation/matrix.py` | Cut ②a — matrix expansion (4 models × 2 modes × N queries × R runs). |
| `backend/app/eval/p2_2_agent_ablation/single_run.py` | Cut ②a — one experimental run: build graph, invoke, extract `agent_trace`, compute auto-metrics. |
| `backend/app/eval/p2_2_agent_ablation/judges.py` | Cut ②a — dual judge: local qwen2.5:7b + cloud BYOK GPT-4o-mini. |
| `backend/app/eval/p2_2_agent_ablation/run_eval.py` | Cut ②a — top-level CLI: `--model`, `--mode`, `--runs`, `--output`, resumable. |
| `backend/app/eval/p2_2_agent_ablation/queries.json` | Cut ②a — 12 HKBU queries (reuse `tests/fixtures/retrieval_eval_queries.json`) + 4 multi-turn plan scenarios. |
| `backend/tests/eval/test_p2_2_harness.py` | Cut ②a — 3 unit tests: matrix expansion, record schema validation, resumability. |
| `study-coach/docs/EVAL.md` | Cut ③ — empirical write-up: latency table, tool-calling correctness table, robustness, plan quality (dual judge), judge agreement, conclusions. |
| `study-coach/docs/agent_loop_vs_deterministic.md` | Cut ③ — portfolio blog post responding to `learn-claude-code`'s "agency = model + minimal harness" thesis. |

### Files to modify

| Path | Change | Cut |
|---|---|---|
| `backend/app/agent/state.py` | Add one field: `agent_trace: NotRequired[dict]` to `CoachState`. | ①d |
| `backend/app/agent/graph.py` | Replace `plan_node` body (lines 175–180) with mode-aware dispatcher (~10 lines). | ①d |
| `backend/app/api/deps.py` | Append `get_planner_mode` factory + `get_planner_agent` factory at end of file. | ①e |
| `backend/app/api/routes.py` | Extend `chat()` signature with `planner_agent` + `planner_mode` Depends, inject into `config.configurable`. | ①e |
| `study-coach/docs/ROADMAP.md` | Update P2.2 block from "planned" to "shipped" with results summary. | ③ |
| `/Users/lianghaozhe/.claude/projects/-Users-lianghaozhe-Downloads-Study-Compaion-and-JadeAI/memory/project_study_coach_refactor.md` | Append P2.2 progress segment + lessons learned. | ③ |

### Files explicitly NOT touched

`planner.py` — deterministic baseline must stay byte-identical so ablation is a clean A/B. `tools/plan.py`, `progress.py`, `tools/schemas.py`, `memory_updater.py`, `judge.py`, `router.py` — all reused via import, never modified.

---

## Cut P2.2-①a — Tool wrappers + closure factory

**Files:**
- Create: `backend/app/agent/planner_agent.py` (partial — tool factory only at this stage)
- Test: `backend/tests/agent/test_planner_agent_tools.py`

**Boundary check (precision-corrected from spec brainstorm):** of the 5 tools, 3 delegate to existing functions and 2 are thin direct wrappers. Imports are non-negotiable:

```python
from app.agent.tools.plan import update_study_plan as update_study_plan_fn
from app.agent.tools.plan import generate_mindmap as generate_mindmap_fn
from app.agent.progress import compute_progress as compute_progress_fn  # NOT in tools/
from app.agent.tools.schemas import Milestone
```

`retriever_search` is a one-liner over `retriever.search()`. `get_existing_plan` is three lines over `goal_repo.list_active_for_user` + `plan_repo.get_by_goal`. No new business logic anywhere.

- [ ] **Step 1: Create the test file with 5 failing tests**

Create `backend/tests/agent/test_planner_agent_tools.py`:

```python
"""Cut P2.2-①a — unit tests for the 5 LLM-facing tool wrappers in planner_agent.

Each test exercises:
  1. Closure injection — the LLM-visible args do not include user_id / repos / llm,
     yet the tool can use them via the factory closure.
  2. JSON return shape — every tool returns a JSON-serializable string the LLM
     can consume as a ToolMessage payload.
  3. Delegation correctness — the wrapper calls into the existing pure functions
     (update_study_plan_fn / generate_mindmap_fn / compute_progress_fn) rather
     than reimplementing the logic.

Business-logic depth is NOT re-tested here; that lives in test_plan_tools.py
and test_progress.py. The contract under test is the wrapper, not the work.
"""
import json
from datetime import datetime

import pytest
from langchain_core.messages import AIMessage
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.agent.planner_agent import _make_planner_tools
from app.db.models import Base
from app.db.repositories import (
    GoalRepository,
    PlanRepository,
    UserRepository,
)


@pytest.fixture
def session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,
    )
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


class StubRetriever:
    def __init__(self, chunks=None):
        self.chunks = chunks or []
        self.calls: list[tuple[str, int]] = []

    def search(self, query, top_k=5):
        self.calls.append((query, top_k))
        return self.chunks[:top_k]


class StubLLM:
    def __init__(self, response_text: str = ""):
        self.response_text = response_text
        self.calls = 0

    async def ainvoke(self, messages, **_kwargs):
        self.calls += 1
        return AIMessage(content=self.response_text)


async def test_retriever_search_uses_closure_retriever_and_returns_json():
    retriever = StubRetriever(chunks=[
        {"chunk_id": "c1", "content": "HyDE = Hypothetical Document Embedding.", "page": 1},
        {"chunk_id": "c2", "content": "Re-rank with BM25.", "page": 3},
    ])
    tools = _make_planner_tools(
        user_id="u1", llm=None, retriever=retriever,
        plan_repo=None, goal_repo=None,
        mastery_scores={}, recent_mistakes=[],
    )
    tool = next(t for t in tools if t.name == "retriever_search")

    # Schema: query is in the public args, retriever is NOT
    assert "query" in tool.args
    assert "top_k" in tool.args
    assert "retriever" not in tool.args
    assert "user_id" not in tool.args

    out = await tool.ainvoke({"query": "HyDE", "top_k": 1})
    parsed = json.loads(out)
    assert retriever.calls == [("HyDE", 1)]
    assert isinstance(parsed, list)
    assert parsed[0]["chunk_id"] == "c1"


async def test_get_existing_plan_returns_null_string_when_no_active_goal(session):
    user = UserRepository(session).get_or_create("fp-get-plan-1")
    goal_repo = GoalRepository(session)
    plan_repo = PlanRepository(session)
    tools = _make_planner_tools(
        user_id=user.id, llm=None, retriever=None,
        plan_repo=plan_repo, goal_repo=goal_repo,
        mastery_scores={}, recent_mistakes=[],
    )
    tool = next(t for t in tools if t.name == "get_existing_plan")

    out = await tool.ainvoke({})
    # No goal, no plan → tool returns literal "null" (LLM-friendly sentinel)
    assert out == "null"


async def test_update_study_plan_upserts_via_closure_and_returns_count(session):
    user = UserRepository(session).get_or_create("fp-upd-1")
    goal_repo = GoalRepository(session)
    plan_repo = PlanRepository(session)
    goal_repo.create(user_id=user.id, title="G")

    tools = _make_planner_tools(
        user_id=user.id, llm=None, retriever=None,
        plan_repo=plan_repo, goal_repo=goal_repo,
        mastery_scores={}, recent_mistakes=[],
    )
    tool = next(t for t in tools if t.name == "update_study_plan")

    milestones = [
        {"title": "Read HyDE §1", "due_at": "2026-05-25", "done": False, "topic": "HyDE"},
        {"title": "Practice HyDE", "due_at": "2026-05-28", "done": False, "topic": "HyDE"},
    ]
    out = await tool.ainvoke({"milestones": milestones})
    parsed = json.loads(out)
    assert parsed["milestones_count"] == 2
    assert parsed["plan_id"]

    # Verify persistence via the same repo (closure path actually wrote)
    goal = goal_repo.list_active_for_user(user.id)[0]
    saved = plan_repo.get_by_goal(goal.id)
    assert saved is not None
    assert len(saved.milestones_json) == 2


async def test_generate_mindmap_delegates_to_closure_llm():
    llm = StubLLM(response_text="```mermaid\nmindmap\n  root((HyDE))\n    Q1\n```\n- HyDE\n  - Q1")
    tools = _make_planner_tools(
        user_id="u", llm=llm, retriever=None,
        plan_repo=None, goal_repo=None,
        mastery_scores={}, recent_mistakes=[],
    )
    tool = next(t for t in tools if t.name == "generate_mindmap")

    # Closure means llm is NOT in args
    assert "llm" not in tool.args
    assert "topic" in tool.args
    assert "milestones" in tool.args

    out = await tool.ainvoke({
        "topic": "HyDE",
        "milestones": [{"title": "M1", "due_at": None, "done": False, "topic": "HyDE"}],
    })
    parsed = json.loads(out)
    assert llm.calls == 1
    assert "mindmap" in parsed["mermaid_src"].lower()
    assert isinstance(parsed["markdown_outline"], str)


async def test_compute_progress_uses_closure_state_and_returns_json(session):
    user = UserRepository(session).get_or_create("fp-prog-1")
    goal_repo = GoalRepository(session)
    plan_repo = PlanRepository(session)
    goal = goal_repo.create(user_id=user.id, title="G")
    plan_repo.create(
        goal_id=goal.id,
        milestones_json=[
            {"title": "M1", "done": True, "topic": "HyDE"},
            {"title": "M2", "done": False, "topic": "BM25", "due_at": "2020-01-01"},
        ],
    )

    tools = _make_planner_tools(
        user_id=user.id, llm=None, retriever=None,
        plan_repo=plan_repo, goal_repo=goal_repo,
        mastery_scores={"BM25": 0.2, "HyDE": 0.8},
        recent_mistakes=["m1", "m2"],
        now_fn=lambda: datetime(2026, 5, 23, 12, 0),
    )
    tool = next(t for t in tools if t.name == "compute_progress")

    # All inputs are closure-injected; tool has no public args
    assert tool.args == {}

    out = await tool.ainvoke({})
    parsed = json.loads(out)
    assert parsed["done_count"] == 1
    assert parsed["total_count"] == 2
    assert "M2" in parsed["overdue"]
    assert parsed["weak_topics"] == ["BM25"]
    assert parsed["recent_mistake_count"] == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd backend && uv run pytest -q tests/agent/test_planner_agent_tools.py`

Expected output: `ImportError: cannot import name '_make_planner_tools' from 'app.agent.planner_agent'` (module does not exist yet). 5 errors.

- [ ] **Step 3: Create `planner_agent.py` with the tool factory**

Create `backend/app/agent/planner_agent.py`:

```python
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
from datetime import datetime
from typing import Callable

from langchain_core.tools import tool
from pydantic import ValidationError

from app.agent.progress import compute_progress as compute_progress_fn
from app.agent.tools.plan import generate_mindmap as generate_mindmap_fn
from app.agent.tools.plan import update_study_plan as update_study_plan_fn
from app.agent.tools.schemas import Milestone
from app.db.repositories import (
    GoalRepository,
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
        Each milestone: {title:str, due_at:str|null, done:bool, topic:str|null}.
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
            return json.dumps({"error": f"invalid milestone shape: {exc}"})
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
            return json.dumps({"error": f"invalid milestone shape: {exc}"})
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd backend && uv run pytest -q tests/agent/test_planner_agent_tools.py`

Expected output: `5 passed`. If any test fails:
- ImportError on `Milestone` → check `app.agent.tools.schemas` exports it (it does — line 74 of schemas.py).
- AssertionError on `tool.args == {}` for `compute_progress` → confirm the @tool decorator picked up the zero-arg signature; LangChain renders it as empty dict.
- AssertionError on `parsed[0]["chunk_id"] == "c1"` → the stub retriever returns dicts directly; make sure `json.dumps(chunks)` is being called (not `json.dumps([chunks])`).

- [ ] **Step 5: Checkpoint — run the full backend suite to verify no regressions**

Run: `cd backend && uv run pytest -q`

Expected output: `162 passed` (157 baseline + 5 new). Any failure outside the new test file is a regression — stop and investigate (the only file you've added is `planner_agent.py`, which is not yet imported by any production code).

---

## Cut P2.2-①b — AgentTrace dataclass

**Files:**
- Modify: `backend/app/agent/planner_agent.py` (append AgentTrace section)
- Test: `backend/tests/agent/test_agent_trace.py`

The trace is the **eval lifeline** — every per-run row in `results.jsonl` dumps `AgentTrace.serialize()`. Get this shape right now; refactoring it after Cut ②b means re-running the matrix.

- [ ] **Step 1: Create the test file with 3 failing tests**

Create `backend/tests/agent/test_agent_trace.py`:

```python
"""Cut P2.2-①b — unit tests for the AgentTrace dataclass.

Trace is the only structured record the eval harness pulls from a run, so
the serialize() shape is contractual. These tests pin:
  - record_iteration / record_tool_call append correctly
  - serialize() emits all expected keys with correct types/counts
  - exit_reason transitions: natural_stop / budget_exhausted / llm_call_failed
"""
import time

from langchain_core.messages import AIMessage

from app.agent.planner_agent import AgentTrace


def _ai(content: str = "ok", tool_calls=None, input_tokens=10, output_tokens=5):
    """Build a stub AIMessage with usage_metadata so trace can record tokens."""
    msg = AIMessage(content=content, tool_calls=tool_calls or [])
    msg.usage_metadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    return msg


def test_record_iteration_and_tool_call_then_serialize_natural_stop():
    trace = AgentTrace(t_start=time.monotonic())
    # First iter: model called retriever_search
    resp1 = _ai(tool_calls=[{"name": "retriever_search", "args": {"query": "HyDE"}, "id": "c1"}],
                input_tokens=100, output_tokens=20)
    trace.record_iteration(resp1, iteration=0)
    trace.record_tool_call("retriever_search", {"query": "HyDE"}, "[]", error=False)
    # Second iter: model emitted final summary (no tool calls → natural stop)
    resp2 = _ai(content="here is your plan", tool_calls=[], input_tokens=120, output_tokens=80)
    trace.record_iteration(resp2, iteration=1)
    trace.exit_reason = "natural_stop"

    out = trace.serialize()
    assert out["total_iterations"] == 2
    assert out["total_tool_calls"] == 1
    assert out["tool_call_breakdown"] == {"retriever_search": 1}
    assert out["tool_errors"] == 0
    assert out["input_tokens"] == 220
    assert out["output_tokens"] == 100
    assert out["exit_reason"] == "natural_stop"
    assert out["llm_error"] is None
    assert isinstance(out["wall_time_s"], float)
    assert out["wall_time_s"] >= 0.0


def test_record_budget_exhaustion_and_tool_error():
    trace = AgentTrace(t_start=time.monotonic())
    trace.record_iteration(_ai(tool_calls=[{"name": "update_study_plan", "args": {}, "id": "c1"}]),
                           iteration=0)
    trace.record_tool_call(
        "update_study_plan", {"milestones": "not-a-list"},
        "Error calling update_study_plan: bad arg type", error=True,
    )
    trace.record_budget_exhaustion(max_iter=10)

    out = trace.serialize()
    assert out["exit_reason"] == "budget_exhausted"
    assert out["tool_errors"] == 1
    assert out["llm_error"] is None


def test_record_llm_error_sets_exit_reason_and_message():
    trace = AgentTrace(t_start=time.monotonic())
    trace.record_llm_error("ConnectionRefusedError: ollama not running")

    out = trace.serialize()
    assert out["exit_reason"] == "llm_call_failed"
    assert "ConnectionRefusedError" in out["llm_error"]
    assert out["total_iterations"] == 0
    assert out["total_tool_calls"] == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd backend && uv run pytest -q tests/agent/test_agent_trace.py`

Expected output: `ImportError: cannot import name 'AgentTrace' from 'app.agent.planner_agent'`. 3 errors.

- [ ] **Step 3: Append AgentTrace to `planner_agent.py`**

Add the following to `backend/app/agent/planner_agent.py`, right after the imports block at top of file:

```python
import time
from collections import Counter
from dataclasses import dataclass, field
```

Then append the dataclass definitions immediately after `_DEFAULT_GOAL_TITLE = "Default Study Goal"` (before `_make_planner_tools`):

```python
@dataclass
class IterationRecord:
    iteration: int
    has_tool_calls: bool
    tool_call_count: int
    input_tokens: int
    output_tokens: int


@dataclass
class ToolCallRecord:
    name: str
    args: dict
    output: str
    error: bool


@dataclass
class AgentTrace:
    """Per-run instrumentation. Serialized into eval results.jsonl rows.

    Field choices are deliberately minimal — anything tightly coupled to a
    specific schema (per-call latency breakdown, full tool output) is out
    because the matrix has 360 runs and the file should stay grep-able.
    """
    t_start: float
    iterations: list[IterationRecord] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    exit_reason: str = "in_flight"
    llm_error: str | None = None

    def record_iteration(self, response, iteration: int) -> None:
        tcs = getattr(response, "tool_calls", None) or []
        usage = getattr(response, "usage_metadata", None) or {}
        self.iterations.append(IterationRecord(
            iteration=iteration,
            has_tool_calls=bool(tcs),
            tool_call_count=len(tcs),
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
        ))

    def record_tool_call(self, name: str, args: dict, output: str, *, error: bool) -> None:
        self.tool_calls.append(ToolCallRecord(
            name=name, args=dict(args or {}), output=str(output)[:500], error=error,
        ))
        # cloud-adapt: production deploy should redact output entirely; truncating
        # to 500 chars keeps it tractable for eval but still leaks user content.

    def record_budget_exhaustion(self, max_iter: int) -> None:
        self.exit_reason = "budget_exhausted"

    def record_llm_error(self, exc: str) -> None:
        self.exit_reason = "llm_call_failed"
        self.llm_error = exc

    def tool_names_called(self) -> list[str]:
        return [tc.name for tc in self.tool_calls if not tc.error]

    def get_existing_plan_returned_nonnull(self) -> bool:
        for tc in self.tool_calls:
            if tc.name == "get_existing_plan" and tc.output != "null":
                return True
        return False

    def last_persisted_plan_id(self) -> str | None:
        for tc in reversed(self.tool_calls):
            if tc.name == "update_study_plan" and not tc.error:
                try:
                    return json.loads(tc.output).get("plan_id")
                except (json.JSONDecodeError, AttributeError):
                    return None
        return None

    def aggregated_retriever_context(self) -> str:
        """Flatten all retriever_search outputs into a single context string for
        consumers (memory_writer / future analytics) that expect the same
        `last_context` shape as the deterministic planner emits."""
        parts: list[str] = []
        for tc in self.tool_calls:
            if tc.name == "retriever_search" and not tc.error:
                try:
                    chunks = json.loads(tc.output)
                except (json.JSONDecodeError, TypeError):
                    continue
                for i, c in enumerate(chunks, start=1):
                    parts.append(f"[{i}] {c.get('content', '')}")
        return "\n".join(parts)

    def serialize(self) -> dict:
        return {
            "total_iterations": len(self.iterations),
            "total_tool_calls": len(self.tool_calls),
            "tool_call_breakdown": dict(Counter(tc.name for tc in self.tool_calls)),
            "tool_errors": sum(1 for tc in self.tool_calls if tc.error),
            "input_tokens": sum(it.input_tokens for it in self.iterations),
            "output_tokens": sum(it.output_tokens for it in self.iterations),
            "wall_time_s": time.monotonic() - self.t_start,
            "exit_reason": self.exit_reason,
            "llm_error": self.llm_error,
        }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd backend && uv run pytest -q tests/agent/test_agent_trace.py`

Expected output: `3 passed`.

If `record_iteration` blows up on a missing `usage_metadata` attribute → confirm the test stub `_ai()` is attaching it; the dataclass uses `getattr(response, "usage_metadata", None) or {}` so a missing attribute should default safely.

- [ ] **Step 5: Checkpoint — run full backend suite**

Run: `cd backend && uv run pytest -q`

Expected output: `165 passed` (162 + 3). Same regression policy as Cut ①a: nothing outside the new test file should be touched.

---

## Cut P2.2-①c — Loop body + plan_action inference + error handling

**Files:**
- Modify: `backend/app/agent/planner_agent.py` (append loop + factory)
- Modify: `backend/app/agent/state.py` (add `agent_trace` field)
- Modify: `backend/app/agent/planner.py` — **DO NOT TOUCH**. Confirm with `git status` (n/a here — non-repo; confirm via `diff` if you're paranoid).
- Test: `backend/tests/agent/test_planner_agent_loop.py`

This is the heart of the experiment. 8 tests carve out: 3 exit conditions × 1 (natural_stop / budget_exhausted / llm_call_failed), tool-error self-correction × 1, plan_action inference × 3 (generate / check_in / no-tools fallback), `_extract_topic` regression × 1.

The `state.py` change is needed BEFORE the loop tests can assert on the `agent_trace` key in the returned state update; it stays a `NotRequired` field so all 157 existing tests remain compatible.

- [ ] **Step 1: Extend `CoachState` with the `agent_trace` field**

Modify `backend/app/agent/state.py` — append one field at the end of the `CoachState` TypedDict body (after line 47 `plan_action: NotRequired[Literal["generate", "check_in"]]`):

```python
    # P2.2 — agent loop instrumentation. Populated only when the agent_loop
    # planner mode is active; deterministic path leaves this field absent.
    agent_trace: NotRequired[dict]
```

Run: `cd backend && uv run pytest -q`

Expected output: `165 passed` — adding a `NotRequired` field can never break existing tests; this is just a guard checkpoint.

- [ ] **Step 2: Create the 8-test failing file**

Create `backend/tests/agent/test_planner_agent_loop.py`:

```python
"""Cut P2.2-①c — loop body + plan_action inference + error handling.

Stub LLM is scripted: each .ainvoke() returns the next preset AIMessage. Some
have tool_calls (forcing the loop to dispatch + iterate), the final one has
content but no tool_calls (natural stop).

Tests:
  1. natural_stop after retriever_search → update_study_plan → final summary
  2. budget_exhausted when LLM never stops calling tools
  3. llm_call_failed degrades cleanly without persisting
  4. tool error becomes a ToolMessage and loop continues (self-correction)
  5. plan_action inference: generate when get_existing_plan absent/null
  6. plan_action inference: check_in when get_existing_plan returned non-null
  7. plan_action inference fallback: generate when zero tools called
  8. _extract_topic regression — closes a P2.1-⑤i loose end
"""
from datetime import datetime

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.agent.planner_agent import build_planner_agent
from app.db.models import Base
from app.db.repositories import (
    GoalRepository,
    MasteryRepository,
    MistakeRepository,
    PlanRepository,
    UserRepository,
)


class ScriptedLLM:
    """Returns the next AIMessage from a preset list per .ainvoke() call.

    The harness wraps this with .bind_tools(tools); since our loop only reads
    the returned AIMessage and its .tool_calls, we don't need a real
    BaseChatModel — just a duck-typed ainvoke + bind_tools no-op.
    """
    def __init__(self, responses: list[AIMessage]):
        self.responses = list(responses)
        self.idx = 0
        self.calls = 0

    def bind_tools(self, _tools):
        return self  # pass-through; tools are dispatched outside the LLM

    async def ainvoke(self, messages, **_kwargs):
        self.calls += 1
        if self.idx >= len(self.responses):
            raise AssertionError("ScriptedLLM exhausted — loop called more times than expected")
        msg = self.responses[self.idx]
        self.idx += 1
        return msg


class CrashingLLM:
    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, messages, **_kwargs):
        raise ConnectionError("ollama unreachable")


class StubRetriever:
    def __init__(self, chunks=None):
        self.chunks = chunks or []

    def search(self, query, top_k=5):
        return self.chunks[:top_k]


def _msg(content="", tool_calls=None, input_tokens=10, output_tokens=5):
    msg = AIMessage(content=content, tool_calls=tool_calls or [])
    msg.usage_metadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    return msg


@pytest.fixture
def session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,
    )
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def _build(session, llm, retriever=None):
    return build_planner_agent(
        llm=llm,
        plan_repo=PlanRepository(session),
        goal_repo=GoalRepository(session),
        mastery_repo=MasteryRepository(session),
        mistake_repo=MistakeRepository(session),
        retriever=retriever or StubRetriever(),
        now_fn=lambda: datetime(2026, 5, 23, 12, 0),
        max_iter=10,
    )


async def test_loop_natural_stop_emits_plan_and_records_trace(session):
    user = UserRepository(session).get_or_create("fp-loop-1")
    goal_repo = GoalRepository(session)
    goal_repo.create(user_id=user.id, title="G")

    llm = ScriptedLLM([
        _msg(tool_calls=[{
            "name": "retriever_search",
            "args": {"query": "HyDE", "top_k": 3},
            "id": "c1",
        }]),
        _msg(tool_calls=[{
            "name": "update_study_plan",
            "args": {"milestones": [
                {"title": "Read HyDE", "due_at": "2026-05-25", "done": False, "topic": "HyDE"},
            ]},
            "id": "c2",
        }]),
        _msg(content="📋 Plan: Read HyDE by 2026-05-25.", tool_calls=[]),
    ])
    agent = _build(session, llm, retriever=StubRetriever(chunks=[
        {"chunk_id": "c1", "content": "HyDE definition", "page": 1},
    ]))

    update = await agent({
        "messages": [HumanMessage(content="make a plan on HyDE")],
        "user_id": user.id,
    })

    assert update["plan_action"] == "generate"
    assert update["active_plan_id"]
    assert "Read HyDE" in update["messages"][0].content
    trace = update["agent_trace"]
    assert trace["exit_reason"] == "natural_stop"
    assert trace["total_iterations"] == 3
    assert trace["total_tool_calls"] == 2
    assert trace["tool_call_breakdown"] == {
        "retriever_search": 1, "update_study_plan": 1,
    }


async def test_loop_budget_exhaustion_degrades_without_persisting(session):
    user = UserRepository(session).get_or_create("fp-loop-2")
    GoalRepository(session).create(user_id=user.id, title="G")

    # LLM keeps calling retriever_search forever; loop must bail at max_iter.
    forever_calls = [
        _msg(tool_calls=[{"name": "retriever_search", "args": {"query": "x"}, "id": f"c{i}"}])
        for i in range(12)
    ]
    llm = ScriptedLLM(forever_calls)
    agent = _build(session, llm)

    update = await agent({
        "messages": [HumanMessage(content="plan on X")],
        "user_id": user.id,
    })

    assert update["agent_trace"]["exit_reason"] == "budget_exhausted"
    # No plan should have been persisted on this path
    assert update.get("active_plan_id") is None
    # User-visible disclaimer text
    assert "reasoning budget" in update["messages"][0].content.lower() \
        or "budget" in update["messages"][0].content.lower()


async def test_loop_llm_error_degrades_with_disclaimer(session):
    user = UserRepository(session).get_or_create("fp-loop-3")
    GoalRepository(session).create(user_id=user.id, title="G")
    agent = _build(session, CrashingLLM())

    update = await agent({
        "messages": [HumanMessage(content="plan on something")],
        "user_id": user.id,
    })

    assert update["agent_trace"]["exit_reason"] == "llm_call_failed"
    assert "ConnectionError" in update["agent_trace"]["llm_error"]
    assert "could not reach" in update["messages"][0].content.lower() \
        or "model" in update["messages"][0].content.lower()


async def test_loop_tool_error_feeds_back_and_self_corrects(session):
    """First update_study_plan call has a bad milestone shape → ToolMessage
    error → LLM retries with correct shape → natural stop."""
    user = UserRepository(session).get_or_create("fp-loop-4")
    GoalRepository(session).create(user_id=user.id, title="G")

    llm = ScriptedLLM([
        _msg(tool_calls=[{
            "name": "update_study_plan",
            "args": {"milestones": [{"WRONG_KEY": "no title"}]},  # invalid → tool error
            "id": "c1",
        }]),
        _msg(tool_calls=[{
            "name": "update_study_plan",
            "args": {"milestones": [
                {"title": "M1", "due_at": None, "done": False, "topic": "X"},
            ]},
            "id": "c2",
        }]),
        _msg(content="Plan ready.", tool_calls=[]),
    ])
    agent = _build(session, llm)

    update = await agent({
        "messages": [HumanMessage(content="plan on X")],
        "user_id": user.id,
    })

    assert update["agent_trace"]["exit_reason"] == "natural_stop"
    assert update["agent_trace"]["tool_errors"] == 1
    # Second call succeeded → plan persisted
    assert update.get("active_plan_id")


async def test_plan_action_generate_when_no_get_existing_plan_called(session):
    user = UserRepository(session).get_or_create("fp-loop-5")
    GoalRepository(session).create(user_id=user.id, title="G")
    llm = ScriptedLLM([_msg(content="just text", tool_calls=[])])
    agent = _build(session, llm)

    update = await agent({
        "messages": [HumanMessage(content="x")],
        "user_id": user.id,
    })
    # No tools called → fallback inference is "generate"
    assert update["plan_action"] == "generate"


async def test_plan_action_check_in_when_get_existing_plan_returned_nonnull(session):
    user = UserRepository(session).get_or_create("fp-loop-6")
    goal_repo = GoalRepository(session)
    goal = goal_repo.create(user_id=user.id, title="G")
    plan_repo = PlanRepository(session)
    plan_repo.create(goal_id=goal.id, milestones_json=[
        {"title": "old M", "done": False, "topic": "HyDE"},
    ])

    llm = ScriptedLLM([
        _msg(tool_calls=[{"name": "get_existing_plan", "args": {}, "id": "c1"}]),
        _msg(content="progress: 0/1 done", tool_calls=[]),
    ])
    agent = _build(session, llm)

    update = await agent({
        "messages": [HumanMessage(content="how is the plan going")],
        "user_id": user.id,
    })

    assert update["plan_action"] == "check_in"


async def test_plan_action_generate_when_get_existing_plan_returned_null(session):
    """get_existing_plan called BUT returned the literal "null" sentinel → generate."""
    user = UserRepository(session).get_or_create("fp-loop-7")
    GoalRepository(session).create(user_id=user.id, title="G")

    llm = ScriptedLLM([
        _msg(tool_calls=[{"name": "get_existing_plan", "args": {}, "id": "c1"}]),
        # Loop now sees "null" — model decides no existing plan, drafts a new one
        _msg(tool_calls=[{
            "name": "update_study_plan",
            "args": {"milestones": [
                {"title": "M1", "due_at": None, "done": False, "topic": "x"},
            ]},
            "id": "c2",
        }]),
        _msg(content="Plan made.", tool_calls=[]),
    ])
    agent = _build(session, llm)

    update = await agent({
        "messages": [HumanMessage(content="plan")],
        "user_id": user.id,
    })

    assert update["plan_action"] == "generate"


async def test_extract_topic_strips_mindmap_suffix_without_corrupting_english():
    """Regression for the P2.1-⑤i character-set vs word-suffix strip bug.
    Topic 'Spam' must not become 'Sp' after suffix stripping."""
    from app.agent.planner_agent import _extract_topic_for_agent_prompt

    assert _extract_topic_for_agent_prompt("make a plan on Spam") == "Spam"
    assert _extract_topic_for_agent_prompt("帮我做学习计划 on HyDE 画脑图") == "HyDE"
    assert _extract_topic_for_agent_prompt("plan on BM25?") == "BM25"
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cd backend && uv run pytest -q tests/agent/test_planner_agent_loop.py`

Expected output: `ImportError: cannot import name 'build_planner_agent' from 'app.agent.planner_agent'`. 8 errors.

- [ ] **Step 4: Append the loop body + factory to `planner_agent.py`**

Add these imports to the top of `backend/app/agent/planner_agent.py` (alongside the existing imports):

```python
import re
from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.config import get_stream_writer

from app.agent.state import CoachState
from app.db.repositories import (
    MasteryRepository,
    MistakeRepository,
)
```

Then append these helpers + constants + factory after the AgentTrace block:

```python
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
    trace.record_tool_call(name, args, str(output), error=False)
    return str(output)


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
```

- [ ] **Step 5: Run the loop tests to verify they pass**

Run: `cd backend && uv run pytest -q tests/agent/test_planner_agent_loop.py`

Expected output: `8 passed`.

Likely failure modes:
- `AssertionError: ScriptedLLM exhausted` → the loop is iterating one more time than expected. Check whether your natural-stop path appends the final response to `messages` before the empty-tool-calls check (it should, so the trace records the final iteration).
- `KeyError: 'plan_action'` on the budget-exhausted case → confirm `_format_degrade_output` includes `plan_action` (it does, falling back to `_infer_plan_action(trace)`).
- `_extract_topic_for_agent_prompt("plan on BM25?") != "BM25"` → the `_TOPIC_PUNCT_RE` strip must run after the `_TOPIC_TRAILING_RE` strip; verify ordering matches `planner.py:_extract_topic` exactly.

- [ ] **Step 6: Checkpoint — run full backend suite**

Run: `cd backend && uv run pytest -q`

Expected output: `173 passed` (165 + 8). Diff inspection (e.g. `diff` between current `planner.py` and HEAD's `planner.py`) should show no changes to `planner.py`. If anything in `tests/agent/test_planner.py` or `tests/agent/test_graph_plan_e2e.py` regressed, you accidentally mutated the deterministic path — back the change out.

---

## Cut P2.2-①d — Graph wiring + mode dispatch

**Files:**
- Modify: `backend/app/agent/graph.py:175-180` (rewrite `plan_node` body)
- Test: `backend/tests/agent/test_graph_plan_agent_e2e.py`

The state field already landed in Cut ①c. This cut wires `plan_node` to dispatch between the two planner factories based on `config.configurable.planner_mode`. All 14 existing graph tests must remain untouched — the new dispatcher's `mode == "deterministic"` branch is byte-equivalent to the old single-branch body.

- [ ] **Step 1: Create the 3-test failing file**

Create `backend/tests/agent/test_graph_plan_agent_e2e.py`:

```python
"""Cut P2.2-①d — graph-level e2e for the agent_loop planner mode.

Three contracts under test:
  1. config.configurable.planner_mode='agent_loop' routes to planner_agent
     callable, NOT to the deterministic planner.
  2. agent_trace ends up in the final state so the eval harness can read it.
  3. Judge sees plan_action and applies the plan rubric (not the tutor one),
     same as deterministic — proving the agent path is observationally
     equivalent to deterministic from the judge's perspective.

Stub LLM is shared between planner_agent (tool-calling) and judge — scripted
to emit a generate→update→summary sequence, then a high-scoring rubric JSON.
"""
from datetime import datetime

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.agent.graph import build_graph
from app.agent.planner import build_planner
from app.agent.planner_agent import build_planner_agent
from app.db.models import Base
from app.db.repositories import (
    GoalRepository,
    MasteryRepository,
    MistakeRepository,
    PlanRepository,
    UserRepository,
)


def _msg(content="", tool_calls=None, input_tokens=10, output_tokens=5):
    m = AIMessage(content=content, tool_calls=tool_calls or [])
    m.usage_metadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    return m


class ScriptedLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.idx = 0

    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, messages, **_kwargs):
        if self.idx >= len(self.responses):
            raise AssertionError("ScriptedLLM exhausted")
        m = self.responses[self.idx]
        self.idx += 1
        return m

    async def astream(self, messages, **_kwargs):
        yield self.responses[0]


class StubRetriever:
    def search(self, query, top_k=5):
        return [{"chunk_id": "c1", "content": "HyDE def", "page": 1}]


@pytest.fixture
def session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,
    )
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def _wire(session, agent_llm, det_llm=None, judge_text=None):
    retriever = StubRetriever()
    graph = build_graph(retriever=retriever, llm=det_llm or agent_llm)
    planner = build_planner(
        llm=det_llm or agent_llm,
        plan_repo=PlanRepository(session),
        goal_repo=GoalRepository(session),
        mastery_repo=MasteryRepository(session),
        mistake_repo=MistakeRepository(session),
        retriever=retriever,
        now_fn=lambda: datetime(2026, 5, 23),
    )
    planner_agent = build_planner_agent(
        llm=agent_llm,
        plan_repo=PlanRepository(session),
        goal_repo=GoalRepository(session),
        mastery_repo=MasteryRepository(session),
        mistake_repo=MistakeRepository(session),
        retriever=retriever,
        now_fn=lambda: datetime(2026, 5, 23),
        max_iter=10,
    )
    judge_llm = ScriptedLLM([_msg(content=judge_text)]) if judge_text else None
    return graph, planner, planner_agent, judge_llm


async def test_graph_mode_agent_loop_dispatches_to_planner_agent_not_deterministic(session):
    user = UserRepository(session).get_or_create("fp-g-mode-agent")

    # Deterministic LLM scripted to crash if invoked → proves it wasn't picked
    class CrashIfCalled:
        async def ainvoke(self, *_a, **_kw):
            raise AssertionError("deterministic planner must not run in agent_loop mode")
        async def astream(self, *_a, **_kw):
            raise AssertionError("deterministic planner must not run in agent_loop mode")
        def bind_tools(self, _t): return self

    agent_llm = ScriptedLLM([
        _msg(tool_calls=[{
            "name": "update_study_plan",
            "args": {"milestones": [
                {"title": "M1", "due_at": "2026-05-30", "done": False, "topic": "HyDE"},
            ]},
            "id": "c1",
        }]),
        _msg(content="Plan done.", tool_calls=[]),
    ])
    graph, _, planner_agent, _ = _wire(session, agent_llm, det_llm=CrashIfCalled())

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="make a plan on HyDE")], "user_id": user.id},
        config={"configurable": {
            "planner_mode": "agent_loop",
            "planner_agent": planner_agent,
            "planner": CrashIfCalled(),  # would crash if dispatcher picked deterministic
        }},
    )

    assert result.get("plan_action") == "generate"
    assert result.get("active_plan_id")
    assert "agent_trace" in result and result["agent_trace"]["exit_reason"] == "natural_stop"


async def test_graph_default_mode_routes_to_deterministic_planner(session):
    """Absent planner_mode key → fallback to deterministic. Existing 157 tests
    pass this implicitly; making it explicit here defends against accidental
    dispatcher rewrites."""
    user = UserRepository(session).get_or_create("fp-g-mode-default")
    det_llm = ScriptedLLM([_msg(content="""[
      {"title": "M1", "due_at": "2026-05-30", "done": false, "topic": "HyDE"}
    ]""")])

    class CrashAgentIfCalled:
        async def ainvoke(self, *_a, **_kw):
            raise AssertionError("agent_loop must not run when mode is unset")
        def bind_tools(self, _t): return self

    graph, planner, planner_agent, _ = _wire(session, CrashAgentIfCalled(), det_llm=det_llm)

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="帮我做学习计划 on HyDE")], "user_id": user.id},
        config={"configurable": {"planner": planner}},  # NO planner_mode key
    )

    assert result.get("plan_action") == "generate"
    assert result.get("agent_trace") is None or "agent_trace" not in result


async def test_graph_judge_runs_plan_rubric_on_agent_loop_output(session):
    """Judge sees plan_action='generate' from agent_loop → applies PLAN_DIMENSIONS
    rubric just like deterministic. Pass-verdict moves through memory_writer."""
    user = UserRepository(session).get_or_create("fp-g-judge-agent")
    agent_llm = ScriptedLLM([
        _msg(tool_calls=[{
            "name": "update_study_plan",
            "args": {"milestones": [
                {"title": "M1", "due_at": "2026-05-30", "done": False, "topic": "HyDE"},
            ]},
            "id": "c1",
        }]),
        _msg(content="Plan generated.", tool_calls=[]),
    ])
    judge_pass = (
        '{"milestone_specificity": 5, "milestone_granularity": 5, '
        '"time_feasibility": 5, "topic_coverage": 4, "actionability": 5, '
        '"reasoning": "ok"}'
    )
    graph, planner, planner_agent, judge_llm = _wire(session, agent_llm, judge_text=judge_pass)

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="make a plan on HyDE")], "user_id": user.id},
        config={"configurable": {
            "planner_mode": "agent_loop",
            "planner_agent": planner_agent,
            "planner": planner,
            "judge_llm": judge_llm,
        }},
    )

    assert result.get("judge_score", 0) >= 0.6
    assert result.get("plan_action") == "generate"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd backend && uv run pytest -q tests/agent/test_graph_plan_agent_e2e.py`

Expected: 3 fails. Likely error: `AssertionError: deterministic planner must not run in agent_loop mode` — because the current `plan_node` only knows about `configurable.planner`, ignoring the new `planner_mode` key, so it picks the deterministic factory regardless.

- [ ] **Step 3: Rewrite `plan_node` to dispatch on mode**

Modify `backend/app/agent/graph.py`. Find this block (lines 175–180):

```python
    async def plan_node(state: CoachState, config) -> dict:
        configurable = (config or {}).get("configurable", {}) or {}
        planner = configurable.get("planner")
        if planner is None:
            return plan_stub_node(state)
        return await planner(state)
```

Replace with:

```python
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
```

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `cd backend && uv run pytest -q tests/agent/test_graph_plan_agent_e2e.py`

Expected: `3 passed`.

- [ ] **Step 5: Re-run all existing graph + plan tests (anti-regression)**

Run: `cd backend && uv run pytest -q tests/agent/test_graph.py tests/agent/test_graph_judge.py tests/agent/test_graph_memory.py tests/agent/test_graph_plan_e2e.py tests/agent/test_graph_quiz_e2e.py tests/agent/test_planner.py`

Expected: all green (no count regression). If any test calling `plan_node` via `config.configurable.planner` (no `planner_mode`) breaks, the dispatcher default-branch is wrong; double-check the `mode == "deterministic"` else-branch is identical to the old body.

- [ ] **Step 6: Checkpoint — full backend suite**

Run: `cd backend && uv run pytest -q`

Expected output: `176 passed` (173 + 3).

---

## Cut P2.2-①e — Production wiring (deps.py + routes.py)

**Files:**
- Modify: `backend/app/api/deps.py` (append `get_planner_mode` + `get_planner_agent`)
- Modify: `backend/app/api/routes.py:83-131` (extend `chat()` signature + config)
- Test: `backend/tests/api/test_routes_plan_agent.py`

This is the cut that gets agent_loop into actual production HTTP traffic. Lesson from Cut ④f (P2.1-④f): unit tests on the factory aren't enough proof — ship the routes integration test in the same cut. That's the second of our 2 tests here.

- [ ] **Step 1: Create the 2-test failing file**

Create `backend/tests/api/test_routes_plan_agent.py`:

```python
"""Cut P2.2-①e — /api/chat with x-planner-mode header.

Two contracts under test:
  1. x-planner-mode: agent_loop routes through the new planner_agent factory
     and produces the same SSE shape (citations → token → done) as the
     deterministic baseline.
  2. Default (no header) and unknown values fall back to deterministic; the
     existing P2.1-⑤g test_chat_plan_generate_emits_citations_token_done
     covers default — here we explicitly verify the unknown-value fallback.
"""
import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.api import deps
from app.main import create_app


def _msg(content="", tool_calls=None, input_tokens=10, output_tokens=5):
    m = AIMessage(content=content, tool_calls=tool_calls or [])
    m.usage_metadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    return m


class ScriptedLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.idx = 0

    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, messages, **_kwargs):
        if self.idx >= len(self.responses):
            raise AssertionError("ScriptedLLM exhausted")
        m = self.responses[self.idx]
        self.idx += 1
        return m

    async def astream(self, messages, **_kwargs):
        yield self.responses[0]


class StubRetriever:
    def add_chunks(self, _chunks):
        pass

    def search(self, query, top_k=5):
        return [{"chunk_id": "c1", "content": "HyDE def",
                 "source": "p.pdf", "page": 1, "score": 0.9}]


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/agent_test.db")
    monkeypatch.setenv("STUDY_COACH_TEST_MODE", "1")
    from app.db import session as session_mod
    session_mod._engine = None
    session_mod._SessionLocal = None

    app = create_app()
    app.state.retriever = StubRetriever()

    # Agent-mode LLM script: tool call → final summary
    agent_script = [
        _msg(tool_calls=[{
            "name": "update_study_plan",
            "args": {"milestones": [
                {"title": "M1", "due_at": "2026-05-30", "done": False, "topic": "HyDE"},
            ]},
            "id": "c1",
        }]),
        _msg(content="📋 Plan: M1 by 2026-05-30."),
    ]
    # Deterministic-mode LLM script: raw milestones JSON (one call only)
    det_script = ["""[
      {"title": "DET-M1", "due_at": "2026-05-30", "done": false, "topic": "HyDE"}
    ]"""]

    def get_llm_override():
        # ChatRequest goes through deps.get_llm once per request; both factories
        # (planner, planner_agent) receive the same LLM. We return a fresh
        # ScriptedLLM keyed by what state the request is in. For these tests we
        # just give a fixed agent script — that's enough because the
        # deterministic factory uses .ainvoke() once and the agent factory
        # uses .ainvoke() twice; their scripts don't share order.
        return ScriptedLLM([_msg(content=det_script[0]), *agent_script])

    app.dependency_overrides[deps.get_llm] = get_llm_override
    app.dependency_overrides[deps.get_judge_dependencies] = lambda: {"llm": None, "same_model": False}

    with TestClient(app) as c:
        yield c


def _parse_events(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: "):]))
    return events


def test_chat_agent_loop_mode_emits_citations_token_done(client):
    response = client.post(
        "/api/chat",
        json={"message": "make a plan on HyDE", "session_id": "sess-agent-1"},
        headers={
            "x-fingerprint": "fp-agent",
            "x-planner-mode": "agent_loop",
        },
    )
    assert response.status_code == 200
    events = _parse_events(response.text)
    types = [e["type"] for e in events]
    assert types[0] == "citations"
    assert "token" in types
    assert types[-1] == "done"
    token_event = next(e for e in events if e["type"] == "token")
    # Agent-mode final summary contained the persisted plan title
    assert "M1" in token_event["text"]


def test_chat_unknown_planner_mode_falls_back_to_deterministic(client):
    response = client.post(
        "/api/chat",
        json={"message": "帮我做学习计划 on HyDE", "session_id": "sess-fallback"},
        headers={
            "x-fingerprint": "fp-fallback",
            "x-planner-mode": "bogus-value",
        },
    )
    assert response.status_code == 200
    events = _parse_events(response.text)
    token_text = "".join(e.get("text", "") for e in events if e["type"] == "token")
    # Deterministic-mode output uses the deterministic LLM script (DET-M1)
    assert "DET-M1" in token_text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd backend && uv run pytest -q tests/api/test_routes_plan_agent.py`

Expected: 2 fails. Both will fail at the route level because `chat()` doesn't yet pull `x-planner-mode` or inject `planner_agent` into config — the requests likely succeed but produce the deterministic output ("DET-M1") in both tests, so the first test's `assert "M1" in token_event["text"]` may pass spuriously (since "DET-M1" contains "M1"). Tighten test 1's assertion if needed during step 4 — preferred shape: assert that the test 1 token text contains "📋 Plan: M1 by 2026-05-30." which is the agent-final-summary string and does NOT appear in the deterministic output.

- [ ] **Step 3: Append `get_planner_mode` + `get_planner_agent` to `deps.py`**

Append to the end of `backend/app/api/deps.py`:

```python
from typing import Literal


def get_planner_mode(
    x_planner_mode: Annotated[str | None, Header()] = None,
) -> Literal["deterministic", "agent_loop"]:
    """Read x-planner-mode header. Default = deterministic. Unknown → deterministic.

    Defensive default: unknown values silently fall back so a typo on the
    client side never breaks production. The eval harness sends the header
    explicitly and never relies on default.
    """
    if x_planner_mode == "agent_loop":
        return "agent_loop"
    return "deterministic"


def get_planner_agent(
    session: Annotated[Session, Depends(get_session)],
    llm: Annotated[object, Depends(get_llm)],
    retriever: Annotated[object, Depends(get_retriever)],
):
    # cloud-adapt: when provider=cloud (BYOK GPT/Claude/Gemini), max_iter can
    # safely be raised to 20-30 here; small local Ollama models cap at 10.
    from app.agent.planner_agent import build_planner_agent
    from app.db.repositories import (
        GoalRepository,
        MasteryRepository,
        MistakeRepository,
        PlanRepository,
    )
    return build_planner_agent(
        llm=llm,
        plan_repo=PlanRepository(session),
        goal_repo=GoalRepository(session),
        mastery_repo=MasteryRepository(session),
        mistake_repo=MistakeRepository(session),
        retriever=retriever,
    )
```

Note: `Literal` import was missing — add `Literal` to the existing `from typing import Annotated` line so it reads `from typing import Annotated, Literal`. If `Literal` is already imported elsewhere in `deps.py` at the time the implementer reads this plan, just skip the extra import.

- [ ] **Step 4: Extend `chat()` in `routes.py`**

Modify `backend/app/api/routes.py`. First, update the imports block (line 16-26):

```python
from .deps import (
    get_document_processor,
    get_graph,
    get_judge_dependencies,
    get_memory_hydrator,
    get_memory_writer,
    get_planner,
    get_planner_agent,
    get_planner_mode,
    get_quiz_master,
    get_retriever,
    get_user_id,
)
```

Then replace the `chat()` function signature + config block (lines 83–112). Locate this:

```python
@router.post("/chat")
async def chat(
    body: ChatRequest,
    user_id: Annotated[str, Depends(get_user_id)],
    graph: Annotated[object, Depends(get_graph)],
    judge: Annotated[dict, Depends(get_judge_dependencies)],
    quiz_master: Annotated[object, Depends(get_quiz_master)],
    planner: Annotated[object, Depends(get_planner)],
    memory_hydrator: Annotated[object, Depends(get_memory_hydrator)],
    memory_writer: Annotated[object, Depends(get_memory_writer)],
):
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
                "memory_hydrator": memory_hydrator,
                "memory_writer": memory_writer,
            }
        }
```

Replace with:

```python
@router.post("/chat")
async def chat(
    body: ChatRequest,
    user_id: Annotated[str, Depends(get_user_id)],
    graph: Annotated[object, Depends(get_graph)],
    judge: Annotated[dict, Depends(get_judge_dependencies)],
    quiz_master: Annotated[object, Depends(get_quiz_master)],
    planner: Annotated[object, Depends(get_planner)],
    planner_agent: Annotated[object, Depends(get_planner_agent)],
    planner_mode: Annotated[str, Depends(get_planner_mode)],
    memory_hydrator: Annotated[object, Depends(get_memory_hydrator)],
    memory_writer: Annotated[object, Depends(get_memory_writer)],
):
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
                "memory_hydrator": memory_hydrator,
                "memory_writer": memory_writer,
            }
        }
```

Leave the rest of `event_stream()` untouched.

- [ ] **Step 5: Run new route tests to verify they pass**

Run: `cd backend && uv run pytest -q tests/api/test_routes_plan_agent.py`

Expected: `2 passed`.

If `test_chat_agent_loop_mode_emits_citations_token_done` fails with `assert "M1" in token_event["text"]` returning a "DET-M1" match (spurious), edit the assertion to `assert "📋 Plan: M1 by 2026-05-30." in token_event["text"]` — that string is uniquely the agent-final-summary content from `agent_script[-1]`.

If `test_chat_unknown_planner_mode_falls_back_to_deterministic` fails because `DET-M1` doesn't appear, the dispatcher in `graph.py` is misreading the mode — verify the `mode = configurable.get("planner_mode", "deterministic")` line returns `"deterministic"` for `"bogus-value"`. (It does, because `get_planner_mode` only matches the exact string `"agent_loop"`.)

- [ ] **Step 6: Re-run all existing routes tests (anti-regression)**

Run: `cd backend && uv run pytest -q tests/api/`

Expected: all green. The dependency-injection signature grew by 2 — any test that uses `dependency_overrides` for old deps continues to work; the new deps are pulled fresh each request with sensible defaults.

- [ ] **Step 7: Checkpoint — full backend suite**

Run: `cd backend && uv run pytest -q`

Expected output: `178 passed` (176 + 2). This crosses the **target threshold for P2.2-①**: per spec §7, the implementation cuts target 24 tests = 181, with 3 of those landing in eval harness (Cut ②a). So `178 = 157 baseline + 21 from ①a-①e`. After Cut ②a we hit 181. (One of those 21 is the `_extract_topic` regression in `test_planner_agent_loop.py`; spec §7 counted it implicitly.)

---

## Cut P2.2-①f — Real-Ollama smoke test (manual)

**Files:**
- No code changes; this is a verification cut.
- Output: append a short table to the plan file's "Verification Log" appendix (see Cut ③).

**Why manual:** automated tests use stub LLMs by necessity — Ollama-tool-calling behavior on small models can't be unit-tested without burning minutes per run. This cut is where we discover whether the 4 models can actually emit valid `tool_calls` at all.

**Pre-requirement:** `ollama pull gemma3:4b qwen3.5:4b qwen2.5:7b gemma4:e4b` — must be done once on the machine running the smoke.

- [ ] **Step 1: Decide the `think=False` parameter shape**

Open a Python REPL and run the following for each thinking-capable model (qwen3.5:4b, gemma4:e4b):

```python
import asyncio
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

async def try_no_think(model: str, mechanism: str):
    """mechanism: 'model_kwargs' | 'extra_body' | 'system_prefix' | 'none'"""
    if mechanism == "model_kwargs":
        llm = ChatOllama(model=model, model_kwargs={"think": False})
    elif mechanism == "extra_body":
        # langchain-ollama may not pass extra_body through; this may no-op
        llm = ChatOllama(model=model, extra_body={"think": False})
    elif mechanism == "system_prefix":
        llm = ChatOllama(model=model)
    else:
        llm = ChatOllama(model=model)
    msg = HumanMessage(content="/no_think\nWhat is 2+2? Reply with just the number." if mechanism == "system_prefix" else "What is 2+2?")
    resp = await llm.ainvoke([msg])
    return resp.content[:300]

print(asyncio.run(try_no_think("qwen3.5:4b", "model_kwargs")))
print(asyncio.run(try_no_think("qwen3.5:4b", "system_prefix")))
print(asyncio.run(try_no_think("gemma4:e4b", "model_kwargs")))
```

Record which mechanism actually suppresses `<think>` blocks (look for the absence of `<think>...</think>` in `resp.content`). The winning mechanism becomes the canonical disable shape used in Cut ②a. If `model_kwargs={"think": False}` works for both, prefer that — single uniform code path.

If none of the mechanisms actually suppress thinking on a given model, log it in the table below as "thinking always on" — that itself is a data point for EVAL.md.

- [ ] **Step 2: For each (model, mode) pair, run one happy-path request**

Start the backend: `cd backend && uv run uvicorn app.main:app --reload --port 8000`

In a second terminal, run for each combination:

```bash
for model in gemma3:4b qwen3.5:4b qwen2.5:7b gemma4:e4b; do
  for mode in deterministic agent_loop; do
    echo "=== model=$model mode=$mode ==="
    curl -N -s -X POST http://localhost:8000/api/chat \
      -H 'content-type: application/json' \
      -H 'x-fingerprint: smoke-test' \
      -H "x-provider: ollama" -H "x-model: $model" \
      -H "x-planner-mode: $mode" \
      -d '{"message": "make a study plan on HyDE", "session_id": "smoke-'"$model"'-'"$mode"'"}' \
      | tee /tmp/smoke-${model//[:.]/_}-${mode}.log | head -80
    echo
  done
done
```

(Each command will stream SSE events to stdout. Look for `data: {"type":"token", ...}` containing markdown milestones.)

- [ ] **Step 3: Record results in a smoke table**

For each cell, record:

| model × mode | tool_calls emitted? | natural_stop reached? | plan persisted? | notes |
|---|---|---|---|---|
| gemma3:4b × deterministic | n/a | n/a | yes | baseline; should always work |
| gemma3:4b × agent_loop | ? | ? | ? | gemma3 has no `tools` capability — expected: 0 tool calls, exit_reason=`natural_stop` with the model outputting plain text instead of structured calls |
| qwen3.5:4b × deterministic | n/a | n/a | yes | baseline |
| qwen3.5:4b × agent_loop | ? | ? | ? | thinking model — expect `<think>` blocks unless step 1 fix in place |
| qwen2.5:7b × deterministic | n/a | n/a | yes | baseline |
| qwen2.5:7b × agent_loop | ? | ? | ? | the most expected-to-work agent_loop cell |
| gemma4:e4b × deterministic | n/a | n/a | yes | baseline |
| gemma4:e4b × agent_loop | ? | ? | ? | thinking + tools, eval main matrix runs this with thinking-OFF |

Surfaces three classes of issues:
1. **Tool calling silently empty** on a model that should support it → check the Ollama Modelfile `Capabilities` line; consider whether `langchain-ollama` is sending the tools field at all (set `langchain.debug = True` or use `ChatOllama(verbose=True)`).
2. **Thinking output bleeds into `content`** → step 1's mechanism didn't work for that model; document in EVAL.md as a known limitation.
3. **Crash mid-loop** → likely a `tool_call_id` mismatch or `ToolMessage.content` size issue. Inspect the captured log; fix in `_safe_invoke_tool` or `_format_final_output`.

- [ ] **Step 4: Decide what to ship vs. defer**

If gemma3:4b emits zero tool_calls (expected per spec §1 Q1a Excluded note: gemma3 has `Capabilities: completion` only): leave the implementation as-is and document the model's behavior in `EVAL.md`. Do NOT add a special-case branch in `planner_agent.py` — "model with no tool capability emits no tool_calls → loop exits at iter=0 with natural_stop" is the cleanest fallback and is itself useful data.

If qwen3.5:4b or gemma4:e4b emits malformed tool_calls that crash `_safe_invoke_tool`: extend `_safe_invoke_tool` to also catch JSON shape errors at the dispatch layer, append a tool-error ToolMessage, log it in the AgentTrace.tool_errors, and let the loop continue. This is already covered by the unknown-tool branch in the implementation; verify it's reachable for malformed calls.

No code changes are checkpoint-able from this cut alone — the verification log lives in the EVAL.md appendix Cut ③ produces.

- [ ] **Step 5: Mark this cut complete only when**

- Each of the 8 cells in the table has a result row filled.
- Any non-expected crash has been root-caused (not just suppressed).
- `cd backend && uv run pytest -q` still shows `178 passed` — no implementation change made by smoke debugging snuck in untested.

---

## Cut P2.2-②a — Eval harness skeleton

**Files:**
- Create: `backend/app/eval/p2_2_agent_ablation/__init__.py`
- Create: `backend/app/eval/p2_2_agent_ablation/matrix.py`
- Create: `backend/app/eval/p2_2_agent_ablation/single_run.py`
- Create: `backend/app/eval/p2_2_agent_ablation/judges.py`
- Create: `backend/app/eval/p2_2_agent_ablation/run_eval.py`
- Create: `backend/app/eval/p2_2_agent_ablation/queries.json`
- Test: `backend/tests/eval/test_p2_2_harness.py`

Three tests: matrix expansion, record schema, resumability. We are NOT writing tests for the judge LLM calls or the actual matrix run — those are real-LLM operations that Cut ②b handles manually. The harness CODE is what we test here.

- [ ] **Step 1: Reuse the existing query fixture + add multi-turn scenarios**

Create `backend/app/eval/p2_2_agent_ablation/queries.json`:

```json
{
  "single_turn_plan": [
    {"id": "plan_hyde", "message": "帮我做学习计划 on HyDE"},
    {"id": "plan_bm25", "message": "make a plan on BM25"},
    {"id": "plan_reranking", "message": "make a study plan on reranking"},
    {"id": "plan_chunking", "message": "帮我做学习计划 on chunking strategies"},
    {"id": "plan_eval", "message": "plan on evaluation metrics for retrieval"},
    {"id": "plan_judge", "message": "make a plan on LLM-as-judge"},
    {"id": "plan_embeddings", "message": "帮我做学习计划 on embedding models"},
    {"id": "plan_hybrid", "message": "make a plan on hybrid retrieval"},
    {"id": "plan_mindmap", "message": "make a plan on HyDE 画脑图"},
    {"id": "plan_mindmap_en", "message": "make a plan on BM25 with mindmap"}
  ],
  "multi_turn_check_in": [
    {
      "id": "check_in_progress",
      "messages": [
        "帮我做学习计划 on HyDE",
        "进度怎么样了"
      ]
    },
    {
      "id": "check_in_edit",
      "messages": [
        "make a plan on BM25",
        "调整第二个里程碑标记为完成"
      ]
    }
  ]
}
```

10 single-turn + 2 multi-turn = 12 logical queries; multi-turn each counts as one "query_id" but emits 2 records (per turn) into the matrix. Total runs:

- Single-turn: 10 × 4 models × 2 modes × 3 runs = 240
- Multi-turn: 2 × 2 turns × 4 models × 2 modes × 3 runs = 96
- Subtotal: 336
- Appendix (gemma4:e4b thinking on/off): 12 × 1 model × 2 thinking states × 2 modes × 3 runs = 144

Spec said 288 + 72 = 360. Plan above gives 336 + 144 = 480 — exceeds. **Decision**: trim multi-turn to single-turn-only for the appendix (just gemma4 thinking on/off on single-turn = 60 runs); revisit if main 336 matrix is inconclusive. Final target: 336 main + 60 appendix = **396 runs** ≈ 1.5–2.5 hours wall time. Adjust the appendix matrix in `matrix.py` accordingly.

If the implementer prefers spec's exact 360, drop 2 single-turn queries (e.g. `plan_chunking` and `plan_eval`) to land at 288 + 72.

- [ ] **Step 2: Create the test file with 3 failing tests**

Create `backend/tests/eval/__init__.py` if missing (empty file).

Create `backend/tests/eval/test_p2_2_harness.py`:

```python
"""Cut P2.2-②a — eval harness unit tests."""
import json
from pathlib import Path

import pytest

from app.eval.p2_2_agent_ablation.matrix import RunSpec, expand_matrix
from app.eval.p2_2_agent_ablation.single_run import validate_record_schema


def test_matrix_expansion_main_run_count_matches_spec():
    """Main matrix (no appendix) over single-turn queries only."""
    specs = expand_matrix(
        models=["gemma3:4b", "qwen3.5:4b", "qwen2.5:7b", "gemma4:e4b"],
        modes=["deterministic", "agent_loop"],
        single_turn_queries=[{"id": f"q{i}", "message": f"m{i}"} for i in range(10)],
        multi_turn_queries=[
            {"id": "mt1", "messages": ["a", "b"]},
            {"id": "mt2", "messages": ["c", "d"]},
        ],
        runs=3,
        thinking=False,
    )

    # 10 × 4 × 2 × 3 single-turn = 240
    # 2 multi-turn × 2 turns × 4 × 2 × 3 = 96
    # Total = 336
    assert len(specs) == 336
    # Every spec has unique run_id
    assert len({s.run_id for s in specs}) == 336
    # The two multi-turn queries each produced "turn_idx" 0 and 1 records
    turn_indices = {(s.query_id, s.turn_idx) for s in specs if s.query_id.startswith("mt")}
    assert (("mt1", 0) in turn_indices and ("mt1", 1) in turn_indices)


def test_record_schema_validation_accepts_full_and_rejects_missing_keys():
    """One test, two assertions — schema validation is one contract surface."""
    full = {
        "run_id": "abc-123",
        "timestamp": "2026-05-23T12:00:00",
        "model": "qwen2.5:7b",
        "mode": "agent_loop",
        "query_id": "plan_hyde",
        "turn_idx": 0,
        "run_idx": 0,
        "operational": {
            "wall_time_s": 4.23,
            "iterations": 3,
            "tool_calls": [{"name": "retriever_search", "success": True, "error": None}],
            "tool_call_count": 1,
            "tool_errors": 0,
            "input_tokens": 1843,
            "output_tokens": 412,
            "exit_reason": "natural_stop",
        },
        "output": {
            "plan_action": "generate",
            "milestones_persisted": 5,
            "milestones_json": [],
            "final_text_excerpt": "Plan: ...",
        },
        "judge_local": {"score": 0.78, "weak_dims": [], "reasoning": "..."},
        "judge_cloud": {"score": 0.82, "weak_dims": [], "reasoning": "...", "model": "gpt-4o-mini"},
    }
    # Must NOT raise on a full record
    validate_record_schema(full)

    # Reject when a required top-level key is absent
    partial = {k: v for k, v in full.items() if k != "timestamp"}
    with pytest.raises(ValueError, match="missing required key"):
        validate_record_schema(partial)


def test_resumable_skips_runs_already_in_results_jsonl(tmp_path):
    from app.eval.p2_2_agent_ablation.run_eval import filter_pending_specs

    all_specs = expand_matrix(
        models=["m1", "m2"],
        modes=["deterministic"],
        single_turn_queries=[{"id": "q1", "message": "x"}],
        multi_turn_queries=[],
        runs=2,
        thinking=False,
    )
    # All specs: 2 models × 1 mode × 1 query × 2 runs = 4
    assert len(all_specs) == 4

    # Pretend 2 are already in results
    results_path = tmp_path / "results.jsonl"
    done_ids = [all_specs[0].run_id, all_specs[2].run_id]
    with results_path.open("w") as f:
        for rid in done_ids:
            f.write(json.dumps({"run_id": rid, "model": "m1", "mode": "deterministic"}) + "\n")

    pending = filter_pending_specs(all_specs, results_path)
    assert len(pending) == 2
    assert all_specs[0].run_id not in {s.run_id for s in pending}
    assert all_specs[1].run_id in {s.run_id for s in pending}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd backend && uv run pytest -q tests/eval/test_p2_2_harness.py`

Expected: 3 ImportErrors (modules don't exist yet).

- [ ] **Step 4: Implement `matrix.py`**

Create `backend/app/eval/p2_2_agent_ablation/__init__.py`:

```python
"""P2.2 Agent Loop Ablation eval harness.

CLI entry: `python -m app.eval.p2_2_agent_ablation.run_eval --output output/results.jsonl`

Schemas:
  - RunSpec: one row in the experiment matrix (model × mode × query × turn × run).
  - results.jsonl record: one row per executed RunSpec; schema validated by
    `single_run.validate_record_schema`.
"""
```

Create `backend/app/eval/p2_2_agent_ablation/matrix.py`:

```python
"""Matrix expansion for the P2.2 agent-loop ablation eval."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    model: str
    mode: Literal["deterministic", "agent_loop"]
    query_id: str
    turn_idx: int            # 0 for single-turn or first turn of multi-turn
    run_idx: int             # which repeat run (0..runs-1)
    thinking: bool           # appendix axis only; main matrix all False
    message: str             # the user message for this turn
    is_multi_turn: bool
    session_key: str         # shared across turns of one multi-turn run


def _run_id(*parts: str) -> str:
    """Deterministic ID so resumability key is reproducible across processes."""
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:16]


def expand_matrix(
    *,
    models: list[str],
    modes: list[str],
    single_turn_queries: list[dict],
    multi_turn_queries: list[dict],
    runs: int,
    thinking: bool = False,
) -> list[RunSpec]:
    """Cartesian product of (model × mode × query × run), with multi-turn
    queries unrolled into one RunSpec per (query, turn). Each multi-turn run
    shares a session_key so the harness can replay turns against the same
    LangGraph checkpointer thread.
    """
    out: list[RunSpec] = []
    for model in models:
        for mode in modes:
            for run_idx in range(runs):
                # Single-turn
                for q in single_turn_queries:
                    sk = _run_id(model, mode, q["id"], str(run_idx), str(thinking), "ST")
                    rid = _run_id(model, mode, q["id"], "0", str(run_idx), str(thinking))
                    out.append(RunSpec(
                        run_id=rid, model=model, mode=mode,
                        query_id=q["id"], turn_idx=0, run_idx=run_idx,
                        thinking=thinking, message=q["message"],
                        is_multi_turn=False, session_key=sk,
                    ))
                # Multi-turn
                for q in multi_turn_queries:
                    sk = _run_id(model, mode, q["id"], str(run_idx), str(thinking), "MT")
                    for ti, msg in enumerate(q["messages"]):
                        rid = _run_id(model, mode, q["id"], str(ti), str(run_idx), str(thinking))
                        out.append(RunSpec(
                            run_id=rid, model=model, mode=mode,
                            query_id=q["id"], turn_idx=ti, run_idx=run_idx,
                            thinking=thinking, message=msg,
                            is_multi_turn=True, session_key=sk,
                        ))
    return out
```

- [ ] **Step 5: Implement `single_run.py`**

Create `backend/app/eval/p2_2_agent_ablation/single_run.py`:

```python
"""One row in results.jsonl, plus the function that produces it."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage


REQUIRED_TOP_LEVEL_KEYS = (
    "run_id", "timestamp", "model", "mode", "query_id",
    "turn_idx", "run_idx",
    "operational", "output", "judge_local", "judge_cloud",
)
REQUIRED_OPERATIONAL_KEYS = (
    "wall_time_s", "iterations", "tool_calls", "tool_call_count",
    "tool_errors", "input_tokens", "output_tokens", "exit_reason",
)


def validate_record_schema(record: dict) -> None:
    """Cheap structural check — DOES NOT validate types deeply."""
    for k in REQUIRED_TOP_LEVEL_KEYS:
        if k not in record:
            raise ValueError(f"record missing required key: {k}")
    op = record["operational"]
    for k in REQUIRED_OPERATIONAL_KEYS:
        if k not in op:
            raise ValueError(f"record.operational missing required key: {k}")


async def run_one(
    *,
    spec,                    # RunSpec
    graph,                   # compiled LangGraph
    judge_local,             # callable: (question, plan_text) -> {score, weak_dims, reasoning}
    judge_cloud,             # callable, may be None if budget exhausted
    config_extras: dict,     # planner/planner_agent/memory_* callables + judge_llm
    user_id: str,
) -> dict:
    """Execute one RunSpec end-to-end, build the record dict, return it."""
    input_state = {
        "messages": [HumanMessage(content=spec.message)],
        "user_id": user_id,
    }
    config = {
        "configurable": {
            **config_extras,
            "thread_id": spec.session_key,
            "planner_mode": spec.mode,
        }
    }
    t0 = datetime.utcnow()
    try:
        final_state = await graph.ainvoke(input_state, config=config)
        err = None
    except Exception as exc:
        final_state = {}
        err = f"{type(exc).__name__}: {exc}"
    elapsed = (datetime.utcnow() - t0).total_seconds()

    trace = final_state.get("agent_trace") or {
        "wall_time_s": elapsed,
        "total_iterations": 0,
        "total_tool_calls": 0,
        "tool_call_breakdown": {},
        "tool_errors": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "exit_reason": "deterministic" if spec.mode == "deterministic" else ("error" if err else "n/a"),
        "llm_error": err,
    }
    final_text = ""
    msgs = final_state.get("messages") or []
    if msgs:
        last = msgs[-1]
        final_text = getattr(last, "content", "") or ""

    plan_action = final_state.get("plan_action")
    persisted = (
        len(final_state.get("active_plan_id") or "") > 0
    )

    judge_local_out = await judge_local(spec.message, final_text) if judge_local else {}
    judge_cloud_out = await judge_cloud(spec.message, final_text) if judge_cloud else {}

    record = {
        "run_id": spec.run_id,
        "timestamp": t0.isoformat(),
        "model": spec.model,
        "mode": spec.mode,
        "query_id": spec.query_id,
        "turn_idx": spec.turn_idx,
        "run_idx": spec.run_idx,
        "operational": {
            "wall_time_s": trace.get("wall_time_s", elapsed),
            "iterations": trace.get("total_iterations", 0),
            "tool_calls": [
                {"name": name, "count": count}
                for name, count in (trace.get("tool_call_breakdown") or {}).items()
            ],
            "tool_call_count": trace.get("total_tool_calls", 0),
            "tool_errors": trace.get("tool_errors", 0),
            "input_tokens": trace.get("input_tokens", 0),
            "output_tokens": trace.get("output_tokens", 0),
            "exit_reason": trace.get("exit_reason", "unknown"),
        },
        "output": {
            "plan_action": plan_action,
            "milestones_persisted": 1 if persisted else 0,
            "milestones_json": [],   # populated by reading PlanRepository if needed
            "final_text_excerpt": final_text[:500],
        },
        "judge_local": judge_local_out,
        "judge_cloud": judge_cloud_out,
    }
    validate_record_schema(record)
    return record
```

- [ ] **Step 6: Implement `judges.py`**

Create `backend/app/eval/p2_2_agent_ablation/judges.py`:

```python
"""Dual judges for the P2.2 eval.

Local: qwen2.5:7b via ChatOllama using app.agent.judge.judge_response with the
       plan rubric (PLAN_DIMENSIONS) so scores are directly comparable to
       production judge_node behavior.

Cloud: BYOK GPT-4o-mini (default) via direct openai/anthropic SDK. Reads
       OPENAI_API_KEY from the environment; falls back to None if missing
       so the harness still runs on machines without cloud access (entries
       just record judge_cloud: {} for those rows).
"""
from __future__ import annotations

import json
import os
from typing import Any

from app.agent.judge import (
    PLAN_DIMENSIONS,
    judge_response,
    load_plan_rubric,
)


def make_local_judge(judge_llm) -> Any:
    """Return an async callable (question, plan_text) -> judge dict."""
    rubric = load_plan_rubric()

    async def judge(question: str, plan_text: str) -> dict:
        result = await judge_response(
            question=question, answer=plan_text, context="",
            rubric=rubric, judge_llm=judge_llm,
            dimensions=PLAN_DIMENSIONS,
        )
        return {
            "score": result["score"],
            "weak_dims": result["weak_dims"],
            "reasoning": result["reasoning"],
        }

    return judge


def make_cloud_judge(model_id: str = "gpt-4o-mini"):
    """Return an async callable, or None if OPENAI_API_KEY is not set.

    cloud-adapt: this is the eval-side BYOK; production path uses x-judge-model
    header.
    """
    if not os.getenv("OPENAI_API_KEY"):
        return None
    from openai import AsyncOpenAI
    client = AsyncOpenAI()
    rubric = load_plan_rubric()

    async def judge(question: str, plan_text: str) -> dict:
        prompt = rubric.format(question=question, answer=plan_text, context="")
        resp = await client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        text = resp.choices[0].message.content or ""
        parsed: dict = {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Naive fallback: search for first {...}
            import re
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except json.JSONDecodeError:
                    parsed = {}
        scores = []
        weak = []
        for dim in PLAN_DIMENSIONS:
            v = parsed.get(dim, 3)
            try:
                fv = float(v)
            except (TypeError, ValueError):
                fv = 3.0
            fv = max(1.0, min(5.0, fv))
            scores.append(fv)
            if fv <= 3:
                weak.append(dim)
        avg = sum(scores) / len(scores) / 5.0
        return {
            "score": round(avg, 4),
            "weak_dims": weak,
            "reasoning": str(parsed.get("reasoning", ""))[:500],
            "model": model_id,
        }

    return judge
```

If the `openai` library is not in `pyproject.toml`: add it as a dev dep so cloud judging works without polluting production deps. Run `cd backend && uv add --dev openai>=1.40` (only if not present).

- [ ] **Step 7: Implement `run_eval.py`**

Create `backend/app/eval/p2_2_agent_ablation/run_eval.py`:

```python
"""CLI entry point for the P2.2 ablation eval.

Usage:
  python -m app.eval.p2_2_agent_ablation.run_eval \\
      --queries app/eval/p2_2_agent_ablation/queries.json \\
      --output app/eval/p2_2_agent_ablation/output/results.jsonl \\
      [--runs 3] [--models gemma3:4b,qwen3.5:4b,qwen2.5:7b,gemma4:e4b]
      [--modes deterministic,agent_loop] [--thinking-appendix]

Resumable: re-running with the same --output skips run_ids that already have
a row in the file. Failures are written as error rows (operational.exit_reason
== "harness_error") rather than retried — failure IS data per spec §6.4.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .matrix import RunSpec, expand_matrix


def filter_pending_specs(specs: list[RunSpec], results_path: Path) -> list[RunSpec]:
    if not results_path.exists():
        return list(specs)
    done = set()
    with results_path.open() as f:
        for line in f:
            try:
                done.add(json.loads(line)["run_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return [s for s in specs if s.run_id not in done]


async def main_async(args):
    """Wire up the graph + planner factories + judges, then iterate specs."""
    # Imports kept local so unit tests can import filter_pending_specs / matrix
    # without spinning up SQLite etc.
    from langchain_ollama import ChatOllama
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from sqlalchemy.pool import StaticPool

    from app.agent.graph import build_graph
    from app.agent.memory_updater import build_memory_hydrator, build_memory_writer
    from app.agent.planner import build_planner
    from app.agent.planner_agent import build_planner_agent
    from app.db.models import Base
    from app.db.repositories import (
        GoalRepository, MasteryRepository, MistakeRepository,
        PlanRepository, TopicRepository, UserRepository,
    )
    from langgraph.checkpoint.memory import InMemorySaver

    from .judges import make_cloud_judge, make_local_judge
    from .single_run import run_one

    with Path(args.queries).open() as f:
        queries_doc = json.load(f)

    models = args.models.split(",")
    modes = args.modes.split(",")
    specs = expand_matrix(
        models=models, modes=modes,
        single_turn_queries=queries_doc.get("single_turn_plan", []),
        multi_turn_queries=queries_doc.get("multi_turn_check_in", []),
        runs=args.runs, thinking=False,
    )
    if args.thinking_appendix:
        specs += expand_matrix(
            models=["gemma4:e4b"], modes=modes,
            single_turn_queries=queries_doc.get("single_turn_plan", []),
            multi_turn_queries=[], runs=args.runs, thinking=True,
        )

    results_path = Path(args.output)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    pending = filter_pending_specs(specs, results_path)
    print(f"Total specs: {len(specs)}; pending: {len(pending)}", file=sys.stderr)

    # cloud-adapt: production deploy would key the judge_llm off x-judge-model;
    # eval pins it to qwen2.5:7b for cross-mode score comparability.
    local_judge_llm = ChatOllama(model="qwen2.5:7b", temperature=0.0)
    judge_local = make_local_judge(local_judge_llm)
    judge_cloud = make_cloud_judge()

    # One DB/graph per model loop iteration so plans persist across multi-turn
    # turns but not across runs of different models (memory_writer would otherwise
    # carry mastery across).
    for spec in pending:
        engine = create_engine(
            f"sqlite:///{results_path.parent}/eval_{spec.session_key}.db",
            connect_args={"check_same_thread": False}, poolclass=StaticPool,
        )
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            user = UserRepository(session).get_or_create(f"eval-{spec.session_key}")
            planner_llm = ChatOllama(
                model=spec.model, temperature=0.7,
                # cloud-adapt: think=False mechanism verified in Cut ①f
            )

            class _NullRetriever:
                def search(self, q, top_k=5): return []
                def add_chunks(self, _): pass
            retriever = _NullRetriever()

            graph = build_graph(retriever=retriever, llm=planner_llm,
                                checkpointer=InMemorySaver())
            config_extras = {
                "planner": build_planner(
                    llm=planner_llm, plan_repo=PlanRepository(session),
                    goal_repo=GoalRepository(session),
                    mastery_repo=MasteryRepository(session),
                    mistake_repo=MistakeRepository(session),
                    retriever=retriever,
                ),
                "planner_agent": build_planner_agent(
                    llm=planner_llm, plan_repo=PlanRepository(session),
                    goal_repo=GoalRepository(session),
                    mastery_repo=MasteryRepository(session),
                    mistake_repo=MistakeRepository(session),
                    retriever=retriever, max_iter=10,
                ),
                "memory_hydrator": build_memory_hydrator(
                    mastery_repo=MasteryRepository(session),
                    mistake_repo=MistakeRepository(session),
                ),
                "memory_writer": build_memory_writer(
                    mastery_repo=MasteryRepository(session),
                    mistake_repo=MistakeRepository(session),
                ),
                "judge_llm": None,  # eval judges run OUT of graph
                "quiz_master": None,
            }
            try:
                record = await run_one(
                    spec=spec, graph=graph,
                    judge_local=judge_local, judge_cloud=judge_cloud,
                    config_extras=config_extras, user_id=user.id,
                )
            except Exception as exc:
                record = {
                    "run_id": spec.run_id, "timestamp": "",
                    "model": spec.model, "mode": spec.mode,
                    "query_id": spec.query_id, "turn_idx": spec.turn_idx,
                    "run_idx": spec.run_idx,
                    "operational": {
                        "wall_time_s": 0.0, "iterations": 0,
                        "tool_calls": [], "tool_call_count": 0,
                        "tool_errors": 0, "input_tokens": 0, "output_tokens": 0,
                        "exit_reason": "harness_error",
                    },
                    "output": {"plan_action": None, "milestones_persisted": 0,
                               "milestones_json": [], "final_text_excerpt": str(exc)[:500]},
                    "judge_local": {}, "judge_cloud": {},
                }

        with results_path.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[{spec.run_id}] model={spec.model} mode={spec.mode} "
              f"exit={record['operational']['exit_reason']}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--queries", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--models", default="gemma3:4b,qwen3.5:4b,qwen2.5:7b,gemma4:e4b")
    p.add_argument("--modes", default="deterministic,agent_loop")
    p.add_argument("--thinking-appendix", action="store_true")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: Run the 3 harness tests to verify they pass**

Run: `cd backend && uv run pytest -q tests/eval/test_p2_2_harness.py`

Expected output: `3 passed`. If `test_matrix_expansion_main_run_count_matches_spec` returns a count != 336, recount per the doc above; if the difference is exactly 48 (multi-turn × turns × models × modes × runs minus your formula), check whether `expand_matrix` is double-counting modes inside the multi-turn nested loop.

- [ ] **Step 9: Checkpoint — full backend suite**

Run: `cd backend && uv run pytest -q`

Expected output: `181 passed` (178 + 3). This is the **24-test target** spec §7 declared.

---

## Cut P2.2-②b — Run the full eval matrix (manual, no automated tests)

**Files:**
- Produce: `backend/app/eval/p2_2_agent_ablation/output/results.jsonl` (gitignored — operator artifact)
- Produce: `backend/app/eval/p2_2_agent_ablation/output/summary.md` (committed — derived tables from results.jsonl)

This cut runs the harness end-to-end on a real Ollama instance. Wall time estimate from spec §6.4: ~1–2 hours for ~360 runs. In practice, expect 2–3 hours for ~396 runs with multi-turn overhead. Run on a machine where Ollama is healthy and not GPU-shared.

- [ ] **Step 1: Pre-flight checks**

```bash
# Models pulled?
ollama list | grep -E "gemma3:4b|qwen3.5:4b|qwen2.5:7b|gemma4:e4b"

# Backend healthy?
cd backend && uv run pytest -q   # must show 181 passed

# Disk space (results.jsonl can grow to 2-5 MB per 100 runs)
df -h .
```

If `OPENAI_API_KEY` is not set, cloud judge will silently skip. Decide before starting whether you want both judges or local-only.

```bash
export OPENAI_API_KEY=sk-...   # optional but recommended
```

- [ ] **Step 2: Kick off the main matrix**

```bash
cd backend
mkdir -p app/eval/p2_2_agent_ablation/output

uv run python -m app.eval.p2_2_agent_ablation.run_eval \
  --queries app/eval/p2_2_agent_ablation/queries.json \
  --output app/eval/p2_2_agent_ablation/output/results.jsonl \
  --runs 3 \
  --models gemma3:4b,qwen3.5:4b,qwen2.5:7b,gemma4:e4b \
  --modes deterministic,agent_loop \
  2>&1 | tee app/eval/p2_2_agent_ablation/output/run.log
```

The harness writes every record as soon as it completes (line-buffered), so killing the process mid-run leaves a partial `results.jsonl`; re-running with the same `--output` resumes.

If you observe a run row with `exit_reason == "harness_error"` more than 3 times in a row, stop the run and inspect `output/run.log` — there's a systemic issue (Ollama OOM, model not pulled, etc.). Per spec §6.4 we don't retry individual rows, but a cascade of harness errors invalidates the run.

- [ ] **Step 3: Kick off the thinking-on/off appendix (gemma4:e4b only)**

```bash
uv run python -m app.eval.p2_2_agent_ablation.run_eval \
  --queries app/eval/p2_2_agent_ablation/queries.json \
  --output app/eval/p2_2_agent_ablation/output/results.jsonl \
  --runs 3 \
  --models gemma4:e4b \
  --modes deterministic,agent_loop \
  --thinking-appendix \
  2>&1 | tee -a app/eval/p2_2_agent_ablation/output/run.log
```

Resumability ensures already-completed rows are skipped; the appendix only adds the `thinking=True` cells.

- [ ] **Step 4: Generate the summary tables**

Create a one-off analysis script at `backend/app/eval/p2_2_agent_ablation/summarize.py`:

```python
"""Read results.jsonl and emit a markdown table for each metric dimension.

Sections: latency / exit_reason distribution / tool-call breakdown /
plan quality (dual judge) / judge agreement.

Output goes to stdout; redirect to summary.md.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def by_cell(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    out: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        out[(r["model"], r["mode"])].append(r)
    return out


def fmt_table(headers: list[str], rows: list[list]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows)
    return f"{head}\n{sep}\n{body}"


def main(path: str):
    rows = load(Path(path))
    cells = by_cell(rows)

    print("# P2.2 Ablation Results Summary\n")
    print(f"Total rows: {len(rows)}\n")

    print("## Latency (median wall_time_s)\n")
    table = []
    for (m, mode), cell_rows in sorted(cells.items()):
        times = [r["operational"]["wall_time_s"] for r in cell_rows
                 if r["operational"]["wall_time_s"] > 0]
        med = statistics.median(times) if times else 0.0
        table.append([m, mode, f"{med:.2f}", len(cell_rows)])
    print(fmt_table(["model", "mode", "median wall_time_s", "n"], table))

    print("\n## Exit reason distribution\n")
    table = []
    for (m, mode), cell_rows in sorted(cells.items()):
        counts: dict[str, int] = defaultdict(int)
        for r in cell_rows:
            counts[r["operational"]["exit_reason"]] += 1
        pieces = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        table.append([m, mode, pieces])
    print(fmt_table(["model", "mode", "exit_reasons"], table))

    print("\n## Tool-call rate (mean per run, agent_loop mode only)\n")
    table = []
    for (m, mode), cell_rows in sorted(cells.items()):
        if mode != "agent_loop":
            continue
        counts = [r["operational"]["tool_call_count"] for r in cell_rows]
        mean = sum(counts) / len(counts) if counts else 0.0
        errors = sum(r["operational"]["tool_errors"] for r in cell_rows)
        table.append([m, f"{mean:.2f}", errors])
    print(fmt_table(["model (agent_loop)", "mean tool_calls/run", "tool_errors total"], table))

    print("\n## Plan quality — local judge mean score\n")
    table = []
    for (m, mode), cell_rows in sorted(cells.items()):
        scores = [r["judge_local"].get("score") for r in cell_rows
                  if r.get("judge_local") and r["judge_local"].get("score") is not None]
        mean = sum(scores) / len(scores) if scores else 0.0
        table.append([m, mode, f"{mean:.3f}", len(scores)])
    print(fmt_table(["model", "mode", "mean local score", "n"], table))

    print("\n## Judge agreement (mean |local - cloud|)\n")
    table = []
    for (m, mode), cell_rows in sorted(cells.items()):
        deltas = []
        for r in cell_rows:
            jl = r.get("judge_local", {}).get("score")
            jc = r.get("judge_cloud", {}).get("score")
            if jl is not None and jc is not None:
                deltas.append(abs(jl - jc))
        mean = sum(deltas) / len(deltas) if deltas else 0.0
        table.append([m, mode, f"{mean:.3f}", len(deltas)])
    print(fmt_table(["model", "mode", "mean |Δ|", "n"], table))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "app/eval/p2_2_agent_ablation/output/results.jsonl")
```

Run it:

```bash
cd backend
uv run python -m app.eval.p2_2_agent_ablation.summarize \
  app/eval/p2_2_agent_ablation/output/results.jsonl \
  > app/eval/p2_2_agent_ablation/output/summary.md

cat app/eval/p2_2_agent_ablation/output/summary.md
```

- [ ] **Step 5: Sanity-check the data before writing EVAL.md**

- Every (model, mode) cell has ~30–36 rows (10 single-turn × 3 runs = 30 single-turn + multi-turn turns ~6 = 36).
- gemma3:4b in agent_loop should show many `natural_stop` rows with `tool_call_count == 0` — that's the "no tool capability" finding.
- qwen2.5:7b in agent_loop should have the highest mean `tool_calls/run`.
- Latency on agent_loop should be 1.5–3× deterministic across most cells (multi-turn LLM calls).
- judge_local mean score on deterministic ≈ 0.6–0.8 baseline (P2.1-⑤ real-run validated 0.7+); agent_loop scores depend on whether the model emitted a plan or degraded.

If any of these "obvious" patterns are violated, suspect a harness bug rather than a finding — re-read `single_run.run_one` and check whether `agent_trace` is being read for the deterministic path (it shouldn't be — deterministic returns no `agent_trace`, and `run_one`'s fallback shape covers that).

- [ ] **Step 6: Mark this cut complete only when**

- `output/results.jsonl` has ≥ 280 valid rows (some harness_error rows tolerable; > 10% harness_error → re-run after root cause).
- `output/summary.md` was regenerated from the final data.
- `cd backend && uv run pytest -q` still 181 passed (no regressions from any debugging code that snuck in).

---

## Cut P2.2-③ — Writeup (EVAL.md + blog + ROADMAP + memory)

**Files:**
- Create: `study-coach/docs/EVAL.md`
- Create: `study-coach/docs/agent_loop_vs_deterministic.md`
- Modify: `study-coach/docs/ROADMAP.md`
- Modify: `/Users/lianghaozhe/.claude/projects/-Users-lianghaozhe-Downloads-Study-Compaion-and-JadeAI/memory/project_study_coach_refactor.md`

No code changes; this is the portfolio crystallization step.

- [ ] **Step 1: Write `docs/EVAL.md`**

Create `study-coach/docs/EVAL.md` with this exact section structure (drop in the numbers from Cut ②b's `summary.md`):

```markdown
# Agent Loop Ablation — Empirical Report

> P2.2 deliverable. Tests the question: on locally-served small Ollama models,
> does an LLM tool-calling agent loop produce better study plans than a
> hand-written deterministic node? Or does it just spend more tokens?

## TL;DR

(2-3 bullets summarizing winner by dimension. Fill from summary.md.)

## Setup

- 4 models: gemma3:4b (4B, no tools/thinking), qwen3.5:4b (4.7B, tools+thinking),
  qwen2.5:7b (7B, tools), gemma4:e4b (8B, tools+thinking).
- 2 modes: deterministic (P2.1-⑤ baseline) vs agent_loop (P2.2 hand-written while-loop).
- 12 queries × 3 runs = 36 trials per (model, mode) cell (+ multi-turn).
- Dual judges: qwen2.5:7b local + gpt-4o-mini cloud, both using the
  PLAN_DIMENSIONS rubric (milestone_specificity / milestone_granularity /
  time_feasibility / topic_coverage / actionability).
- Appendix: gemma4:e4b thinking on vs off on the same matrix.
- Wall time: ~2.5 hours total.

## Results

### Latency

(Paste table from summary.md.)

### Tool-calling correctness

(Paste table.)

### Robustness — exit_reason distribution

(Paste table.)

### Plan quality (dual judge means)

(Paste table.)

### Judge agreement

(Paste table. Add ≥ 1 sentence on the self-preference bias if qwen-judge-rating-qwen-cells differ from cross-model cells.)

## Findings

(3-5 numbered conclusions. Each = 1-2 sentences. Examples:)

1. **Agent loops cost more, win on specific queries** — `qwen2.5:7b` agent_loop scored X% higher than its deterministic baseline on plan_mindmap queries but Y% lower on plan_hyde.
2. **Tool capability matters more than parameter count** — gemma3:4b (no tools) failed every agent_loop run; qwen2.5:7b (7B + tools) outperformed gemma4:e4b (8B + tools+thinking) by Z%, suggesting raw parameter count is not the bottleneck.
3. **Thinking is a tax, not a feature, at this scale** — gemma4:e4b thinking-ON spent A% more tokens for a ~0% quality delta. (Appendix.)
4. **Tool-error self-correction works on the new-gen tier** — qwen3.5:4b and gemma4:e4b recovered from B% of tool errors within the budget; older tier instead consumed the budget without recovery.
5. **Dual judges disagree on UX subjectivity** — cells where the markdown formatting differed showed |local - cloud| spread up to C, while milestone-only plans agreed within D.

## Limitations

- N = 36 per cell. Statistical power for small effects is limited.
- Local judge model overlaps with one of the planner models (qwen2.5:7b judging qwen2.5:7b plans) — self-preference bias visible in those cells.
- All queries are HKBU-domain; transferability to other corpora untested.
- Ollama tool-calling on gemma3:4b/4-tier is an industry-state issue; results
  reflect today's mid-2026 stack, not theoretical ceilings.

## Smoke verification log (from Cut ①f)

| model × mode | tool_calls? | natural_stop? | persisted? | notes |
| ... | ... | ... | ... | ... |

(Paste the table filled during Cut ①f Step 3.)
```

Fill the section bodies from the actual `summary.md`. The TL;DR is the only section to write last (after all sections have data).

- [ ] **Step 2: Write the portfolio blog post**

Create `study-coach/docs/agent_loop_vs_deterministic.md`. Outline:

```markdown
# Did I build a real agent? An empirical answer.

[Hook: link to learn-claude-code's central thesis from LEARN-CLAUDE-CODE-README-zh.md —
"Agency comes from the model + harness lets agency land". Quote directly.]

## The thesis I was responding to

[2-3 paragraphs summarizing learn-claude-code's argument. Be honest about where
it convinced me and where I was suspicious. Tone: respectful, not strawman.]

## What I built

[1-2 paragraphs introducing Study Coach's Planner: deterministic baseline
shipped in P2.1-⑤, agent-loop variant shipped in P2.2. The point: SAME tools,
SAME LangGraph topology, SAME judge — only the planner node body differs.
Fair A/B.]

## What I measured

[Brief — refer reader to EVAL.md for full tables. Cite the matrix size:
4 models × 2 modes × 12 queries × 3 runs.]

## The findings that hold

[2-3 findings from EVAL.md that survived sanity-checking, written for a reader
who hasn't read the EVAL.md tables. Tone: data-first, theory-second.]

## Where the thesis lands

[1-2 paragraphs: which parts of learn-claude-code's thesis I now believe more
strongly, which parts I think need an asterisk, which were untested by my data.

Specifically address: does "minimal harness" still hold when the model can't
emit valid tool calls? My answer: the harness has to absorb that failure
gracefully — and that absorption layer IS engineering, not just letting the
model do its thing.]

## What I would do next

[1 short paragraph: P2.3 = quiz path same ablation; future direction = mixed
mode where the dispatcher chooses based on per-query model confidence.
Implicit: I'm not done.]

## Reproduce

```
git clone study-coach
cd backend
uv run pytest -q          # 181 tests pass
uv run python -m app.eval.p2_2_agent_ablation.run_eval --output /tmp/r.jsonl
```

[End of post.]
```

Cap at ~1200 words. Read it aloud once — if any sentence sounds like marketing copy, cut it.

- [ ] **Step 3: Update `docs/ROADMAP.md`**

Find the P2.2 block in `study-coach/docs/ROADMAP.md` (currently marked planned/in-progress). Update it to:

```markdown
## P2.2 — Agent Loop Ablation ✅

Shipped 2026-MM-DD. 181 backend tests passing. EVAL.md has the full data.

Outcome: agent_loop variant lives alongside deterministic; mode switch via
`x-planner-mode` HTTP header. Run the harness:

    cd backend && uv run python -m app.eval.p2_2_agent_ablation.run_eval \
        --queries app/eval/p2_2_agent_ablation/queries.json \
        --output /tmp/r.jsonl

Key references:
- Spec: docs/superpowers/specs/2026-05-22-p2-2-agent-loop-ablation-design.md
- Plan: docs/superpowers/plans/2026-05-23-p2-2-agent-loop-ablation.md
- Report: docs/EVAL.md
- Blog: docs/agent_loop_vs_deterministic.md
```

Date stamp on actual ship date.

- [ ] **Step 4: Update memory**

Append a new P2.2 segment to `/Users/lianghaozhe/.claude/projects/-Users-lianghaozhe-Downloads-Study-Compaion-and-JadeAI/memory/project_study_coach_refactor.md`. Replace the `⬜ Phase 2.2 (planned, not yet started ...)` bullet with `✅ Phase 2.2 (shipped 2026-MM-DD): Agent Loop Ablation — ...` and include:

- Total backend tests: 181 (was 157).
- Files added (planner_agent.py, eval/p2_2_agent_ablation/*).
- Files modified (state.py, graph.py, deps.py, routes.py).
- Cut-by-cut headline numbers (test count delta per cut).
- 3-5 lessons learned from the actual cuts (not the plan), explicitly:
  - what surprised you about Ollama tool calling on the 4 models
  - whether the `_extract_topic` regression test caught anything in real runs
  - any reshape forced by an unexpected LangChain `tool_calls` shape
- Cloud-adapt hooks count: `grep -rn "# cloud-adapt:" backend/app | wc -l` should be ≥ 5 (spec §11 listed 6).

- [ ] **Step 5: Final verification — full suite + lint**

```bash
cd backend && uv run pytest -q
```

Expected: `181 passed`. Then:

```bash
grep -rn "# cloud-adapt:" backend/app/agent/planner_agent.py
grep -rn "# cloud-adapt:" backend/app/api/deps.py
grep -rn "# cloud-adapt:" backend/app/agent/graph.py
```

Expected: at least one anchor in each of those files. If a file has none, you skipped a marker; re-read the spec §11 list and add the missing ones.

- [ ] **Step 6: Invoke `superpowers:verification-before-completion`**

This is the final discipline gate per CLAUDE.md. The verification skill will require fresh evidence (re-run pytest, re-grep, etc.) before allowing a "P2.2 complete" claim. Do not skip.

---

## Self-Review (run after final cut)

This section is for the implementer to use after Cut ③ but before reporting P2.2 done.

### Spec coverage check

| Spec §            | Covered by cut(s)            | Notes |
|---|---|---|
| §1.Q1a 4-model matrix | ②b CLI default `--models` | gemma3:4b appears in matrix per spec; agent_loop expected to "fail informatively" |
| §1.Q1b thinking-OFF main | ②b non-appendix run | mechanism nailed in ①f |
| §1.Q1b thinking-ON appendix | ②b `--thinking-appendix` | gemma4:e4b only |
| §1.Q2 5 tools, no done() | ①a tool factory | no done/todo tools defined |
| §1.Q2 max_iter=10 | ①c factory default | cloud-adapt comment notes raise-to-30 path |
| §1.Q3 dual judge | ②a judges.py | local=qwen2.5:7b + cloud=gpt-4o-mini |
| §1.Q4 header flag | ①e deps + routes | default deterministic, unknown→deterministic |
| §1.Q5 phase-differentiated detail | this plan structure | ① cuts are TDD-deep, ② is harness-only, ③ is outline |
| §2 graph topology unchanged | ①d plan_node dispatch | judge/memory_hydrator/memory_writer 100% reused |
| §3 closure factory | ①a `_make_planner_tools` | 5 tools, user_id never in args |
| §4 loop control flow | ①c `planner_agent_node` + `_safe_invoke_tool` | 4 exit conditions handled |
| §4.4 plan_action inference | ①c `_infer_plan_action` + 3 inference tests | |
| §4.5 SSE-compatible final | ①c `_format_final_output` | citations + token + done same as deterministic |
| §4.6 AgentTrace shape | ①b dataclass + 3 tests | serialize() returns spec-§4.6 dict |
| §5 state/deps/routes | ①d + ①e | `agent_trace` NotRequired field added |
| §6 eval harness | ②a + ②b | matrix.py / single_run.py / judges.py / run_eval.py |
| §7 testing strategy | ①a-①e + ②a | 24 new tests landed |
| §11 cloud-adapt hooks | comments embedded inline | ≥ 5 grep anchors visible |

### Placeholder scan

- Search this plan for "TBD" / "TODO" / "fill in details" / "Similar to Task N" / "..." with no surrounding code. (Should find none in implementation steps; the EVAL.md template has intentional fill-from-data slots that aren't placeholders.)
- Search for ungrounded type references — every type used (`RunSpec`, `AgentTrace`, `IterationRecord`, `ToolCallRecord`, `Milestone`, `PlanRepository`, etc.) must trace to a `Create` step or an existing file referenced above.

### Type consistency

- `_make_planner_tools` signature: `user_id: str, llm, retriever, plan_repo: PlanRepository | None, goal_repo: GoalRepository | None, mastery_scores: dict[str, float], recent_mistakes: list[str], now_fn` — used same in Cut ①c factory, deps.py factory, and eval harness.
- `AgentTrace.serialize()` keys: `total_iterations / total_tool_calls / tool_call_breakdown / tool_errors / input_tokens / output_tokens / wall_time_s / exit_reason / llm_error` — Cut ①b tests pin these, Cut ②a record schema reads these.
- `planner_mode` value space: `"deterministic" | "agent_loop"` — pinned in `get_planner_mode` (deps), checked in graph dispatch, used in matrix/spec.

### Cuts in correct order

①a (tools) → ①b (trace) → ①c (loop, needs both) → ①d (graph, needs ①c factory) → ①e (routes, needs ①d) → ①f (smoke, needs ①e end-to-end) → ②a (harness, needs ①c+①d for invoke) → ②b (matrix run) → ③ (writeup, needs ②b data).

Implementer must NOT reorder. Specifically: do not start ①c before ①b (trace dataclass) — the loop tests assert on `update["agent_trace"]["exit_reason"]` which requires AgentTrace to exist.

### Risk pre-mitigations baked into specific cuts

| Risk | Pre-mitigation | Cut |
|---|---|---|
| gemma3:4b can't tool-call | Loop natural-stops at iter=0; no special-case branch added | ①f Step 4 |
| `think=False` mechanism unstable across models | Smoke verifies before matrix runs | ①f Step 1 |
| AgentTrace bloats production response | `# cloud-adapt:` redact marker at `record_tool_call` | ①b |
| `_extract_topic` regression (P2.1-⑤i char-set bug) | Regression test in `test_planner_agent_loop.py` | ①c |
| LangChain `tool_calls` shape varies by langchain-ollama version | Defensive `getattr(response, "tool_calls", None) or []` in loop body | ①c |
| Long results.jsonl unreadable | `summarize.py` produces summary.md | ②b |

---

## Execution Handoff

**Plan complete and saved to** `study-coach/docs/superpowers/plans/2026-05-23-p2-2-agent-loop-ablation.md`.

Two execution options:

1. **Subagent-Driven (recommended)** — Dispatch a fresh subagent per cut, review between cuts, fast iteration. Best for this plan because each cut has a tight TDD scope and a clean "1 test file + 1 implementation diff" boundary; subagent context stays small.

2. **Inline Execution** — Execute cuts sequentially in this session via `superpowers:executing-plans`; batch execution with the user reviewing at each `Checkpoint` step.

Tell me which one to invoke next.


