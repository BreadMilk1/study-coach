# P2.3 Quiz Agent Loop Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship an LLM tool-calling agent-loop variant of Quiz GENERATE that lives in parallel with the deterministic `quiz_master.py` from P2.1-④, then run a head-to-head ablation matrix (4 models × 2 modes × 12 queries × 3 runs + gemma4:e4b thinking-ON appendix ≈ 396 records) and append `EVAL.md` + cap the blog with a "Quiz task: does persistence rescue replicate?" section.

**Architecture:** A closure factory `_make_quiz_tools(...)` builds 2 LangChain `@tool` wrappers (`retriever_search` + `persist_quiz_question`) — minimal surface so the LLM-as-creator/tool-as-persistence pattern stays clean. A hand-written async `while`-loop (`max_iter=6`, exit on empty `response.tool_calls`, tool errors fed back as `ToolMessage` for self-correction via Pydantic `QuizQuestionPersist` schema, LLM errors → degrade) drives the model. The loop is embedded in the existing LangGraph `quiz_node` via a **state-aware-then-mode-aware** dispatcher: GRADE turns (`active_quiz_question_id` truthy) always route to deterministic `quiz_master` regardless of mode; GENERATE turns route based on `config.configurable.quiz_mode`. New HTTP header `x-quiz-mode: deterministic|agent_loop` selects mode per request, defaulting to `deterministic`. `AgentTrace` (P2.2) refactored into shared `app/agent/agent_trace.py` so Plan and Quiz agents share instrumentation.

**Tech Stack:** Python 3.11, LangChain 0.3 + langchain-ollama 1.1 (`bind_tools` + `@tool` decorator), LangGraph 0.2, FastAPI, SQLAlchemy, Pydantic 2 (Literal + field_validator + min_length), pytest-asyncio (`asyncio_mode = "auto"`).

**Spec:** `study-coach/docs/superpowers/specs/2026-05-24-p2-3-quiz-agent-loop-ablation-design.md`

**Discipline reminders for the implementer:**
- This repo is **not** a git repo — never run `git init / commit / push`. Use the project's existing `# checkpoint` convention (verify full test suite green) at the end of every cut.
- Every `# cloud-adapt:` comment in new files must be marker-only — do NOT implement cloud branches. See spec §11 for the canonical grep anchor list.
- TDD: every cut writes the failing test(s) first, runs them to confirm RED, then writes the minimal implementation, then runs the full suite. No exceptions.
- Project working directory is `study-coach/backend/` for all `uv run pytest` commands. Use `cd backend && uv run pytest -q ...`.
- Baseline before P2.3 starts: **181 backend tests passing** (P2.2 ship-state). Target after Cut ②a: **201 tests passing** (+ 20 net new — 1 in ①a, 4 in ①b, 6 in ①c, 3 in ①d, 3 in ①e, 3 in ②a).
- Do not modify any non-P2.3 file unless the cut says so. Critical byte-identical files: `quiz_master.py`, `planner.py`, `tools/quiz.py`, `tools/plan.py`, `progress.py`, `judge.py`, `router.py`, `memory_updater.py`. If you find an unrelated bug, log it as a follow-up and stop.
- The **deterministic `quiz_master.py` MUST stay byte-identical** through every cut — that is the fair-A/B guarantee for the ablation.
- The Cut ①a refactor moves `AgentTrace`/`IterationRecord`/`ToolCallRecord` out of `planner_agent.py`. P2.2's existing 3 `test_agent_trace.py` tests MUST pass byte-identical after the move (just with an updated import path) — this is the gate that proves the refactor is non-breaking.

---

## File Structure

### Files to create

| Path | Responsibility |
|---|---|
| `backend/app/agent/agent_trace.py` | Cut ①a — Shared `AgentTrace` + `IterationRecord` + `ToolCallRecord` dataclasses moved from `planner_agent.py`. Adds `last_persisted_question_id()` method for Quiz agent. ~170 lines. |
| `backend/app/agent/quiz_master_agent.py` | Cuts ①b/①c — Closure-factory tool wrappers + while-loop body + `build_quiz_master_agent` factory. Mirror of `planner_agent.py` shape but smaller (~330 lines, 2 tools vs 5). |
| `backend/tests/agent/test_quiz_master_agent_tools.py` | Cut ①b — 4 unit tests for the 2 tool wrappers (closure injection, JSON return shape, valid-args persistence, invalid-args error JSON). |
| `backend/tests/agent/test_quiz_master_agent_loop.py` | Cut ①c — 6 unit tests for the loop body: natural_stop / budget_exhausted / llm_error / tool_error_self_correction / valid-persist round-trip / invalid-schema retry. |
| `backend/tests/agent/test_graph_quiz_agent_e2e.py` | Cut ①d — 3 graph-level e2e tests with stub LLM: GENERATE mode switching, GRADE always deterministic (state-aware), agent_trace lands in state. |
| `backend/tests/api/test_routes_quiz_agent.py` | Cut ①e — 3 route-level integration tests: `x-quiz-mode` header routing + SSE contract identical + multi-turn dispatcher coherence (GENERATE agent_loop → GRADE deterministic). |
| `backend/app/eval/p2_3_quiz_ablation/__init__.py` | Cut ②a — package marker. |
| `backend/app/eval/p2_3_quiz_ablation/matrix.py` | Cut ②a — `RunSpec` + `expand_matrix` for Quiz (mode field name → `quiz_mode`). |
| `backend/app/eval/p2_3_quiz_ablation/single_run.py` | Cut ②a — one experimental run: build graph, invoke, extract Quiz-state fields (`active_quiz_question_id` / `quiz_action`), compute auto-metrics. |
| `backend/app/eval/p2_3_quiz_ablation/judges.py` | Cut ②a — dual judge wired to `QUIZ_DIMENSIONS` rubric + `judge_quiz.txt` prompt (local qwen2.5:7b + cloud MiniMax-M2.7). |
| `backend/app/eval/p2_3_quiz_ablation/run_eval.py` | Cut ②a — top-level CLI: `--queries`, `--output`, `--runs`, `--thinking-appendix`, resumable, `reasoning=spec.thinking` patch ported from P2.2. |
| `backend/app/eval/p2_3_quiz_ablation/queries.json` | Cut ②a — 10 single-turn GENERATE + 2 multi-turn GENERATE→"A"→GRADE. |
| `backend/app/eval/p2_3_quiz_ablation/summarize.py` | Cut ②a — markdown table generator (fork of P2.2 `summarize.py`, column labels swapped). |
| `backend/tests/eval/test_p2_3_harness.py` | Cut ②a — 3 unit tests: matrix expansion, record schema validation, resumability. |

### Files to modify

| Path | Change | Cut |
|---|---|---|
| `backend/app/agent/planner_agent.py` | DELETE inline `IterationRecord`/`ToolCallRecord`/`AgentTrace` (lines ~61-161 of current file). REPLACE with `from app.agent.agent_trace import AgentTrace, IterationRecord, ToolCallRecord` near top. Remove unused `time`/`Counter`/`dataclass`/`field` imports if they become unused. **No logic change.** | ①a |
| `backend/tests/agent/test_agent_trace.py` | Update import from `from app.agent.planner_agent import AgentTrace` to `from app.agent.agent_trace import AgentTrace`. | ①a |
| `backend/app/agent/tools/schemas.py` | Append `QuizQuestionPersist` Pydantic model with `Literal` answer, `min_length=4 max_length=4` options, prefix validator. | ①b |
| `backend/app/agent/graph.py` | Replace `quiz_node` body with state-aware (GRADE always deterministic) + mode-aware dispatcher (~15 lines net). | ①d |
| `backend/app/api/deps.py` | Append `get_quiz_mode` factory + `get_quiz_master_agent` factory at end of file. | ①e |
| `backend/app/api/routes.py` | Extend `chat()` signature with `quiz_master_agent` + `quiz_mode` Depends, inject into `config.configurable`. | ①e |
| `study-coach/docs/ROADMAP.md` | Update P2.3 from "candidate" to "shipped" with results summary. | ③ |
| `study-coach/docs/EVAL.md` | Append P2.3 Quiz Ablation section (mirror P2.2 8-section structure). | ③ |
| `study-coach/docs/agent_loop_vs_deterministic.md` | Append "Update from P2.3" section answering the 3 predictions or note "see new sister blog `docs/quiz_ablation_followup.md`". Implementer picks based on length. | ③ |
| `/Users/lianghaozhe/.claude/projects/-Users-lianghaozhe-Downloads-Study-Compaion-and-JadeAI/memory/project_study_coach_refactor.md` | Append P2.3 progress segment + lessons learned. | ③ |

### Files explicitly NOT touched

- `quiz_master.py` — deterministic baseline must stay byte-identical for ablation fairness.
- `planner.py`, `planner_agent.py` logic (only its imports change in ①a) — P2.2 ablation fairness preserved.
- `tools/quiz.py`, `tools/plan.py`, `progress.py`, `judge.py`, `router.py`, `memory_updater.py` — reused via import, never modified.
- `state.py` — no schema change needed (`agent_trace`, `active_quiz_question_id`, `quiz_action` all exist from P2.1-④ + P2.2).
- `prompts/judge_quiz.txt` — reused as-is.

---

## Cut P2.3-①a — AgentTrace refactor extraction

**Files:**
- Create: `backend/app/agent/agent_trace.py`
- Modify: `backend/app/agent/planner_agent.py` (remove inline classes, add import)
- Modify: `backend/tests/agent/test_agent_trace.py` (update import path; add 1 new test for `last_persisted_question_id`)

**Boundary check:** This is a pure move + 1 method addition. P2.2's 3 existing `test_agent_trace.py` tests must pass byte-identical (only their import line changes). If a single P2.2 test fails after this cut, the refactor is broken — STOP.

- [ ] **Step 1: Create `backend/app/agent/agent_trace.py` with the moved code**

Open the existing `backend/app/agent/planner_agent.py` and identify the block from `@dataclass class IterationRecord:` through end of `class AgentTrace:` (approximately lines 61-161). Copy that entire block (including the `last_persisted_plan_id` and `aggregated_retriever_context` methods).

Create `backend/app/agent/agent_trace.py` with:

```python
"""Shared agent-loop instrumentation dataclasses.

Extracted from `planner_agent.py` in P2.3 Cut ①a so the Quiz agent (and any
future agent) can use the same trace shape without cross-module reach.

Per-run instrumentation is the only structured record the eval harness pulls,
so the `serialize()` output shape is contractual. Anything tightly coupled to
a specific tool name (`last_persisted_plan_id`, `get_existing_plan_returned_nonnull`,
`last_persisted_question_id`) lives on this class as helpers — they're parallel
small methods, not abstractions. Refactor to a single `last_persisted_id(tool_name,
key)` when a 3rd consumer arrives (YAGNI).
"""
from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field


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
    because matrices have hundreds of runs and the file should stay grep-able.
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

    def last_persisted_question_id(self) -> str | None:
        """Return the question_id from the most recent successful
        persist_quiz_question tool call. Returns None if no successful
        persistence happened in this trace."""
        for tc in reversed(self.tool_calls):
            if tc.name == "persist_quiz_question" and not tc.error:
                try:
                    return json.loads(tc.output).get("question_id")
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

- [ ] **Step 2: Update `backend/app/agent/planner_agent.py` — remove inline classes, add import**

Edit `planner_agent.py`:

1. Near the top of the file (after the existing `from __future__ import annotations` and below it), ensure these imports are present:

```python
from app.agent.agent_trace import AgentTrace, IterationRecord, ToolCallRecord
```

2. Remove the now-redundant standalone imports at top of file IF they are only used by the moved classes:
   - `import time` — KEEP if `time.monotonic()` is still used elsewhere in `planner_agent.py` (it IS — line `trace = AgentTrace(t_start=time.monotonic())`)
   - `from collections import Counter` — REMOVE (only used inside `AgentTrace.serialize`)
   - `from dataclasses import dataclass, field` — REMOVE (only used by the moved dataclasses)
   - `import json` — KEEP (still used elsewhere in planner_agent.py for JSON parsing in tool wrappers / output formatting)

3. Delete the entire block of `@dataclass class IterationRecord:`, `@dataclass class ToolCallRecord:`, and `@dataclass class AgentTrace:` (with all methods including `last_persisted_plan_id`, `aggregated_retriever_context`, `serialize`).

4. The rest of `planner_agent.py` (the `_make_planner_tools`, `_AGENT_SYSTEM_PROMPT`, all helpers, `build_planner_agent`) stays byte-identical.

Verify no logic change by reading the diff line count: lines removed should equal lines deleted from the dataclass block; lines added = 1 import statement.

- [ ] **Step 3: Update `backend/tests/agent/test_agent_trace.py` import**

Edit `backend/tests/agent/test_agent_trace.py`:

Change line 13 (or wherever the `AgentTrace` import lives) from:

```python
from app.agent.planner_agent import AgentTrace
```

to:

```python
from app.agent.agent_trace import AgentTrace
```

DO NOT touch any other content in this test file — the 3 existing tests must pass byte-identical with only the import path swapped.

- [ ] **Step 4: Run the existing 3 AgentTrace tests to verify refactor is non-breaking**

Run: `cd backend && uv run pytest -q tests/agent/test_agent_trace.py`

Expected output: `3 passed`. If ANY test fails, the refactor broke something — STOP and reconcile. Diff the original `planner_agent.py` against the new `agent_trace.py` block to find missing lines.

- [ ] **Step 5: Add the failing test for `last_persisted_question_id`**

Append to `backend/tests/agent/test_agent_trace.py`:

```python
def test_last_persisted_question_id_returns_id_from_latest_successful_call():
    """Quiz agent inference helper: walk tool_calls in reverse, return question_id
    from the most recent successful persist_quiz_question call. Skip errored calls."""
    trace = AgentTrace(t_start=time.monotonic())
    # First successful persist
    trace.record_tool_call(
        "persist_quiz_question",
        {"topic": "HyDE", "prompt": "...", "options": ["A) x", "B) y", "C) z", "D) w"], "answer": "A", "explanation": "..."},
        '{"question_id": "q-first", "topic_id": "t-1", "persisted": true}',
        error=False,
    )
    # Subsequent errored persist (validation failure)
    trace.record_tool_call(
        "persist_quiz_question",
        {"topic": "BM25", "prompt": "...", "options": ["A) x"], "answer": "A", "explanation": "..."},
        '{"error": "invalid at options: List should have at least 4 items"}',
        error=True,
    )
    # Then another successful persist (LLM self-corrected)
    trace.record_tool_call(
        "persist_quiz_question",
        {"topic": "BM25", "prompt": "...", "options": ["A) x", "B) y", "C) z", "D) w"], "answer": "B", "explanation": "..."},
        '{"question_id": "q-corrected", "topic_id": "t-2", "persisted": true}',
        error=False,
    )

    assert trace.last_persisted_question_id() == "q-corrected"


def test_last_persisted_question_id_returns_none_when_no_successful_persist():
    trace = AgentTrace(t_start=time.monotonic())
    trace.record_tool_call(
        "persist_quiz_question", {}, '{"error": "..."}', error=True,
    )
    # Only errored persists + a retriever_search → returns None
    trace.record_tool_call("retriever_search", {"query": "x"}, "[]", error=False)
    assert trace.last_persisted_question_id() is None
```

- [ ] **Step 6: Run the new tests to verify they fail (RED)**

Run: `cd backend && uv run pytest -q tests/agent/test_agent_trace.py::test_last_persisted_question_id_returns_id_from_latest_successful_call tests/agent/test_agent_trace.py::test_last_persisted_question_id_returns_none_when_no_successful_persist`

Expected: 2 tests FAIL with `AttributeError: 'AgentTrace' object has no attribute 'last_persisted_question_id'`.

Wait — if you copied the full block including `last_persisted_question_id` per Step 1, this method exists already and the tests will pass immediately (which is fine — the "RED" was structural: tests would have failed before the move).

If tests pass on first run: good, you implemented the method during Step 1. The TDD intent is satisfied — the method was added in the same cut as the test.

- [ ] **Step 7: Checkpoint — run the full backend suite**

Run: `cd backend && uv run pytest -q`

Expected output: `182 passed` (181 baseline + 1 net new; the test file has 2 new tests but they exercise the same method so they count as a single conceptual addition... actually they're 2 separate test functions so count is `181 + 2 = 183`. Adjust the targets in the discipline reminder accordingly).

CORRECTION: target is `181 baseline + 2 new = 183 tests passing`. Any test failure outside the new tests is a regression — investigate before proceeding.

- [ ] **Step 8: Mark cut ①a complete**

Cut ①a is done. Conditions met:
- `app/agent/agent_trace.py` exists with all 3 dataclasses + new `last_persisted_question_id` method.
- `app/agent/planner_agent.py` imports from `agent_trace` and has no inline dataclass definitions.
- `tests/agent/test_agent_trace.py` import path updated; 5 tests passing (3 original + 2 new).
- Full suite: 183 tests passing.
- No git operations (this is not a git repo).

---

## Cut P2.3-①b — Tool wrappers + QuizQuestionPersist schema + closure factory

**Files:**
- Create: `backend/app/agent/quiz_master_agent.py` (partial — tools only at this stage)
- Modify: `backend/app/agent/tools/schemas.py` (append `QuizQuestionPersist` model)
- Test: `backend/tests/agent/test_quiz_master_agent_tools.py`

**Boundary check:** of the 2 tools, `retriever_search` is identical in shape to its planner-agent twin (closure-injected retriever, top_k arg). `persist_quiz_question` is the new business — it validates via Pydantic then persists via 3 repos (`goal_repo` for active goal lookup, `topic_repo` for upsert-on-name, `question_repo` for create). Schema validation rejection returns `{"error": "..."}` JSON the LLM can self-correct from.

- [ ] **Step 1: Append `QuizQuestionPersist` to `backend/app/agent/tools/schemas.py`**

Open `backend/app/agent/tools/schemas.py`. After the existing Pydantic models (around the end of the file), append:

```python
from typing import Literal

from pydantic import field_validator


class QuizQuestionPersist(BaseModel):
    """Schema for the persist_quiz_question agent tool.

    Strictly more constrained than Milestone (P2.2):
    - options: exactly 4 strings, each prefixed "A) "/"B) "/"C) "/"D) "
    - answer: Literal["A","B","C","D"]
    - prompt and explanation: non-empty strings
    - topic: non-empty string (consumer creates Goal/Topic rows if missing)
    """
    topic: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    options: list[str] = Field(min_length=4, max_length=4)
    answer: Literal["A", "B", "C", "D"]
    explanation: str = Field(min_length=1)

    @field_validator("options")
    @classmethod
    def _check_option_prefixes(cls, v: list[str]) -> list[str]:
        expected_prefixes = ["A) ", "B) ", "C) ", "D) "]
        for i, (opt, prefix) in enumerate(zip(v, expected_prefixes)):
            if not opt.startswith(prefix):
                raise ValueError(f"option[{i}] must start with {prefix!r}, got: {opt!r}")
        return v
```

Note: `BaseModel` and `Field` should already be imported at the top of `schemas.py` from existing P2.1-④/⑤ code. If `Literal` / `field_validator` are not imported, add them to the import block at top of file (do NOT scatter imports throughout the module).

- [ ] **Step 2: Create the failing test file with 4 tests**

Create `backend/tests/agent/test_quiz_master_agent_tools.py`:

```python
"""Cut P2.3-①b — unit tests for the 2 LLM-facing tool wrappers in quiz_master_agent.

Each test exercises:
  1. Closure injection — LLM-visible args do not include user_id / repos, yet
     the tool can use them via the factory closure.
  2. JSON return shape — every tool returns a JSON-serializable string.
  3. For persist_quiz_question: round-trip persistence on valid input, and
     {"error": ...} JSON on Pydantic validation failure (LLM self-correct path).

Business logic depth is NOT re-tested here — Pydantic validation is exercised
in app/tests/agent/test_quiz_schema.py (if such a test exists) or by Pydantic
itself. The contract under test is the WRAPPER, not the validator.
"""
import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.agent.quiz_master_agent import _make_quiz_tools
from app.db.models import Base
from app.db.repositories import (
    GoalRepository,
    QuestionRepository,
    TopicRepository,
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


def test_retriever_search_uses_closure_retriever_and_returns_json():
    retriever = StubRetriever(chunks=[
        {"chunk_id": "c1", "content": "HyDE = Hypothetical Document Embedding.", "page": 1},
        {"chunk_id": "c2", "content": "Embed the hypothetical doc, search corpus.", "page": 2},
    ])
    tools = _make_quiz_tools(
        user_id="u1", retriever=retriever,
        topic_repo=None, question_repo=None, goal_repo=None,
    )
    tool = next(t for t in tools if t.name == "retriever_search")

    # Schema: query/top_k in public args, retriever NOT
    assert "query" in tool.args
    assert "top_k" in tool.args
    assert "retriever" not in tool.args
    assert "user_id" not in tool.args

    out = tool.invoke({"query": "HyDE", "top_k": 1})
    parsed = json.loads(out)
    assert retriever.calls == [("HyDE", 1)]
    assert isinstance(parsed, list)
    assert parsed[0]["chunk_id"] == "c1"


def test_persist_quiz_question_creates_goal_topic_question_on_valid_input(session):
    user = UserRepository(session).get_or_create("fp-quiz-1")
    goal_repo = GoalRepository(session)
    topic_repo = TopicRepository(session)
    question_repo = QuestionRepository(session)

    tools = _make_quiz_tools(
        user_id=user.id, retriever=None,
        topic_repo=topic_repo, question_repo=question_repo, goal_repo=goal_repo,
    )
    tool = next(t for t in tools if t.name == "persist_quiz_question")

    # Schema: topic/prompt/options/answer/explanation in public args
    assert "topic" in tool.args
    assert "prompt" in tool.args
    assert "options" in tool.args
    assert "answer" in tool.args
    assert "explanation" in tool.args
    assert "user_id" not in tool.args

    out = tool.invoke({
        "topic": "HyDE",
        "prompt": "What does HyDE stand for?",
        "options": ["A) Hypothesis-Driven Experimentation",
                    "B) Hypothetical Document Embedding",
                    "C) High-Yield Data Encoding",
                    "D) Hybrid Document Engine"],
        "answer": "B",
        "explanation": "HyDE = Hypothetical Document Embedding (Gao et al. 2022).",
    })
    parsed = json.loads(out)
    assert parsed["persisted"] is True
    assert "question_id" in parsed
    assert "topic_id" in parsed

    # Round-trip via repos: question exists, topic exists, goal auto-created
    q = question_repo.get_by_id(parsed["question_id"])
    assert q is not None
    assert q.answer == "B"
    assert q.options_json == ["A) Hypothesis-Driven Experimentation",
                              "B) Hypothetical Document Embedding",
                              "C) High-Yield Data Encoding",
                              "D) Hybrid Document Engine"]
    assert goal_repo.list_active_for_user(user.id), "goal auto-created"


def test_persist_quiz_question_returns_error_json_on_wrong_options_count(session):
    user = UserRepository(session).get_or_create("fp-quiz-2")
    goal_repo = GoalRepository(session)
    topic_repo = TopicRepository(session)
    question_repo = QuestionRepository(session)

    tools = _make_quiz_tools(
        user_id=user.id, retriever=None,
        topic_repo=topic_repo, question_repo=question_repo, goal_repo=goal_repo,
    )
    tool = next(t for t in tools if t.name == "persist_quiz_question")

    out = tool.invoke({
        "topic": "BM25",
        "prompt": "What is BM25?",
        "options": ["A) An embedding model", "B) A ranking function"],  # only 2!
        "answer": "B",
        "explanation": "BM25 is a ranking function used by search engines.",
    })
    parsed = json.loads(out)
    assert "error" in parsed
    assert "options" in parsed["error"]  # Pydantic flags the field name
    # No persistence happened
    assert not question_repo.list_for_topic("nonexistent")  # sanity — confirm clean


def test_persist_quiz_question_returns_error_json_on_invalid_option_prefix(session):
    user = UserRepository(session).get_or_create("fp-quiz-3")
    goal_repo = GoalRepository(session)
    topic_repo = TopicRepository(session)
    question_repo = QuestionRepository(session)

    tools = _make_quiz_tools(
        user_id=user.id, retriever=None,
        topic_repo=topic_repo, question_repo=question_repo, goal_repo=goal_repo,
    )
    tool = next(t for t in tools if t.name == "persist_quiz_question")

    out = tool.invoke({
        "topic": "reranking",
        "prompt": "What is a reranker?",
        "options": ["a) bad prefix", "B) ok", "C) ok", "D) ok"],  # lowercase a)
        "answer": "B",
        "explanation": "A reranker reorders search results by relevance.",
    })
    parsed = json.loads(out)
    assert "error" in parsed
    # field_validator surfaces the index + expected prefix
    assert "option[0]" in parsed["error"] or "must start with" in parsed["error"]
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cd backend && uv run pytest -q tests/agent/test_quiz_master_agent_tools.py`

Expected output: `ImportError: cannot import name '_make_quiz_tools' from 'app.agent.quiz_master_agent'`. 4 errors (collection-time error).

If you see Pydantic schema errors instead, your `schemas.py` `QuizQuestionPersist` has a bug — fix it before proceeding.

- [ ] **Step 4: Create `backend/app/agent/quiz_master_agent.py` with the tool factory**

Create the file with file header + imports + `_make_quiz_tools` function:

```python
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
from datetime import datetime
from typing import Callable

from langchain_core.tools import tool
from pydantic import ValidationError

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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd backend && uv run pytest -q tests/agent/test_quiz_master_agent_tools.py`

Expected output: `4 passed`. If any test fails:
- `KeyError: 'options'` in `tool.args` → confirm `@tool` decorator picked up the type hints; recent LangChain versions render `list[str]` correctly but older versions need `from __future__ import annotations` at top of file.
- `AssertionError: 'topic_id' in parsed` → verify the JSON return shape in `persist_quiz_question` matches the test expectation exactly.
- `list_for_topic` not found → that's a test-side issue; if `QuestionRepository` doesn't have `list_for_topic`, change the sanity check to `question_repo.get_by_id("nonexistent")` returning `None`.

- [ ] **Step 6: Checkpoint — run the full backend suite**

Run: `cd backend && uv run pytest -q`

Expected output: `187 passed` (183 from Cut ①a + 4 new). Any regression outside the new file is a fail — investigate before proceeding.

- [ ] **Step 7: Mark cut ①b complete**

`quiz_master_agent.py` has tools only; the `build_quiz_master_agent` factory comes in Cut ①c.

---

## Cut P2.3-①c — Loop body + system prompt + final output formatting

**Files:**
- Modify: `backend/app/agent/quiz_master_agent.py` (append loop + helpers + factory)
- Test: `backend/tests/agent/test_quiz_master_agent_loop.py`

**Boundary check:** The loop body is structurally identical to `planner_agent.py:build_planner_agent`. We are NOT generalizing this — the second copy lives independently (each loop has its own system prompt + tool list + output formatter; abstracting now is 3rd-consumer territory per YAGNI).

- [ ] **Step 1: Create the failing test file with 6 tests**

Create `backend/tests/agent/test_quiz_master_agent_loop.py`:

```python
"""Cut P2.3-①c — unit tests for the quiz_master_agent loop body.

Test surface (matches spec §7 layer C):
  1. natural_stop — model emits final summary with no tool calls
  2. budget_exhausted — model keeps calling tools past max_iter
  3. llm_call_failed — LLM ainvoke raises (e.g. Ollama 400 for no-tools model)
  4. tool_error_self_correction — invalid schema → ToolMessage → model retries
  5. valid_persist_round_trip — full happy path: retriever_search → persist → summary
  6. quiz_action_always_generate — agent never sees GRADE turns by contract; verify
     the inference helper returns "generate" unconditionally
"""
import json
from datetime import datetime

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.agent.agent_trace import AgentTrace
from app.agent.quiz_master_agent import (
    build_quiz_master_agent,
    _infer_quiz_action,
)
from app.db.models import Base
from app.db.repositories import (
    GoalRepository,
    QuestionRepository,
    TopicRepository,
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


class ScriptedLLM:
    """LLM that emits a scripted sequence of responses. Useful for asserting
    exit conditions deterministically without spinning up Ollama."""
    def __init__(self, responses: list[AIMessage]):
        self.responses = list(responses)
        self.bound_tools = None
        self.calls = 0

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    async def ainvoke(self, messages, **_kwargs):
        if self.calls >= len(self.responses):
            raise IndexError("ScriptedLLM ran out of responses")
        resp = self.responses[self.calls]
        self.calls += 1
        return resp


class FailingLLM:
    bound_tools = None
    def bind_tools(self, tools):
        self.bound_tools = tools
        return self
    async def ainvoke(self, messages, **_kwargs):
        raise ConnectionRefusedError("ollama is down")


def _ai(content="", tool_calls=None, input_tokens=10, output_tokens=5):
    msg = AIMessage(content=content, tool_calls=tool_calls or [])
    msg.usage_metadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    return msg


async def test_natural_stop_returns_final_summary(session):
    """LLM calls retriever → persist → summary (no tool_calls). Loop exits as natural_stop."""
    user = UserRepository(session).get_or_create("fp-loop-1")
    goal_repo = GoalRepository(session)
    topic_repo = TopicRepository(session)
    question_repo = QuestionRepository(session)

    valid_persist_args = {
        "topic": "HyDE",
        "prompt": "What does HyDE stand for?",
        "options": ["A) Hypothesis-Driven Experimentation",
                    "B) Hypothetical Document Embedding",
                    "C) High-Yield Data Encoding",
                    "D) Hybrid Document Engine"],
        "answer": "B",
        "explanation": "HyDE = Hypothetical Document Embedding.",
    }
    llm = ScriptedLLM([
        _ai(tool_calls=[{"name": "retriever_search",
                         "args": {"query": "HyDE"}, "id": "tc-1"}]),
        _ai(tool_calls=[{"name": "persist_quiz_question",
                         "args": valid_persist_args, "id": "tc-2"}]),
        _ai(content="Quiz on HyDE:\n\nWhat does HyDE stand for?\nA) ...\nB) ...\nReply with A-D."),
    ])

    agent = build_quiz_master_agent(
        llm=llm,
        topic_repo=topic_repo, question_repo=question_repo, goal_repo=goal_repo,
        retriever=None,
        now_fn=lambda: datetime(2026, 5, 24),
    )
    result = await agent({
        "messages": [HumanMessage(content="quiz me on HyDE")],
        "user_id": user.id,
    })

    assert result["quiz_action"] == "generate"
    assert result["active_quiz_question_id"] is not None
    assert "agent_trace" in result
    assert result["agent_trace"]["exit_reason"] == "natural_stop"
    assert result["agent_trace"]["total_iterations"] == 3
    assert result["agent_trace"]["total_tool_calls"] == 2
    assert "Quiz on HyDE" in result["messages"][0].content
    assert result["messages"][0].__class__.__name__ == "AIMessage"


async def test_budget_exhausted_degrades_gracefully(session):
    """LLM never stops calling tools — loop terminates at max_iter."""
    user = UserRepository(session).get_or_create("fp-loop-2")
    goal_repo = GoalRepository(session)
    topic_repo = TopicRepository(session)
    question_repo = QuestionRepository(session)

    # Always emit a tool call, never a final summary
    looping_responses = [
        _ai(tool_calls=[{"name": "retriever_search",
                         "args": {"query": "x"}, "id": f"tc-{i}"}])
        for i in range(10)
    ]
    llm = ScriptedLLM(looping_responses)

    agent = build_quiz_master_agent(
        llm=llm,
        topic_repo=topic_repo, question_repo=question_repo, goal_repo=goal_repo,
        retriever=None,
        max_iter=3,  # tight budget for test speed
    )
    result = await agent({
        "messages": [HumanMessage(content="quiz me on chunking")],
        "user_id": user.id,
    })

    assert result["degraded"] is True
    assert result["agent_trace"]["exit_reason"] == "budget_exhausted"
    assert "budget" in result["messages"][0].content.lower() or "⚠️" in result["messages"][0].content
    assert result.get("active_quiz_question_id") is None  # no persistence


async def test_llm_call_failed_degrades_gracefully(session):
    """LLM raises (e.g. gemma3:4b → Ollama 400). Loop catches and degrades."""
    user = UserRepository(session).get_or_create("fp-loop-3")
    goal_repo = GoalRepository(session)
    topic_repo = TopicRepository(session)
    question_repo = QuestionRepository(session)

    agent = build_quiz_master_agent(
        llm=FailingLLM(),
        topic_repo=topic_repo, question_repo=question_repo, goal_repo=goal_repo,
    )
    result = await agent({
        "messages": [HumanMessage(content="quiz me on HyDE")],
        "user_id": user.id,
    })

    assert result["degraded"] is True
    assert result["agent_trace"]["exit_reason"] == "llm_call_failed"
    assert "ConnectionRefused" in (result["agent_trace"]["llm_error"] or "")


async def test_tool_error_self_correction_via_toolmessage(session):
    """LLM emits invalid options (3 items), tool returns {"error": ...}, LLM
    retries with valid 4-option payload. Loop exits natural_stop."""
    user = UserRepository(session).get_or_create("fp-loop-4")
    goal_repo = GoalRepository(session)
    topic_repo = TopicRepository(session)
    question_repo = QuestionRepository(session)

    bad_args = {
        "topic": "BM25",
        "prompt": "What is BM25?",
        "options": ["A) bad", "B) bad", "C) bad"],  # only 3
        "answer": "A",
        "explanation": "BM25 is...",
    }
    good_args = {
        "topic": "BM25",
        "prompt": "What is BM25?",
        "options": ["A) embedding", "B) ranking function",
                    "C) tokenizer", "D) reranker"],
        "answer": "B",
        "explanation": "BM25 is a probabilistic ranking function.",
    }
    llm = ScriptedLLM([
        _ai(tool_calls=[{"name": "persist_quiz_question",
                         "args": bad_args, "id": "tc-bad"}]),
        _ai(tool_calls=[{"name": "persist_quiz_question",
                         "args": good_args, "id": "tc-good"}]),
        _ai(content="Quiz on BM25 ready"),
    ])

    agent = build_quiz_master_agent(
        llm=llm,
        topic_repo=topic_repo, question_repo=question_repo, goal_repo=goal_repo,
    )
    result = await agent({
        "messages": [HumanMessage(content="quiz me on BM25")],
        "user_id": user.id,
    })

    assert result["quiz_action"] == "generate"
    assert result["active_quiz_question_id"] is not None  # eventually persisted
    assert result["agent_trace"]["exit_reason"] == "natural_stop"
    assert result["agent_trace"]["tool_errors"] == 1  # bad call counted
    # Tool was called twice; second succeeded
    breakdown = result["agent_trace"]["tool_call_breakdown"]
    assert breakdown.get("persist_quiz_question") == 2


async def test_valid_persist_round_trip_writes_active_quiz_question_id(session):
    """Happy path: LLM persists once, summary emitted, active_quiz_question_id
    set to the persisted question_id so next turn routes to GRADE deterministic."""
    user = UserRepository(session).get_or_create("fp-loop-5")
    goal_repo = GoalRepository(session)
    topic_repo = TopicRepository(session)
    question_repo = QuestionRepository(session)

    persist_args = {
        "topic": "embeddings",
        "prompt": "What is an embedding?",
        "options": ["A) A vector representation",
                    "B) A type of database",
                    "C) A search algorithm",
                    "D) A loss function"],
        "answer": "A",
        "explanation": "An embedding is a dense vector representation of text.",
    }
    llm = ScriptedLLM([
        _ai(tool_calls=[{"name": "persist_quiz_question",
                         "args": persist_args, "id": "tc-1"}]),
        _ai(content="Quiz ready"),
    ])

    agent = build_quiz_master_agent(
        llm=llm,
        topic_repo=topic_repo, question_repo=question_repo, goal_repo=goal_repo,
    )
    result = await agent({
        "messages": [HumanMessage(content="quiz me on embeddings")],
        "user_id": user.id,
    })

    # Verify question_id was extracted from tool output and surfaced to state
    persisted_id = result["active_quiz_question_id"]
    assert persisted_id is not None
    fetched = question_repo.get_by_id(persisted_id)
    assert fetched is not None
    assert fetched.answer == "A"


def test_infer_quiz_action_always_returns_generate():
    """Agent never sees GRADE turns by dispatcher contract. _infer_quiz_action
    is here for state-contract uniformity with deterministic quiz_master."""
    from app.agent.quiz_master_agent import _infer_quiz_action
    import time
    trace = AgentTrace(t_start=time.monotonic())
    # Empty trace → still generate (sensible default)
    assert _infer_quiz_action(trace) == "generate"
    # Trace with persist → still generate
    trace.record_tool_call("persist_quiz_question", {}, '{"question_id":"q"}', error=False)
    assert _infer_quiz_action(trace) == "generate"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd backend && uv run pytest -q tests/agent/test_quiz_master_agent_loop.py`

Expected output: `ImportError: cannot import name 'build_quiz_master_agent' from 'app.agent.quiz_master_agent'`. 6 errors (collection-time).

- [ ] **Step 3: Append loop body + helpers + factory to `quiz_master_agent.py`**

Open `backend/app/agent/quiz_master_agent.py` and append (after the `_make_quiz_tools` function):

```python
import time
from typing import Literal

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.config import get_stream_writer

from app.agent.agent_trace import AgentTrace
from app.agent.state import CoachState


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


def _safe_writer():
    """get_stream_writer() with a no-op fallback for direct unit-test calls."""
    try:
        return get_stream_writer()
    except RuntimeError:
        return lambda _payload: None


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
        output = await handler.ainvoke(args) if hasattr(handler, "ainvoke") else handler.invoke(args)
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


def _format_final_output(writer, trace: AgentTrace, last_response) -> dict:
    final_text = getattr(last_response, "content", "") or ""
    if not isinstance(final_text, str):
        final_text = "".join(
            (b.get("text", "") if isinstance(b, dict) else str(b))
            for b in final_text
        )

    writer({"type": "citations", "citations": []})
    writer({"type": "token", "text": final_text})

    return {
        "messages": [AIMessage(content=final_text)],
        "citations": [],
        "active_quiz_question_id": trace.last_persisted_question_id(),
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

    return {
        "messages": [AIMessage(content=text)],
        "citations": [],
        # active_quiz_question_id intentionally NOT set — no confirmed persistence
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
    truthy) to deterministic quiz_master regardless of configured mode. If
    somehow called on a GRADE turn anyway, the agent will still run a fresh
    GENERATE — defensive but signals dispatcher misuse.
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
                return _format_final_output(writer, trace, response)

            for tc in tool_calls:
                output = await _safe_invoke_tool(tool_map, tc, trace)
                messages.append(ToolMessage(
                    content=str(output), tool_call_id=tc.get("id", ""),
                ))

        trace.record_budget_exhaustion(max_iter)
        return _format_degrade_output(writer, trace, "budget_exhausted")

    return quiz_master_agent_node
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd backend && uv run pytest -q tests/agent/test_quiz_master_agent_loop.py`

Expected output: `6 passed`. Common issues:
- `AttributeError: 'AgentTrace' object has no attribute 'last_persisted_question_id'` → Cut ①a didn't add the method. Go back and fix.
- `KeyError: 'agent_trace'` in result → confirm `_format_final_output` / `_format_degrade_output` both include `agent_trace` in the returned dict.
- `tool_errors == 0` in test 4 → the `_safe_invoke_tool` JSON `{"error":...}` detection branch isn't recording error=True; check the `is_tool_error` flag is correctly set.

- [ ] **Step 5: Checkpoint — run the full backend suite**

Run: `cd backend && uv run pytest -q`

Expected output: `193 passed` (187 from Cut ①b + 6 new).

- [ ] **Step 6: Mark cut ①c complete**

Loop body is wired. Next cuts integrate it into the graph and routes.

---

## Cut P2.3-①d — Graph dispatcher: state-aware + mode-aware

**Files:**
- Modify: `backend/app/agent/graph.py` (rewrite `quiz_node` body — currently lines that delegate to `configurable.quiz_master`)
- Test: `backend/tests/agent/test_graph_quiz_agent_e2e.py`

**Boundary check:** This is the ONE place where the GRADE-always-deterministic guarantee is enforced. Verify the order of branches BEFORE the implementation: state check FIRST, mode check SECOND.

- [ ] **Step 1: Read the existing `quiz_node` in `graph.py` to find the exact location**

Run: `cd backend && grep -n "quiz_node\|quiz_stub_node" app/agent/graph.py`

Note the line numbers. You're looking for a function `quiz_node` (or `async def quiz_node`) that currently delegates to `configurable.get("quiz_master")` — that's the function body being replaced.

- [ ] **Step 2: Create the failing test file with 3 tests**

Create `backend/tests/agent/test_graph_quiz_agent_e2e.py`:

```python
"""Cut P2.3-①d — graph-level e2e tests for the Quiz mode-aware dispatcher.

Three assertions:
  1. quiz_mode="agent_loop" + GENERATE turn → routes to agent (agent_trace set)
  2. quiz_mode="agent_loop" + GRADE turn (active_quiz_question_id set) →
     routes to deterministic quiz_master (state-aware override)
  3. quiz_mode="deterministic" (default) + GENERATE → routes to quiz_master
     (baseline path, no agent_trace in state)
"""
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.agent.graph import build_graph
from app.db.models import Base
from app.db.repositories import (
    GoalRepository, MasteryRepository, MistakeRepository,
    QuestionRepository, TopicRepository, UserRepository,
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


async def _stub_quiz_master(state):
    return {
        "messages": [AIMessage(content="DETERMINISTIC-PATH: generated question X")],
        "citations": [],
        "active_quiz_question_id": "deterministic-q-1",
        "quiz_action": "generate",
    }


async def _stub_quiz_master_grade(state):
    """Stub that handles GRADE turn — sets quiz_action='grade'."""
    return {
        "messages": [AIMessage(content="DETERMINISTIC-GRADE: ✓ correct")],
        "citations": [],
        "active_quiz_question_id": None,
        "quiz_action": "grade",
    }


async def _stub_quiz_master_agent(state):
    return {
        "messages": [AIMessage(content="AGENT-PATH: generated question Y")],
        "citations": [],
        "active_quiz_question_id": "agent-q-1",
        "quiz_action": "generate",
        "agent_trace": {
            "total_iterations": 2,
            "total_tool_calls": 2,
            "tool_call_breakdown": {"retriever_search": 1, "persist_quiz_question": 1},
            "tool_errors": 0,
            "input_tokens": 100,
            "output_tokens": 50,
            "wall_time_s": 2.5,
            "exit_reason": "natural_stop",
            "llm_error": None,
        },
    }


def _qm_branch(state):
    """Dispatch deterministic stub by state shape: GRADE if active_quiz_question_id set."""
    if state.get("active_quiz_question_id"):
        return _stub_quiz_master_grade(state)
    return _stub_quiz_master(state)


async def test_quiz_mode_agent_loop_routes_to_agent_on_generate_turn(session):
    user = UserRepository(session).get_or_create("fp-graph-1")
    graph = build_graph(checkpointer=None)
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="quiz me on HyDE")], "user_id": user.id},
        config={"configurable": {
            "quiz_master": _qm_branch,
            "quiz_master_agent": _stub_quiz_master_agent,
            "quiz_mode": "agent_loop",
        }},
    )
    # Route hit the agent stub
    assert "AGENT-PATH" in result["messages"][-1].content
    assert result.get("agent_trace") is not None
    assert result["agent_trace"]["exit_reason"] == "natural_stop"


async def test_quiz_mode_agent_loop_routes_to_deterministic_on_grade_turn(session):
    """State-aware override: active_quiz_question_id truthy → deterministic GRADE
    regardless of quiz_mode configured."""
    user = UserRepository(session).get_or_create("fp-graph-2")
    graph = build_graph(checkpointer=None)
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="A")],
            "user_id": user.id,
            "active_quiz_question_id": "pre-existing-q-99",
        },
        config={"configurable": {
            "quiz_master": _qm_branch,
            "quiz_master_agent": _stub_quiz_master_agent,
            "quiz_mode": "agent_loop",  # still agent_loop, but GRADE state wins
        }},
    )
    # Route hit the deterministic stub even though mode is agent_loop
    assert "DETERMINISTIC-GRADE" in result["messages"][-1].content
    assert result.get("quiz_action") == "grade"
    assert result.get("agent_trace") is None  # deterministic doesn't populate


async def test_quiz_mode_deterministic_default_routes_to_quiz_master(session):
    user = UserRepository(session).get_or_create("fp-graph-3")
    graph = build_graph(checkpointer=None)
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="quiz me on chunking")], "user_id": user.id},
        config={"configurable": {
            "quiz_master": _qm_branch,
            "quiz_master_agent": _stub_quiz_master_agent,
            "quiz_mode": "deterministic",
        }},
    )
    assert "DETERMINISTIC-PATH" in result["messages"][-1].content
    assert result.get("agent_trace") is None
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cd backend && uv run pytest -q tests/agent/test_graph_quiz_agent_e2e.py`

Expected output: 3 tests FAIL — `quiz_master_agent` is in configurable but the current `quiz_node` doesn't read `quiz_mode` yet, so all 3 routes hit `quiz_master` → 2/3 tests fail (the AGENT-PATH and DETERMINISTIC-GRADE assertions don't match).

- [ ] **Step 4: Rewrite `quiz_node` in `graph.py`**

Edit `backend/app/agent/graph.py`. Locate the existing `quiz_node` function (use the line numbers from Step 1). Replace the function body with:

```python
async def quiz_node(state, config) -> dict:
    """State-aware + mode-aware Quiz dispatcher.

    Routing precedence:
    1. GRADE turn (active_quiz_question_id truthy) → ALWAYS deterministic
       quiz_master, regardless of configured mode. P2.3 agent_loop never sees
       GRADE turns by design.
    2. GENERATE turn → mode-aware: agent_loop → quiz_master_agent;
       deterministic (default) → quiz_master.
    Missing dependencies → fall back to quiz_stub_node.
    """
    configurable = (config or {}).get("configurable", {}) or {}

    # State-aware: GRADE turn always deterministic
    if state.get("active_quiz_question_id"):
        quiz_master = configurable.get("quiz_master")
        if quiz_master is None:
            return await quiz_stub_node(state)
        return await quiz_master(state)

    # GENERATE turn: mode-aware dispatch
    mode = configurable.get("quiz_mode", "deterministic")
    if mode == "agent_loop":
        agent = configurable.get("quiz_master_agent")
        if agent is None:
            return await quiz_stub_node(state)
        return await agent(state)
    quiz_master = configurable.get("quiz_master")
    if quiz_master is None:
        return await quiz_stub_node(state)
    return await quiz_master(state)
```

Note: if the existing `quiz_node` was synchronous OR did not take `config`, update its signature to match `async def quiz_node(state, config) -> dict:`. The LangGraph `add_node` registration line elsewhere in `graph.py` must continue to pass `quiz_node` as a callable; LangGraph auto-injects `config` when the function signature accepts it.

If `quiz_stub_node` is not async, you have two options:
- Keep `quiz_stub_node` sync and remove the `await` before it: `return quiz_stub_node(state)`.
- Or make it async (1-line change).

Prefer the first to minimize diff.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd backend && uv run pytest -q tests/agent/test_graph_quiz_agent_e2e.py`

Expected output: `3 passed`.

Common issues:
- `TypeError: object dict can't be used in 'await' expression` → `_stub_quiz_master_grade` or `_qm_branch` return values are not awaitable. Either:
  - Wrap the stub return with `async def`, or
  - Adjust the test stub to be `async def _stub_quiz_master_grade(state):` and call it normally.
- `quiz_master` stub not invoked when expected → `_qm_branch` discrimination on `active_quiz_question_id` is failing. Add a print statement to debug.

- [ ] **Step 6: Checkpoint — run the full backend suite**

Run: `cd backend && uv run pytest -q`

Expected output: `196 passed` (193 from Cut ①c + 3 new).

Existing P2.1-④ Quiz e2e tests (`test_graph_quiz_e2e.py` or similar) must still pass — they don't touch `quiz_mode` so they fall through to deterministic and behave as before. If any P2.1-④ test fails, the dispatcher rewrite broke a regression — investigate.

- [ ] **Step 7: Mark cut ①d complete**

Graph dispatcher in place. Routes wiring in Cut ①e closes production.

---

## Cut P2.3-①e — deps.py + routes.py production wiring

**Files:**
- Modify: `backend/app/api/deps.py` (append 2 factories)
- Modify: `backend/app/api/routes.py` (extend `chat()` signature + configurable keys)
- Test: `backend/tests/api/test_routes_quiz_agent.py`

**Boundary check:** This is what makes P2.3 reach production users via HTTP. Until this cut, the agent variant only exists in the graph; routes don't inject it.

- [ ] **Step 1: Create the failing test file with 3 tests**

Create `backend/tests/api/test_routes_quiz_agent.py`:

```python
"""Cut P2.3-①e — route-level integration tests for x-quiz-mode + multi-turn dispatcher.

Three assertions:
  1. x-quiz-mode=agent_loop → SSE stream emits citations → token → done with
     content from the agent path (stub identifiable string).
  2. x-quiz-mode header missing (or = "deterministic") → SSE from deterministic
     quiz_master path.
  3. Multi-turn: same session_id, turn 1 = "quiz me", turn 2 = "A". Turn 1 in
     agent_loop mode routes to agent; turn 2 (active_quiz_question_id set by
     turn 1) routes to deterministic GRADE.
"""
import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from sqlalchemy.orm import Session

from app.api.deps import (
    get_graph, get_judge_dependencies, get_llm, get_quiz_master,
    get_planner, get_planner_agent, get_planner_mode,
    get_quiz_master_agent, get_quiz_mode,
    get_memory_hydrator, get_memory_writer, get_user_id,
)
from app.main import create_app


@pytest.fixture
def client_with_stubs(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/p2_3_routes.db")
    monkeypatch.setenv("STUDY_COACH_TEST_MODE", "1")
    app = create_app()

    # Identifiable stubs so the test asserts which path served the request
    async def agent_stub(state):
        return {
            "messages": [AIMessage(content="AGENT-QUIZ-PATH")],
            "citations": [],
            "active_quiz_question_id": "agent-q-1",
            "quiz_action": "generate",
            "agent_trace": {"exit_reason": "natural_stop", "total_iterations": 2,
                            "total_tool_calls": 2, "tool_call_breakdown": {},
                            "tool_errors": 0, "input_tokens": 0,
                            "output_tokens": 0, "wall_time_s": 0.1,
                            "llm_error": None},
        }

    async def quiz_master_stub(state):
        if state.get("active_quiz_question_id"):
            return {
                "messages": [AIMessage(content="DETERMINISTIC-GRADE-✓")],
                "citations": [],
                "active_quiz_question_id": None,
                "quiz_action": "grade",
            }
        return {
            "messages": [AIMessage(content="DETERMINISTIC-QUIZ-PATH")],
            "citations": [],
            "active_quiz_question_id": "deterministic-q-1",
            "quiz_action": "generate",
        }

    # Override deps to inject stubs (preserve everything else)
    app.dependency_overrides[get_quiz_master] = lambda: quiz_master_stub
    app.dependency_overrides[get_quiz_master_agent] = lambda: agent_stub
    # Other deps default to real values; routing logic is what we test.

    with TestClient(app) as c:
        yield c


def test_quiz_mode_header_agent_loop_routes_to_agent_path(client_with_stubs):
    response = client_with_stubs.post(
        "/api/chat",
        json={"message": "quiz me on HyDE", "session_id": "test-sess-1"},
        headers={"x-fingerprint": "fp-r-1", "x-quiz-mode": "agent_loop"},
        stream=True,  # if your test client supports streaming; otherwise omit
    )
    assert response.status_code == 200
    body = response.text  # full SSE body
    assert "AGENT-QUIZ-PATH" in body
    assert "citations" in body
    assert "token" in body
    assert "done" in body


def test_quiz_mode_header_absent_defaults_to_deterministic(client_with_stubs):
    response = client_with_stubs.post(
        "/api/chat",
        json={"message": "quiz me on chunking", "session_id": "test-sess-2"},
        headers={"x-fingerprint": "fp-r-2"},
    )
    assert response.status_code == 200
    body = response.text
    assert "DETERMINISTIC-QUIZ-PATH" in body
    assert "AGENT-QUIZ-PATH" not in body


def test_multi_turn_agent_loop_generate_then_deterministic_grade(client_with_stubs):
    """Turn 1 in agent_loop mode → AGENT path persists active_quiz_question_id.
    Turn 2 (same session) → state-aware dispatcher overrides mode → deterministic GRADE."""
    # Turn 1: GENERATE in agent_loop mode
    r1 = client_with_stubs.post(
        "/api/chat",
        json={"message": "quiz me on BM25", "session_id": "test-sess-3"},
        headers={"x-fingerprint": "fp-r-3", "x-quiz-mode": "agent_loop"},
    )
    assert r1.status_code == 200
    assert "AGENT-QUIZ-PATH" in r1.text

    # Turn 2: same session → checkpointer has active_quiz_question_id = "agent-q-1"
    r2 = client_with_stubs.post(
        "/api/chat",
        json={"message": "A", "session_id": "test-sess-3"},
        headers={"x-fingerprint": "fp-r-3", "x-quiz-mode": "agent_loop"},  # still agent_loop
    )
    assert r2.status_code == 200
    assert "DETERMINISTIC-GRADE-✓" in r2.text
    assert "AGENT-QUIZ-PATH" not in r2.text  # GRADE didn't go through agent
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd backend && uv run pytest -q tests/api/test_routes_quiz_agent.py`

Expected output: `ImportError: cannot import name 'get_quiz_master_agent' from 'app.api.deps'` (collection error).

- [ ] **Step 3: Append `get_quiz_mode` and `get_quiz_master_agent` to `deps.py`**

Open `backend/app/api/deps.py` and append at the end of file:

```python
def get_quiz_mode(
    x_quiz_mode: Annotated[str | None, Header()] = None,
) -> Literal["deterministic", "agent_loop"]:
    """Read x-quiz-mode header. Default = deterministic. Unknown → deterministic."""
    if x_quiz_mode == "agent_loop":
        return "agent_loop"
    return "deterministic"


def get_quiz_master_agent(
    session: Annotated[Session, Depends(get_session)],
    llm: Annotated[object, Depends(get_llm)],
    retriever: Annotated[object, Depends(get_retriever)],
):
    # cloud-adapt: cloud BYOK can raise max_iter from 6 to 12 here
    from app.agent.quiz_master_agent import build_quiz_master_agent
    from app.db.repositories import (
        GoalRepository, QuestionRepository, TopicRepository,
    )
    return build_quiz_master_agent(
        llm=llm,
        topic_repo=TopicRepository(session),
        question_repo=QuestionRepository(session),
        goal_repo=GoalRepository(session),
        retriever=retriever,
    )
```

If `Literal` or `Annotated` or `Header` is not already imported at the top of `deps.py`, add it to the import block. Do NOT scatter imports.

- [ ] **Step 4: Extend `chat()` in `routes.py`**

Open `backend/app/api/routes.py`. Locate the `async def chat(...)` signature. Add 2 new `Depends` parameters to the signature (placement after the existing `planner_mode` Depends keeps Quiz-related kwargs grouped):

```python
quiz_master_agent: Annotated[object, Depends(get_quiz_master_agent)],
quiz_mode: Annotated[str, Depends(get_quiz_mode)],
```

Make sure `get_quiz_master_agent` and `get_quiz_mode` are added to the existing `from .deps import (...)` import block.

In the `config["configurable"]` dict (inside the `event_stream` async generator or wherever the config is built), add 2 keys:

```python
"quiz_master_agent": quiz_master_agent,
"quiz_mode": quiz_mode,
```

DO NOT change anything else in `routes.py` — keep the SSE contract bytes unchanged.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd backend && uv run pytest -q tests/api/test_routes_quiz_agent.py`

Expected output: `3 passed`.

Common issues:
- `KeyError: 'quiz_master_agent'` in `quiz_node` → routes.py didn't inject the key into configurable. Re-check Step 4.
- TestClient stream handling issue → some FastAPI TestClient versions return the body as text directly. If `response.text` is empty, switch to `response.iter_lines()` or `b"".join(response.iter_bytes()).decode()`.
- `405 Method Not Allowed` on multi-turn test → check that `/api/chat` accepts POST with `session_id` in body; that's the existing P2.1-④f / P2.2 contract.

- [ ] **Step 6: Checkpoint — run the full backend suite**

Run: `cd backend && uv run pytest -q`

Expected output: `199 passed` (196 from Cut ①d + 3 new).

If any existing test fails (e.g. existing `tests/api/test_routes_quiz.py` for deterministic Quiz), it likely means a `Depends` ordering issue in the chat signature. Revisit Step 4.

- [ ] **Step 7: Mark cut ①e complete**

Backend production wiring is done. The remaining test target is Cut ②a (eval harness, +3 tests → 202 total) per the original plan. Discipline reminder said "201" — adjust if your actual count differs by 1-2; the spec's count was approximate.

---

## Cut P2.3-①f — Real-Ollama smoke test (manual)

**Files:** No code changes. This is a manual verification step before running the matrix.

**Discipline:** Don't skip this. Cut P2.2-①f Phase B was where the `ChatOllama(reasoning=False)` discovery happened — without that smoke, the matrix would have taken 30 hours instead of 5.

### Phase A — Pre-flight check

- [ ] **A1: Confirm Ollama is running and 4 models are pulled**

Run:
```bash
ollama list | grep -E "gemma3:4b|qwen3.5:4b|qwen2.5:7b|gemma4:e4b"
```

Expected: 4 lines, one per model. If any missing:
```bash
ollama pull gemma3:4b qwen3.5:4b qwen2.5:7b gemma4:e4b
```

- [ ] **A2: Confirm 199 backend tests still pass**

Run: `cd backend && uv run pytest -q`

Expected: `199 passed`.

### Phase B — `reasoning=False` mechanism sanity check (CRITICAL)

P2.2 Cut ①f discovered that `ChatOllama(reasoning=False)` is the only kwarg that forwards to Ollama API `think` field. Verify this still holds before running the matrix.

- [ ] **B1: Time a single qwen3.5:4b agent call with `reasoning=False`**

Create a tiny smoke script `backend/scripts/smoke_quiz_reasoning_off.py`:

```python
"""Phase B smoke — verify ChatOllama(reasoning=False) is fast on qwen3.5:4b."""
import asyncio
import time

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage


async def main():
    llm = ChatOllama(
        model="qwen3.5:4b",
        temperature=0.7,
        reasoning=False,  # critical
    )
    t0 = time.monotonic()
    response = await llm.ainvoke([HumanMessage(content="What is 2+2? Answer in one word.")])
    elapsed = time.monotonic() - t0
    print(f"qwen3.5:4b reasoning=False elapsed: {elapsed:.1f}s")
    print(f"response: {response.content[:80]}")
    assert elapsed < 30, f"reasoning=False mechanism broken — expected <30s, got {elapsed:.1f}s"


if __name__ == "__main__":
    asyncio.run(main())
```

Run: `cd backend && uv run python scripts/smoke_quiz_reasoning_off.py`

Expected output:
- elapsed < 15 seconds (typical 5-10s on 16GB Mac)
- response is a short answer like "Four" or "4"

If elapsed > 30s, the `reasoning=False` mechanism is broken or langchain-ollama upgraded and changed the kwarg name. STOP — re-investigate before running the matrix; the P2.2 finding may have regressed.

### Phase C — 8-cell happy-path smoke

- [ ] **C1: Start the backend in test mode (one Terminal)**

Run: `cd backend && STUDY_COACH_TEST_MODE=0 uv run uvicorn app.main:app --host 127.0.0.1 --port 8000`

Wait until you see `Uvicorn running on http://127.0.0.1:8000`.

- [ ] **C2: Run 8 happy-path requests (one for each (model, mode) cell)**

For each cell, POST a single GENERATE request via `curl` and watch the SSE output. Use a separate Terminal:

```bash
for model in gemma3:4b qwen3.5:4b qwen2.5:7b gemma4:e4b; do
  for mode in deterministic agent_loop; do
    echo "=== model=$model mode=$mode ==="
    curl -N -X POST http://127.0.0.1:8000/api/chat \
      -H "content-type: application/json" \
      -H "x-fingerprint: smoke-$model-$mode" \
      -H "x-model: $model" \
      -H "x-quiz-mode: $mode" \
      -d '{"message": "quiz me on HyDE"}' 2>&1 | head -50
    echo ""
  done
done
```

Per-cell expected outcomes:

| Model | Mode | Expected |
|---|---|---|
| gemma3:4b | deterministic | SSE 3 events (citations → token → done); MCQ persisted; ~10-20s |
| gemma3:4b | agent_loop | SSE 3 events; DEGRADE message `"⚠️ Could not reach the quiz model"`; ~5s; agent_trace.exit_reason="llm_call_failed" |
| qwen3.5:4b | deterministic | SSE 3 events; MCQ persisted (possibly garbled JSON → fallback); ~5-15s |
| qwen3.5:4b | agent_loop | SSE 3 events; MCQ persisted (schema rescue may kick in); ~30-90s |
| qwen2.5:7b | deterministic | SSE 3 events; MCQ persisted; ~10-20s |
| qwen2.5:7b | agent_loop | SSE 3 events; MCQ persisted; ~20-40s |
| gemma4:e4b | deterministic | SSE 3 events; MCQ persisted; ~30-90s (thinking-OFF slower than expected; thinking-ON default is suspected — verify) |
| gemma4:e4b | agent_loop | SSE 3 events; MCQ persisted; ~30-90s |

- [ ] **C3: Verify gemma3:4b agent_loop produces expected Ollama 400 degrade**

If gemma3:4b agent_loop returns a real persisted question instead of a degrade, the Ollama gemma3 manifest may have added `tools` capability since P2.2. Update spec §12 prediction P1 to "TRUE (verified mid-2026)" or "FALSE (gemma3 now supports tools)" — either is a finding.

If it produces the degrade message as expected, P2.2's negative-data-point finding replicates. Confirm `_format_degrade_output` cleanly handles the error.

- [ ] **C4: Spot-check at least 1 persisted question for grounding quality**

Pick `qwen2.5:7b agent_loop` (highest baseline-quality cell). Check the latest question in the DB:

```bash
sqlite3 backend/study_coach.db "SELECT prompt, answer, explanation FROM questions ORDER BY id DESC LIMIT 1;"
```

Expected: prompt references HyDE concepts from HKBU PDF (HyDEGenerator, Hypothetical Document Embedding, embedding-based search). Distractors should be plausible.

If the question is hallucinated (e.g. "HyDE = Hypothesis-Driven Experimentation"), the retriever_search tool may not be wired correctly or the LLM ignored grounding. Investigate before the matrix.

- [ ] **C5: Kill the backend**

`Ctrl+C` in Terminal 1.

- [ ] **C6: Mark Cut ①f complete and log Phase B/C outcomes**

Write a short summary at the top of the next cut (②a) noting:
- Phase B `reasoning=False` elapsed: ___ seconds
- Phase C gemma3:4b agent_loop outcome: degrade as expected? YES / NO
- Phase C qwen2.5:7b agent_loop question quality: grounded? YES / NO
- Any model-specific quirks to note for ②b matrix run

---

## Cut P2.3-②a — Eval harness fork

**Files:**
- Create: 7 files under `backend/app/eval/p2_3_quiz_ablation/`
- Test: `backend/tests/eval/test_p2_3_harness.py`

**Boundary check:** Fork from `p2_2_agent_ablation/`, do NOT refactor it into a shared base class. Code duplication is OK here per CLAUDE.md "smaller, focused files over abstractions". The forked files share ~80% of structure but each is independently editable for future task-specific tweaks.

- [ ] **Step 1: Create the package marker**

Create `backend/app/eval/p2_3_quiz_ablation/__init__.py`:

```python
"""P2.3 Quiz Agent Loop Ablation — eval harness.

Fork of p2_2_agent_ablation/. Reads x-quiz-mode header for mode selection;
uses QUIZ_DIMENSIONS rubric for both local and cloud judges.

Schema parity with P2.2 single_run records — only mode field name (quiz_mode
vs planner_mode) and output extraction (quiz_action vs plan_action) differ.
"""
```

- [ ] **Step 2: Create `matrix.py`**

Create `backend/app/eval/p2_3_quiz_ablation/matrix.py`:

```python
"""Matrix expansion: build RunSpec list from models × modes × queries × runs."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class RunSpec:
    run_id: str           # deterministic md5 hash of all other fields
    model: str
    mode: Literal["deterministic", "agent_loop"]
    thinking: bool        # reasoning=spec.thinking forwarded to ChatOllama
    query_id: str
    turn_idx: int         # 0 for single-turn or first of multi-turn; 1 for grade
    run_idx: int          # 0..runs-1 — repetitions for statistical power
    message: str          # the prompt text for this turn
    session_key: str      # langgraph thread_id (same across multi-turn turns)


def _run_id(model: str, mode: str, thinking: bool, query_id: str, turn_idx: int, run_idx: int) -> str:
    raw = f"{model}|{mode}|{thinking}|{query_id}|{turn_idx}|{run_idx}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def expand_matrix(
    *,
    models: list[str],
    modes: list[Literal["deterministic", "agent_loop"]],
    single_turn_queries: list[dict],   # [{"id":..., "message":...}]
    multi_turn_queries: list[dict],    # [{"id":..., "messages": [t0, t1, ...]}]
    runs: int = 3,
    thinking_appendix: bool = False,
) -> list[RunSpec]:
    """Expand matrix. Appendix is gemma4:e4b thinking-ON, single-turn only."""
    specs: list[RunSpec] = []
    for model in models:
        for mode in modes:
            # Single-turn
            for q in single_turn_queries:
                for r in range(runs):
                    specs.append(RunSpec(
                        run_id=_run_id(model, mode, False, q["id"], 0, r),
                        model=model, mode=mode, thinking=False,
                        query_id=q["id"], turn_idx=0, run_idx=r,
                        message=q["message"],
                        session_key=f"{model}|{mode}|{q['id']}|r{r}",
                    ))
            # Multi-turn
            for q in multi_turn_queries:
                for r in range(runs):
                    for turn, msg in enumerate(q["messages"]):
                        specs.append(RunSpec(
                            run_id=_run_id(model, mode, False, q["id"], turn, r),
                            model=model, mode=mode, thinking=False,
                            query_id=q["id"], turn_idx=turn, run_idx=r,
                            message=msg,
                            session_key=f"{model}|{mode}|{q['id']}|r{r}",
                        ))
    # Appendix: gemma4:e4b thinking-ON, single-turn only, both modes
    if thinking_appendix:
        for mode in modes:
            for q in single_turn_queries:
                for r in range(runs):
                    specs.append(RunSpec(
                        run_id=_run_id("gemma4:e4b", mode, True, q["id"], 0, r),
                        model="gemma4:e4b", mode=mode, thinking=True,
                        query_id=q["id"], turn_idx=0, run_idx=r,
                        message=q["message"],
                        session_key=f"gemma4:e4b|{mode}|{q['id']}|r{r}|think",
                    ))
    return specs


def filter_pending_specs(specs: list[RunSpec], done_run_ids: set[str]) -> list[RunSpec]:
    """Resumability — drop specs whose run_id is already in results.jsonl."""
    return [s for s in specs if s.run_id not in done_run_ids]
```

- [ ] **Step 3: Create `single_run.py`**

Create `backend/app/eval/p2_3_quiz_ablation/single_run.py`:

```python
"""One row in results.jsonl + the function that produces it."""
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
    graph,
    judge_local,             # callable: (question, quiz_text) -> {score, weak_dims, reasoning}
    judge_cloud,             # callable or None
    config_extras: dict,     # quiz_master / quiz_master_agent / memory_* callables
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
            "quiz_mode": spec.mode,
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
        "exit_reason": "error" if err else ("deterministic" if spec.mode == "deterministic" else "n/a"),
        "llm_error": err,
    }
    final_text = ""
    msgs = final_state.get("messages") or []
    if msgs:
        last = msgs[-1]
        final_text = getattr(last, "content", "") or ""

    quiz_action = final_state.get("quiz_action")
    question_id = final_state.get("active_quiz_question_id") or ""
    persisted = 1 if question_id else 0

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
            "quiz_action": quiz_action,
            "question_persisted": persisted,
            "question_id": question_id,
            "final_text_excerpt": final_text[:500],
        },
        "judge_local": judge_local_out,
        "judge_cloud": judge_cloud_out,
    }
    validate_record_schema(record)
    return record
```

- [ ] **Step 4: Create `judges.py`**

Create `backend/app/eval/p2_3_quiz_ablation/judges.py`:

```python
"""Dual-judge for Quiz: qwen2.5:7b local + MiniMax-M2.7 cloud.

Both use QUIZ_DIMENSIONS rubric loaded from app/agent/prompts/judge_quiz.txt.
Cloud judge is skipped (returns {}) when MINIMAX_API_KEY env is unset.
"""
from __future__ import annotations

import os
from pathlib import Path

from langchain_ollama import ChatOllama

from app.agent.judge import judge_response


_QUIZ_DIMENSIONS = (
    "question_quality",
    "option_plausibility",
    "answer_correctness",
    "explanation_clarity",
    "difficulty_calibration",
)

_RUBRIC_PATH = Path(__file__).resolve().parents[2] / "agent" / "prompts" / "judge_quiz.txt"


def _load_rubric() -> str:
    return _RUBRIC_PATH.read_text(encoding="utf-8")


def make_local_judge():
    """qwen2.5:7b local judge using QUIZ_DIMENSIONS."""
    llm = ChatOllama(model="qwen2.5:7b", temperature=0.0, reasoning=False)
    rubric = _load_rubric()

    async def _judge(question: str, answer: str) -> dict:
        return await judge_response(
            question=question, answer=answer, llm=llm,
            rubric=rubric, dimensions=_QUIZ_DIMENSIONS,
            context="",  # context retrieval would inflate cost; leave blank for eval
        )
    return _judge


def make_cloud_judge():
    """MiniMax-M2.7 cloud judge using QUIZ_DIMENSIONS.

    Returns None if MINIMAX_API_KEY not set (CI / local without budget)."""
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        return None

    # cloud-adapt: same OpenAI-compatible endpoint pattern as P2.2
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=api_key, base_url="https://api.minimaxi.com/v1")
    rubric = _load_rubric()

    async def _judge(question: str, answer: str) -> dict:
        prompt = rubric.format(question=question, context="", answer=answer)
        try:
            resp = await client.chat.completions.create(
                model="MiniMax-M2.7",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.01,  # MiniMax range (0, 1], 0.0 rejected
                max_tokens=400,
            )
            content = resp.choices[0].message.content or ""
        except Exception as exc:
            return {"score": 0.0, "weak_dims": [], "reasoning": f"cloud judge error: {exc}",
                    "model": "MiniMax-M2.7"}

        # Reuse the same greedy regex extraction as P2.2 / judge.py
        import re, json
        match = re.search(r"\{[\s\S]*\}", content)
        if not match:
            return {"score": 0.0, "weak_dims": [], "reasoning": "no JSON in cloud judge output",
                    "model": "MiniMax-M2.7"}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"score": 0.0, "weak_dims": [], "reasoning": "cloud judge JSON parse error",
                    "model": "MiniMax-M2.7"}

        dims_values = [int(parsed.get(d, 3)) for d in _QUIZ_DIMENSIONS]
        score = sum(dims_values) / (5 * len(_QUIZ_DIMENSIONS))  # 1-5 → 0-1
        weak_dims = [d for d, v in zip(_QUIZ_DIMENSIONS, dims_values) if v <= 2]
        return {"score": round(score, 3), "weak_dims": weak_dims,
                "reasoning": parsed.get("reasoning", ""), "model": "MiniMax-M2.7"}

    return _judge
```

- [ ] **Step 5: Create `run_eval.py`**

Create `backend/app/eval/p2_3_quiz_ablation/run_eval.py`:

```python
"""CLI entry point for the P2.3 Quiz Ablation matrix run."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.agent.graph import build_graph
from app.agent.judge import build_judge_llm  # if exists; else simpler ChatOllama
from app.agent.memory_updater import build_memory_hydrator, build_memory_writer
from app.agent.planner import build_planner
from app.agent.quiz_master import build_quiz_master
from app.agent.quiz_master_agent import build_quiz_master_agent
from app.db.models import Base
from app.db.repositories import (
    GoalRepository, MasteryRepository, MistakeRepository, PlanRepository,
    QuestionRepository, TopicRepository, UserRepository,
)
from app.eval.p2_3_quiz_ablation.judges import make_cloud_judge, make_local_judge
from app.eval.p2_3_quiz_ablation.matrix import expand_matrix, filter_pending_specs
from app.eval.p2_3_quiz_ablation.single_run import run_one
from langchain_ollama import ChatOllama


DEFAULT_MODELS = ["gemma3:4b", "qwen3.5:4b", "qwen2.5:7b", "gemma4:e4b"]
DEFAULT_MODES = ["deterministic", "agent_loop"]


def _load_done_run_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    done = set()
    with output_path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
                done.add(rec.get("run_id"))
            except json.JSONDecodeError:
                continue
    return done


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True, help="Path to queries.json")
    parser.add_argument("--output", required=True, help="Path to results.jsonl")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--thinking-appendix", action="store_true")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    args = parser.parse_args()

    queries = json.loads(Path(args.queries).read_text())
    specs = expand_matrix(
        models=args.models, modes=args.modes,
        single_turn_queries=queries["single_turn_quiz"],
        multi_turn_queries=queries["multi_turn_grade"],
        runs=args.runs,
        thinking_appendix=args.thinking_appendix,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done_run_ids(output_path)
    pending = filter_pending_specs(specs, done)
    print(f"[run_eval] total={len(specs)} done={len(done)} pending={len(pending)}")

    # Eval DB — separate from dev DB to avoid pollution
    eval_db = output_path.parent / "eval.db"
    engine = create_engine(f"sqlite:///{eval_db}", echo=False)
    Base.metadata.create_all(engine)

    local_judge = make_local_judge()
    cloud_judge = make_cloud_judge()
    print(f"[run_eval] cloud judge: {'enabled' if cloud_judge else 'disabled (no MINIMAX_API_KEY)'}")

    completed = 0
    with output_path.open("a") as out_f:
        for spec in pending:
            # Per-spec session for fresh state isolation
            with Session(engine) as session:
                planner_llm = ChatOllama(
                    model=spec.model, temperature=0.7, reasoning=spec.thinking,
                )
                user = UserRepository(session).get_or_create(f"eval-user-{spec.run_id[:6]}")
                config_extras = {
                    "quiz_master": build_quiz_master(
                        llm=planner_llm,
                        topic_repo=TopicRepository(session),
                        question_repo=QuestionRepository(session),
                        mistake_repo=MistakeRepository(session),
                        mastery_repo=MasteryRepository(session),
                        goal_repo=GoalRepository(session),
                        retriever=None,
                    ),
                    "quiz_master_agent": build_quiz_master_agent(
                        llm=planner_llm,
                        topic_repo=TopicRepository(session),
                        question_repo=QuestionRepository(session),
                        goal_repo=GoalRepository(session),
                        retriever=None,
                    ),
                    "planner": build_planner(
                        llm=planner_llm,
                        plan_repo=PlanRepository(session),
                        goal_repo=GoalRepository(session),
                        mastery_repo=MasteryRepository(session),
                        mistake_repo=MistakeRepository(session),
                        retriever=None,
                    ),
                    "memory_hydrator": build_memory_hydrator(
                        MasteryRepository(session), MistakeRepository(session),
                    ),
                    "memory_writer": build_memory_writer(
                        MasteryRepository(session), MistakeRepository(session),
                    ),
                    "judge_llm": None,  # judge disabled inside loop; eval has its own judges
                }
                graph = build_graph(checkpointer=None)
                record = await run_one(
                    spec=spec, graph=graph,
                    judge_local=local_judge, judge_cloud=cloud_judge,
                    config_extras=config_extras, user_id=user.id,
                )
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
            completed += 1
            if completed % 10 == 0:
                print(f"[run_eval] completed {completed}/{len(pending)}")

    print(f"[run_eval] done. results at {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 6: Create `queries.json`**

Create `backend/app/eval/p2_3_quiz_ablation/queries.json` (content from spec §6.4):

```json
{
  "single_turn_quiz": [
    {"id": "quiz_hyde", "message": "quiz me on HyDE"},
    {"id": "quiz_bm25", "message": "测一下 BM25"},
    {"id": "quiz_reranking", "message": "quiz me on reranking"},
    {"id": "quiz_chunking", "message": "考一下 chunking strategies"},
    {"id": "quiz_eval", "message": "quiz me on retrieval evaluation"},
    {"id": "quiz_judge", "message": "测一下 LLM-as-judge"},
    {"id": "quiz_embeddings", "message": "quiz me on embedding models"},
    {"id": "quiz_hybrid", "message": "quiz me on hybrid retrieval"},
    {"id": "quiz_hyde_zh", "message": "测我一下 HyDE"},
    {"id": "quiz_rrf", "message": "quiz me on reciprocal rank fusion"}
  ],
  "multi_turn_grade": [
    {"id": "quiz_hyde_then_grade", "messages": ["quiz me on HyDE", "A"]},
    {"id": "quiz_bm25_then_grade", "messages": ["quiz me on BM25", "A"]}
  ]
}
```

- [ ] **Step 7: Create `summarize.py` (fork from p2_2 with column labels swapped)**

Create `backend/app/eval/p2_3_quiz_ablation/summarize.py`. The simplest path: copy `backend/app/eval/p2_2_agent_ablation/summarize.py` byte-for-byte, then replace these strings:
- `plan_action` → `quiz_action`
- `milestones_persisted` → `question_persisted`
- `PLAN_DIMENSIONS` → `QUIZ_DIMENSIONS`
- Section heading labels mentioning "Plan" → "Quiz"

Run: `cd backend && cp app/eval/p2_2_agent_ablation/summarize.py app/eval/p2_3_quiz_ablation/summarize.py` and then use Edit to swap the strings.

If you prefer the explicit version, write it inline — the file is ~200 lines of Counter + tabulate logic. Reuse what works.

- [ ] **Step 8: Create the failing test file with 3 tests**

Create `backend/tests/eval/test_p2_3_harness.py`:

```python
"""Cut P2.3-②a — unit tests for the eval harness."""
import json

import pytest

from app.eval.p2_3_quiz_ablation.matrix import (
    RunSpec, expand_matrix, filter_pending_specs,
)
from app.eval.p2_3_quiz_ablation.single_run import validate_record_schema


def test_expand_matrix_main_count():
    """4 models × 2 modes × 10 single-turn × 3 runs + 2 multi-turn × 2 turns × 4 × 2 × 3 = 336."""
    specs = expand_matrix(
        models=["gemma3:4b", "qwen3.5:4b", "qwen2.5:7b", "gemma4:e4b"],
        modes=["deterministic", "agent_loop"],
        single_turn_queries=[{"id": f"q{i}", "message": f"m{i}"} for i in range(10)],
        multi_turn_queries=[
            {"id": "mt1", "messages": ["turn0", "turn1"]},
            {"id": "mt2", "messages": ["turn0", "turn1"]},
        ],
        runs=3,
        thinking_appendix=False,
    )
    # 4 × 2 × 10 × 3 = 240 single + 4 × 2 × 2 × 2 × 3 = 96 multi = 336
    assert len(specs) == 336


def test_expand_matrix_with_appendix_adds_60():
    """Appendix: gemma4:e4b thinking-ON × 2 modes × 10 single × 3 runs = 60."""
    specs = expand_matrix(
        models=["gemma3:4b", "qwen3.5:4b", "qwen2.5:7b", "gemma4:e4b"],
        modes=["deterministic", "agent_loop"],
        single_turn_queries=[{"id": f"q{i}", "message": f"m{i}"} for i in range(10)],
        multi_turn_queries=[
            {"id": "mt1", "messages": ["turn0", "turn1"]},
            {"id": "mt2", "messages": ["turn0", "turn1"]},
        ],
        runs=3,
        thinking_appendix=True,
    )
    assert len(specs) == 336 + 60


def test_validate_record_schema_rejects_missing_keys():
    record = {
        "run_id": "abc", "timestamp": "t", "model": "m", "mode": "deterministic",
        "query_id": "q", "turn_idx": 0, "run_idx": 0,
        "operational": {"wall_time_s": 1, "iterations": 0},  # missing keys
        "output": {}, "judge_local": {}, "judge_cloud": {},
    }
    with pytest.raises(ValueError, match="record.operational missing required key"):
        validate_record_schema(record)
```

- [ ] **Step 9: Run the harness tests to verify they pass**

Run: `cd backend && uv run pytest -q tests/eval/test_p2_3_harness.py`

Expected output: `3 passed`.

- [ ] **Step 10: Checkpoint — run the full backend suite**

Run: `cd backend && uv run pytest -q`

Expected output: `202 passed` (199 from Cut ①e + 3 new). The discipline reminder said "201"; the actual count may be 202-203 depending on how `test_agent_trace.py` net new tests were counted. Either is acceptable — the gate is "all green".

- [ ] **Step 11: Mark Cut ②a complete**

Eval harness is ready. Cut ②b runs the matrix.

---

## Cut P2.3-②b — Run the full matrix (manual)

**Files:** No code changes. This is a manual matrix execution producing `results.jsonl`.

**Wall time:** ~5h on 16GB Apple Silicon Mac.
**Cost:** ~$3-3.5 MiniMax-M2.7 API.

- [ ] **Step 1: Confirm pre-flight green**

```bash
cd backend && uv run pytest -q
```

Expected: 202+ passed.

```bash
ollama list | grep -E "gemma3:4b|qwen3.5:4b|qwen2.5:7b|gemma4:e4b"
```

Expected: 4 lines.

```bash
echo $MINIMAX_API_KEY | head -c 8
```

Expected: 8 chars of API key visible (or empty — cloud judge will skip).

- [ ] **Step 2: Run the matrix in resumable mode**

```bash
cd backend && uv run python -m app.eval.p2_3_quiz_ablation.run_eval \
  --queries app/eval/p2_3_quiz_ablation/queries.json \
  --output app/eval/p2_3_quiz_ablation/output/results.jsonl \
  --runs 3 \
  --thinking-appendix \
  2>&1 | tee app/eval/p2_3_quiz_ablation/output/run.log
```

Expected progress lines every 10 completions: `[run_eval] completed 10/396`, `[run_eval] completed 20/396`, etc.

If the process dies mid-way (Ollama OOM, network blip, etc.), re-run the same command. Resumability picks up where it left off via `_load_done_run_ids`.

Typical milestones:
- 0-15 min: pre-flight + first ~20 fast cells (gemma3 deterministic, gemma3 agent_loop degrade)
- 15-90 min: qwen2.5:7b and deterministic gemma4 cells
- 90-180 min: qwen3.5:4b agent_loop cells (slowest — even with reasoning=False, ~70-90s each)
- 180-280 min: gemma4:e4b agent_loop cells
- 280-300 min: appendix thinking-ON gemma4 cells

- [ ] **Step 3: Verify the output**

After completion:

```bash
wc -l backend/app/eval/p2_3_quiz_ablation/output/results.jsonl
```

Expected: ~396 lines (no harness_error should reduce count; check `[run_eval]` log for "completed 396/396").

```bash
cd backend && uv run python -c "
import json
from collections import Counter
exits = Counter()
errors = 0
with open('app/eval/p2_3_quiz_ablation/output/results.jsonl') as f:
    for line in f:
        rec = json.loads(line)
        exits[rec['operational']['exit_reason']] += 1
        if rec['operational'].get('exit_reason') == 'error':
            errors += 1
print(exits)
print(f'harness_error rows: {errors}')
"
```

Expected output (approximate, based on P2.2 pattern):
- `deterministic`: ~210 (all deterministic-mode runs)
- `natural_stop`: ~140 (most agent_loop GENERATE successes)
- `llm_call_failed`: ~42 (all gemma3:4b agent_loop runs)
- `budget_exhausted`: ≤ 5 (edge cases)
- `harness_error rows: 0` (or very few — investigate any)

- [ ] **Step 4: Run the summarizer to generate a markdown table**

```bash
cd backend && uv run python -m app.eval.p2_3_quiz_ablation.summarize \
  app/eval/p2_3_quiz_ablation/output/results.jsonl \
  > app/eval/p2_3_quiz_ablation/output/summary.md
```

Open `summary.md` and confirm it contains tables for: latency / exit_reason distribution / tool calling / persistence / judge scores.

- [ ] **Step 5: Sanity-check at least 3 cells visually**

For each of these 3 cells, manually read 2-3 results.jsonl rows:
- `gemma3:4b agent_loop` — every row should have `exit_reason: "llm_call_failed"` with `llm_error` containing "400" or "does not support tools".
- `qwen3.5:4b deterministic` (thinking-OFF) — verify persistence rate; if much higher than P2.2's 10%, the LLM may be emitting valid JSON for Quiz where it didn't for Plan (different finding).
- `gemma4:e4b agent_loop` — verify natural_stop rate and persistence rate near 100% (would replicate P2.2 champion finding).

- [ ] **Step 6: Mark Cut ②b complete**

Log the actual numbers (exit_reason counts, total records, wall time, MiniMax cost) at the top of Cut ③ for the writeup.

---

## Cut P2.3-③ — EVAL append + blog continuation + memory + ROADMAP

**Files:**
- Modify: `study-coach/docs/EVAL.md` (append P2.3 section)
- Modify or Create: `study-coach/docs/agent_loop_vs_deterministic.md` (append section) OR `study-coach/docs/quiz_ablation_followup.md` (new sibling blog)
- Modify: `study-coach/docs/ROADMAP.md` (P2.3 → shipped)
- Modify: `/Users/lianghaozhe/.claude/projects/-Users-lianghaozhe-Downloads-Study-Compaion-and-JadeAI/memory/project_study_coach_refactor.md` (append P2.3 segment)

**This cut has NO TDD steps — it's writeup. Discipline: data-first, narrative-second. Numbers come from results.jsonl, not from intuition.**

- [ ] **Step 1: Append "P2.3 Quiz Ablation" section to EVAL.md**

The section should mirror P2.2's 8-subsection structure but reference back to P2.2 for shared methodology. Required subsections:

1. **TL;DR** — 4-6 bullets answering the 3 predictions from spec §12 with actual numbers.
2. **Setup** — short, link to P2.2 setup, note Quiz-specific deltas (2-tool surface, max_iter=6).
3. **Results table 1: Latency** — same shape as P2.2 §1, 8 cells.
4. **Results table 2: Robustness (exit_reason distribution)** — confirm gemma3 collapse repeats or doesn't.
5. **Results table 3: Tool calling correctness** — for agent_loop cells, expected pattern is `[retriever_search, persist_quiz_question]` (~2 tool calls/run). Note error rates.
6. **Results table 4: Quiz quality (Local judge qwen2.5:7b)** — full 8-cell table.
7. **Results table 5: Quiz quality (Cloud judge MiniMax-M2.7)** — full 8-cell table.
8. **Results table 6: Judge agreement** — same `|local − cloud|` analysis.
9. **Results table 7: Persistence — the schema rescue test** — most important table for prediction P2. Compare:
   - qwen3.5:4b deterministic persistence rate (Quiz) vs P2.2 (Plan 10%)
   - qwen3.5:4b agent_loop persistence rate (Quiz) vs P2.2 (Plan 86%)
   - gemma4:e4b same comparison
10. **Findings (3 numbered)** — directly answer P1, P2, P3 from spec §12.
11. **Limitations** — link to P2.2 limitations (statistical power, judge bias). Add Quiz-specific ones (multi-turn GRADE quality not separately analyzed).

Write directly into the existing `study-coach/docs/EVAL.md` under a new top-level heading `# P2.3 Quiz Agent Loop Ablation — Empirical Report`. Use the same date stamp format as P2.2.

- [ ] **Step 2: Append continuation to blog OR create sister blog**

Decision rule based on length:
- If your EVAL.md section + commentary together would fit in a 600-word blog continuation → append to `study-coach/docs/agent_loop_vs_deterministic.md` as a new section `## P2.3 update: does persistence rescue replicate on Quiz?`.
- If it needs > 600 words to do justice to a stand-alone narrative → create `study-coach/docs/quiz_ablation_followup.md` and cross-link.

Blog content checklist (whichever path):
- Open by referencing the 3 predictions from the original blog's `What I would do next`.
- For each prediction, state actual data + interpretation (1-2 paragraphs each).
- Address "schema strictness vs tool count" — clarify the disambiguation of "simpler" (P3 prediction).
- Close with a synthesis: does the agent_loop pattern's value transfer? Conditional on what?
- Reproduce footer (matrix command) like P2.2.

- [ ] **Step 3: Update ROADMAP.md**

Move the P2.3 candidate block to "shipped" status. Add results summary:

```markdown
#### P2.3 Quiz Agent Loop Ablation (Done, 2026-MM-DD)

**202+ backend tests passing** (181 baseline + 20+ new). 5 implementation cuts (①a-①e) + 1 smoke (①f) + 2 eval cuts (②a-②b) + 1 writeup (③).

[Headline findings: copy 5-bullet from EVAL.md TL;DR]

**Spec / Plan / Report docs**:
- Spec: docs/superpowers/specs/2026-05-24-p2-3-quiz-agent-loop-ablation-design.md
- Plan: docs/superpowers/plans/2026-05-24-p2-3-quiz-agent-loop-ablation.md
- Report: docs/EVAL.md (appended section)
- Blog: docs/agent_loop_vs_deterministic.md (appended) OR docs/quiz_ablation_followup.md
- Raw data: backend/app/eval/p2_3_quiz_ablation/output/results.jsonl
```

- [ ] **Step 4: Update memory file**

Edit `/Users/lianghaozhe/.claude/projects/-Users-lianghaozhe-Downloads-Study-Compaion-and-JadeAI/memory/project_study_coach_refactor.md`.

Append a `**Phase 2.3 (shipped 2026-MM-DD): Quiz Agent Loop Ablation.**` block in the same shape as the existing P2.2 block. Include:
- Total tests passing (e.g. "202 backend tests passing (+21 over P2.2's 181)")
- Cuts shipped (8 total: 5 impl + 1 smoke + 2 eval + 1 writeup)
- 3 lessons learned (whatever surfaced — refactor cleanliness, schema-rescue replication result, multi-turn dispatcher coherence)
- Cross-links to spec/plan/report

- [ ] **Step 5: Final regression check**

Run: `cd backend && uv run pytest -q`

Expected: 202+ passed (no documentation changes should affect tests; this is a paranoia check).

- [ ] **Step 6: Mark Cut ③ complete and P2.3 ship-state**

P2.3 is shipped. Backend test count is final. EVAL/blog/ROADMAP/memory all updated. Raw data preserved in `results.jsonl`.

---

## Verification Criteria (from spec §9)

- [ ] 202+ tests passing (181 baseline + 20+ new across cuts ①a/①b/①c/①d/①e/②a)
- [ ] P2.2's 3 AgentTrace tests pass byte-identical after Cut ①a refactor (import-path-only change)
- [ ] 4 models × 2 modes pass real Ollama smoke test in Cut ①f (one happy-path GENERATE each + degrade verified for gemma3:4b agent_loop)
- [ ] Eval matrix ~396 records complete, `results.jsonl` persisted with 0 harness_error (or documented few)
- [ ] `docs/EVAL.md` has new section "P2.3 Quiz Ablation" with table + finding addressing all 3 blog predictions
- [ ] Blog continuation (appended or sister file) explicitly answers the 3 predictions from `docs/agent_loop_vs_deterministic.md`
- [ ] Memory `project_study_coach_refactor.md` has P2.3 segment
- [ ] ROADMAP.md updated — P2.3 from candidate to shipped
- [ ] `# cloud-adapt:` hooks grep-able in `agent_trace.py`, `quiz_master_agent.py`, `eval/p2_3_quiz_ablation/judges.py`, `deps.py::get_quiz_master_agent`

---

## Self-Review

Spec coverage check:
- §1 Q1 (agent_loop scope) → Cut ①d dispatcher logic enforces it. ✓
- §1 Q2 (tool surface) → Cut ①b. ✓
- §1 Q3 (matrix shape) → Cut ②a matrix.py + queries.json. ✓
- §1 Q4 (judges) → Cut ②a judges.py. ✓
- §1 Q5 (thinking appendix) → Cut ②a matrix.py + Cut ②b CLI `--thinking-appendix`. ✓
- §1 Approach A → all of Cuts ①a-①e. ✓
- §2 Architecture & topology → Cuts ①a/①d. ✓
- §3 Tool wrappers & schemas → Cut ①b. ✓
- §4 Loop control flow → Cut ①c. ✓
- §5 State/deps/routes wiring → Cuts ①d/①e. ✓
- §6 Eval harness → Cut ②a. ✓
- §7 Testing strategy → distributed across cuts. ✓
- §8 Cut skeleton → this plan's cut structure. ✓
- §9 Verification criteria → above. ✓
- §10 Out of scope → respected (no GPT-4o-mini, no mastery read tools, no agent_loop on GRADE). ✓
- §11 Cloud-adapt hooks → markers placed in all new files. ✓
- §12 Predictions to test → Cut ③ Step 1 EVAL.md section + Step 2 blog. ✓

Placeholder scan: zero TBD / TODO / "fill in later" markers in code blocks. Manual cuts (①f / ②b / ③) describe explicit checklists with commands and expected outputs.

Type consistency check:
- `_make_quiz_tools(...)` signature matches across Cut ①b code block, Cut ①c usage, and Cut ①e factory wiring. ✓
- `build_quiz_master_agent(...)` factory kwargs (llm / topic_repo / question_repo / goal_repo / retriever / now_fn / max_iter / system_prompt) consistent across ①c body, ①e deps wiring, ②a run_eval wiring. ✓
- `QuizQuestionPersist` field names (topic / prompt / options / answer / explanation) consistent across schemas.py, tool wrapper, test fixtures, run_eval queries. ✓
- `last_persisted_question_id` method name consistent across `agent_trace.py`, `quiz_master_agent.py` consumer, and Cut ①a test. ✓
- `quiz_mode` configurable key consistent across `graph.py`, `routes.py`, `deps.py`, `run_eval.py`. ✓

Plan ready. Begin Cut ①a.
