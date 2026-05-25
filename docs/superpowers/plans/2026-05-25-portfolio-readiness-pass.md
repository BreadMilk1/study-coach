# Portfolio Readiness Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Study Coach ready for resume, interview, and portfolio review by adding reviewer-facing architecture docs, demo readiness, deploy hardening, and an Agent visibility UI without expanding into broad P4 product work.

**Architecture:** Keep the current FastAPI + LangGraph + Vue architecture. Add portfolio-safe custom SSE events around existing graph state and `AgentTrace`, then render those events in a compact frontend panel shared by Chat / Plan / Quiz. Keep demo/deploy artifacts explicit and local-first; private course PDFs stay outside the repo.

**Tech Stack:** FastAPI `StreamingResponse`, LangGraph `stream_mode="custom"`, Python 3.11, Vue 3 + Pinia + Vite, Tailwind 4, Docker Compose, uv, pnpm.

**Spec:** `docs/superpowers/specs/2026-05-25-portfolio-readiness-pass-design.md`

**Git rule:** Project instructions override generic plan guidance. Do not commit unless the user explicitly approves. Because this repo has no baseline commit yet, Task 0 must be completed before implementation begins.

---

## File Structure

Planned new files:

- `.env.example` — root-level local/Compose environment template.
- `compose.yaml` — local production-like backend/frontend services with persistent volumes.
- `docs/DEMO.md` — reviewer demo path using a user-owned PDF path.
- `backend/.env.example` — backend-specific environment template.
- `backend/scripts/readiness_check.py` — non-destructive local readiness checker.
- `backend/tests/scripts/test_readiness_check.py` — unit tests for readiness checks.
- `frontend/src/stores/agentRun.ts` — Pinia store for additive agent visibility events.
- `frontend/src/components/AgentRunPanel.vue` — compact portfolio-safe trace panel.

Planned modified files:

- `README.md` — link architecture/demo/deploy/readiness assets.
- `docs/ARCHITECTURE.md` — replace contract-level doc with v2 reviewer-facing architecture.
- `docs/ROADMAP.md` — mark Portfolio Readiness Pass as the active P4 slice.
- `backend/app/api/routes.py` — emit request-level `agent_trace` SSE after graph completion.
- `backend/app/agent/graph.py` — emit `agent_step` / `judge` events from existing nodes.
- `backend/app/agent/agent_trace.py` — add redacted portfolio serialization helper.
- `backend/tests/agent/test_agent_trace.py` — verify redaction helper.
- `backend/tests/agent/test_graph_judge.py` — verify judge SSE event shape.
- `backend/tests/api/test_routes_graph_stream.py` — verify additive SSE events do not break `citations/token/done`.
- `frontend/src/lib/api.ts` — parse new SSE events and forward to callbacks.
- `frontend/src/stores/chat.ts` — attach latest agent run to assistant messages.
- `frontend/src/views/Chat.vue` — render `AgentRunPanel` for assistant messages.
- `frontend/src/views/PlanTimeline.vue` — reset/update run store during check-in and render panel.
- `frontend/src/views/QuizAdaptive.vue` — reset/update run store during quiz streams and render panel.

Do not touch:

- mobile layout.
- auth/OAuth.
- i18n.
- shared plans/public links.
- group study.
- drag reorder / Gantt / activity heatmap.
- private course PDFs or paths outside `study-coach/`.

---

## Task 0: Git Baseline Gate

**Purpose:** Decide whether implementation runs in an isolated worktree. This is not a code task.

**Files:**

- Read: `.gitignore`
- Read: `AGENTS.md`
- No planned writes unless user approves baseline commit / worktree setup.

- [ ] **Step 1: Check git state**

Run:

```bash
git status --short --branch
git worktree list
```

Expected current state before implementation:

```text
## No commits yet on main
?? .gitignore
?? AGENTS.md
...
```

- [ ] **Step 2: Ask user for git execution mode**

Ask:

```text
Implementation is ready to start, but this repo has no baseline commit.

Choose one:
1. Approve a baseline commit, then use superpowers:using-git-worktrees for feat/portfolio-readiness-pass.
2. Do not commit; implement in the current workspace and review by diff/status only.

Which mode?
```

- [ ] **Step 3A: If user chooses baseline commit + worktree**

Use `superpowers:using-git-worktrees`.

Commands to run after approval:

```bash
git add .gitignore AGENTS.md LEARN-CLAUDE-CODE-README-zh.md README.md backend design-system docs frontend
git commit -m "chore: establish study coach baseline"
```

Then prefer a global worktree to avoid project-local `.worktrees/` ignore churn:

```bash
mkdir -p ~/.config/superpowers/worktrees/study-coach
git worktree add ~/.config/superpowers/worktrees/study-coach/feat-portfolio-readiness-pass -b feat/portfolio-readiness-pass
```

Baseline verification in the worktree:

```bash
cd ~/.config/superpowers/worktrees/study-coach/feat-portfolio-readiness-pass/backend
uv run pytest -q
cd ../frontend
pnpm build
```

Expected:

- Backend tests pass.
- Frontend build passes.
- If either fails, stop and ask whether to investigate baseline failure before feature work.

- [ ] **Step 3B: If user chooses current workspace**

Record in final implementation notes:

```text
No baseline commit/worktree was used at user request. Reviews are diff/status based, not commit-range based.
```

Proceed in current workspace.

---

## Task 1: Reviewer-Facing Architecture v2

**Purpose:** Make the project understandable as an Agent engineering portfolio artifact without requiring source-code reading.

**Files:**

- Modify: `docs/ARCHITECTURE.md`
- Modify: `README.md`
- Modify: `docs/ROADMAP.md`

- [ ] **Step 1: Replace `docs/ARCHITECTURE.md` with v2 structure**

Use these sections exactly:

```markdown
# Study Coach — Architecture v2

> Portfolio-facing architecture for the Exam Coach Agent.

## 1. Elevator Pitch
## 2. System Topology
## 3. Agent Graph
## 4. Tool Registry
## 5. Data Model
## 6. Runtime Flows
## 7. Agent Visibility And Trace Safety
## 8. Eval-To-Product Decisions
## 9. Failure Boundaries
## 10. Deployment Boundary
## 11. References
```

Required diagram content for `## 2. System Topology`:

```text
Vue 3 SPA
  -> FastAPI /api
    -> LangGraph graph
      -> Tutor / Planner / QuizMaster
      -> Judge Guard
      -> Memory Writer
    -> SQLAlchemy repositories -> SQLite/Postgres
    -> Chroma collection
    -> LLM provider via BYOK headers or Ollama
```

Required graph content for `## 3. Agent Graph`:

```text
memory_hydrator
  -> router
    -> tutor -> judge
    -> planner -> judge
    -> quiz -> judge
  -> memory_writer
```

Required links in `## 11. References`:

```markdown
- [Eval report](./EVAL.md)
- [Plan agent-loop ablation](./agent_loop_vs_deterministic.md)
- [Quiz ablation follow-up](./quiz_ablation_followup.md)
- [P3 frontend productization](./p3_frontend_productize.md)
- [Demo guide](./DEMO.md)
```

- [ ] **Step 2: Add failure boundary table**

Include this table in `docs/ARCHITECTURE.md`:

```markdown
| Failure | Boundary | User-visible behavior | Portfolio point |
|---|---|---|---|
| No corpus / empty retrieval | Retriever + Quiz UI | EmptyCorpusBanner or refusal-style guidance | Agent loop can refuse unsupported generation |
| Same-model judge | API dependency + UI metadata | bias warning / same_model=true | judge bias is surfaced, not hidden |
| Model lacks tool calling | Planner/Quiz agent loop | degraded answer + exit_reason=llm_call_failed | agent loop capability is provider-dependent |
| Schema validation failure | Pydantic tool schema | tool error goes back to model for self-correction | schema is part of the harness |
| Ollama unavailable | provider/readiness | readiness warning or degraded LLM call | local-first has explicit operational boundary |
```

- [ ] **Step 3: Add README links**

In `README.md`, add a short "Portfolio Review Path" after `What This Project Demonstrates`:

```markdown
## Portfolio Review Path

For reviewers, read these in order:

1. `docs/ARCHITECTURE.md` — system and agent graph.
2. `docs/DEMO.md` — local demo path with your own PDF.
3. `docs/EVAL.md` — retrieval and agent-loop evidence.
4. `docs/agent_loop_vs_deterministic.md` and `docs/quiz_ablation_followup.md` — portfolio writeups.
```

- [ ] **Step 4: Update roadmap**

In `docs/ROADMAP.md`, under P4, add:

```markdown
### P4.1 — Portfolio Readiness Pass

- [x] Scope locked in `docs/superpowers/specs/2026-05-25-portfolio-readiness-pass-design.md`
- [ ] `ARCHITECTURE.md v2`
- [ ] demo readiness
- [ ] Agent visibility UI
- [ ] deploy hardening
- [ ] final review gates
```

Do not mark implementation boxes done until corresponding tasks are complete.

- [ ] **Step 5: Verify docs**

Run:

```bash
rg -n "ARCHITECTURE.md|DEMO.md|EVAL.md|agent_loop_vs_deterministic.md|quiz_ablation_followup.md" README.md docs/ARCHITECTURE.md docs/ROADMAP.md
rg -n "course-corpus|private-course-pdf|local-demo-corpus" README.md docs || true
```

Expected:

- First command finds the intended references.
- Second command prints no matches.

- [ ] **Step 6: Self-review**

Check:

- Architecture doc answers "why this is an Agent project".
- No private corpus path or specific course PDF filename appears.
- No non-goal product feature is added to scope.

---

## Task 2: Demo Readiness And Deploy Hardening

**Purpose:** Let a reviewer run the project locally with their own PDF and understand how it would run in a local production-like environment.

**Files:**

- Create: `.env.example`
- Create: `backend/.env.example`
- Create: `compose.yaml`
- Create: `docs/DEMO.md`
- Create: `backend/scripts/readiness_check.py`
- Create: `backend/tests/scripts/test_readiness_check.py`
- Modify: `README.md`
- Modify: `docs/ROADMAP.md`

- [ ] **Step 1: Add root `.env.example`**

Create `.env.example`:

```dotenv
# Study Coach local/Compose defaults
# Copy to .env for local Compose usage. Do not commit .env.

BACKEND_PORT=8000
FRONTEND_PORT=5173

DATABASE_URL=sqlite:///./data/study_coach.db
CHROMA_PATH=./data/chroma

OLLAMA_HOST=http://host.docker.internal:11434
EMBED_MODEL=nomic-embed-text

CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

- [ ] **Step 2: Add backend `.env.example`**

Create `backend/.env.example`:

```dotenv
# Backend-only local development defaults
# Copy to backend/.env if your shell tooling loads env files.

DATABASE_URL=sqlite:///./study_coach.db
CHROMA_PATH=./chroma_data
OLLAMA_HOST=http://localhost:11434
EMBED_MODEL=nomic-embed-text
CORS_ORIGINS=http://localhost:5173
STUDY_COACH_TEST_MODE=0
```

- [ ] **Step 3: Add `compose.yaml`**

Create `compose.yaml`:

```yaml
services:
  backend:
    image: python:3.11-slim
    working_dir: /app/backend
    command: >
      sh -lc "pip install uv &&
              uv sync &&
              uv run uvicorn app.main:app --host 0.0.0.0 --port 8000"
    env_file:
      - .env
    environment:
      DATABASE_URL: ${DATABASE_URL:-sqlite:///./data/study_coach.db}
      CHROMA_PATH: ${CHROMA_PATH:-./data/chroma}
      OLLAMA_HOST: ${OLLAMA_HOST:-http://host.docker.internal:11434}
      EMBED_MODEL: ${EMBED_MODEL:-nomic-embed-text}
      CORS_ORIGINS: ${CORS_ORIGINS:-http://localhost:5173}
    volumes:
      - ./backend:/app/backend
      - study-coach-data:/app/backend/data
    ports:
      - "${BACKEND_PORT:-8000}:8000"

  frontend:
    image: node:22-slim
    working_dir: /app/frontend
    command: sh -lc "corepack enable && pnpm install && pnpm dev --host 0.0.0.0"
    volumes:
      - ./frontend:/app/frontend
      - frontend-node-modules:/app/frontend/node_modules
    ports:
      - "${FRONTEND_PORT:-5173}:5173"
    depends_on:
      - backend

volumes:
  study-coach-data:
  frontend-node-modules:
```

Note: if Compose errors because `.env` is missing, document `cp .env.example .env` in `docs/DEMO.md`. Do not commit `.env`.

- [ ] **Step 4: Add readiness checker**

Create `backend/scripts/readiness_check.py`:

```python
from __future__ import annotations

import argparse
import importlib
import os
import socket
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse


def check_imports() -> tuple[bool, str]:
    for name in ("fastapi", "chromadb", "langgraph", "sqlalchemy", "app.main"):
        importlib.import_module(name)
    return True, "python imports ok"


def check_writable_path(path: Path) -> tuple[bool, str]:
    path.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path, delete=True):
        pass
    return True, f"writable: {path}"


def check_ollama(host: str) -> tuple[bool, str]:
    parsed = urlparse(host)
    target_host = parsed.hostname or "localhost"
    target_port = parsed.port or 11434
    try:
        with socket.create_connection((target_host, target_port), timeout=1.5):
            return True, f"ollama reachable: {host}"
    except OSError as exc:
        return False, f"ollama not reachable at {host}: {exc}"


def check_demo_pdf(path: str | None) -> tuple[bool, str]:
    if not path:
        return True, "demo pdf not provided; upload your own PDF in the UI"
    p = Path(path).expanduser()
    if p.exists() and p.suffix.lower() == ".pdf":
        return True, f"demo pdf found: {p}"
    return False, f"demo pdf missing or not a PDF: {p}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-pdf", default=None)
    parser.add_argument("--strict-ollama", action="store_true")
    args = parser.parse_args(argv)

    checks: list[tuple[str, bool, str, bool]] = []

    ok, msg = check_imports()
    checks.append(("imports", ok, msg, True))

    chroma_path = Path(os.environ.get("CHROMA_PATH", "./chroma_data"))
    ok, msg = check_writable_path(chroma_path)
    checks.append(("chroma_path", ok, msg, True))

    ok, msg = check_demo_pdf(args.demo_pdf)
    checks.append(("demo_pdf", ok, msg, False))

    provider = os.environ.get("X_PROVIDER", os.environ.get("PROVIDER", "ollama"))
    if provider == "ollama":
        ok, msg = check_ollama(os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
        checks.append(("ollama", ok, msg, args.strict_ollama))

    failed_required = False
    for name, ok, msg, required in checks:
        marker = "PASS" if ok else ("FAIL" if required else "WARN")
        print(f"[{marker}] {name}: {msg}")
        if required and not ok:
            failed_required = True

    return 1 if failed_required else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Add readiness tests**

Create `backend/tests/scripts/test_readiness_check.py`:

```python
from pathlib import Path

from scripts import readiness_check


def test_check_demo_pdf_accepts_missing_optional_path():
    ok, msg = readiness_check.check_demo_pdf(None)
    assert ok is True
    assert "upload your own PDF" in msg


def test_check_demo_pdf_rejects_non_pdf(tmp_path: Path):
    f = tmp_path / "notes.txt"
    f.write_text("x")
    ok, msg = readiness_check.check_demo_pdf(str(f))
    assert ok is False
    assert "not a PDF" in msg


def test_check_writable_path_creates_directory(tmp_path: Path):
    target = tmp_path / "chroma"
    ok, msg = readiness_check.check_writable_path(target)
    assert ok is True
    assert target.exists()
    assert "writable" in msg
```

- [ ] **Step 6: Add demo guide**

Create `docs/DEMO.md` with these sections:

```markdown
# Study Coach Demo Guide

## Data Boundary
Use a PDF you own or are allowed to process locally. The author may validate with private course PDFs outside this repository; those files are not part of the repo and are not required for reviewers.

## Local Dev Demo
1. Start Ollama.
2. Start backend.
3. Start frontend.
4. Open Library and upload your PDF.
5. Ask a tutor question.
6. Generate a plan.
7. Generate a quiz.
8. Switch deterministic / agent_loop and compare behavior.
9. Inspect the Agent Run panel.

## Readiness Check
Run `cd backend && uv run python scripts/readiness_check.py --demo-pdf /path/to/your.pdf`.

## Compose Demo
Run `cp .env.example .env`, then `docker compose up`.

## Expected Interview Talking Points
- The agent graph routes intent before generation.
- Planner and Quiz can run deterministic or agent_loop paths.
- Judge Guard scores outputs and exposes weak dimensions.
- Tool traces are redacted to avoid leaking source text.
```

- [ ] **Step 7: Update README**

Add a short "Demo readiness" paragraph:

```markdown
For a reviewer walkthrough, see `docs/DEMO.md`. Run `backend/scripts/readiness_check.py` before the demo to check imports, Chroma path, optional PDF path, and Ollama reachability.
```

- [ ] **Step 8: Run verification**

Run:

```bash
cd backend
uv run pytest tests/scripts/test_readiness_check.py -q
uv run python scripts/readiness_check.py
cd ..
rg -n "private course PDFs|/path/to/your.pdf|readiness_check" docs/DEMO.md README.md
rg -n "course-corpus|private-course-pdf|local-demo-corpus" README.md docs .env.example backend/.env.example compose.yaml backend/scripts/readiness_check.py || true
```

Expected:

- Readiness tests pass.
- `readiness_check.py` exits `0` when Ollama is missing unless `--strict-ollama` is provided.
- Private PDF scan prints no matches.

- [ ] **Step 9: Self-review**

Check:

- `.env.example` has no secrets.
- Compose does not assume public deployment.
- Demo guide does not require private course files.
- Roadmap P4.1 demo/deploy boxes are still unchecked unless implementation completed.

---

## Task 3: Backend Agent Visibility Events

**Purpose:** Emit additive, portfolio-safe SSE events without breaking existing `citations/token/done` consumers.

**Files:**

- Modify: `backend/app/agent/agent_trace.py`
- Modify: `backend/app/agent/graph.py`
- Modify: `backend/app/api/routes.py`
- Modify: `backend/tests/agent/test_agent_trace.py`
- Modify: `backend/tests/agent/test_graph_judge.py`
- Modify: `backend/tests/api/test_routes_graph_stream.py`

- [ ] **Step 1: Add redacted trace serialization test**

Append to `backend/tests/agent/test_agent_trace.py`:

```python
from app.agent.agent_trace import AgentTrace


def test_portfolio_summary_redacts_tool_outputs():
    trace = AgentTrace(t_start=0.0)
    trace.record_tool_call(
        "retriever_search",
        {"query": "HyDE"},
        '[{"content":"PRIVATE SOURCE TEXT","page":1}]',
        error=False,
    )

    summary = trace.to_portfolio_summary(mode="agent_loop", intent="quiz")

    assert summary["type"] == "agent_trace"
    assert summary["mode"] == "agent_loop"
    assert summary["intent"] == "quiz"
    assert summary["tool_call_breakdown"] == {"retriever_search": 1}
    assert "PRIVATE SOURCE TEXT" not in str(summary)
```

Expected failure before implementation:

```text
AttributeError: 'AgentTrace' object has no attribute 'to_portfolio_summary'
```

- [ ] **Step 2: Implement redacted helper**

In `backend/app/agent/agent_trace.py`, add:

```python
    def to_portfolio_summary(self, *, mode: str, intent: str | None = None) -> dict:
        """Redacted SSE-safe summary for the portfolio UI.

        Unlike serialize(), this deliberately excludes tool args, tool outputs,
        token counts, and raw LLM errors because those can contain private
        source text or provider details.
        """
        return {
            "type": "agent_trace",
            "mode": mode,
            "intent": intent,
            "iterations": len(self.iterations),
            "tool_call_breakdown": dict(Counter(tc.name for tc in self.tool_calls)),
            "tool_errors": sum(1 for tc in self.tool_calls if tc.error),
            "exit_reason": self.exit_reason,
        }
```

- [ ] **Step 3: Add judge event test**

Append to `backend/tests/agent/test_graph_judge.py`:

```python
@pytest.mark.asyncio
async def test_judge_emits_portfolio_safe_judge_event():
    retriever = StubRetriever(_CHUNKS)
    llm = RecordingTutorLLM(["HyDE is a query rewriting technique."])
    judge_llm = StubJudgeLLM([_PASS])
    graph = build_graph(retriever=retriever, llm=llm)

    events = []
    async for event in graph.astream(
        {"messages": [HumanMessage(content="What is HyDE?")]},
        config={"configurable": {"judge_llm": judge_llm, "same_model": True}},
        stream_mode="custom",
    ):
        events.append(event)

    judge_events = [e for e in events if e.get("type") == "judge"]
    assert judge_events
    assert judge_events[-1]["score"] >= 0.6
    assert judge_events[-1]["same_model"] is True
    assert "reasoning" in judge_events[-1]
```

- [ ] **Step 4: Emit `agent_step` and `judge` events**

In `backend/app/agent/graph.py`:

- In `router_node`, call `writer = get_stream_writer()` via safe helper is not available in sync node. Add a local helper:

```python
def _safe_writer():
    try:
        return get_stream_writer()
    except RuntimeError:
        return lambda _payload: None
```

- In `router_node`, emit:

```python
writer = _safe_writer()
...
writer({"type": "agent_step", "node": "router", "action": "end", "label": base_intent})
```

- In `tutor_node`, before retrieval:

```python
writer({"type": "agent_step", "node": "tutor", "action": "start", "label": "RAG tutor"})
```

- In `quiz_node`, emit:

```python
_safe_writer()({
    "type": "agent_step",
    "node": "quiz",
    "action": "start",
    "label": mode if not state.get("active_quiz_question_id") else "grade",
})
```

- In `plan_node`, emit:

```python
_safe_writer()({
    "type": "agent_step",
    "node": "plan",
    "action": "start",
    "label": mode,
})
```

- In `judge_node`, after `result = await judge_response(...)`, emit:

```python
writer({
    "type": "judge",
    "score": result["score"],
    "weak_dims": result["weak_dims"],
    "reasoning": result["reasoning"],
    "same_model": bool(configurable.get("same_model", False)),
})
```

- [ ] **Step 5: Pass `same_model` into graph config**

In `backend/app/api/routes.py`, inside `config["configurable"]`, add:

```python
"same_model": judge["same_model"],
```

- [ ] **Step 6: Emit final `agent_trace` in route**

In `backend/app/api/routes.py`, capture the final graph state from `astream`.

Implementation pattern:

```python
final_state = None
async for chunk in graph.astream(input_state, stream_mode="custom", config=config):
    final_state = chunk if isinstance(chunk, dict) and "messages" in chunk else final_state
    yield _sse(chunk)
...
yield _sse({"type": "done"})
```

If `stream_mode="custom"` does not expose final state, do not guess. Instead, emit mode/intent events from graph nodes and leave `agent_trace` emission for nodes that have access to `agent_trace`.

Preferred safer pattern: emit `agent_trace` from `plan_node` / `quiz_node` right after awaiting the agent result:

```python
result = await agent(state)
trace = result.get("agent_trace")
if trace:
    writer = _safe_writer()
    writer({
        "type": "agent_trace",
        "mode": "agent_loop",
        "intent": "plan",
        "iterations": trace.get("total_iterations", 0),
        "tool_call_breakdown": trace.get("tool_call_breakdown", {}),
        "exit_reason": trace.get("exit_reason"),
    })
return result
```

Use this safer node-level pattern unless route tests prove route-level final-state capture is reliable.

- [ ] **Step 7: Add route SSE test**

Append to `backend/tests/api/test_routes_graph_stream.py`:

```python
def test_chat_sse_includes_additive_agent_visibility_events(client):
    with client.stream(
        "POST",
        "/api/chat",
        json={"message": "What is HyDE?"},
        headers=_HEADERS,
    ) as resp:
        assert resp.status_code == 200
        events = _read_sse_events(resp)

    types = [e["type"] for e in events]
    assert "agent_step" in types
    assert "judge" in types
    assert types[0] in {"agent_step", "citations"}
    assert "citations" in types
    assert "token" in types
    assert types[-1] == "done"
```

- [ ] **Step 8: Run backend tests**

Run:

```bash
cd backend
uv run pytest tests/agent/test_agent_trace.py tests/agent/test_graph_judge.py tests/api/test_routes_graph_stream.py -q
```

Expected:

- All selected tests pass.
- Existing `citations/token/done` assertions remain valid or are updated only to allow additive events before `citations`.

- [ ] **Step 9: Self-review**

Check:

- No raw tool output is emitted in `agent_trace`.
- `citations/token/done` still exist.
- Same-model warning is structured in `judge.same_model`.
- No unrelated graph refactor.

---

## Task 4: Frontend Agent Visibility UI

**Purpose:** Render additive agent events in a compact, accessible panel that supports interview explanation.

**Files:**

- Create: `frontend/src/stores/agentRun.ts`
- Create: `frontend/src/components/AgentRunPanel.vue`
- Modify: `frontend/src/lib/api.ts`
- Modify: `frontend/src/stores/chat.ts`
- Modify: `frontend/src/views/Chat.vue`
- Modify: `frontend/src/views/PlanTimeline.vue`
- Modify: `frontend/src/views/QuizAdaptive.vue`

- [ ] **Step 1: Add agent run store**

Create `frontend/src/stores/agentRun.ts`:

```ts
import { defineStore } from 'pinia'

export type AgentMode = 'agent_loop' | 'deterministic'
export type AgentIntent = 'tutor' | 'quiz' | 'plan'

export interface AgentStepEvent {
  type: 'agent_step'
  node: string
  action: 'start' | 'end'
  label?: string
}

export interface JudgeEvent {
  type: 'judge'
  score: number
  weak_dims: string[]
  reasoning?: string
  same_model?: boolean
}

export interface AgentTraceEvent {
  type: 'agent_trace'
  mode: AgentMode
  intent?: AgentIntent
  iterations?: number
  tool_call_breakdown?: Record<string, number>
  tool_errors?: number
  exit_reason?: string
}

export type AgentRunEvent = AgentStepEvent | JudgeEvent | AgentTraceEvent

interface AgentRunState {
  events: AgentRunEvent[]
}

export const useAgentRun = defineStore('agentRun', {
  state: (): AgentRunState => ({ events: [] }),
  getters: {
    latestJudge: (state): JudgeEvent | null =>
      [...state.events].reverse().find((e): e is JudgeEvent => e.type === 'judge') ?? null,
    latestTrace: (state): AgentTraceEvent | null =>
      [...state.events].reverse().find((e): e is AgentTraceEvent => e.type === 'agent_trace') ?? null,
    steps: (state): AgentStepEvent[] =>
      state.events.filter((e): e is AgentStepEvent => e.type === 'agent_step'),
  },
  actions: {
    reset() { this.events = [] },
    record(event: AgentRunEvent) { this.events.push(event) },
  },
})
```

- [ ] **Step 2: Extend API callbacks**

In `frontend/src/lib/api.ts`, add imports/types:

```ts
import type { AgentRunEvent } from '../stores/agentRun'
```

Extend `ChatStreamCallbacks`:

```ts
onAgentEvent?: (event: AgentRunEvent) => void
```

In event parsing:

```ts
else if (
  event.type === 'agent_step' ||
  event.type === 'judge' ||
  event.type === 'agent_trace'
) cb.onAgentEvent?.(event as AgentRunEvent)
```

- [ ] **Step 3: Attach run metadata to chat messages**

In `frontend/src/stores/chat.ts`:

```ts
import type { AgentRunEvent } from './agentRun'
```

Extend `Message`:

```ts
agentEvents?: AgentRunEvent[]
```

Initialize in `startAssistant()`:

```ts
agentEvents: [],
```

Add action:

```ts
recordAgentEvent(m: Message, event: AgentRunEvent) {
  if (!m.agentEvents) m.agentEvents = []
  m.agentEvents.push(event)
}
```

- [ ] **Step 4: Add panel component**

Create `frontend/src/components/AgentRunPanel.vue`:

```vue
<script setup lang="ts">
import { computed } from 'vue'
import { Activity, BrainCircuit, Gauge, ShieldCheck } from 'lucide-vue-next'
import type { AgentRunEvent } from '../stores/agentRun'

const props = defineProps<{ events: AgentRunEvent[] }>()

const latestJudge = computed(() =>
  [...props.events].reverse().find(e => e.type === 'judge'),
)
const latestTrace = computed(() =>
  [...props.events].reverse().find(e => e.type === 'agent_trace'),
)
const steps = computed(() => props.events.filter(e => e.type === 'agent_step'))
const toolSummary = computed(() => {
  const trace: any = latestTrace.value
  const entries = Object.entries(trace?.tool_call_breakdown ?? {})
  return entries.map(([name, count]) => `${name} × ${count}`).join(', ')
})
</script>

<template>
  <section v-if="events.length"
           class="mt-3 rounded-lg border border-border bg-surface/80 p-3 text-xs text-fg-muted"
           aria-label="Agent run details">
    <div class="flex items-center gap-2 text-fg">
      <Activity class="h-4 w-4 text-primary" />
      <span class="font-medium">Agent run</span>
    </div>

    <div class="mt-3 grid gap-2 sm:grid-cols-3">
      <div class="rounded-md border border-border bg-white/[0.03] p-2">
        <div class="flex items-center gap-1 text-fg-dim">
          <BrainCircuit class="h-3.5 w-3.5" />
          Route
        </div>
        <div class="mt-1 font-mono text-fg">
          {{ (latestTrace as any)?.intent ?? steps.at(-1)?.label ?? 'running' }}
        </div>
      </div>

      <div class="rounded-md border border-border bg-white/[0.03] p-2">
        <div class="flex items-center gap-1 text-fg-dim">
          <Gauge class="h-3.5 w-3.5" />
          Mode
        </div>
        <div class="mt-1 font-mono text-fg">
          {{ (latestTrace as any)?.mode ?? 'deterministic' }}
        </div>
      </div>

      <div class="rounded-md border border-border bg-white/[0.03] p-2">
        <div class="flex items-center gap-1 text-fg-dim">
          <ShieldCheck class="h-3.5 w-3.5" />
          Judge
        </div>
        <div class="mt-1 font-mono text-fg">
          <template v-if="latestJudge">
            {{ Math.round((latestJudge as any).score * 100) }}%
          </template>
          <template v-else>pending</template>
        </div>
      </div>
    </div>

    <div v-if="toolSummary" class="mt-2 font-mono text-[11px]">
      tools: {{ toolSummary }}
    </div>
    <div v-if="(latestTrace as any)?.exit_reason" class="mt-1 font-mono text-[11px]">
      exit: {{ (latestTrace as any).exit_reason }}
    </div>
    <div v-if="(latestJudge as any)?.weak_dims?.length" class="mt-1">
      weak: {{ (latestJudge as any).weak_dims.join(', ') }}
    </div>
    <div v-if="(latestJudge as any)?.same_model" class="mt-2 rounded-md border border-warning/30 bg-warning-bg px-2 py-1 text-warning">
      Judge uses the same model as generator; self-preference bias possible.
    </div>
  </section>
</template>
```

- [ ] **Step 5: Wire Chat view**

In `frontend/src/views/Chat.vue`:

- Import:

```ts
import AgentRunPanel from '../components/AgentRunPanel.vue'
```

- In `streamChat` callbacks:

```ts
onAgentEvent: (event) => chat.recordAgentEvent(assistant, event),
```

- Under citations block:

```vue
<AgentRunPanel
  v-if="m.role === 'assistant' && m.agentEvents?.length"
  :events="m.agentEvents"
/>
```

- [ ] **Step 6: Wire Plan view**

In `frontend/src/views/PlanTimeline.vue`:

- Import:

```ts
import { useAgentRun } from '../stores/agentRun'
import AgentRunPanel from '../components/AgentRunPanel.vue'
```

- Setup:

```ts
const agentRun = useAgentRun()
```

- In `checkIn()` before stream:

```ts
agentRun.reset()
```

- In stream callbacks:

```ts
onAgentEvent: (event) => agentRun.record(event),
```

- In template after button:

```vue
<AgentRunPanel class="mt-4" :events="agentRun.events" />
```

- [ ] **Step 7: Wire Quiz view**

In `frontend/src/views/QuizAdaptive.vue`:

- Import:

```ts
import { useAgentRun } from '../stores/agentRun'
import AgentRunPanel from '../components/AgentRunPanel.vue'
```

- Setup:

```ts
const agentRun = useAgentRun()
```

- In `send()` before `quiz.startStream()`:

```ts
agentRun.reset()
```

- In stream callbacks:

```ts
onAgentEvent: (event) => agentRun.record(event),
```

- In template under quiz content:

```vue
<AgentRunPanel class="mt-6" :events="agentRun.events" />
```

- [ ] **Step 8: Build frontend**

Run:

```bash
cd frontend
pnpm build
```

Expected:

- Build passes.
- No TypeScript errors from union narrowing.

- [ ] **Step 9: Optional browser smoke**

If dev servers are available:

```bash
cd backend && uv run uvicorn app.main:app --reload --port 8000
cd frontend && pnpm dev
```

Open `http://localhost:5173` and verify:

- Chat answer shows AgentRunPanel after stream starts.
- Plan check-in shows AgentRunPanel.
- Quiz generation shows AgentRunPanel.

- [ ] **Step 10: Self-review**

Check:

- UI uses existing dark design tokens.
- No emoji icons.
- No raw source text in panel.
- Panel is compact and not a full trace debugger.

---

## Task 5: Integration Verification And Drift Review

**Purpose:** Prove the pass meets the spec and did not drift into unrelated P4 work.

**Files:**

- Modify: `docs/ROADMAP.md`
- No new production files unless verification finds required fixes.

- [ ] **Step 1: Run backend verification**

Run:

```bash
cd backend
uv run pytest -q
uv run python scripts/readiness_check.py
```

Expected:

- Full backend suite passes.
- Readiness check exits `0`, with Ollama warning allowed when not strict.

- [ ] **Step 2: Run frontend verification**

Run:

```bash
cd frontend
pnpm build
```

Expected:

- Build passes.

- [ ] **Step 3: Run docs/private-data scan**

Run:

```bash
rg -n "course-corpus|private-course-pdf|local-demo-corpus" README.md docs .env.example backend/.env.example compose.yaml backend/scripts/readiness_check.py frontend/src || true
rg -n "OAuth|i18n|shared plans|group study|drag-reorder|activity heatmap|mobile UI" README.md docs frontend/src backend/app || true
```

Expected:

- First command prints no private course PDF paths or filenames.
- Second command only finds non-goal mentions in spec/roadmap context, not implemented features.

- [ ] **Step 4: Update ROADMAP checkboxes**

In `docs/ROADMAP.md`, mark P4.1 items complete only after corresponding verification passes:

```markdown
- [x] `ARCHITECTURE.md v2`
- [x] demo readiness
- [x] Agent visibility UI
- [x] deploy hardening
- [x] final review gates
```

- [ ] **Step 5: Spec compliance self-review**

Create a local checklist in final notes, not a committed file:

```text
Spec compliance:
- Reviewer-facing architecture: PASS/FAIL
- Demo readiness: PASS/FAIL
- Agent visibility UI: PASS/FAIL
- Deploy hardening: PASS/FAIL
- Review gates: PASS/FAIL
- Private PDF boundary: PASS/FAIL
- Non-goal drift: PASS/FAIL
```

- [ ] **Step 6: Code quality review gate**

Use `superpowers:requesting-code-review` or the `subagent-driven-development` code-quality reviewer prompt.

Review context:

```text
Implemented Portfolio Readiness Pass from docs/superpowers/specs/2026-05-25-portfolio-readiness-pass-design.md.
Focus on:
- SSE backwards compatibility.
- trace redaction / no private source leakage.
- frontend state correctness.
- Docker/env safety.
- scope drift into non-goals.
```

If subagent tools are unavailable, perform manual review with the same checklist and record that limitation in final response.

- [ ] **Step 7: Finishing branch decision**

Only after tests/build/reviews pass, use `superpowers:finishing-a-development-branch` if work was done on a feature branch/worktree.

If no worktree/branch was used, present:

```text
Implementation complete in current workspace. No branch cleanup is available because user chose no baseline commit/worktree.
```

---

## Plan Self-Review

Spec coverage:

- Reviewer-facing architecture maps to Task 1.
- Demo readiness maps to Task 2.
- Deploy hardening maps to Task 2.
- Agent visibility UI maps to Tasks 3 and 4.
- Review and drift control maps to Task 5.
- Git timing maps to Task 0.

Placeholder scan:

- No unresolved placeholder instructions are present.
- Any "optional" step is explicitly a smoke check and not required for correctness.

Type consistency:

- Backend event names match the spec: `agent_step`, `judge`, `agent_trace`.
- Frontend event union uses the same event names.
- Mode values remain existing `agent_loop | deterministic`.
- Intent values remain existing `tutor | quiz | plan`.

Scope check:

- No mobile, OAuth, i18n, shared plans, group study, drag-reorder, Gantt, heatmap, or public deployment task is included.
- Private course PDFs remain outside repo; demo accepts a generic `--demo-pdf /path/to/your.pdf`.

Git check:

- No automatic commit is planned.
- Worktree is gated on explicit user approval because the repo has no baseline commit.
