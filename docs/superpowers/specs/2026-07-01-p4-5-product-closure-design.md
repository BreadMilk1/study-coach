# P4.5 Product Closure for Portfolio Demo - Design Spec

> Brainstormed 2026-07-01.
> Upstream: P3 frontend productization, P4 product polish, P4d chat persistence, P4e recoverable Agent Run Trace.
> Target: close the Chat-first reviewer demo path before larger harness, durable memory, or multi-agent orchestration work.

---

## 1. Problem

Study Coach already has a strong agent engineering base:

- FastAPI + LangGraph graph with Tutor, QuizMaster, Planner, Judge Guard, Memory Hydrator, and Memory Writer.
- Hybrid retrieval with citations and reranking.
- deterministic and `agent_loop` Plan / Quiz modes.
- P2.2 / P2.3 ablation evidence across local Ollama models.
- P4d persisted chat sessions and P4e recoverable Agent Run Trace.

The remaining risk is not missing feature width. It is demo trust.

A reviewer can see many features, but the most important Chat-first path can still break in ways that undermine the portfolio story:

- A visible quiz question may be streamed even when `persist_quiz_question` failed, so the next user answer cannot be graded.
- The Debug panel can show tool errors, but it is not yet shaped as a clear pass/fail evidence trail for the demo path.
- Backend route tests currently pass only when live Ollama is available in at least one quiz route test file, which weakens the "no live Ollama required" verification baseline.
- There is no checked-in `docs/DEMO.md` stable reviewer path, even though the Portfolio Readiness spec called for one.

P4.5 closes those gaps without starting a full harness platform or durable learning memory redesign.

---

## 2. Goal

Make one reviewer-facing demo path stable, explainable, and testable:

1. Upload a user-owned PDF in Library.
2. Ask a grounded question in Chat and receive citations.
3. Ask "quiz me on ..." in Chat.
4. If an MCQ is shown, it is already persisted and can be graded on the next `A/B/C/D` reply.
5. Refresh the frontend and restore chat history plus Debug / Agent Run evidence.
6. Confirm the run through a documented CLI and frontend validation path.

The product headline is:

> Study Coach can demonstrate a recoverable, source-grounded learning loop from PDF upload to Chat quiz, grading, and Agent Run evidence.

---

## 3. Scope

### 3.1 Test Isolation Hardening

Fix the existing route-level quiz tests so backend verification does not require live Ollama.

Required behavior:

- `backend/tests/api/test_routes_quiz_agent.py` stubs `get_judge_dependencies`.
- The test continues to verify route / dispatcher behavior.
- `cd backend && uv run pytest -q` should pass without starting Ollama.

Rationale:

- This is a small precondition for trustworthy P4.5 validation.
- It does not change product behavior.

### 3.2 Strong Consistency Quiz Closure

Adopt this rule:

> Chat must not display an answerable MCQ unless the question has been successfully persisted and the next user option can be graded.

Required behavior:

- Successful quiz generation sets or preserves a valid `active_quiz_question_id`.
- If the agent loop cannot persist a question, it emits a clear degraded response instead of a usable MCQ.
- A degraded response may mention that the quiz question could not be saved, but must not invite the user to answer `A/B/C/D`.
- Deterministic and agent-loop quiz paths should keep the same user-facing contract: visible MCQ means gradeable MCQ.

### 3.3 Tool-Layer Tolerance

Allow narrow normalization before rejecting `persist_quiz_question` arguments.

In scope:

- Normalize answer values like `"A)"`, `"A) ..."`, or `"a"` to `"A"` when unambiguous.
- Normalize option lists that contain four option texts but omit `A) ` / `B) ` / `C) ` / `D) ` prefixes.
- Preserve strict validation after normalization: exactly four options, valid answer letter, non-empty prompt, non-empty explanation.

Out of scope:

- Inferring the correct answer from explanation text.
- Parsing arbitrary free-form quiz prose into a question.
- Frontend-created temporary quiz state.

Rationale:

- The user-observed failure came from near-miss formatting (`A` vs `A)`, missing option prefixes), not from lack of domain content.
- Tool-level tolerance is more reliable than prompt-only fixes.

### 3.4 Debug Evidence

Keep Debug Mode as the existing evidence surface. Do not build a new Agent Runs page.

Required behavior:

- The latest assistant message shows `agent_run.exit_reason`, `tool_errors`, tool call names, and safe output previews.
- `persist_quiz_question` success or failure is visible through the existing redacted tool call list.
- Refreshing the frontend preserves the relevant `agent_run` evidence for restored assistant messages.
- Raw PDF text must not be exposed in Debug output.

Nice to have, not required:

- A small visual distinction for degraded runs or tool errors.
- A short "persisted question id" preview when available.

### 3.5 Demo Guide and Roadmap Sync

Add documentation before implementation planning:

- `docs/DEMO.md`: Chat-first reviewer path, current validation commands, expected P4.5 gates.
- `docs/ROADMAP.md`: add P4.5 as planned / in-progress, not as shipped.

README and ARCHITECTURE should be updated after implementation, when behavior is verified.

---

## 4. Non-Goals

P4.5 does not include:

- A full Agent Run Inspector page.
- Normalized `agent_runs` / `agent_run_tool_calls` tables.
- OpenTelemetry, Grafana, token-cost dashboards, or replay analytics.
- A new durable learning memory layer.
- Session-scoped memory redesign.
- Full Chroma multi-user / multi-document filtering.
- Multi-agent orchestration beyond the existing LangGraph node dispatch.
- A new frontend test framework.
- A broad UI redesign.
- Solving every Quiz view flow. The main path is Chat-first; Quiz view gets only a smoke-level check.

---

## 5. Acceptance Criteria

### 5.1 Backend

- `cd backend && uv run pytest -q` passes without live Ollama.
- A regression test covers the case where the quiz agent emits final text after failed `persist_quiz_question`; expected result is degraded / non-answerable output and no `active_quiz_question_id`.
- Tool schema tests cover answer and option-prefix normalization.
- A route-level Chat test proves a generated Chat MCQ can be followed by an `A/B/C/D` grade turn in the same `session_id`.

### 5.2 Frontend

- `cd frontend && pnpm build` passes.
- With Debug Mode enabled, the Chat page shows restored Agent Run evidence after refresh.
- Tool errors in Agent Run remain visible but redacted.
- No UI copy instructs the user to answer a quiz that was not persisted.

### 5.3 Manual Demo

The documented demo path in `docs/DEMO.md` can be followed with a user-owned PDF:

1. Start backend and frontend.
2. Upload a PDF.
3. Ask a grounded Chat question.
4. Ask a Chat quiz question.
5. Answer with a letter.
6. Refresh the browser.
7. Confirm history and Debug / Agent Run evidence are restored.

### 5.4 Documentation

- `docs/DEMO.md` names generic PDF paths only.
- `docs/ROADMAP.md` records P4.5 as a planned closure slice.
- README and ARCHITECTURE are not marked as fully updated until implementation is verified.

---

## 6. Design Notes

### 6.1 Where to Fix Quiz Consistency

Prefer backend enforcement over frontend recovery.

The frontend should not parse assistant prose into temporary quiz state. The backend owns the invariant because it already owns:

- `active_quiz_question_id` in graph state.
- `persist_quiz_question` tool side effects.
- deterministic quiz grading.
- session checkpointer state.

### 6.2 Expected Agent-Loop Behavior

Agent-loop Quiz generation has two valid outcomes:

1. **Success**: persisted question id exists, final text is answerable.
2. **Degrade**: no persisted question id exists, final text is not answerable.

`exit_reason == "natural_stop"` alone is not enough to call a Quiz run successful. P4.5 should distinguish "natural stop with persisted question" from "natural stop after tool errors without persistence."

### 6.3 Debug Surface

The current `assistant_artifacts.v1` envelope remains enough for P4.5:

```json
{
  "schema": "assistant_artifacts.v1",
  "citations": [],
  "agent_run": null
}
```

Do not add a table until there is a real cross-session analytics or replay use case.

---

## 7. Suggested Implementation Slices

This spec intentionally stops short of detailed code steps. A separate implementation plan should break the work into small cuts:

1. Test isolation patch for quiz route tests.
2. Tool normalization tests and implementation.
3. Quiz agent strong-consistency regression and implementation.
4. Route-level Chat generate-then-grade test.
5. Debug panel evidence polish if needed.
6. Demo guide verification and final docs sync.

---

## 8. Resolved Defaults for Implementation Plan

- Use a quiz-specific degraded message. It should say that Study Coach could not save a gradeable quiz question and invite the user to retry or switch mode, but it must not include answer options.
- Debug output may expose safe persistence metadata such as `question_id` or `persisted: true`. It must not expose raw PDF text or full tool output.
- The Chat-first demo should prefer `agent_loop` with a known tool-capable model (`gemma4:e4b` in the local Ollama setup) because P4.5 is meant to show Agent Run evidence. Deterministic mode remains the fast fallback check.
