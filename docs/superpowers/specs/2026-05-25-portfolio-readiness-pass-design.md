# Portfolio Readiness Pass — Design Spec

> Make Study Coach ready for resume, interview, and portfolio review without turning P4 into a broad product backlog.

- **Date**: 2026-05-25
- **Status**: Approved for planning after brainstorm
- **Baseline**: P3 product shell shipped; README says 214 backend tests and frontend production build passing; `study-coach/` has no commits yet on `main`.
- **Primary audience**: reviewer / interviewer / portfolio reader who needs to understand why this is an Agent engineering project in 5-10 minutes.
- **Private demo data rule**: local course PDFs may be used for local validation and screenshots, but must not be copied into `study-coach/`, committed, published, or referenced as public downloadable assets.

---

## 1. Problem

Study Coach already has a strong Agent core:

- FastAPI + LangGraph graph with router, Tutor, Planner, QuizMaster, Judge Guard, and Memory Writer.
- Hybrid RAG with citations and reranking.
- deterministic vs `agent_loop` Plan / Quiz modes.
- P2.2 / P2.3 ablation evidence instead of generic "agents are better" claims.
- P3 Vue product shell with Overview / Chat / Plan / Quiz / Mistake Bank / Library / Settings.

The remaining portfolio risk is not missing product width. It is reviewer friction:

- A reviewer cannot quickly see the agent graph, tool calls, judge output, and failure boundaries from the app.
- A reviewer cannot confidently run a stable demo path without knowing local setup details.
- `ARCHITECTURE.md` is still contract-level and says final rationale / ER / ADR content ships at Phase 4.
- Deploy/configuration boundaries are implicit in code and README, not explicit enough for a public-facing project.

This pass should make the existing Agent engineering legible, runnable, and defensible.

---

## 2. Non-goals

These stay out of scope unless explicitly requested later:

- Mobile UI.
- OAuth / email auth.
- i18n.
- shared plans / public links.
- group study mode.
- drag-reorder milestones / Gantt timeline.
- activity heatmap.
- real public deployment.
- copying or publishing HKBU course PDFs.
- broad refactors unrelated to portfolio readiness.

---

## 3. Deliverables

### 3.1 Reviewer-facing architecture

Upgrade `docs/ARCHITECTURE.md` to v2 while preserving useful contract details.

Required sections:

- Elevator pitch: "Exam Coach Agent" in one paragraph.
- System topology: frontend, API, LangGraph graph, retrieval, SQL, Chroma, LLM provider.
- Agent graph: memory hydrator -> router -> Tutor / Planner / QuizMaster -> Judge Guard -> Memory Writer.
- Tool registry: deterministic tools, agent-loop tools, side effects, failure behavior.
- Data model: ER-style text diagram for users / documents / goals / topics / plans / questions / mastery / mistakes / sessions / messages / citations.
- Data flow walkthroughs:
  - Upload PDF -> chunks -> Chroma + SQL document row.
  - Chat question -> RAG -> Tutor -> Judge -> SSE tokens/citations.
  - Quiz generate -> retrieve -> persist question -> judge -> answer grade -> mastery/mistake update.
  - Plan generate/check-in -> milestones -> mindmap -> progress.
- Failure boundaries:
  - no corpus / empty retrieval.
  - judge parse failure.
  - same-model judge bias.
  - model lacks tool calling.
  - local Ollama unavailable.
  - schema validation failure in agent loops.
- Eval-to-product decisions:
  - why Plan and Quiz expose deterministic vs `agent_loop`.
  - why Quiz defaults can prefer `agent_loop` for grounding-sensitive tasks.
  - when deterministic is a better UX fit.
- Deployment/config boundary:
  - local dev.
  - Docker Compose.
  - BYOK cloud providers.
  - private local demo corpus.

Acceptance:

- A reader can answer "what makes this an agent project?" without opening code.
- The doc links to `docs/EVAL.md`, `docs/agent_loop_vs_deterministic.md`, `docs/quiz_ablation_followup.md`, and `docs/p3_frontend_productize.md`.
- No private PDF paths are presented as public assets.

### 3.2 Demo readiness

Add a stable local demo path that uses private local PDFs only when the user explicitly points to them.

Required artifacts:

- `.env.example` with safe defaults and comments.
- `backend/scripts/readiness_check.py` or equivalent script that checks:
  - Python dependency environment can import the app.
  - DB migration path can initialize.
  - configured Chroma path is writable.
  - Ollama host is reachable when provider is `ollama`, but reports actionable warning instead of hard failing when not running.
  - optional demo PDF path exists if provided.
- demo guide in README or `docs/DEMO.md`:
  - start backend.
  - start frontend.
  - upload a PDF from a user-owned local path.
  - run Chat, Plan, Quiz.
  - switch deterministic / `agent_loop`.
  - inspect agent visibility panel.

Private course PDF handling:

- The guide may say: "For the author's local validation, HKBU course PDFs live outside this repo."
- It must not name those files as required public assets.
- It must not copy them into `study-coach/`.
- Any script argument should accept a generic path like `--demo-pdf /path/to/your.pdf`.

Acceptance:

- A reviewer with their own PDF can follow the demo path.
- The author can still validate locally with HKBU PDFs.
- `.gitignore` continues to exclude DB, Chroma, `.env`, build outputs, and runtime data.

### 3.3 Agent visibility UI

Expose enough agent run evidence in the frontend for interview demos.

Current SSE contract:

- Backend emits `citations`, `token`, and `done`.
- Frontend `streamChat()` only handles those three event types.
- The same-model judge bias warning is currently emitted as a `token`.

New event contract:

```ts
type AgentEvent =
  | { type: 'agent_step'; node: string; action: 'start' | 'end'; label?: string }
  | { type: 'tool_call'; name: string; status: 'start' | 'success' | 'error'; summary?: string }
  | { type: 'judge'; score: number; weak_dims: string[]; reasoning?: string; same_model?: boolean }
  | { type: 'agent_trace'; mode: 'deterministic' | 'agent_loop'; intent?: 'tutor' | 'quiz' | 'plan'; iterations?: number; tool_call_breakdown?: Record<string, number>; exit_reason?: string }
```

Implementation constraints:

- Keep `citations`, `token`, and `done` backward compatible.
- New events must be additive and safe to ignore.
- Do not stream raw private document text or full tool outputs into UI trace.
- Summaries must be redacted/truncated.
- Use existing `AgentTrace.serialize()` where available for Planner / Quiz agent loops.
- Deterministic paths should still show mode and intent, even if they have no tool-loop trace.
- UI should match `design-system/MASTER.md`: dark cinema, no emojis, lucide icons, readable contrast, accessible buttons.

Frontend shape:

- Add an `AgentRunPanel` component visible from Chat / Plan / Quiz, likely as a compact side panel or collapsible card under the latest assistant message.
- Show:
  - route / intent.
  - current mode.
  - tool calls summary.
  - judge score and weak dimensions.
  - exit reason and iteration count when available.
  - same-model judge warning as structured metadata instead of only inline prose.
- Avoid building a full trace debugger. The goal is portfolio evidence, not LangSmith replacement.

Acceptance:

- A demo can visually show: "router chose quiz/plan/tutor", "the model used retriever/tool calls", "Judge Guard scored it", and "mode was deterministic or agent_loop".
- No sensitive corpus text is exposed in trace.
- Existing chat streaming still works if new events are ignored.

### 3.4 Deploy hardening

Add deploy-ready configuration without forcing a public deploy.

Required artifacts:

- root `.env.example` for project-level compose / common defaults.
- backend `.env.example` if backend-specific variables need clearer ownership.
- Dockerfile(s) only if required by the chosen Compose strategy.
- `compose.yaml` or `docker-compose.yml` for local production-like startup.
- README / docs section explaining:
  - local dev (`uvicorn`, `pnpm dev`).
  - compose path.
  - persistent volumes for DB / Chroma.
  - Ollama requirement and host configuration.
  - BYOK headers remain user-provided and server does not persist API keys.

Official-doc constraints checked:

- Docker Compose env handling should use explicit `env_file` / `environment` rather than relying on undocumented behavior. Docker's Compose docs state that a container's environment is not set until an explicit service config entry does it, and document both `environment` and `env_file`: <https://docs.docker.com/compose/how-tos/environment-variables/set-environment-variables/>.
- Vite only exposes `VITE_*` values to client code and warns not to put sensitive data in them because they are bundled at build time: <https://vite.dev/guide/env-and-mode.html>.
- FastAPI `StreamingResponse` remains the right mechanism for SSE-like streaming because it streams an async generator or normal generator response body: <https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse>.
- LangGraph supports custom streaming via `get_stream_writer()` and `stream_mode="custom"`, matching the current backend architecture: <https://docs.langchain.com/oss/python/langgraph/streaming>.

Acceptance:

- A reviewer can understand how the project would run in a containerized local setup.
- No secret or API key is committed.
- Compose does not require public deployment.

### 3.5 Review and drift control

This pass must include explicit review gates.

Spec self-review:

- Check every deliverable has an acceptance criterion.
- Check non-goals are not implemented.
- Check private PDF rule appears in docs/scripts where relevant.

Implementation plan self-review:

- Every task must map to a deliverable.
- Every code task must include verification.
- No task may introduce mobile/OAuth/i18n/shared-plan/product-width work.

Implementation review:

- Use `superpowers:subagent-driven-development` for execution when tasks are independent enough.
- Per task:
  - implementer self-review.
  - spec compliance review.
  - code quality review.
- Final:
  - full diff review against this spec.
  - run backend tests relevant to changed backend surfaces.
  - run frontend build for UI changes.
  - run readiness script.
  - manually validate demo flow if local LLM/runtime is available.

If subagent tooling is unavailable or conflicts with the environment, fall back to inline execution but keep the same gates: self-review, spec compliance review, code quality review, final review.

---

## 4. Architecture choices

### 4.1 Agent visibility source

Recommended approach: emit portfolio-safe custom events from existing graph nodes and route layer.

Why:

- Current architecture already uses LangGraph `stream_mode="custom"`.
- It avoids adding a second tracing subsystem.
- It keeps trace visibility close to the runtime behavior being demonstrated.

Rejected:

- Reading DB messages/citations after the run: not real-time and current message persistence is not the main UI path.
- Building a full observability backend: too large for readiness pass.
- Streaming raw LangGraph debug events: too noisy and risks leaking internals/private text.

### 4.2 Demo corpus

Recommended approach: generic user-owned PDF path, no checked-in demo corpus.

Why:

- Local course PDFs are private course materials and should not be published.
- The product is a PDF-based study coach; a reviewer can use their own PDF.
- Keeping the demo script path-based preserves author validation without repository risk.

Rejected:

- Commit course PDFs: unacceptable.
- Generate a synthetic PDF in this pass: possible but lower priority; can be added later if a public demo needs zero external input.

### 4.3 Deploy hardening level

Recommended approach: local production-like Compose, not public deployment.

Why:

- The portfolio needs deploy credibility, not Fly/Railway operations work.
- BYOK and local Ollama create environment-specific concerns better documented before public hosting.
- Compose + `.env.example` provides enough reviewer confidence.

Rejected:

- Full public deployment: out of scope.
- No deploy artifacts: leaves the reviewer with "works on author's machine" concern.

---

## 5. File impact map

Expected new files:

- `.env.example`
- `compose.yaml`
- `docs/DEMO.md`
- `docs/superpowers/plans/2026-05-25-portfolio-readiness-pass.md`
- `backend/scripts/readiness_check.py`
- `frontend/src/components/AgentRunPanel.vue`
- `frontend/src/stores/agentRun.ts`

Expected modified files:

- `README.md`
- `docs/ARCHITECTURE.md`
- `docs/ROADMAP.md`
- `backend/app/api/routes.py`
- `backend/app/agent/graph.py`
- `backend/app/agent/planner_agent.py`
- `backend/app/agent/quiz_master_agent.py`
- `frontend/src/lib/api.ts`
- `frontend/src/stores/chat.ts`
- `frontend/src/views/Chat.vue`
- `frontend/src/views/PlanTimeline.vue`
- `frontend/src/views/QuizAdaptive.vue`

This list is directional. The implementation plan may reduce it after code inspection.

---

## 6. Verification strategy

Minimum verification:

- Backend:
  - relevant route / graph tests for new SSE events.
  - tests that trace events are redacted and additive.
  - `uv run pytest -q` if backend event changes touch shared graph behavior.
- Frontend:
  - `pnpm build`.
  - if feasible, browser smoke for Chat / Plan / Quiz with AgentRunPanel visible.
- Scripts:
  - readiness check success path for local filesystem/config.
  - readiness check warning path for missing Ollama or missing optional PDF.
- Docs:
  - link check by grep for referenced docs.
  - private PDF scan: no local course-corpus path in public docs/scripts.

Manual demo acceptance:

1. Start backend and frontend.
2. Upload a local PDF.
3. Ask a Tutor question and see citations + judge metadata.
4. Generate a plan and see mode/trace metadata.
5. Generate a quiz and see retrieval/tool/trace metadata.
6. Switch mode and verify UI labels reflect deterministic vs `agent_loop`.

---

## 7. Risks

- **Trace leakage**: raw tool outputs may contain private document text. Mitigation: only emit names, counts, scores, redacted summaries, exit reasons.
- **SSE contract breakage**: changing stream parsing can break chat. Mitigation: additive event types; old events unchanged; tests for `citations -> token* -> done`.
- **Scope creep**: Agent visibility can become full observability. Mitigation: one compact panel, no history browser, no LangSmith clone.
- **Docker/Ollama networking**: containers may not reach host Ollama uniformly across OSes. Mitigation: document `OLLAMA_HOST` and provide warning-oriented readiness check.
- **No commits in repo**: this repository has no commits yet; review-by-SHA workflows may need diff-based review instead of commit-range review unless the user explicitly asks to commit.

---

## 8. Completion definition

This pass is complete when:

- `ARCHITECTURE.md v2` explains the project as an Agent engineering portfolio artifact.
- Demo docs and readiness check let a reviewer run the app with their own PDF.
- Agent visibility UI shows mode, tool/trace summary, and judge metadata without leaking corpus text.
- Deploy hardening artifacts document local production-like startup and env boundaries.
- README / ROADMAP point to the new readiness assets.
- Verification results are recorded in the final response.
- Self-review and code review gates have been run, with critical/important issues resolved or explicitly deferred with rationale.
