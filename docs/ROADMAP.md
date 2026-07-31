# Study Coach — ROADMAP

> Phase history + priority backlog. Current state: P4.5 product closure verified; P5 local-first data lifecycle automated remediation and current-head Chrome acceptance complete (2026-07-31).
> Lives in repo (vs. plan in `~/.claude/plans/`) so it survives across Claude sessions.

## Current Snapshot (2026-07-31)

- **Project shape**: portfolio-grade Exam Coach Agent, refactored from `HKBU_StudyCompanion` and informed by JadeAI engineering patterns.
- **Backend**: FastAPI + LangGraph + Chroma hybrid retrieval + SQLAlchemy/Alembic + BYOK LLM provider + signed JWT local anonymous identity. Google OAuth remains frozen in backend code and is not surfaced by the frontend.
- **Agent graph**: memory hydrator → router → Tutor / QuizMaster / Planner (det + agent_loop) → Judge Guard → memory writer.
- **Frontend**: Vue 3 + Pinia + Tailwind 4 + vue-i18n (en/zh-CN). 8 views: Overview / Chat / Plan / Quiz / Mistake Bank / Library / Settings / Onboarding. Mobile responsive (Chat/Quiz/Plan).
- **Eval evidence**: P2.0 retrieval eval; primary agent-loop matrices P2.2 Plan (396 runs) + P2.3 Quiz (396 runs) = 792; P2.3 no-retriever pilot adds 396, for 1,188 raw runs total.
- **Architecture docs**: ARCHITECTURE.md v2 (Mermaid ER + 6 ADRs + deployment topology + security model).
- **Deploy**: host-run is the canonical reviewer path; Docker Compose is an alternative mutually exclusive Ollama path (stop host Ollama first — both bind `127.0.0.1:11434`). Cloud deployment is deferred; the retained Fly files are an unverified scaffold.
- **Verification baseline**: 411 backend tests passing; 130 frontend tests across 14 Vitest files passing (541 total); frontend production build and `docker compose config` passing. Existing Vite >500 kB chunk warning remains accepted. Automated remediation and current-head Chrome acceptance completed on 2026-07-31.
- **Recent (2026-07-31)**: Final PR remediation and Chrome acceptance: learning routes require an existing user row (Factory reset old JWT cannot recreate orphan learning data; same JWT may still idempotently retry Factory reset); summary exposes `current_user_exists`; a deterministic staged recovery fingerprint makes response-lost and delayed tabs converge on one replacement user before unlock; pre-multipart ASGI upload request cap; CancelledError temp cleanup; dialect-based SQLite FK; native lifecycle dialogs explicitly intercept Esc keydown so store state and modal visibility cannot diverge. Path A Chrome verified real Ollama, 49/61-chunk PDFs, Chat/Quiz/Plan/mistake/mastery state, learning reset preservation, re-upload, cross-tab acknowledgement, Factory defaults/reload, old-JWT refusal/retry, and three-tab convergence to one user.
- **Recent (2026-07-28)**: PR review hardening prevents route side effects before the startup choice, makes anonymous provisioning retryable without stale Pinia overwrite, leases identity-mutating auth during reset, preserves a same-scope recovery latch after partial reset, cleans partial temp uploads, and adds Settings save confirmation. Manual lifecycle validation on that date verified Save → refresh → `Connected`, upload → learning reset → re-upload at 49 chunks, then Factory reset → default Settings, empty Library/counts, second-tab reload, and exactly one new anonymous user — historical evidence only until current-head browser recheck.
- **Next**: choose the next portfolio slice; multi-user auth remains deliberately deferred to a separate worktree.
- **Recent (2026-07-20)**: P4.5 manual browser demo passed with user-owned PDFs — grounded Chat answer, agent-loop Quiz generation, deterministic grading, and refresh restore all verified. Follow-up hardening prevents answer/explanation disclosure during GENERATE and strips trailing retrieval metadata before explanation persistence.
- **Recent (2026-07-02)**: P4.5 automated Product Closure implemented — Chat quiz MCQs are persisted before display, failed quiz persistence degrades safely, `persist_quiz_question` tolerates narrow local-model formatting near-misses, route tests no longer depend on live Ollama, and `docs/DEMO.md` documents the reviewer path. The final manual public-review gate was completed on 2026-07-20.
- **Recent (2026-06-15)**: Recoverable Agent Run Trace shipped — Planner / QuizMaster `agent_loop` nodes emit SSE `agent_run` events, `/api/chat` persists them in `messages.tool_calls_json` assistant artifact envelopes, `GET /api/chat/sessions/{id}/messages` restores them after refresh, and Debug Mode TracePanel displays node/mode/exit reason, tool counts, token counts, latency, and redacted tool-call previews.
- **Recent (2026-06-10)**: Chat persistence slice shipped — `/api/chat` now creates/reuses `ChatSession`, persists user/assistant `Message` rows plus assistant `Citation` rows, emits SSE `session` events, restores current Chat session after frontend refresh, refuses retrieval when the current user has no Library documents, and `GET /api/users/me/stats.total_sessions` reflects real session count.

## P0 — Done

- [x] `PROJECTS_OVERVIEW.md` — dual-project comparison (HKBU_StudyCompanion vs. JadeAI)
- [x] `~/.claude/plans/hey-1-hkbu-...md` — refactor plan, decisions, phase breakdown
- [x] `docs/ARCHITECTURE.md` — contract-level spec (state / tools / DB / routes / provider / stores)
- [x] `docs/ROADMAP.md` — this file
- [x] memory locked: stack, LLM dual-track, contract decisions, skill usage by phase

## P1 — Done (Phase 1: minimal closed loop)

Goal: upload 1 PDF → ask 1 question → see streamed answer with one clickable citation. Single tool, single LLM, no agent loop yet.

- [x] `backend/` skeleton: FastAPI + `uv` / `pyproject.toml` + `app/main.py` + healthcheck
- [x] `app/db/` SQLAlchemy 2.x + SQLite + Alembic init + `users` / `documents` tables
- [x] `app/rag/document_processor.py` — port from `HKBU_StudyCompanion 2/src/document_processor.py`
- [x] `app/rag/retriever.py` — Chroma backend; first dense path, later rebuilt as hybrid in P2.0
- [x] `app/llm/provider.py` — `init_chat_model` + BYOK header parsing
- [x] `app/agent/state.py` + `app/agent/graph.py` — initial single-node graph wrapping `rag_search` tool
- [x] `app/api/documents.py` / `routes.py` — upload + ingest
- [x] `app/api/chat.py` / `routes.py` — SSE streaming endpoint
- [x] `frontend/` Vite + Vue 3 + Pinia + Tailwind scaffold
- [x] Chat view: message list, SSE consumer, citation chip
- [x] BYOK Settings view: provider/model/api_key/base_url
- [x] tests: `pytest` for retriever + chat endpoint (mock LLM)
- [x] **Verification**: minimal upload → ask → streamed answer + citation path delivered

**Skill triggers for P1**: `superpowers:tdd` (modules), `superpowers:dispatching-parallel-agents` (rag / db / frontend parallel), `superpowers:verification-before-completion` (before each tick).

## P2 — Done (Phase 2: old four features → agent tools)

### P2.0 — Retrieval Foundation (Done)

Pre-flight: Phase 1 answers were "missing content actually in the PDFs". Diagnosed not as model but as **retrieval + prompt**. Rebuilt the foundation with TDD before bolting agent layer on top. Full report: [PHASE_2.0_REPORT.md](./PHASE_2.0_REPORT.md).

- [x] Retrieval eval harness + 15 ground-truth queries (HKBU 12 + 3 cross-lang)
- [x] BM25 + Dense + RRF fusion (`HybridRetriever`, `rank_bm25` dep)
- [x] Cross-encoder reranker (`jina-reranker-v2-base-multilingual` via `fastembed`)
- [x] Concat-then-split chunking (preserves cross-page concepts)
- [x] Shared prompt module (`agent/prompt.py`): system role + grounded约束 + source/page injection
- [x] Citation 5-field (chunk_id / source / page / span_start / span_end), ARCHITECTURE.md synced
- [x] `main.py` production wiring uses `RerankingRetriever(HybridRetriever(Retriever))`
- [x] **Verification**: 36 tests passing; Hit@5 0.733 → 0.933 (+27%), MRR 0.633 → 0.822 (+30%) on 15 queries

### P2.1 — Agent-ify (Done)

#### P2.1-① Multi-node graph skeleton (Done, 2026-05-19)

- [x] `app/agent/router.py` — keyword-based intent classifier (`quiz` > `plan` > `tutor`)
- [x] `CoachState` extended with `intent: NotRequired[Literal["tutor","quiz","plan"]]`
- [x] `app/agent/graph.py` rewritten as 4-node `StateGraph`: router → conditional_edges → {tutor | quiz_stub | plan_stub} → END
- [x] Tutor node async + `get_stream_writer()` double-emit: `{type:"citations",...}` first, then per-LLM-chunk `{type:"token","text":...}`
- [x] Quiz/Plan stub nodes also emit `citations: []` to keep frontend SSE contract (`citations → token* → done`) uniform across branches
- [x] `app/api/deps.py` adds `get_graph(retriever, llm)`; per-request build (preserves BYOK header model swap)
- [x] `app/api/routes.py` chat handler rewritten to `graph.astream(stream_mode="custom")` — production now flows through LangGraph (closes P2.0 known limitation #4)
- [x] **Verification**: 49 tests passing (36 baseline + 5 router + 5 graph + 3 routes_graph_stream); `test_chat_streams_citations_then_tokens_then_done` still green (zero frontend change)

#### P2.1-② Judge Guard middleware + x-judge-model (Done, 2026-05-19)

- [x] `app/agent/judge.py` — `judge_response()` LLM-as-judge utility with 6-dimension rubric (relevance / accuracy / citation_quality / accessibility / example_quality / learner_level_fit), markdown-fenced JSON parse, neutral fallback on parse failure
- [x] `app/agent/prompts/judge_tutor.txt` — externalised rubric template with bias-aware instruction + 4 calibrated few-shot examples; `rubric: str` parameter on `judge_response` lets future P2.1-④/⑤ Quiz/Plan judges reuse same utility with different rubrics (A→C evolution per design discussion)
- [x] `CoachState` extended with `judge_score / retry_count / weak_dims / judge_reasoning / degraded / last_context`
- [x] `app/agent/graph.py` adds Judge node after Tutor; uses `Command[Literal["tutor","__end__"]]` for retry-or-end routing; tutor `_retry_hint()` injects previous score + weak_dims so the LLM can self-correct (PDCA "Act")
- [x] Retry budget = 2 (threshold 0.6); on exhaustion, degrade path appends visible `⚠️ Self-check note: ...` disclaimer to AIMessage and END
- [x] `app/api/deps.py` adds `get_judge_dependencies()` — distinct judge LLM when `x-judge-model` set; same-LLM fallback + `same_model=True` flag otherwise
- [x] `app/api/routes.py` injects judge_llm via `graph.astream(config={"configurable":{"judge_llm":...}})`; same-model paths inline-emit bias warning as `type:"token"` right after `citations` so frontend stays zero-change
- [x] **Verification**: 63 tests passing (49 from P2.1-① + 6 judge unit + 5 graph_judge integration + 3 same-model warning routes tests); P2.0 SSE strict-equality contract still green; frontend zero-change

#### P2.1-③ DB schema + repositories + Memory Updater (Done, 2026-05-20)

- [x] **Cut ①** Alembic init + baseline migration `a03f432cd12f_p1_baseline_users_documents` (captures `users` + `documents`); env.py honors `set_main_option` (tests) > `DATABASE_URL` env (CLI) > default
- [x] **Cut ②** 9 new tables + repositories TDD (Goal/Topic/Plan/Question/Mastery/Mistake/ChatSession/Message/Citation); autogen migration `cae9687d6295_p2_1_3_memory_schema`. **Naming note**: SQL `sessions` table maps to Python `ChatSession` class to avoid `sqlalchemy.orm.Session` collision
- [x] **Cut ③** Memory Updater factories (`app/agent/memory_updater.py`): `build_memory_hydrator()` loads `mastery_scores` + `recent_mistakes` from DB into state; `build_memory_writer()` drains `pending_mastery_delta` + `pending_mistake` to DB; both no-op without `user_id`. CoachState gains 5 NotRequired fields.
- [x] **Cut ④** Graph topology: `START → memory_hydrator → router → {tutor → judge | quiz_stub | plan_stub} → memory_writer → END`. Hydrator/writer injected via `RunnableConfig.configurable` (mirror of judge_llm pattern); absent → no-op so existing 14 graph tests pass zero-change. Judge Command annotation flipped to `[tutor, memory_writer]`.
- [x] **Verification**: 86 tests passing (63 baseline + 4 alembic + 11 repos + 5 memory_updater unit + 3 graph_memory integration). Zero P2.0/P2.1-①/② regressions.
- [x] **Follow-up (2026-05-21)**: dev DB stamped + upgraded to head (`alembic stamp a03f432cd12f && alembic upgrade head`); `session.py` `create_all` retired in favor of `migrate_to_head()` called from `create_app()`; alembic is now single source of truth. 88 tests passing (+2 migrate_to_head unit tests).

**Deferred follow-ups (not in P2.1-③ scope)**:
- ~~`sessions`/`messages`/`citations` write paths land in P2.1-④/⑤ when real Quiz/Plan nodes produce them~~ Closed 2026-06-05 by Chat persistence slice.

#### P2.1-④ Quiz chain (Done, 2026-05-21)

5 cuts, 113 backend tests passing (+25 over P2.1-③ follow-up). Zero regressions.

- [x] **Cut ④a** SM-2 lite scheduler (`app/srs/sm2.py`): pure function, derives repetitions implicitly from prior interval (0→1, 1→6, else *ease). 6 tests.
- [x] **Cut ④b** 4 Quiz tools (`app/agent/tools/{quiz,schemas}.py`): `update_mastery` / `record_mistake` (uses SM-2) / `grade_quiz_answer` (case+whitespace-tolerant) / `generate_quiz` (LLM + tolerant JSON parse). Plain functions with explicit repo args. 8 tests.
- [x] **Cut ④c** QuizMaster deterministic node (`app/agent/quiz_master.py`): `state.active_quiz_question_id` present → GRADE (mastery ±, record_mistake on wrong); absent → GENERATE (extract topic, auto-create goal/topic, persist Q). 3 tests. State extension: `active_quiz_question_id: NotRequired[str | None]`.
- [x] **Cut ④d** Quiz-specific Judge rubric (`app/agent/prompts/judge_quiz.txt` + `QUIZ_DIMENSIONS`): 5 dims (question_quality / option_plausibility / answer_correctness / explanation_clarity / difficulty_calibration). `judge_response(dimensions=None)` refactored backward-compat. 5 tests.
- [x] **Cut ④e** Graph wire-up: `quiz_node` delegates to `config.quiz_master` (fallback stub); `quiz → judge` edge; Judge picks rubric on `state.intent`; quiz weak → degrade direct (no retry). State-aware router: `active_quiz_question_id` truthy forces intent="quiz" for multi-turn. 3 e2e tests through full graph.
- [x] **Cut ④f** Production wiring (follow-up): `deps.py` factories for quiz_master / memory_hydrator / memory_writer / checkpointer; `routes.py` injects them + `user_id` into state + `thread_id` into config; `main.py` mounts app-level `InMemorySaver`; `build_graph(checkpointer=None)`. 1 multi-turn quiz e2e through `/api/chat`. **Total 114 tests passing**.
- [x] **Cut ④g** Judge GRADE skip + disclaimer fix (from real-Ollama validation): `quiz_master` sets `quiz_action: "generate"|"grade"`; `judge_node` short-circuits on grade (quiz rubric doesn't apply to deterministic grade output); `_degrade_disclaimer(retry_count)` parameterized so quiz weak no longer lies "after 2 retries". **118 tests passing**.
- [x] **Cut ④h** RAG-grounded quiz generation (from real-Ollama validation: gemma3:4b had hallucinated wrong HyDE meaning because generator never saw HKBU chunks): `quiz_master(retriever=None)` calls `retriever.search(topic_name)`; `generate_quiz(context_chunks=None)` prompt embeds chunks with strict grounding instruction; `Topic.source_chunks` persisted per quiz via `TopicRepository.set_source_chunks`. Ungrounded fallback when retrieval returns empty. **124 tests passing**.

**Deferred ablation point (P2.2/P3)**: LLM tool-calling agent QuizMaster vs deterministic baseline (latency / correctness comparison; portfolio differentiation HKBU class report didn't have).

**Real-run E2E validation (2026-05-21, ④g + ④h)**: Same browser session, `quiz me on HyDE` before vs after ④h:
- BEFORE: question topic = "Hypothesis-Driven Experimentation" (gemma3:4b hallucinated off-corpus)
- AFTER: question topic = "HyDEGenerator's primary function = rewrite user query into a hypothetical answer document before embedding" (precisely the HKBU §HyDE chapter); 4 options drew from PDF vocabulary; grade feedback cited `Chunk 2 and 5`; `topics.source_chunks` persisted 5 cross-page chunks
- ④g verified: GRADE turn produced zero `[JUDGE/quiz]` log entries + no `⚠️ Self-check ... after 2 retries` false disclaimer

#### P2.1-④i Cosmetic: Quiz body char-by-char vertical render (Done — cannot reproduce)

**Symptom (one-time, ④h E2E)**: Quiz GENERATE path rendered char-by-char vertically. Not reproducible in 2026-05-21 follow-up test — SSE raw frames correct, Python emit text correct, bubble renders normally.

**Investigation result**: All 4 suspect paths ruled out by code inspection (`quiz_master.py:192` one-shot emit, `tools/quiz.py:146` ainvoke, `routes.py:79` json.dumps escaping, frontend `api.ts:37-44` split+parse correct). Likely gemma3:4b one-off malformed output or stale browser JS cache.

#### P2.1-⑤ Plan chain (Done, 2026-05-22)

9 cuts (⑤a-⑤i), **157 backend tests passing** (+33 over P2.1-④h). Zero regressions. Real-Ollama E2E verified across 4 multi-turn scenarios.

- [x] **Cut ⑤a** `compute_progress` pure function (`app/agent/progress.py`): done/overdue/weak_topics derived deterministically from plan + mastery + mistakes. Datetime injection seam. Tolerant `due_at` parse (ISO string / datetime / None / garbage). 5 tests.
- [x] **Cut ⑤b** `PlanRepository.update_milestones(*, goal_id, milestones)`: upsert — creates if absent, overwrites + bumps `updated_at` if present. 2 tests.
- [x] **Cut ⑤c** Plan tools (`app/agent/tools/plan.py` + new `Milestone` / `PlanPatchOut` / `MindmapOut` schemas): `update_study_plan` (sync DB upsert wrapper), `generate_mindmap` (async LLM call with 3-tier tolerant mermaid parsing — fenced → bare → outline-only fallback). 5 tests.
- [x] **Cut ⑤d** Plan judge rubric (`app/agent/prompts/judge_plan.txt` + `PLAN_DIMENSIONS`): 5 dims (milestone_specificity / milestone_granularity / time_feasibility / topic_coverage / actionability) disjoint from tutor/quiz. `judge_response(dimensions=)` signature unchanged from ④d. 2 calibrated few-shots. 4 tests.
- [x] **Cut ⑤e** `planner_node` deterministic (`app/agent/planner.py`, ~300 lines mirrors quiz_master.py shape): GENERATE path (extract topic → resolve/create goal → retrieve chunks → LLM milestones JSON → persist → optional mindmap) + CHECK-IN path (compute_progress → LLM adjust → tolerant schema validate → persist or schema-skip). 7 tests. State extension: `active_plan_id: NotRequired[str|None]`, `plan_action: NotRequired[Literal["generate","check_in"]]`. **5 review-driven inline fixes** applied (rstrip char-stripping bug, dead `_has_check_in_keyword` removal, duplicate `_MINDMAP_KEYWORDS` entry, shadowed regex pattern, defensive `Milestone.model_validate` unwrap in CHECK-IN fallback).
- [x] **Cut ⑤f** Graph wire: `plan_stub_node` → `plan_node` delegator (mirrors quiz_node pattern, retains plan_stub_node as fallback); state-aware router extended with `active_plan_id` → intent=plan; `judge_node` extends to plan rubric on GENERATE, skips on CHECK-IN (mirror ④g grade-skip). Edge `plan → judge` (new). PLAN_KEYWORDS extended with `进度` / `check in` / `check-in` / `调整`. 4 tests.
- [x] **Cut ⑤g** Production wiring: `deps.py::get_planner` factory (session-scoped repos + retriever + llm; lazy import); `routes.py` injects `planner` into config["configurable"]. 2 tests (single-turn SSE shape + two-turn GENERATE→CHECK-IN with shared session_id via InMemorySaver checkpointer). One Cut ④f test (`test_chat_via_graph_plan_path_emits_empty_citations_then_stub_tokens`) rewritten to assert real-planner behavior (was asserting stub fallback that no longer fires in production wiring). **Total 153 tests.**
- [x] **Cut ⑤h** Real-Ollama E2E (manual, 2026-05-22): 4 multi-turn scenarios verified — see "Real-run validation" below. Surfaced 3 UX bugs which become Cut ⑤i.
- [x] **Cut ⑤i** UX fixes from real-Ollama testing (+4 tests = **157 total**):
  - **Fix A** `planner.py` `_has_create_plan_keyword` helper + `force_generate` flag: explicit create-plan phrases (`帮我做` / `make a plan` / etc.) override the "active_plan_id → CHECK-IN" rule and force GENERATE. Closes the silently-dropped-mindmap bug when user asks for a fresh plan mid-session.
  - **Fix B** `graph.py::router_node` intent-aware override: replaces blunt `active_plan_id → plan` with `if active_plan_id and base_intent=="plan" → plan; if active_plan_id and base_intent=="tutor" → tutor; else → base_intent`. Lets users ask tutor questions mid-plan without switching session (spec §10 caveat now solved). Pre-existing Cut ⑤f test fixture message updated from `"把第二个里程碑标记完成"` to `"调整第二个里程碑标记为完成"` so its intent assertion remains valid under new semantics.
  - **Fix C** `planner.py` CHECK-IN recomputes `final_progress` against post-update milestones via duck-typed `FinalPlan` shim, so "Done: X / Y" denominator matches the displayed Updated Plan list length.

**Deferred ablation point (P2.2/P3)**: LLM tool-calling agent Planner / QuizMaster vs deterministic baseline (latency / correctness comparison; portfolio differentiation point — HKBU class report had neither true agent loop nor head-to-head ablation against one).

**Real-run E2E validation (2026-05-22)** — feedback.md captures 4 scenarios with stale `active_plan_id` from prior session (worst case):
- `帮我做学习计划 on HyDE` → 7 milestones grounded in HKBU PDF (`generate_hypothetical_document`, `LLMAsJudge`, `TokenAnalyzer`, `ablation runs`, `use_hyde: bool = True`); judge passed at 0.80 with `milestone_granularity` weak_dim flagged
- `What is HyDE?` (same session, active_plan_id set) → **Tutor path engaged** (Fix B), full answer with 5 citations from PDF pages 2/3/4/5
- `帮我做学习计划 on RAG 画脑图` (same session) → **fresh GENERATE on RAG** (Fix A force_generate detected `帮我做`) + valid mermaid `mindmap` block + markdown outline; topic correctly switched from HyDE to RAG
- `进度怎么样了` (same session) → CHECK-IN with `Done: 0 / 9` matching 9-item Updated Plan list (Fix C); LLM grew plan from 7 → 9 by adding milestones aimed at `weak_topics` hint (hypothetical document embedding, quantum computing)

**Spec / Plan docs:**
- Spec: `docs/superpowers/specs/2026-05-22-p2-1-5-plan-chain-design.md`
- Plan: `docs/superpowers/plans/2026-05-22-p2-1-5-plan-chain.md`

**Known limitations (carried to P2.2/P3):**
- **Stale memory noise**: weak_topics may include items from prior test sessions (e.g. "quantum computing" surfacing on a HyDE plan). Memory Hydrator is user-scoped, not session-scoped. Fix = add "clear mastery" UX or session-scoped hydration.
- **No frontend Plan Timeline view** — backend SSE contract unchanged; UI is P3 polish (see ROADMAP P3).
- **Plan rubric** judge runs but `milestone_granularity` flagged weak even on reasonable 7-milestone plans → threshold may need calibration after more E2E data.

#### P2.2 Agent Loop Ablation (Done, 2026-05-23)

**181 backend tests passing** (157 baseline + 24 new). 6 implementation cuts (①a-①e + ②a) via `superpowers:subagent-driven-development`; 3 manual cuts (①f real-Ollama smoke + ②b matrix run + ③ writeup). Spec at `docs/superpowers/specs/2026-05-22-p2-2-agent-loop-ablation-design.md`; plan at `docs/superpowers/plans/2026-05-23-p2-2-agent-loop-ablation.md`; report at `docs/EVAL.md`; blog at `docs/agent_loop_vs_deterministic.md`.

- [x] **Cut ①a** Tool wrappers + closure factory (`_make_planner_tools` in new `app/agent/planner_agent.py`): 5 LangChain `@tool` wrappers (`retriever_search` / `get_existing_plan` / `update_study_plan` / `generate_mindmap` / `compute_progress`). 3 delegate to existing pure functions (`tools/plan.py`, `progress.py`); 2 are direct repo wrappers. closure injects user_id / repos / llm so LLM-visible args stay clean. 5 tests, 162 passing.
- [x] **Cut ①b** `AgentTrace` dataclass + `IterationRecord` + `ToolCallRecord`: instrumentation backbone for eval. `serialize()` emits 9-key flat dict (iterations / tool_call_breakdown / tool_errors / input_tokens / output_tokens / wall_time_s / exit_reason / llm_error). 3 tests, 165 passing.
- [x] **Cut ①c** Loop body + `_infer_plan_action` + error handling (`build_planner_agent` factory): hand-written `while iteration in range(max_iter)` with 4 exit paths (`natural_stop` / `budget_exhausted` / `llm_call_failed` / tool errors recoverable via ToolMessage). `state.py` extended with `agent_trace: NotRequired[dict]`. Mirror of deterministic factory shape. 8 tests, 173 passing.
- [x] **Cut ①d** Graph mode-aware dispatch: `graph.py:plan_node` reads `configurable.planner_mode` and routes to `planner_agent` callable or falls through to deterministic `planner`. 3 tests, 176 passing.
- [x] **Cut ①e** Production wiring: `deps.py` adds `get_planner_mode` (reads `x-planner-mode` HTTP header) + `get_planner_agent` factory; `routes.py:chat()` signature extended with 2 new Depends; `configurable` dict gets 2 new keys. 2 route-level tests, 178 passing.
- [x] **Cut ①f** Real-Ollama smoke (manual, 2026-05-23):
  - **Phase A** pre-flight: 4 models pulled, backend tests green, Ollama API reachable.
  - **Phase B** `think=False` mechanism discovery: tested `ChatOllama(think=False)`, `model_kwargs={"think": False}`, `reasoning=False`, `/no_think` prefix, baseline. **Verdict**: `ChatOllama(reasoning=False)` is the only ChatOllama 1.1 kwarg that actually forwards to Ollama API `think` field. qwen3.5:4b speedup 813s → 7.3s (**32× verified**). `model_kwargs={"think": False}` is silently dropped because `think` is a TOP-LEVEL Ollama API param, not an `options` sub-field.
  - **Phase C** 8-cell happy-path smoke: every (model, mode) cell ran one request through `/api/chat`, all 8 emitted SSE `citations → token → done`. `gemma3:4b agent_loop` predicted-failed (Ollama API 400 `does not support tools` — 5s degrade via `_format_degrade_output`); other 7 cells produced valid plans (3-7 milestones each, grounded in HKBU PDF content). 8 judges passed 0.60-0.96.
- [x] **Cut ②a follow-up patch** (from ①f Phase B finding): `run_eval.py:ChatOllama(...)` got `reasoning=spec.thinking` forwarded. Without this, main matrix would have taken ~30 hours (qwen3.5:4b 813s × 84 specs). With it, ②b ran in ~5 hours.
- [x] **Cut ②a** Eval harness skeleton: new `app/eval/p2_2_agent_ablation/` package (6 files, 504 lines): `matrix.py` (RunSpec + expand_matrix) / `single_run.py` (validate_record_schema + run_one) / `judges.py` (local qwen2.5:7b + cloud MiniMax-M2.7 via OpenAI-compatible endpoint) / `run_eval.py` (resumable CLI with `filter_pending_specs`) / `queries.json` (10 single-turn + 2 multi-turn) / `__init__.py`. `summarize.py` added in ③. 3 harness tests, 181 passing.
- [x] **Cut ②b** Full matrix run (manual, 2026-05-23): **396 runs in ~5 hours** on 16GB Apple Silicon Mac. 0 harness_error rows; 1 budget_exhausted; 42 expected llm_call_failed (gemma3 agent_loop); 353 successful (natural_stop or deterministic). Total cost: ~$3 MiniMax API.
- [x] **Cut ③** Writeup: `docs/EVAL.md` (1100+ lines empirical report with 8 sections) + `docs/agent_loop_vs_deterministic.md` (1200-word portfolio blog responding to learn-claude-code's "agency = model + minimal harness" thesis).

**Headline results (from `docs/EVAL.md`)**:
- **gemma3:4b agent_loop: 100% llm_call_failed** (Ollama 400 `does not support tools` — predicted negative data point ✓; degrade handler clean).
- **Tool schemas rescue thinking models when reasoning=False**: `qwen3.5:4b` deterministic persistence collapses to 10% (garbled JSON without thinking), but agent_loop persists 86% because `Milestone.model_validate(m)` forces valid output. Same on `gemma4:e4b` (43% → 94%). **Tool schema as structural reasoning substitute** — refines learn-claude-code's "minimal harness" claim.
- **`gemma4:e4b` is agent_loop champion**: 100% natural_stop, 2.74 tool calls/run, 94% persistence, top-tied judge scores (local 0.856 / cloud 0.606).
- **`qwen2.5:7b` is "simplicity wins" tier**: deterministic tied agent_loop on local judge (both 0.770) but deterministic is 3× faster and 2.25× more persistent. Agent loop adds cost without quality for this tier.
- **Judges disagree systematically**: mean |local−cloud| 0.19-0.31; local qwen2.5:7b consistently +0.19 to +0.30 more generous than cloud MiniMax-M2.7 (self-preference bias visible when local judges its own model).
- **Latency cost of agent_loop**: 3-20× over deterministic on same model. qwen3.5:4b worst (20× at 73s vs 3.7s).

**Cloud-adapt hooks** (grep `# cloud-adapt:` returns ≥ 9 markers): tool description verbosity, max_iter scaling, agent_trace redaction, dispatcher provider-default switch, mindmap_default for cloud BYOK, MiniMax endpoint swap, think mechanism. All marker-only per spec §11.

**Spec deviations recorded in EVAL.md**:
- §1.Q1a gemma3:4b failure mode is Ollama API 400 (more severe than predicted "silent empty tool_calls"). Degrade handler still works.
- §1.Q1b main matrix `think=False` mechanism is `ChatOllama(reasoning=False)`, NOT `model_kwargs` (which `options`-routes the param into oblivion).
- §6.4 wall-time estimate 1-2 hours is too optimistic; actual ~5 hours with all model + judge swaps on 16GB Mac.

**Known limitations (carried to P3)**:
- **Deterministic path has no token cost data** (`agent_trace` is agent_loop-only). Add instrumentation in P3.
- **N = 36-72 per cell** — statistical power limited for ±0.05 effects.
- **Self-preference bias** between qwen2.5:7b local judge and qwen2.5:7b planner; cross-judge swap (non-thinking cloud judge, e.g. GPT-4o-mini) would isolate thinking-vs-non-thinking judge bias from local-vs-cloud bias.

**Spec / Plan / Report docs:**
- Spec: `docs/superpowers/specs/2026-05-22-p2-2-agent-loop-ablation-design.md`
- Plan: `docs/superpowers/plans/2026-05-23-p2-2-agent-loop-ablation.md`
- Report: `docs/EVAL.md`
- Blog: `docs/agent_loop_vs_deterministic.md`
- Raw data: `backend/app/eval/p2_2_agent_ablation/output/results.jsonl` (396 rows)
- Auto-summary: `backend/app/eval/p2_2_agent_ablation/output/summary.md`

#### P2.3 Quiz Agent Loop Ablation (Done, 2026-05-24)

**202 backend tests passing** (181 baseline + 21 new). 5 implementation cuts (①a-①e) via `superpowers:subagent-driven-development` + 3 manual cuts (①f real-Ollama smoke + ②b 396-record matrix run + ③ writeup). Spec at `docs/superpowers/specs/2026-05-24-p2-3-quiz-agent-loop-ablation-design.md`; plan at `docs/superpowers/plans/2026-05-24-p2-3-quiz-agent-loop-ablation.md`; report at `docs/EVAL.md` (P2.3 section appended, ~210 new lines).

- [x] **Cut ①a** AgentTrace refactor (`app/agent/agent_trace.py`): moved shared `AgentTrace`/`IterationRecord`/`ToolCallRecord` out of `planner_agent.py`; +1 method `last_persisted_question_id` for Quiz consumers; +2 new tests; P2.2's 3 trace tests pass byte-identical with import-path-only change.
- [x] **Cut ①b** Tool wrappers + `QuizQuestionPersist` schema (`app/agent/quiz_master_agent.py` + `tools/schemas.py`): 2 LLM-visible tools (`retriever_search` + `persist_quiz_question`); strict Pydantic schema (`Literal["A","B","C","D"]` answer + `min/max_length=4` options + `field_validator` for "A)/B)/C)/D) " prefix). 4 tool unit tests.
- [x] **Cut ①c** Loop body + factory (`build_quiz_master_agent`): `max_iter=6` while-loop, 4 exit paths (`natural_stop`/`budget_exhausted`/`llm_call_failed`/tool_error self-correction), mirror of P2.2 `build_planner_agent` shape. 6 loop tests.
- [x] **Cut ①d** Graph dispatcher (`graph.py:quiz_node`): state-aware (GRADE always routes to deterministic regardless of mode) + mode-aware (GENERATE reads `configurable.quiz_mode`). 3 graph e2e tests.
- [x] **Cut ①e** Production wiring: `deps.py` adds `get_quiz_mode` (reads `x-quiz-mode` header) + `get_quiz_master_agent` factory; `routes.py:chat()` extends signature + configurable. 3 route integration tests.
- [x] **Cut ①f** Real-Ollama smoke (manual): Phase B verified `ChatOllama(reasoning=False)` mechanism holds (qwen3.5:4b 3.0s); Phase C 8-cell happy-path confirmed grounded MCQ generation from HKBU PDF; `gemma3:4b agent_loop` degraded cleanly per P1 prediction.
- [x] **Cut ②a** Eval harness fork (`app/eval/p2_3_quiz_ablation/`): forked from `p2_2_agent_ablation/` with `quiz_mode` field + QUIZ_DIMENSIONS rubric + production retriever wired via `_build_default_retriever()`. 3 harness tests.
- [x] **Cut ②b** 396-record matrix run (manual, ~5h, ~$3 MiniMax): 0 harness_error. Distribution: `deterministic=196, natural_stop=110, llm_call_failed=33` (all gemma3 agent_loop), `budget_exhausted=10` (all qwen3.5:4b agent_loop), `n/a=45` (multi-turn GRADE dispatcher-routed), `error=2` (gemma3 det parse).
- [x] **Cut ③** EVAL append + blog continuation + ROADMAP/memory update.

**Headline findings (from `docs/EVAL.md` §Findings)**:
- **Schema rescue REVERSES on Quiz vs Plan**: P2.2 qwen3.5:4b agent_loop 86% persist (Plan, lax Milestone) → P2.3 50% persist (Quiz, strict QuizQuestionPersist), and deterministic Quiz hits 75%. Strict schema + small-model self-correct capability < what 6-iter budget allows → 25% `budget_exhausted` on qwen3.5:4b. **Schema-as-harness is conditional on a 2D fit (schema-strictness × model-capability)**, refining P2.2 §Finding 2 from a 1D claim.
- **Precision-recall trade-off**: filtered to `natural_stop`, agent_loop quality beats deterministic by +0.05 to +0.16 on cloud judge across qwen3.5/qwen2.5/gemma4. But agent_loop completes only 50-75% of runs. Mode choice depends on UX axis.
- **Alignment safety dimension (new, from pilot)**: `retriever=None` pilot run revealed agent_loop with `retriever_search` returning `"[]"` makes well-aligned LLMs decline to fabricate (21-39% refusal); deterministic prompt path has no equivalent signal and silently generates from training. Production implication: agent_loop should be default for grounding-essential tasks.

**Spec / Plan / Report docs**:
- Spec: `docs/superpowers/specs/2026-05-24-p2-3-quiz-agent-loop-ablation-design.md`
- Plan: `docs/superpowers/plans/2026-05-24-p2-3-quiz-agent-loop-ablation.md`
- Report: `docs/EVAL.md` (P2.3 section)
- Blog continuation: `docs/quiz_ablation_followup.md` (sister to `agent_loop_vs_deterministic.md`)
- Raw data main: `backend/app/eval/p2_3_quiz_ablation/output/results.jsonl` (396 rows, with production retriever)
- Raw data pilot: `backend/app/eval/p2_3_quiz_ablation/output/results_no_retriever.jsonl` (396 rows, retriever=None — alignment-safety pilot)

**Known limitations (carried to P3)**:
- **`max_iter=6` too tight for qwen3.5:4b** — follow-up cut at max_iter=12 would isolate "retry budget" from "schema strictness" as separate variables.
- Same statistical power limits as P2.2 (N=36-72/cell).
- Cross-judge swap (non-thinking cloud judge like GPT-4o-mini) still deferred — would isolate thinking-judge bias from cloud-vs-local bias.

#### P2.1 — Remaining / Carried Forward

- [x] Tools: ✅ `generate_quiz` / `grade_quiz_answer` / `record_mistake` / `update_mastery` delivered in P2.1-④. ✅ `update_study_plan` / `generate_mindmap` delivered in P2.1-⑤. Remaining: `hyde_rag_search` (judge_response delivered in P2.1-②).
- [x] DB tables: ✅ delivered in P2.1-③ above
- [x] `app/srs/sm2.py` ✅ delivered in P2.1-④a
- [x] Replace `quiz_stub` ✅ delivered in P2.1-④e (real QuizMaster). ✅ `plan_stub` → real Planner node delivered in P2.1-⑤e/⑤f (Reviewer = `judge_node` plan-rubric branch per spec Approach 2).
- [x] Frontend product shell delivered in P3: Overview, Plan Timeline, adaptive Quiz, Mistake Bank, Library, Settings.
- [ ] Goal Setup wizard remains P4.
- [ ] Full 1-week simulated study session remains P4 verification target.

**Skill triggers for P2**: `superpowers:systematic-debugging` (LangGraph trace bugs), `superpowers:tdd`.

## P3 — Done (Phase 3: productize)

### P3 — Done (shipped 2026-05-25)

**214 backend tests passing** (current baseline; P3 shipped at 213). 14 cuts via `superpowers:subagent-driven-development`. Spec at `docs/superpowers/specs/2026-05-25-p3-frontend-productize-design.md`; plan at `docs/superpowers/plans/2026-05-25-p3-frontend-productize.md`; report at `docs/EVAL.md` §"P3 Frontend Productize"; blog at `docs/p3_frontend_productize.md`.

Shipped:
- 4 new backend GET endpoints (plans/current, documents, mistakes/due, mastery) + 1 new repo method (MasteryRepository.list_for_user_detailed) + 1 new joined query (MistakeRepository.list_due_with_details)
- 4 new views (Overview / PlanTimeline / QuizAdaptive / MistakeBank) + 11 new components + 5 new Pinia stores (plan / quiz / mistakes / mastery / documents) + 1 derived store (overview)
- 3 new frontend deps: `mermaid` (Plan mindmap), `chart.js` + `vue-chartjs` (Overview radar), `lucide-vue-next` (icons)
- Design system MASTER.md locked: Modern Dark Cinema + Inter / JetBrains Mono / Noto Sans SC + indigo primary
- P2.3 §F3 alignment-safety operationalized via dual-channel `<EmptyCorpusBanner>`
- ModeChip per-view default + per-message override → settings persistence

- [x] **UI/UX polish** with `ui-ux-pro-max` — Modern Dark Cinema style locked at `design-system/MASTER.md`; Tailwind 4 `@theme` tokens; full responsive (>=768px) + dark mode default. shadcn-vue migration deferred (current component count well under threshold).
- [x] Eval page: P2.2 / P2.3 ablation surfaced indirectly via `<ModeChip>` per-view default + Settings dropdowns — user can A/B both modes side-by-side. Dedicated `/eval` page not built; the production UX *is* the artifact.
- [x] `docs/EVAL.md` — judge guardrail methodology + ablation results (P2.2 + P2.3 + P3 sections, ~1100 + ~210 + ~120 lines respectively)
- [ ] i18n (en/zh) via `vue-i18n` — deferred to P4; current UI is English-default with copy primed for extraction
- [ ] Real OAuth (NextAuth-equivalent in Python: `Authlib`) + email login; FingerprintJS becomes anonymous tier
- [ ] Shared plans (read-only public link with token, like JadeAI `share/[token]`)
- [ ] Group study mode (multi-user goal + shared mistake bank)
- [ ] Docker Compose + deploy (fly.io / railway)
- [ ] `docs/ARCHITECTURE.md` v2 (full JadeAI-grade: ER diagram, ADRs, deployment topology)
- [ ] `docs/PROMPT_ENGINEERING.md` — port class report + new agent-prompt design notes

**Skill triggers for P3**: `ui-ux-pro-max:ui-ux-pro-max` (UI), `example-skills:frontend-design` (visual polish), `superpowers:requesting-code-review` (pre-merge audit), `superpowers:finishing-a-development-branch` (release prep).

## P4 — Done (deploy, demo readiness, ARCHITECTURE.md v2)

**Shipped 2026-05-27. Then-current baseline: 252 backend tests, frontend build passing.**

### P4a — Deploy & Auth Hardening
- [x] Google OAuth + JWT auth (`app/auth.py`, `app/api/auth_routes.py`)
- [x] `get_current_user` dependency; all routes migrated from `x-fingerprint` to `Authorization: Bearer`
- [x] Anonymous login (guest tier) + guest→Google upgrade
- [x] User model extended: `google_id`, `email` columns
- [x] Frontend auth integration: auto-provision anonymous JWT, `authHeaders()` on all API calls
- [x] Real Google One Tap sign-in (GIS, `GET /api/auth/config`)
- [x] Docker Compose (backend + frontend + ollama)
- [x] Initial fly.io configuration scaffold (`fly.toml`, `Dockerfile.fly`); later review classified it as deferred and unverified rather than a supported fallback
- [x] `GET /api/health` returns `ollama_enabled` flag

### P4b — Product Polish
- [x] Agent visibility Debug Mode (SSE `trace` / `agent_run` events + TracePanel)
- [x] Goal Setup 3-step wizard (Onboarding view + `POST /api/goals`)
- [x] i18n with `vue-i18n` (en/zh-CN), language switcher, full view extraction
- [x] Streak + coverage computation (`GET /api/users/me/stats`, radar integration)
- [x] Activity heatmap (GitHub-style 30-day grid on Overview)
- [x] Gantt vertical timeline on Plan page
- [x] ~~Drag-reorder milestones~~ (removed — unreliable with native HTML5 DnD)
- [x] Mistake "Mark understood" (SM-2 quality=5 → 100-year future due date)
- [x] MCQ format hardening (prompt constraint CRITICAL FORMATTING section)

### P4c — Mobile & Docs
- [x] Mobile responsive layout (MobileNav bottom tab bar, Chat/Quiz/Plan <768px)
- [x] `useMediaQuery` composable
- [x] ARCHITECTURE.md v2: Mermaid ER diagram, 5 ADRs at P4 delivery, deployment topology, security model, A-tier expansion placeholders

### P4d — Chat Persistence & Activity Evidence
- [x] `/api/chat` creates or reuses a persisted `ChatSession` and emits `{type:"session", session_id}` before graph events
- [x] User and assistant turns persist to `messages`; assistant citations persist to `citations`
- [x] `GET /api/chat/sessions/current` and `GET /api/chat/sessions/{id}/messages` restore the current Chat view after frontend refresh
- [x] `/api/chat` refuses before retrieval when the current user has no uploaded Library documents
- [x] `GET /api/users/me/stats.total_sessions` now counts real sessions instead of returning `0`

### P4e — Recoverable Agent Run Trace
- [x] `AgentTrace.serialize_public()` exposes UI-safe agent-loop summaries without changing eval-facing `serialize()`
- [x] Planner / QuizMaster `agent_loop` paths stream `{type:"agent_run", run}` after the assistant token output
- [x] Assistant messages persist `citations` and `agent_run` in `messages.tool_calls_json` schema `assistant_artifacts.v1`; legacy plain citation lists still restore source names
- [x] Chat restore maps `agent_run` back to frontend `Message.agentRun`
- [x] Debug Mode TracePanel renders latest Agent Run summary and redacted tool-call previews

### P4.5 — Product Closure for Portfolio Demo (Verified)
- [x] Scope locked in `docs/superpowers/specs/2026-07-01-p4-5-product-closure-design.md`
- [x] Reviewer demo guide drafted in `docs/DEMO.md`
- [x] Quiz route test isolation: backend suite should not require live Ollama
- [x] Chat quiz strong consistency: visible MCQ must be persisted and gradeable
- [x] `persist_quiz_question` tolerates narrow format near-misses (`A)` answers, missing option prefixes) before strict validation
- [x] Failed quiz persistence degrades without showing an answerable MCQ
- [x] Debug Mode shows recoverable, redacted evidence of persist success/failure after refresh
- [x] Manual-demo follow-up (2026-07-20): agent-loop Quiz GENERATE renders only the persisted question/options, never model-authored answer/explanation; trailing retrieval metadata is removed before explanation persistence
- [x] Automated verification: backend tests and frontend build
- [x] Manual browser demo: upload user-owned PDF → grounded Chat answer → Chat quiz → grade → refresh restore

### Deferred to beyond P4
- PROMPT_ENGINEERING.md (skipped — content already in ADRs + EVAL.md)
- Library auto-redirect after upload
- Full Chroma corpus isolation: add `user_id` / `document_id` metadata to chunks and pass filters through dense, BM25, reranking, deterministic nodes, and agent-tool retrieval. Current P4d guard only refuses retrieval when the current user has no Library document rows.
- shadcn-vue migration

## P5 — Done (2026-07-31)

Design approved 2026-07-20; implementation, automated verification, and destructive current-head Chrome acceptance are complete. P5 deliberately treats Study Coach as a local-first, single-user portfolio product; multi-user auth and Chroma ownership move to a later isolated worktree.

- [x] Product direction, data scopes, failure semantics, startup gate, Danger Zone, and notification design approved.
- [x] Design spec: `docs/superpowers/specs/2026-07-20-p5-local-first-data-lifecycle-design.md`.
- [x] P5.0: remove incomplete Google OAuth UI/GIS runtime and align local-first README/product copy.
- [x] P5.1: backend summary and idempotent two-scope reset across Chroma, graph state, retriever caches, SQLite, local-mode/loopback deployment guard, Docker Chroma path fix, and unique temporary-upload cleanup.
- [x] P5.2: Vitest foundation and capability-aware, required once-per-tab startup gate (Continue plus Start fresh when local reset is enabled).
- [x] P5.3: Settings Danger Zone, confirmation flows, global notifications, accessibility, cross-tab lifecycle handling, and stale-request-safe client refresh.
- [x] P5.4 automated verification and architecture/product documentation sync (re-verified 2026-07-31 remediation HEAD): full backend 411/411; frontend 130/130 across 14 files; production build and Compose render passed. Host-run is the canonical reviewer path; Compose is an alternative mutually exclusive Ollama path. Compose routes backend embeddings through `http://ollama:11434`, binds Ollama to loopback, and pre-pulls `nomic-embed-text`, `gemma3:4b`, and `qwen2.5:7b`. Clean image build previously reduced backend/frontend contexts to 47.96 kB / 4.77 kB with runtime HTML/direct-health/proxied-health smoke HTTP 200; the existing Vite chunk warning is accepted.
- [x] P5.4 final current-head Chrome acceptance: startup choice and Esc/backdrop blocking, Continue + same-tab refresh, real Ollama Chat/Quiz/Plan, learning reset with cross-tab acknowledgement and Settings preservation, 49-chunk re-import, Factory reset default restoration and peer-tab reload, old-JWT write refusal/idempotent retry, and concurrent three-tab recovery to exactly one new anonymous user.

**Deferred after P5:** a separate `feature/multi-user-auth` worktree for guest upgrade, Google OAuth, SQL/Chroma ownership, legacy migration, and data-continuity tests.

## Out of scope (won't do unless asked)

- Mobile-native apps (Flutter / React Native) — web is enough for portfolio
- Voice mode (TTS / STT)
- Multi-modal input (image / video lectures)
- Real-time collaboration (operational transform)
- Plugin marketplace / extensions
