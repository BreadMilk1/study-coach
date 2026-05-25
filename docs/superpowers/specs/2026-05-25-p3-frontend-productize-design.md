# P3 Frontend Productize — Design Spec

> Phase 3 deliverable. Productizes the stable backend (202 tests, dual-mode dispatchers shipped P2.2 + P2.3) into a portfolio-grade 7-view Vue 3 app. Backend stays byte-identical except for 4 minimal `GET` endpoints (3 new resource fetchers + 1 documents-list promoted from §4.5 during brainstorm).

- **Date**: 2026-05-25
- **Baseline**: 202 backend tests passing; only `POST /api/chat` + `POST /api/documents` + `GET /api/health` exist; frontend is P1 minimal (Chat/Library/Settings).
- **Scope (decided in brainstorm)**: Bundle 2 · All Mid — 4 new views at "Mid" fidelity (mermaid mindmap, radar chart, SM-2 schedule display, alignment-safety banner).
- **Anchor decisions**:
  - Q1 — Routing: **Dashboard-first** (Option C). `/` is Overview hero; Chat moves to `/chat`.
  - Q2 — Per-view fidelity: **All Mid** (Bundle 2). Mid upgrades are pure-frontend (mermaid / chart / banner / SM-2 display); 0 backend cost vs MVP.
  - Q3 — Mode UX: **View-default + per-message override** (Option C). Plan/Quiz default to `agent_loop`; per-view `<ModeChip>` flips a single message to deterministic for demo.
- **Drives portfolio narrative for**: P2.2 + P2.3 ablations (`docs/agent_loop_vs_deterministic.md` + `docs/quiz_ablation_followup.md`) — UI must surface the dual-mode finding and the P2.3 §F3 alignment-safety property.

---

## 1. Architecture overview

### 1.1 Stack (locked, no additions outside the two below)

| layer | tech | version | notes |
|---|---|---|---|
| Framework | Vue 3.5 (Composition API + `<script setup lang="ts">`) | existing | |
| State | Pinia 3 | existing | one store per backend resource; no cross-store coupling |
| Router | vue-router 5 | existing | history mode; 7 routes |
| Styling | Tailwind 4 (via `@tailwindcss/vite`) | existing | dark theme `#11162a` / indigo accents (P1 baseline) |
| Build | Vite 8 + TS 6 + vue-tsc | existing | `vue-tsc -b && vite build` is the only build gate |

**New deps (3 only)**:
- `mermaid` (~3 MB gzip) — Plan view mindmap render
- `chart.js` 4 + `vue-chartjs` 5 — Overview radar chart (locked by `ui-ux-pro-max` phase)
- `lucide-vue-next` — icon library (locked by `ui-ux-pro-max` phase)

**Visual system locked**: see `design-system/MASTER.md` (style = Modern Dark Cinema (Inter System), palette = P1 dark+indigo formalized into tokens, typography = Inter + JetBrains Mono + Noto Sans SC fallback). All cuts must conform unless they declare an override in `design-system/pages/<page>.md`.

### 1.2 Shell modification (`App.vue` + `router.ts`)

- Left nav grows from 3 → 7 links (Overview, Chat, Plan, Quiz, Mistakes, Library, Settings).
- `/` was Chat, becomes Overview. Chat moves to `/chat`.
- All existing `<RouterLink to="/">` (currently only in `Chat.vue` empty-state) updated to `/library` target stays.
- Existing dark theme palette unchanged (style refinement is the `ui-ux-pro-max` phase, not this spec).

---

## 2. Routing topology + Pinia stores

```
/             Overview.vue       ← stores/overview.ts (derived; aggregates mastery + plan + mistakes)
/chat         Chat.vue           ← stores/chat.ts (existing)
/plan         PlanTimeline.vue   ← stores/plan.ts
/quiz         QuizAdaptive.vue   ← stores/quiz.ts
/mistakes     MistakeBank.vue    ← stores/mistakes.ts
/library      Library.vue        ← existing
/settings     Settings.vue       ← stores/settings.ts (extended)
```

**Store separation principle**: 1 store ↔ 1 backend resource. `overview.ts` is derived only — it imports `usePlan`, `useMistakes`, `useMastery` via Pinia composition and exposes `computed` rollups; no independent fetch logic.

**Settings store new fields**:
```ts
default_planner_mode: 'agent_loop' | 'deterministic'  // default 'agent_loop'
default_quiz_mode:    'agent_loop' | 'deterministic'  // default 'agent_loop'
judge_model:          string | null                   // optional x-judge-model override
```

**Persisted to `localStorage`** alongside existing BYOK fields. No backend write.

---

## 3. Per-view component breakdown (Bundle 2 Mid)

### 3.1 Overview (`/`) — portfolio hero

```
<UploadGate>                ← only when /api/documents returns chunks_count=0
<header> "Today is N | Overdue: N | Streak: -"  (streak placeholder)
<grid>
  <MasteryCard>             top-5 topic scores + bar
  <PlanProgressCard>        done/total + next-due milestone
  <MistakesDueCard>         count + first 3 question stems
  <WeakTopicsChips>         weak_topic names as chips → click filters to /quiz?topic=X
  <RadarChart>              5-dim mastery radar (deferred dim choice to ui-ux-pro-max)
</grid>
```

`<UploadGate>` is dismissible per-session (in-memory). Reappears on refresh until upload happens.

### 3.2 PlanTimeline (`/plan`)

```
<ViewHeader> "Study Plan" + <ModeChip mode="planner">
<MilestoneList>
  ↳ MilestoneRow: <CheckIcon> title (due_at colour: red overdue / amber today / grey future)
                  optional topic chip
<MindmapPanel collapsible>   mermaid render of last generate_mindmap output (parsed from chat history)
<Footer> [Check-in progress] button → POSTs to /api/chat ("进度怎么样了") with default mode
```

- **Mindmap source**: parse from chat token stream when planner emits `\`\`\`mermaid\n...\n\`\`\`` block; store latest in `plan.ts`. No backend endpoint for mindmap (it's a chat artifact, not persisted in DB at present).
- **Check-in button**: routes user to /chat with prefilled message AND also triggers SSE inline (single-call). UX detail: which behavior wins is deferred to A2 implementation (test both, pick the one that feels less jarring).

### 3.3 QuizAdaptive (`/quiz`)

```
<ViewHeader> "Quiz" + <DifficultySelector easy|med|hard> + <ModeChip mode="quiz">
<EmptyCorpusBanner v-if="needsUpload">    ← P2.3 §F3 alignment-safety
  "Quiz needs source material to ground questions. Upload a PDF →" <UploadButton>
<MCQCard v-else>
  question prompt
  4 radio options A) / B) / C) / D)
  [Submit] button (disabled until selection)
<GradeResult v-if="lastGrade">
  ✓/✗ + correct answer + explanation
  [Next question] button → re-submit GENERATE with same difficulty
```

**`needsUpload` triggers**:
1. On view mount: `useLibrary().totalChunks === 0` (fast path).
2. During GENERATE stream: regex on token buffer matches `/(unable to retrieve|no .* found in.*source|please provide.*context)/i` (slow path; catches retriever-empty refusal even if Library cache is stale).

**`difficulty` injection**: prepended to chat message as `"[difficulty:hard] quiz me on {topic}"`. Backend `quiz_master` already extracts topic via regex; difficulty hint is forwarded to LLM via existing `generate_quiz` system prompt slot (verify with TDD in A6; if backend doesn't support, document deviation and ship "selector is UI-only, hint visible in chat history" as fallback).

**`mistake_id` query param**: when present, fetches the mistake's `question_id` → topic → prefills chat message as `"quiz me on {topic}"` and skips generate (uses existing GRADE flow on the same question_id). Routing from Mistake Bank Redo button.

### 3.4 MistakeBank (`/mistakes`)

```
<ViewHeader> "Mistake Bank · N due today"
<MistakeList>
  ↳ MistakeRow:
      question stem (truncated)
      topic chip
      next-review: "in 3 days" (computed from due_at)
      ease: 2.5 / interval: 6d  (small label, SM-2 details)
      [Redo] → router.push('/quiz?mistake_id=' + id)
<EmptyState v-if="no mistakes">   "No mistakes due. Take a quiz to start tracking."
```

---

## 4. Backend `GET` endpoints (4 new — minimum required by Bundle 2)

All added to `app/api/routes.py`, mirror existing dep-injection pattern. **No model/migration changes.** Each endpoint TDD red→green per cut.

### 4.1 `GET /api/plans/current?user_id={uuid}`

```python
# Response 200
{
  "plan_id": "uuid",
  "goal_id": "uuid",
  "goal_title": "Master HyDE for HKBU exam",
  "milestones": [
    {"title": "Read chapter on HyDE", "due_at": "2026-05-26T00:00:00", "done": false, "topic": "HyDE"}
  ],
  "updated_at": "2026-05-25T10:00:00"
}
# Response 404
{"detail": "no active plan for user"}
```

**Backing repo method (new)**: `PlanRepository.get_current_for_user(user_id) -> Plan | None` — picks the most recent plan for user's active goal. Resolves goal via existing `GoalRepository.get_active_for_user` (already exists per P2.1-③ schema).

### 4.2 `GET /api/mistakes/due?user_id={uuid}&limit=20`

```python
# Response 200
[
  {
    "mistake_id": "uuid",
    "question": {
      "id": "uuid", "prompt": "...", "options": ["A) ...","B) ...","C) ...","D) ..."],
      "answer": "B", "explanation": "..."
    },
    "due_at": "2026-05-25T08:00:00",
    "srs_interval_days": 6,
    "srs_ease": 2.5,
    "topic_name": "HyDE"
  }
]
```

**Backing repo method (new)**: `MistakeRepository.list_due_for_user(user_id, limit, now)` — `WHERE user_id = X AND due_at <= now ORDER BY due_at LIMIT N`, joins `questions` + `topics` for the embedded payload.

### 4.3 `GET /api/mastery?user_id={uuid}`

```python
# Response 200
{
  "scores": [
    {"topic_id": "uuid", "topic_name": "HyDE", "score": 0.72, "last_reviewed": "2026-05-24T..."}
  ],
  "weak_topics": ["BM25", "RRF"],          # score < 0.5, sorted by score asc, max 5
  "overdue_milestones_count": 3            # joins active plan; counts done=false AND due_at < now
}
```

**Backing repo methods (new)**: `MasteryRepository.list_for_user(user_id) -> list[Mastery]` (joined with Topic for `topic_name`); `weak_topics` + `overdue_milestones_count` derived in route handler (not repo) — pure functions over the loaded lists.

### 4.4 `user_id` injection from frontend

- `lib/api.ts` reads `useSettings().fingerprint` (existing P1 field) and appends `?user_id=` to all 3 `GET` calls.
- `POST /api/chat` already passes user_id in body — no change.
- **No auth header.** P3 stays in FingerprintJS-anonymous tier per CLAUDE.md scope.

### 4.5 Existing endpoints reused

- `POST /api/chat` — unchanged; mode headers injected from settings + per-view `<ModeChip>` override.
- `POST /api/documents` + (implicit) `GET /api/documents` — **`GET /api/documents` does NOT exist today**. Library.vue lists docs uploaded in current session via local state. For Overview's `<UploadGate>` we need `totalChunks` count. **Decision**: add `GET /api/documents?user_id=X → [{id, filename, chunks_count}]` as a **4th** cut (A4.5) — minimal,15 lines. Not in original "3 new GET" promise; flagged as scope expansion at brainstorm review time but unavoidable for Bundle 2 Overview correctness.

**Updated count: 4 new GET endpoints** (plans/current, mistakes/due, mastery, documents).

---

## 5. Mode dispatch flow

### 5.1 Headers (production-wired since P2.2 / P2.3)

| header | values | default | injected when |
|---|---|---|---|
| `x-planner-mode` | `agent_loop` / `deterministic` | server-side: `deterministic` | from /plan view only (or explicit override) |
| `x-quiz-mode` | `agent_loop` / `deterministic` | server-side: `deterministic` | from /quiz view only (or explicit override) |
| `x-judge-model` | model id (e.g. `qwen2.5:7b`) | unset (same as main) | from Settings, all views |

### 5.2 View defaults (frontend layer)

```ts
// stores/settings.ts:  user-tunable defaults
default_planner_mode: 'agent_loop'   // P2.2 Finding 4: agent_loop optimizes quality (filter natural_stop)
default_quiz_mode:    'agent_loop'   // P2.3 Finding 3: alignment-safety net via tool feedback
judge_model:          null           // optional cross-judge

// Per-view derivation (in each view's setup):
const mode = ref(settings.default_planner_mode)        // /plan default
const mode = ref(settings.default_quiz_mode)           // /quiz default
// /chat: no mode chip, sends no override headers
```

### 5.3 `<ModeChip>` per-message override

- Tiny pill in view header: `agent_loop ⇄ deterministic`.
- Click flips local `mode.value` for the **next chat send only**, then auto-restores to view default.
- `streamChat()` signature extended:
  ```ts
  streamChat(text, settings, { plannerMode?, quizMode?, judgeModel? }, callbacks)
  ```
  When overrides are absent → headers omitted → backend uses server defaults.

---

## 6. P2.3 §Finding 3 alignment-safety UX

**Source of truth**: `docs/EVAL.md` §"P2.3 Quiz Agent Loop Ablation" §Finding 3 — agent_loop with empty retriever causes well-aligned models to decline fabrication. The product implication is: never show this refusal as an error.

### 6.1 Detection (dual-channel for robustness)

1. **Pre-flight**: on Quiz/Overview mount, fetch `GET /api/documents` → if `sum(chunks_count) === 0` → render `<EmptyCorpusBanner>`/`<UploadGate>` instead of MCQ/widgets.
2. **In-flight (Quiz)**: during GENERATE token stream, regex match `/(I'm unable to retrieve|no .* available|cannot quiz.*without|please provide.*context)/i` against accumulated text. On match: abort SSE consume, render banner overlay, do not display refusal raw text to user.

### 6.2 Banner content

```
┌────────────────────────────────────────────────┐
│  📚  Quiz needs your study materials           │
│                                                │
│  Upload a PDF in Library to start generating   │
│  questions grounded in your sources.           │
│                                                │
│         [Upload PDF →]                         │
└────────────────────────────────────────────────┘
```

Click → `router.push('/library')` with `?return=/quiz` query so Library auto-redirects back after successful ingest.

### 6.3 Why both channels

- Pre-flight catches the obvious empty-corpus case (Library never used).
- In-flight catches mid-session corpus deletion, stale chunks_count cache, or future cloud BYOK model with different refusal phrasing.

`# cloud-adapt:` marker in `<EmptyCorpusBanner>` detection regex — cloud GPT-4 / DeepSeek may use different refusal language; extend regex when wiring cloud BYOK.

---

## 7. Cut sequencing (vertical slice + shared infra first)

14 cuts; each cut = TDD red→green + chrome-devtools MCP screenshot/behavior verification before next cut.

| # | cut | type | new tests |
|---|---|---|---|
| A0 | Shell + Pinia store skeletons + `lib/api.ts` user_id injection + mode header injection scaffold | frontend infra | type-check only (frontend has no test runner per §10) |
| A1 | `GET /api/plans/current` + `PlanRepository.get_current_for_user` | backend TDD | +3 (repo + route + 404) |
| A2 | `PlanTimeline.vue` + `MilestoneList` + `ModeChip` + connect A1 | frontend + visual | DevTools screenshot gate |
| A3 | `MindmapPanel` (mermaid integration) — parse mermaid from chat history | frontend | DevTools screenshot |
| A4 | `GET /api/documents` (4.5 above) + `DocumentRepository.list_for_user` | backend TDD | +3 |
| A5 | `GET /api/mistakes/due` + `MistakeRepository.list_due_for_user` | backend TDD | +3 |
| A6 | `MistakeBank.vue` + Redo flow + `?mistake_id` query support in router | frontend | DevTools screenshot + click-through |
| A7 | `QuizAdaptive.vue` + `DifficultySelector` + `MCQCard` + `GradeResult` | frontend | DevTools screenshot + full GENERATE→GRADE click-through |
| A8 | `EmptyCorpusBanner` + dual-channel detection (pre-flight + in-flight regex) | frontend | DevTools — manually clear corpus + verify banner shows |
| A9 | `GET /api/mastery` + `MasteryRepository.list_for_user` + handler-side derivations | backend TDD | +4 (repo + route + weak_topics + overdue_count) |
| A10 | `Overview.vue` + 4 widget cards + `UploadGate` | frontend | DevTools — full screenshot of /  |
| A11 | `RadarChart` (chart lib pick + integration) | frontend | DevTools |
| A12 | `Settings.vue` extension: default modes + judge_model fields | frontend | DevTools — change + reload persistence |
| A13 | Full E2E: chrome-devtools 7-view screenshot batch + 1 study-loop video doc + EVAL.md update + ROADMAP § P3 entry + sister blog post | docs + verification | DevTools batch |

**Estimated wall time**: ~10 work days (matches Bundle 2 estimate; backend cuts ~30 min each given existing repo patterns + alembic untouched; frontend cuts 2-4 h each including chrome-devtools verification).

**Vertical slice discipline**: Every backend cut (A1/A4/A5/A9) ships its consumer view in the next cut. Avoids "REST endpoints exist but nothing uses them" tech debt.

---

## 8. File modification map

### 8.1 Backend (additive only — byte-identical existing files except `routes.py`)

| file | change | risk |
|---|---|---|
| `app/api/routes.py` | +4 `@router.get(...)` handlers, ~80 lines | low — pattern matches existing handlers |
| `app/db/repositories/plan_repo.py` | +`get_current_for_user(user_id)` | low |
| `app/db/repositories/document_repo.py` | +`list_for_user(user_id)` | low |
| `app/db/repositories/mistake_repo.py` | +`list_due_for_user(user_id, limit, now)` | low |
| `app/db/repositories/mastery_repo.py` | +`list_for_user(user_id)` | low |
| `app/api/deps.py` | +4 repo dep factories if not already present | low |
| `tests/` | +4 test files (~12-15 new tests total) | adds to 202 baseline |

### 8.2 Frontend

**New files (12)**:
- `src/views/Overview.vue`, `PlanTimeline.vue`, `QuizAdaptive.vue`, `MistakeBank.vue`
- `src/components/MilestoneList.vue`, `MindmapPanel.vue`, `ModeChip.vue`, `MCQCard.vue`, `DifficultySelector.vue`, `GradeResult.vue`, `EmptyCorpusBanner.vue`, `UploadGate.vue`, `MistakeRow.vue`, `WeakTopicsChips.vue`, `RadarChart.vue`, `MasteryCard.vue`, `PlanProgressCard.vue`, `MistakesDueCard.vue`
- `src/stores/plan.ts`, `quiz.ts`, `mistakes.ts`, `mastery.ts`, `overview.ts`, `documents.ts`

**Modified files (5)**:
- `src/App.vue` — nav 3→7 links
- `src/router.ts` — `/` route rewired to Overview; +5 routes
- `src/lib/api.ts` — user_id injection + mode header injection + new `streamChat` overrides param + 4 new GET functions
- `src/stores/settings.ts` — 3 new fields + localStorage persistence
- `src/views/Chat.vue` — RouterLink targets unchanged; just verify still works at `/chat`

**New deps**: `mermaid` + 1 chart lib (chosen in `ui-ux-pro-max` phase, default to `@unovis/vue` if no opinion surfaces).

---

## 9. Testing strategy

### 9.1 Backend (TDD red→green per cut, matches P2.X discipline)

- **Per repo method**: 1 unit test against in-memory SQLite (StaticPool fixture — already exists from P2.1-③).
- **Per route**: 1 happy-path test (FastAPI `TestClient` + dep overrides) + 1 edge test (404 or empty list).
- **Total new tests**: ~14-16; ends ~216-218 baseline.
- **`run_eval.py` matrices untouched** — P2.2/P2.3 reproducibility preserved.

### 9.2 Frontend

- **No JS test framework** introduced (per §10 Out of scope rationale).
- **vue-tsc** compile is the only static gate (`pnpm build` must pass per cut).
- **chrome-devtools MCP** is the per-cut verification gate:
  1. `pnpm dev` start dev server in background.
  2. `mcp__chrome-devtools__navigate_page` to the route under test.
  3. `mcp__chrome-devtools__take_snapshot` + `take_screenshot` for visual regression spot-check.
  4. `mcp__chrome-devtools__click` / `fill` to exercise primary interaction.
  5. `mcp__chrome-devtools__list_console_messages` to catch runtime errors.
- **No cut completes until both type-check passes AND DevTools snapshot is clean.**

### 9.3 E2E (A13)

- Manual chrome-devtools driven script: full study loop = upload PDF → generate plan → quiz → wrong answer → mistake appears in bank → redo → grade correct → mastery score moves on Overview.
- 7 screenshots (one per view) committed to `docs/screenshots/p3/` as portfolio artifacts.
- 1 video/GIF stitched from screenshots (or skipped — flagged for A13 implementer).

---

## 10. Out of scope (explicit non-goals)

- **OAuth / multi-user** — stays FingerprintJS anonymous; P4.
- **Drag-reorder milestones / Gantt time-axis / activity heatmap** — Bundle 4 features deferred.
- **Mistake batch mode / "mark as understood" without re-quiz** — Bundle 4.
- **i18n / theme switcher** — P4 (current dark theme is the only theme).
- **shadcn-vue full component library migration** — components written ad-hoc per cut; full library swap is P4.
- **Frontend unit test framework (Vitest)** — vue-tsc + chrome-devtools verification is sufficient at P3 scope; introducing Vitest is a multi-cut commitment we don't need for shipping.
- **Playwright E2E framework** — chrome-devtools MCP replaces it for this stage.
- **`POST /api/plans/{id}/milestone-toggle`** — milestone done-state toggle via plan agent CHECK-IN message; no direct REST. (Promoted from Mid; would shift to Full.)
- **`POST /api/mistakes/{id}/understood`** — Bundle 4.
- **Real-time updates** (WebSocket / SSE for stores other than chat) — stores refetch on view mount; user explicitly clicks refresh otherwise.

---

## 11. Cloud-adapt markers (P3 placements)

Per CLAUDE.md memory `feedback_cloud_model_adaptation_hooks`, mark (do not implement) where cloud BYOK GPT / DeepSeek would flip behavior:

- `src/components/EmptyCorpusBanner.vue` — refusal phrase regex (cloud models phrase differently).
- `src/lib/api.ts` — mode header injection (cloud BYOK may need additional `x-provider` / `x-api-key` flow already wired in P1).
- `src/stores/settings.ts` — default modes (cloud BYOK could default `quiz_mode` to deterministic since hallucination risk is lower with stronger models).
- `src/components/ModeChip.vue` — visibility (cloud BYOK might hide the chip — single-mode UX).
- `src/components/RadarChart.vue` — dim labels (cloud BYOK could expose extra judge dims).

Total markers: ≥5. All `# cloud-adapt:` HTML/TS comment style.

---

## 12. Verification gate per cut (mirrors P2.2/P2.3 discipline)

Each cut is "done" only when:
1. **Backend cut**: `cd backend && uv run pytest` passes (count strictly increases by expected delta, 0 regressions).
2. **Frontend cut**: `cd frontend && pnpm build` passes (vue-tsc clean).
3. **Verification cut**: chrome-devtools MCP exercise:
   - navigate → snapshot → key click → console messages clean.
   - Visual screenshot saved to `docs/screenshots/p3/cut-A{N}.png`.
4. **2-stage subagent review** (per `superpowers:subagent-driven-development`): implementer-subagent → code-quality reviewer subagent → fix issues → green.

**Do not proceed to next cut without all 4.**

---

## 13. Deliverable per cut + final

Per cut: code + tests + DevTools screenshot.

Final (A13):
- `docs/EVAL.md` appended §P3 — UX integration summary referencing P2.2/P2.3 findings.
- `docs/ROADMAP.md` §P3 marked done; §P4 list refined based on what surfaced.
- `docs/p3_frontend_productize.md` — sister blog post to `agent_loop_vs_deterministic.md` + `quiz_ablation_followup.md`; covers visual style choices, mode-dispatch UX rationale, alignment-safety banner story.
- `docs/screenshots/p3/` — 14 screenshots (one per cut) + 7 final view screenshots.
- Memory `project_study_coach_refactor` updated with P3 completion + lessons learned.

---

## References

- Brainstorm session: 2026-05-25 (this date)
- P2.2 report: `docs/EVAL.md` §"Agent Loop Ablation"
- P2.3 report: `docs/EVAL.md` §"P2.3 Quiz Agent Loop Ablation" — §Finding 3 is the primary product driver for §6 above.
- P2.3 blog: `docs/quiz_ablation_followup.md` — §"The dimension P2.2 didn't anticipate" is what `<EmptyCorpusBanner>` operationalizes.
- ARCHITECTURE.md §5 (FastAPI Routes): target REST table — P3 ships 4 of the not-yet-existing GET endpoints from that list.
- CLAUDE.md memories: `feedback_cloud_model_adaptation_hooks` (cloud-adapt comments), `feedback_scaffold_write_readfirst` (Read before Write on any scaffolded config).
- Not a git repo — no `git init/commit/push`; spec lives at this path as the source of truth.
