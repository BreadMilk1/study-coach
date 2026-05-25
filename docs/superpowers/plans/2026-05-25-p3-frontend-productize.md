# P3 Frontend Productize Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Productize the stable Study Coach backend (202 tests, dual-mode dispatchers shipped P2.2 + P2.3) into a portfolio-grade 7-view Vue 3 app. Ship 4 new views (Overview / PlanTimeline / QuizAdaptive / MistakeBank) backed by 4 minimal `GET` REST endpoints + an upgraded shell + Pinia stores per backend resource. Visual system locked to Modern Dark Cinema (Inter + JetBrains Mono + indigo) per `design-system/MASTER.md`. Operationalize P2.3 §Finding 3 alignment-safety via `<EmptyCorpusBanner>` (dual-channel detection).

**Architecture:** Vertical-slice cuts (A0-A13). Each backend cut adds 1 route handler + 0-1 repo method + 2-3 tests; the very next cut consumes it in its view. Frontend cuts use the existing `streamChat` SSE infrastructure for chat-bearing flows; new `getJSON()` helpers added to `lib/api.ts` for the 4 `GET` endpoints. State management = 1 Pinia store per backend resource; `overview.ts` is derived (composes the other stores via `computed`). Mode dispatch via existing `x-fingerprint` / `x-planner-mode` / `x-quiz-mode` / `x-judge-model` headers; per-view defaults from `settings.ts`, per-message override from `<ModeChip>` ref. `<EmptyCorpusBanner>` detects via pre-flight (`/api/documents` `chunks_count===0`) + in-flight (refusal-phrase regex on chat token buffer). No unit-test framework for frontend — `vue-tsc -b` build + chrome-devtools MCP screenshot+behavior verification is the per-cut gate.

**Tech Stack:** Python 3.11 + FastAPI + SQLAlchemy 2 (backend — byte-identical except `routes.py` + `repositories.py` append-only); Vue 3.5 + Pinia 3 + vue-router 5 + Tailwind 4 + Vite 8 + TS 6 + new deps `mermaid` + `chart.js` 4 + `vue-chartjs` 5 + `lucide-vue-next`.

**Spec:** `study-coach/docs/superpowers/specs/2026-05-25-p3-frontend-productize-design.md`
**Design system:** `study-coach/design-system/MASTER.md`

**Discipline reminders for the implementer (read EVERY cut):**

- This repo is **not** a git repo — never run `git init / commit / push / status`. The project's convention is a `# checkpoint` mental marker at end of every cut = full backend test suite + full frontend build both green + chrome-devtools verification logged.
- TDD: every backend cut writes the failing test(s) first, runs them to confirm RED, then writes the minimal implementation, then runs the FULL `uv run pytest -q` suite green.
- Project working dirs:
  - Backend: `cd "study-coach/backend" && uv run pytest -q`
  - Frontend: `cd "study-coach/frontend" && pnpm build` (uses `vue-tsc -b && vite build`); `pnpm dev` for the live preview
- **Baseline before P3 starts: 202 backend tests passing** (P2.3 ship-state). Expected per-cut deltas listed inline; total after A13: **~218 tests passing** (+16 net new).
- Every `# cloud-adapt:` comment in new files must be marker-only — DO NOT implement cloud branches. The MASTER.md §11 and spec §11 list the 5 anchor placements.
- Do not modify any file unless the cut says so. Critical byte-identical files for P3: ALL existing backend code except `app/api/routes.py` (handlers appended only) and `app/db/repositories.py` (methods appended only). Critical byte-identical files in `app/agent/*` — ZERO change.
- No emoji in any UI text or icon. Use `lucide-vue-next` icons only.
- No `<style scoped>` — Tailwind utility classes only (matches existing P1 `App.vue` + `Chat.vue` convention). The only `<style>` block tolerated is the existing one in `App.vue` that defines `.nav-link` (kept until A0 replaces it with utility classes).
- After scaffold commands like `pnpm add` / `pnpm install`: per memory `feedback_scaffold_write_readfirst`, ALWAYS Read the modified config (`package.json` / `tailwind.config.*` / `vite.config.ts`) BEFORE Write. Scaffolders silently rewrite config and Edit-without-Read will lose state.
- Backend repos live in `app/db/repositories.py` (single file). Append new methods at the end of the relevant class. DO NOT split into per-resource files.
- Frontend tests are NOT being introduced — `vue-tsc` compile + chrome-devtools MCP is the gate. If you find yourself wanting to add Vitest, STOP and re-read spec §10.

---

## Verification template (referenced by every frontend cut)

When a cut says "**run the standard frontend verification**", do all of:

1. `cd "study-coach/frontend" && pnpm build` — must exit 0 (vue-tsc clean + vite build clean).
2. `cd "study-coach/frontend" && pnpm dev` in a background bash task — wait for "Local: http://localhost:5173/" line.
3. Backend up: `cd "study-coach/backend" && uv run uvicorn app.main:app --port 8000` in a separate background task (only if the cut touches API; A0 can skip if no API consumed yet).
4. Open chrome-devtools MCP: `mcp__chrome-devtools__new_page(url="http://localhost:5173/<route>")` for the route under test.
5. `mcp__chrome-devtools__take_snapshot` — capture the a11y tree. Verify all components from the cut are present (greppable component names in tree).
6. `mcp__chrome-devtools__take_screenshot(filePath="study-coach/docs/screenshots/p3/cut-A<N>.png")` — save the visual.
7. `mcp__chrome-devtools__list_console_messages(types=["error","warn"])` — must be empty (or all entries pre-existing baseline noise listed in the cut).
8. Run the cut's interaction script (e.g. click a button, fill a form) and `take_snapshot` after — verify state transition. The cut spec lists what to click.
9. Stop the dev server and uvicorn background tasks before declaring cut done.

**If any step fails, the cut is NOT done.** Fix and rerun all 9 steps.

---

## File Structure

### Files to create

| Path | Responsibility | Cut |
|---|---|---|
| `backend/tests/db/test_repositories_p3.py` | Repo method tests for MasteryRepository.list_for_user_detailed | A9 |
| `backend/tests/api/test_routes_p3_documents.py` | Route test for GET /api/documents | A4 |
| `backend/tests/api/test_routes_p3_plans.py` | Route test for GET /api/plans/current (200 + 404) | A1 |
| `backend/tests/api/test_routes_p3_mistakes.py` | Route test for GET /api/mistakes/due (with + without due rows) | A5 |
| `backend/tests/api/test_routes_p3_mastery.py` | Route test for GET /api/mastery (scores + weak_topics + overdue_count) | A9 |
| `frontend/src/views/Overview.vue` | Overview hero view: UploadGate + 4 widget cards + RadarChart | A10 |
| `frontend/src/views/PlanTimeline.vue` | Plan view: MilestoneList + MindmapPanel + ModeChip | A2 |
| `frontend/src/views/QuizAdaptive.vue` | Quiz view: DifficultySelector + MCQCard + GradeResult + ModeChip | A7 |
| `frontend/src/views/MistakeBank.vue` | Mistake bank: due list + Redo flow | A6 |
| `frontend/src/components/ModeChip.vue` | Per-view mode toggle pill | A2 |
| `frontend/src/components/MilestoneList.vue` | List of milestones with status color + icon | A2 |
| `frontend/src/components/MindmapPanel.vue` | Collapsible mermaid renderer (lazy import) | A3 |
| `frontend/src/components/MistakeRow.vue` | One row in MistakeBank | A6 |
| `frontend/src/components/DifficultySelector.vue` | Segmented control easy/med/hard | A7 |
| `frontend/src/components/MCQCard.vue` | Question prompt + 4 radio options + submit | A7 |
| `frontend/src/components/GradeResult.vue` | ✓/✗ + correct answer + explanation | A7 |
| `frontend/src/components/EmptyCorpusBanner.vue` | P2.3 §F3 alignment-safety UX | A8 |
| `frontend/src/components/UploadGate.vue` | Overview banner: docs=0 → upload prompt | A10 |
| `frontend/src/components/MasteryCard.vue` | Top-5 mastery + bar (pure Tailwind) | A10 |
| `frontend/src/components/PlanProgressCard.vue` | Done/total + next-due milestone summary | A10 |
| `frontend/src/components/MistakesDueCard.vue` | Count + first 3 question stems | A10 |
| `frontend/src/components/WeakTopicsChips.vue` | Weak topic chips → click filters /quiz | A10 |
| `frontend/src/components/RadarChart.vue` | chart.js radar wrapper | A11 |
| `frontend/src/stores/plan.ts` | Plan state: milestones + mindmap (parsed from chat) | A2 |
| `frontend/src/stores/quiz.ts` | Quiz state: current MCQ + difficulty + needsUpload signal | A7 |
| `frontend/src/stores/mistakes.ts` | Mistakes due list + last-fetched timestamp | A6 |
| `frontend/src/stores/mastery.ts` | Mastery scores + weak_topics + overdue_count | A10 |
| `frontend/src/stores/documents.ts` | Uploaded docs list + totalChunks computed | A4 |
| `frontend/src/stores/overview.ts` | Derived store: imports + composes other stores | A10 |
| `frontend/src/lib/parse.ts` | Pure utility: extract mermaid block + detect refusal regex | A3, A8 |
| `study-coach/docs/screenshots/p3/cut-A<N>.png` | One per cut, saved by chrome-devtools | every frontend cut |

### Files to modify

| Path | Change | Cut |
|---|---|---|
| `backend/app/db/repositories.py` | Append `MasteryRepository.list_for_user_detailed(user_id) -> list[tuple[Topic, Mastery]]` | A9 |
| `backend/app/api/routes.py` | Append 4 `@router.get(...)` handlers + 1 Pydantic response model per endpoint | A1, A4, A5, A9 |
| `frontend/package.json` | + `mermaid` `chart.js` `vue-chartjs` `lucide-vue-next` | A0, A3, A11 |
| `frontend/src/style.css` | + Google Fonts `@import` + `:root` font vars + reduced-motion gate | A0 |
| `frontend/tailwind.config.*` (or `vite.config.ts` if Tailwind 4 vite plugin) | + color tokens from MASTER.md §1 + fontFamily | A0 |
| `frontend/src/App.vue` | Nav 3→7 links with section labels; nav `w-48`→`w-56`; replace `.nav-link` style block with Tailwind classes | A0 |
| `frontend/src/router.ts` | `/` rewired to Overview placeholder; +5 routes (chat moves to /chat) | A0 |
| `frontend/src/lib/api.ts` | + `getJSON<T>(path)` generic + 4 typed wrappers; extend `streamChat` signature with `{ plannerMode?, quizMode? }` overrides | A0, A1, A4, A5, A9 |
| `frontend/src/stores/settings.ts` | + `defaultPlannerMode` `defaultQuizMode` fields (init `agent_loop`); `llmHeaders` extended with optional override params | A0 |
| `frontend/src/views/Chat.vue` | Update empty-state RouterLink (still `/library` — verify); no logic change | A0 |
| `frontend/src/views/Settings.vue` | Add default-mode selectors + judge_model input | A12 |
| `study-coach/docs/EVAL.md` | Append §P3 UX integration summary | A13 |
| `study-coach/docs/ROADMAP.md` | Mark §P3 done; refine §P4 | A13 |
| `study-coach/docs/p3_frontend_productize.md` | New sister blog post | A13 |
| `/Users/lianghaozhe/.claude/projects/-Users-lianghaozhe-Downloads-Study-Compaion-and-JadeAI/memory/project_study_coach_refactor.md` | Append P3 progress + lessons learned | A13 |

### Files explicitly NOT touched

- `backend/app/agent/**/*.py` — all agent code, BYTE-IDENTICAL.
- `backend/app/db/models.py`, `backend/alembic/**` — no schema change.
- `backend/app/llm/**`, `backend/app/rag/**`, `backend/app/srs/**` — unchanged.
- `backend/app/api/deps.py` — `get_user_id` (the `x-fingerprint` resolver) already exists; no new dep needed for the 4 GETs (they just `Depends(get_user_id)` + `Depends(get_session)`).
- `backend/app/api/routes.py` `POST /api/chat` and `POST /api/documents` handlers — append GETs only, do not edit existing handler bodies.

---

## A0 — Shell + design tokens + Pinia/store scaffolds + lib/api.ts extension

**Goal:** Reskin to the locked design system; rewire the router so `/` is Overview placeholder + Chat moves to `/chat` + 4 new view stubs exist; extend the API client to inject mode-override headers; add `defaultPlannerMode`/`defaultQuizMode` to settings store; install `lucide-vue-next`.

**Files:**
- Modify: `frontend/package.json`, `frontend/src/style.css`, `frontend/src/App.vue`, `frontend/src/router.ts`, `frontend/src/lib/api.ts`, `frontend/src/stores/settings.ts`
- Create: stub `frontend/src/views/{Overview,PlanTimeline,QuizAdaptive,MistakeBank}.vue` (5-line placeholder each so router resolves)
- Create/Modify: `frontend/tailwind.config.ts` (may not exist yet — Tailwind 4 uses Vite plugin; check)

**Test count delta:** 0 (no backend touched, no frontend tests).

- [ ] **Step 1: Inspect current Tailwind 4 wiring**

Tailwind 4 doesn't always have a `tailwind.config.*` — it's often configured inline in `vite.config.ts` or via `@theme` in CSS. Read first:

Run: `cd "study-coach/frontend" && cat vite.config.ts && echo --- && cat src/style.css && echo --- && ls tailwind*`

Note which pattern is in use. The next step adapts to it.

- [ ] **Step 2: Install new dep `lucide-vue-next`**

Run: `cd "study-coach/frontend" && pnpm add lucide-vue-next`

Per memory `feedback_scaffold_write_readfirst`: AFTER this command, before any Edit, Read `frontend/package.json` to see what `pnpm` did. Confirm `lucide-vue-next` is listed.

- [ ] **Step 3: Replace `frontend/src/style.css` with design-system tokens**

Read the current file first (`Read frontend/src/style.css`). Then Write the full new content:

```css
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Noto+Sans+SC:wght@400;500;700&display=swap');

@import 'tailwindcss';

@theme {
  /* Color tokens — see design-system/MASTER.md §1 */
  --color-bg: #0b0e1a;
  --color-surface: #11162a;
  --color-surface-2: #171c34;
  --color-border: rgba(255, 255, 255, 0.05);
  --color-border-strong: rgba(255, 255, 255, 0.12);

  --color-fg: #e6e6ec;
  --color-fg-muted: #b0b6c5;
  --color-fg-dim: rgba(255, 255, 255, 0.4);

  --color-primary: #6366f1;
  --color-primary-2: #818cf8;
  --color-primary-bg: rgba(99, 102, 241, 0.15);
  --color-primary-ring: rgba(99, 102, 241, 0.4);

  --color-success: #10b981;
  --color-warning: #f59e0b;
  --color-danger: #f43f5e;
  --color-success-bg: rgba(16, 185, 129, 0.12);
  --color-warning-bg: rgba(245, 158, 11, 0.12);
  --color-danger-bg: rgba(244, 63, 94, 0.12);

  /* Typography */
  --font-sans: 'Inter', 'Noto Sans SC', system-ui, sans-serif;
  --font-mono: 'JetBrains Mono', ui-monospace, monospace;
}

:root {
  color-scheme: dark;
}

html, body, #app {
  height: 100%;
  background: var(--color-bg);
  color: var(--color-fg);
  font-family: var(--font-sans);
  font-feature-settings: 'cv11', 'ss01';
}

code, .mono { font-family: var(--font-mono); }

/* Reduced motion */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    transition-duration: 0.01ms !important;
    animation-duration: 0.01ms !important;
  }
}
```

The `@theme` block is Tailwind 4 idiom — colors become `bg-bg` / `bg-surface` / `text-fg` / `text-fg-muted` / `bg-primary` / etc. Verify by spinning `pnpm dev` after this step and inspecting any element. If colors don't resolve, check that `vite.config.ts` has the Tailwind 4 vite plugin enabled.

- [ ] **Step 4: Create 4 view stub files**

Each placeholder so router can resolve. Example for `Overview.vue` (replicate the shape for the 3 others, swapping name):

```vue
<script setup lang="ts">
// A0 stub — filled in cut A10
</script>

<template>
  <div class="h-full overflow-y-auto p-8">
    <h1 class="text-2xl font-semibold">Overview</h1>
    <p class="text-fg-muted mt-2 text-sm">Placeholder — filled in cut A10.</p>
  </div>
</template>
```

Create:
- `frontend/src/views/Overview.vue` (title: "Overview", cut ref: A10)
- `frontend/src/views/PlanTimeline.vue` (title: "Plan", cut ref: A2)
- `frontend/src/views/QuizAdaptive.vue` (title: "Quiz", cut ref: A7)
- `frontend/src/views/MistakeBank.vue` (title: "Mistake Bank", cut ref: A6)

- [ ] **Step 5: Rewire `frontend/src/router.ts`**

```ts
import { createRouter, createWebHistory } from 'vue-router'
import Overview from './views/Overview.vue'
import Chat from './views/Chat.vue'
import PlanTimeline from './views/PlanTimeline.vue'
import QuizAdaptive from './views/QuizAdaptive.vue'
import MistakeBank from './views/MistakeBank.vue'
import Library from './views/Library.vue'
import Settings from './views/Settings.vue'

export const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', name: 'overview', component: Overview },
    { path: '/chat', name: 'chat', component: Chat },
    { path: '/plan', name: 'plan', component: PlanTimeline },
    { path: '/quiz', name: 'quiz', component: QuizAdaptive },
    { path: '/mistakes', name: 'mistakes', component: MistakeBank },
    { path: '/library', name: 'library', component: Library },
    { path: '/settings', name: 'settings', component: Settings },
  ],
})
```

- [ ] **Step 6: Rewrite `frontend/src/App.vue` nav 3→7 + remove `<style>` block**

```vue
<script setup lang="ts">
import { RouterLink, RouterView } from 'vue-router'
import {
  LayoutDashboard, MessageSquare, ListTodo, BookOpen,
  AlertTriangle, FolderOpen, Settings as SettingsIcon,
} from 'lucide-vue-next'

const navSections = [
  {
    label: 'Study',
    items: [
      { to: '/',          icon: LayoutDashboard, text: 'Overview' },
      { to: '/chat',      icon: MessageSquare,   text: 'Chat' },
      { to: '/plan',      icon: ListTodo,        text: 'Plan' },
      { to: '/quiz',      icon: BookOpen,        text: 'Quiz' },
    ],
  },
  {
    label: 'Review',
    items: [
      { to: '/mistakes',  icon: AlertTriangle,   text: 'Mistakes' },
    ],
  },
  {
    label: 'System',
    items: [
      { to: '/library',   icon: FolderOpen,      text: 'Library' },
      { to: '/settings',  icon: SettingsIcon,    text: 'Settings' },
    ],
  },
]
</script>

<template>
  <div class="h-full flex">
    <nav class="w-56 bg-surface p-4 flex flex-col gap-1 border-r border-border">
      <h1 class="text-lg font-semibold mb-4 px-2">Study Coach</h1>
      <template v-for="section in navSections" :key="section.label">
        <div class="px-2 text-[10px] uppercase tracking-wider text-fg-dim mt-3 mb-1">
          {{ section.label }}
        </div>
        <RouterLink
          v-for="item in section.items"
          :key="item.to"
          :to="item.to"
          class="flex items-center gap-3 px-3 py-2 rounded-md text-sm text-fg-muted hover:bg-white/5 transition-colors"
          active-class="!bg-primary-bg !text-fg"
        >
          <component :is="item.icon" class="w-4 h-4" />
          {{ item.text }}
        </RouterLink>
      </template>
      <div class="mt-auto text-xs text-fg-dim px-2">P3 · productized shell</div>
    </nav>
    <main class="flex-1 overflow-hidden">
      <RouterView />
    </main>
  </div>
</template>
```

The `<style>` block is gone (utility classes replace `.nav-link`). Q1 brainstorm picked routing "C: Dashboard-first" but the design-system §3 nav width is `w-56`; sections are **B-style 3-group** because 7 items needs structure — this is a small step beyond brainstorm Q1's "C" but improves scannability and was already implied by MASTER.md §3.

- [ ] **Step 7: Extend `frontend/src/stores/settings.ts`**

```ts
import { defineStore } from 'pinia'

export type Provider = 'ollama' | 'openai' | 'anthropic' | 'gemini'
export type Mode = 'agent_loop' | 'deterministic'

interface SettingsState {
  provider: Provider
  model: string
  apiKey: string
  baseUrl: string
  judgeModel: string
  defaultPlannerMode: Mode
  defaultQuizMode: Mode
}

const STORAGE_KEY = 'study-coach:settings'

function loadInitial(): SettingsState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw) {
      const parsed = JSON.parse(raw)
      // back-fill new fields (existing localStorage may not have them)
      return {
        defaultPlannerMode: 'agent_loop',
        defaultQuizMode: 'agent_loop',
        ...parsed,
      }
    }
  } catch {
    /* empty */
  }
  return {
    provider: 'ollama',
    model: 'gemma3:4b',
    apiKey: '',
    baseUrl: '',
    judgeModel: '',
    defaultPlannerMode: 'agent_loop',
    defaultQuizMode: 'agent_loop',
  }
}

export const useSettings = defineStore('settings', {
  state: () => loadInitial(),
  actions: {
    persist() {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(this.$state))
    },
  },
})

export interface ModeOverrides {
  plannerMode?: Mode
  quizMode?: Mode
}

export function llmHeaders(s: SettingsState, overrides: ModeOverrides = {}): Record<string, string> {
  const h: Record<string, string> = {
    'x-provider': s.provider,
    'x-model': s.model,
  }
  if (s.apiKey) h['x-api-key'] = s.apiKey
  if (s.baseUrl) h['x-base-url'] = s.baseUrl
  if (s.judgeModel) h['x-judge-model'] = s.judgeModel
  if (overrides.plannerMode) h['x-planner-mode'] = overrides.plannerMode
  if (overrides.quizMode) h['x-quiz-mode'] = overrides.quizMode
  return h
}
```

- [ ] **Step 8: Extend `frontend/src/lib/api.ts` — `streamChat` overrides + `getJSON` helper**

Read the current file first. Then change the `streamChat` signature and add `getJSON`:

```ts
import { getFingerprint } from './fingerprint'
import { llmHeaders, type ModeOverrides } from '../stores/settings'
import type { Citation } from '../stores/chat'

interface ChatStreamCallbacks {
  onCitations?: (cs: Citation[]) => void
  onToken?: (text: string) => void
  onDone?: () => void
  onError?: (err: unknown) => void
}

export async function streamChat(
  message: string,
  settings: any,
  cb: ChatStreamCallbacks,
  overrides: ModeOverrides = {},
): Promise<void> {
  try {
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-fingerprint': getFingerprint(),
        ...llmHeaders(settings, overrides),
      },
      body: JSON.stringify({ message }),
    })
    if (!resp.ok || !resp.body) throw new Error(`chat failed: ${resp.status}`)
    const reader = resp.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n\n')
      buffer = lines.pop() ?? ''
      for (const line of lines) {
        const trimmed = line.trim()
        if (!trimmed.startsWith('data: ')) continue
        const json = trimmed.slice(6)
        try {
          const event = JSON.parse(json)
          if (event.type === 'token') cb.onToken?.(event.text)
          else if (event.type === 'citations') cb.onCitations?.(event.citations)
          else if (event.type === 'done') cb.onDone?.()
        } catch { /* ignore malformed */ }
      }
    }
  } catch (e) {
    cb.onError?.(e)
  }
}

export async function uploadDocument(file: File): Promise<{
  document_id: string
  filename: string
  chunks_count: number
}> {
  const form = new FormData()
  form.append('file', file)
  const resp = await fetch('/api/documents', {
    method: 'POST',
    headers: { 'x-fingerprint': getFingerprint() },
    body: form,
  })
  if (!resp.ok) throw new Error(`upload failed: ${resp.status}`)
  return resp.json()
}

// P3 — typed GET helper. Backend resolves user via x-fingerprint header.
export async function getJSON<T>(path: string): Promise<T> {
  const resp = await fetch(path, {
    headers: { 'x-fingerprint': getFingerprint() },
  })
  if (resp.status === 404) {
    // Caller decides whether 404 is data-empty or hard error.
    const err: any = new Error(`${path} returned 404`)
    err.status = 404
    throw err
  }
  if (!resp.ok) throw new Error(`${path} failed: ${resp.status}`)
  return resp.json() as Promise<T>
}
```

- [ ] **Step 9: Run frontend verification**

Run: `cd "study-coach/frontend" && pnpm build` — must exit 0.

Open chrome-devtools to `http://localhost:5173/` (dev server) — verify:
- Left nav shows 7 links in 3 sections (Study / Review / System).
- Background is dark `#0b0e1a` (not pure black).
- `/` shows the "Overview placeholder" text.
- Click each of the 7 nav links — each route loads its placeholder/existing view, no console errors.
- Chat at `/chat` (moved) still loads.
- `take_screenshot(filePath="study-coach/docs/screenshots/p3/cut-A0.png")`.

- [ ] **Step 10: Checkpoint**

Confirm backend test suite is still 202 passing (no backend file touched). Run from `study-coach/backend`: `uv run pytest -q | tail -5`. Expect `202 passed`. Then declare A0 done.

---

## A1 — GET /api/plans/current

**Goal:** Return the user's active plan (active goal → its plan) as JSON, 404 if no active plan.

**Files:**
- Modify: `backend/app/api/routes.py` (append handler)
- Create: `backend/tests/api/test_routes_p3_plans.py`

**Test count delta:** +2 (200 case + 404 case). Baseline → 204.

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/api/test_routes_p3_plans.py`:

```python
"""Cut A1 — GET /api/plans/current — happy + 404."""
import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/p3_plans.db")
    monkeypatch.setenv("STUDY_COACH_TEST_MODE", "1")
    from app.db import session as session_mod
    session_mod._engine = None
    session_mod._SessionLocal = None
    app = create_app()
    yield TestClient(app)


def test_get_plans_current_404_when_no_plan(client):
    resp = client.get("/api/plans/current", headers={"x-fingerprint": "fp-1"})
    assert resp.status_code == 404
    assert resp.json() == {"detail": "no active plan for user"}


def test_get_plans_current_returns_active_plan(client):
    # Seed: create user, goal, plan via repos directly.
    from app.db.session import get_session_scope
    from app.db.repositories import (
        UserRepository, GoalRepository, PlanRepository,
    )
    with get_session_scope() as s:
        user = UserRepository(s).get_or_create("fp-2")
        goal = GoalRepository(s).create(user_id=user.id, title="Master HyDE", exam_date=None)
        PlanRepository(s).update_milestones(
            goal_id=goal.id,
            milestones=[
                {"title": "Read HyDE chapter", "due_at": "2026-05-26T00:00:00", "done": False, "topic": "HyDE"},
                {"title": "Quiz on HyDE", "due_at": None, "done": False, "topic": "HyDE"},
            ],
        )
    resp = client.get("/api/plans/current", headers={"x-fingerprint": "fp-2"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["goal_title"] == "Master HyDE"
    assert len(body["milestones"]) == 2
    assert body["milestones"][0]["title"] == "Read HyDE chapter"
    assert "plan_id" in body and "goal_id" in body and "updated_at" in body
```

Note: the test uses `get_session_scope` (context manager) — verify it exists in `app/db/session.py`. If only `get_session` (generator) exists, adapt to it: `s = next(get_session()); ... ; s.commit()`.

- [ ] **Step 2: Run test to verify RED**

Run: `cd "study-coach/backend" && uv run pytest tests/api/test_routes_p3_plans.py -v`
Expected: both tests FAIL with 404 (because the route doesn't exist yet — FastAPI returns 404 on unknown path with "Not Found" detail).

- [ ] **Step 3: Add the route handler to `backend/app/api/routes.py`**

Add Pydantic models near top of file (after the `_SAME_MODEL_WARNING` block, before `@router.get("/health")`):

```python
class MilestoneOut(BaseModel):
    title: str
    due_at: str | None = None
    done: bool = False
    topic: str | None = None


class PlanCurrentOut(BaseModel):
    plan_id: str
    goal_id: str
    goal_title: str
    milestones: list[MilestoneOut]
    updated_at: str
```

Add handler at the end of `routes.py`:

```python
@router.get("/plans/current", response_model=PlanCurrentOut)
def get_plans_current(
    user_id: Annotated[str, Depends(get_user_id)],
    session: Annotated[Session, Depends(get_session)],
):
    from fastapi import HTTPException
    from app.db.repositories import GoalRepository, PlanRepository

    goals = GoalRepository(session).list_active_for_user(user_id)
    if not goals:
        raise HTTPException(status_code=404, detail="no active plan for user")
    goal = goals[0]  # one active goal per user (P2.1-③ invariant)
    plan = PlanRepository(session).get_by_goal(goal.id)
    if plan is None:
        raise HTTPException(status_code=404, detail="no active plan for user")
    return PlanCurrentOut(
        plan_id=plan.id,
        goal_id=goal.id,
        goal_title=goal.title,
        milestones=[MilestoneOut(**m) for m in plan.milestones_json],
        updated_at=plan.updated_at.isoformat(),
    )
```

- [ ] **Step 4: Run tests to verify GREEN**

Run: `cd "study-coach/backend" && uv run pytest tests/api/test_routes_p3_plans.py -v` — both PASS.

Then: `uv run pytest -q | tail -5` — expect **204 passed** (202 + 2 new).

- [ ] **Step 5: Checkpoint**

A1 done. The next cut (A2) consumes this endpoint.

---

## A2 — PlanTimeline view + MilestoneList + ModeChip

**Goal:** `/plan` route shows real milestones from the new endpoint. ModeChip toggles `x-planner-mode` for the next chat send.

**Files:**
- Create: `frontend/src/components/MilestoneList.vue`, `frontend/src/components/ModeChip.vue`, `frontend/src/stores/plan.ts`
- Modify: `frontend/src/views/PlanTimeline.vue` (replace placeholder)
- Modify: `frontend/src/lib/api.ts` (add `getCurrentPlan` typed wrapper)

**Test count delta:** 0 (frontend). Backend stays 204.

- [ ] **Step 1: Add typed wrapper to `frontend/src/lib/api.ts`**

Append at the bottom:

```ts
export interface MilestoneDto {
  title: string
  due_at: string | null
  done: boolean
  topic: string | null
}

export interface PlanCurrentDto {
  plan_id: string
  goal_id: string
  goal_title: string
  milestones: MilestoneDto[]
  updated_at: string
}

export function getCurrentPlan(): Promise<PlanCurrentDto> {
  return getJSON<PlanCurrentDto>('/api/plans/current')
}
```

- [ ] **Step 2: Create `frontend/src/stores/plan.ts`**

```ts
import { defineStore } from 'pinia'
import { getCurrentPlan, type PlanCurrentDto } from '../lib/api'

interface PlanState {
  plan: PlanCurrentDto | null
  loading: boolean
  error: string | null
  noActive: boolean        // true if backend returned 404
  mindmapMermaid: string | null  // set by A3
}

export const usePlan = defineStore('plan', {
  state: (): PlanState => ({
    plan: null,
    loading: false,
    error: null,
    noActive: false,
    mindmapMermaid: null,
  }),
  actions: {
    async fetch() {
      this.loading = true
      this.error = null
      this.noActive = false
      try {
        this.plan = await getCurrentPlan()
      } catch (e: any) {
        if (e?.status === 404) {
          this.noActive = true
          this.plan = null
        } else {
          this.error = e?.message ?? 'failed'
        }
      } finally {
        this.loading = false
      }
    },
    setMindmap(mermaid: string) {
      this.mindmapMermaid = mermaid
    },
  },
})
```

- [ ] **Step 3: Create `frontend/src/components/ModeChip.vue`**

```vue
<script setup lang="ts">
import { computed } from 'vue'
import { ArrowLeftRight } from 'lucide-vue-next'
import type { Mode } from '../stores/settings'

const props = defineProps<{
  mode: Mode
  defaultMode: Mode
}>()
const emit = defineEmits<{ (e: 'toggle'): void }>()

const overridden = computed(() => props.mode !== props.defaultMode)
</script>

<template>
  <button
    type="button"
    :aria-pressed="overridden"
    @click="emit('toggle')"
    class="inline-flex items-center gap-2 rounded-full border border-primary-ring bg-primary-bg px-3 py-1 text-xs font-mono text-primary hover:bg-primary/20 transition-colors"
    :title="overridden ? `Overridden — default is ${defaultMode}` : `Default mode for this view`"
  >
    {{ mode }}
    <ArrowLeftRight class="w-3 h-3 opacity-60" />
  </button>
</template>
```

- [ ] **Step 4: Create `frontend/src/components/MilestoneList.vue`**

```vue
<script setup lang="ts">
import { computed } from 'vue'
import { CheckCircle2, AlertCircle, AlertTriangle, Circle } from 'lucide-vue-next'
import type { MilestoneDto } from '../lib/api'

const props = defineProps<{ milestones: MilestoneDto[] }>()

function statusOf(m: MilestoneDto): 'success' | 'warning' | 'danger' | 'neutral' {
  if (m.done) return 'success'
  if (!m.due_at) return 'neutral'
  const due = new Date(m.due_at).getTime()
  const now = Date.now()
  const dayMs = 86_400_000
  if (due < now - dayMs) return 'danger'           // overdue (yesterday or earlier)
  if (due < now + dayMs) return 'warning'          // due today
  return 'neutral'
}

const rows = computed(() =>
  props.milestones.map(m => ({ m, status: statusOf(m) })),
)

const iconFor = { success: CheckCircle2, warning: AlertCircle, danger: AlertTriangle, neutral: Circle }
const colorFor = {
  success: 'text-success',
  warning: 'text-warning',
  danger: 'text-danger',
  neutral: 'text-fg-muted',
}
</script>

<template>
  <ul class="flex flex-col gap-2">
    <li v-for="(r, i) in rows" :key="i"
        class="flex items-start gap-3 rounded-lg border border-border bg-surface p-3">
      <component :is="iconFor[r.status]" :class="['w-5 h-5 mt-0.5 shrink-0', colorFor[r.status]]" />
      <div class="flex-1 min-w-0">
        <div class="text-sm font-medium" :class="r.m.done ? 'line-through text-fg-muted' : ''">
          {{ r.m.title }}
        </div>
        <div class="mt-1 flex gap-2 text-xs text-fg-muted">
          <span v-if="r.m.topic"
                class="font-mono px-2 py-0.5 rounded-md bg-primary-bg text-primary">
            {{ r.m.topic }}
          </span>
          <span v-if="r.m.due_at" class="font-mono">due {{ new Date(r.m.due_at).toLocaleDateString() }}</span>
        </div>
      </div>
    </li>
  </ul>
</template>
```

- [ ] **Step 5: Replace `frontend/src/views/PlanTimeline.vue`**

```vue
<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { usePlan } from '../stores/plan'
import { useSettings, type Mode } from '../stores/settings'
import { streamChat } from '../lib/api'
import MilestoneList from '../components/MilestoneList.vue'
import ModeChip from '../components/ModeChip.vue'

const planStore = usePlan()
const settings = useSettings()
const mode = ref<Mode>(settings.defaultPlannerMode)
const checkInLoading = ref(false)

onMounted(() => planStore.fetch())

function toggleMode() {
  mode.value = mode.value === 'agent_loop' ? 'deterministic' : 'agent_loop'
}

async function checkIn() {
  checkInLoading.value = true
  const nextMode = mode.value
  await streamChat(
    '进度怎么样了',
    settings.$state,
    {
      onDone: () => { planStore.fetch(); mode.value = settings.defaultPlannerMode },
      onError: () => { mode.value = settings.defaultPlannerMode },
    },
    { plannerMode: nextMode },
  )
  checkInLoading.value = false
}
</script>

<template>
  <div class="h-full overflow-y-auto p-8">
    <div class="max-w-4xl mx-auto">
      <header class="flex items-center justify-between mb-6">
        <div>
          <h1 class="text-2xl font-semibold">Plan</h1>
          <p v-if="planStore.plan" class="text-sm text-fg-muted mt-1">
            {{ planStore.plan.goal_title }} ·
            <span class="font-mono">{{ planStore.plan.milestones.length }} milestones</span>
          </p>
        </div>
        <ModeChip :mode="mode" :default-mode="settings.defaultPlannerMode" @toggle="toggleMode" />
      </header>

      <div v-if="planStore.loading" class="text-fg-muted text-sm">Loading…</div>

      <div v-else-if="planStore.noActive"
           class="rounded-lg border border-border bg-surface p-6 text-center">
        <p class="text-fg-muted">No active plan yet.</p>
        <p class="text-xs text-fg-dim mt-2">
          Go to <RouterLink to="/chat" class="underline">Chat</RouterLink> and ask
          <span class="font-mono">帮我做学习计划 on &lt;topic&gt;</span>.
        </p>
      </div>

      <template v-else-if="planStore.plan">
        <MilestoneList :milestones="planStore.plan.milestones" />
        <div class="mt-6 flex justify-end">
          <button @click="checkIn" :disabled="checkInLoading"
                  class="rounded-md bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary-2 disabled:opacity-40 transition-colors">
            {{ checkInLoading ? 'Checking in…' : 'Check-in progress' }}
          </button>
        </div>
      </template>
    </div>
  </div>
</template>
```

- [ ] **Step 6: Run the standard frontend verification**

Per the template at the top of this plan. Route to test: `/plan`.

Interaction script:
1. Pre-seed via Chat: visit `/chat`, send `帮我做学习计划 on HyDE` (real backend with Ollama is needed; if Ollama isn't running, skip the data-population step and verify the `noActive` empty state instead).
2. Navigate to `/plan` — verify MilestoneList renders ≥1 milestone with status icon + topic chip.
3. Click `<ModeChip>` — verify it flips to `deterministic`, then on `<Check-in progress>` click, gets sent and on done reverts to `agent_loop`.
4. Console must be clean (no errors).

Save screenshot to `study-coach/docs/screenshots/p3/cut-A2.png`.

- [ ] **Step 7: Checkpoint**

Backend tests still 204. Frontend builds clean. Done.

---

## A3 — MindmapPanel (mermaid integration, parse from chat history)

**Goal:** When Planner emits a `\`\`\`mermaid\n...\n\`\`\`` block in chat tokens, capture and render it as a collapsible panel in `/plan`.

**Files:**
- Create: `frontend/src/lib/parse.ts` (extract mermaid block utility — pure)
- Create: `frontend/src/components/MindmapPanel.vue`
- Modify: `frontend/src/stores/chat.ts` (after each `done`, scan token buffer; if `extractMermaid()` returns text, push to `usePlan().setMindmap()`)
- Modify: `frontend/src/views/PlanTimeline.vue` (mount `<MindmapPanel>` when `planStore.mindmapMermaid` non-null)
- Modify: `frontend/package.json` (`pnpm add mermaid`)

**Test count delta:** 0. Backend stays 204.

- [ ] **Step 1: Install mermaid**

Run: `cd "study-coach/frontend" && pnpm add mermaid`

After install: Read `frontend/package.json` (Write-after-Read discipline).

- [ ] **Step 2: Create `frontend/src/lib/parse.ts`**

```ts
// Pure utilities — no Vue imports. Easy to test by inspection.

const MERMAID_RE = /```mermaid\n([\s\S]*?)```/

export function extractMermaid(text: string): string | null {
  const m = MERMAID_RE.exec(text)
  return m ? m[1].trim() : null
}

// P2.3 §Finding 3 — agent-loop refusal phrases.
// cloud-adapt: cloud GPT/DeepSeek may use different phrasing — extend list.
const REFUSAL_PATTERNS = [
  /i'?m\s+unable\s+to\s+retrieve/i,
  /no\s+(information|content|context)\s+(was\s+)?(found|available|retrieved)/i,
  /cannot\s+(quiz|generate)\s+(you\s+)?without/i,
  /please\s+(provide|upload)\s+(more\s+)?(context|source|pdf)/i,
  /retriever\s+returned\s+(no|empty)/i,
]

export function looksLikeEmptyCorpusRefusal(text: string): boolean {
  return REFUSAL_PATTERNS.some(re => re.test(text))
}
```

- [ ] **Step 3: Create `frontend/src/components/MindmapPanel.vue`**

```vue
<script setup lang="ts">
import { ref, onMounted, watch, nextTick } from 'vue'
import { ChevronDown, ChevronRight } from 'lucide-vue-next'

const props = defineProps<{ mermaid: string }>()
const open = ref(true)
const container = ref<HTMLDivElement | null>(null)
const renderError = ref<string | null>(null)

async function render() {
  if (!open.value || !container.value) return
  try {
    const mermaid = (await import('mermaid')).default
    mermaid.initialize({ startOnLoad: false, theme: 'dark', securityLevel: 'loose' })
    container.value.innerHTML = ''
    const { svg } = await mermaid.render(`mm-${Date.now()}`, props.mermaid)
    container.value.innerHTML = svg
    renderError.value = null
  } catch (e: any) {
    renderError.value = e?.message ?? 'render failed'
  }
}

onMounted(render)
watch(() => props.mermaid, () => nextTick(render))
watch(open, render)
</script>

<template>
  <section class="rounded-lg border border-border bg-surface mt-6">
    <button @click="open = !open"
            class="w-full flex items-center gap-2 p-4 text-sm font-medium hover:bg-white/5 transition-colors">
      <component :is="open ? ChevronDown : ChevronRight" class="w-4 h-4" />
      Mindmap
    </button>
    <div v-show="open" class="px-4 pb-4">
      <div v-if="renderError" class="text-xs text-danger font-mono">{{ renderError }}</div>
      <div ref="container" class="overflow-x-auto"></div>
    </div>
  </section>
</template>
```

- [ ] **Step 4: Modify `frontend/src/stores/chat.ts` to harvest mermaid + refusal signal**

Read current `chat.ts` first. Then change the `finish()` action so it scans the just-completed assistant message for mermaid + refusal:

(Show the change patterns — implementer reads current file and applies):

Inside the chat store (action `finish()` or wherever an assistant message marks "done"), add:

```ts
import { extractMermaid, looksLikeEmptyCorpusRefusal } from '../lib/parse'
import { usePlan } from './plan'
import { useQuiz } from './quiz'  // created in A7 — until then, guard the import

// inside finish() after the message is finalized:
const last = this.messages[this.messages.length - 1]
if (last && last.role === 'assistant') {
  const mm = extractMermaid(last.content)
  if (mm) usePlan().setMindmap(mm)
  // Quiz refusal — A8 will use it; safe to set unconditionally.
  // useQuiz().setNeedsUpload(looksLikeEmptyCorpusRefusal(last.content))
}
```

The `useQuiz()` import + call is commented until A7 creates the store. A8 uncomments it.

- [ ] **Step 5: Mount `<MindmapPanel>` in `PlanTimeline.vue`**

Inside `PlanTimeline.vue`'s template, after `<MilestoneList>`, add:

```vue
<MindmapPanel v-if="planStore.mindmapMermaid" :mermaid="planStore.mindmapMermaid" />
```

And in the `<script setup>`:

```ts
import MindmapPanel from '../components/MindmapPanel.vue'
```

- [ ] **Step 6: Run the standard frontend verification**

Route: `/plan`.

Interaction script:
1. Pre-seed in Chat: `帮我做学习计划 on RAG 画脑图` (real Ollama backend needed for the mermaid block to land).
2. Navigate to `/plan` — verify the `<MindmapPanel>` renders with a real mermaid SVG.
3. Click the panel header — verify toggle open/close.
4. Console clean.

Save `cut-A3.png`.

- [ ] **Step 7: Checkpoint**

---

## A4 — GET /api/documents

**Goal:** List user's uploaded documents (id + filename + chunks_count).

**Files:**
- Modify: `backend/app/api/routes.py` (append handler)
- Create: `backend/tests/api/test_routes_p3_documents.py`

**Test count delta:** +2. Baseline → 206.

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/api/test_routes_p3_documents.py`:

```python
"""Cut A4 — GET /api/documents."""
import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/p3_docs.db")
    monkeypatch.setenv("STUDY_COACH_TEST_MODE", "1")
    from app.db import session as session_mod
    session_mod._engine = None
    session_mod._SessionLocal = None
    app = create_app()
    yield TestClient(app)


def test_get_documents_empty(client):
    resp = client.get("/api/documents", headers={"x-fingerprint": "fp-1"})
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_documents_lists_uploaded(client):
    from app.db.session import get_session
    from app.db.repositories import UserRepository, DocumentRepository
    s = next(get_session())
    user = UserRepository(s).get_or_create("fp-2")
    DocumentRepository(s).create(user_id=user.id, filename="a.pdf", hash_="h1", chunks_count=10)
    DocumentRepository(s).create(user_id=user.id, filename="b.pdf", hash_="h2", chunks_count=5)
    resp = client.get("/api/documents", headers={"x-fingerprint": "fp-2"})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 2
    assert {d["filename"] for d in body} == {"a.pdf", "b.pdf"}
    assert all(set(d.keys()) >= {"id", "filename", "chunks_count"} for d in body)
```

If `get_session` is a generator (not contextmanager), adapt as shown — `s = next(get_session())` and don't worry about cleanup (test DB is throwaway).

- [ ] **Step 2: Run test to verify RED**

Run: `cd "study-coach/backend" && uv run pytest tests/api/test_routes_p3_documents.py -v` — both FAIL (404 "Not Found").

- [ ] **Step 3: Add handler to `backend/app/api/routes.py`**

Add Pydantic model:

```python
class DocumentOut(BaseModel):
    id: str
    filename: str
    chunks_count: int
```

Add handler:

```python
@router.get("/documents", response_model=list[DocumentOut])
def get_documents(
    user_id: Annotated[str, Depends(get_user_id)],
    session: Annotated[Session, Depends(get_session)],
):
    docs = DocumentRepository(session).list_for_user(user_id)
    return [DocumentOut(id=d.id, filename=d.filename, chunks_count=d.chunks_count) for d in docs]
```

(`DocumentRepository.list_for_user` already exists per repos.py line 66.)

- [ ] **Step 4: Run tests to verify GREEN**

Run: `cd "study-coach/backend" && uv run pytest tests/api/test_routes_p3_documents.py -v` — both PASS.

Full suite: `uv run pytest -q | tail -5` → **206 passed**.

- [ ] **Step 5: Checkpoint**

---

## A5 — GET /api/mistakes/due

**Goal:** Return mistakes whose `due_at <= now` for the user, embedded with question stem + topic name.

**Files:**
- Modify: `backend/app/api/routes.py` (append handler)
- Create: `backend/tests/api/test_routes_p3_mistakes.py`

**Test count delta:** +2. Baseline → 208.

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/api/test_routes_p3_mistakes.py`:

```python
"""Cut A5 — GET /api/mistakes/due."""
from datetime import datetime, timedelta
import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/p3_mistakes.db")
    monkeypatch.setenv("STUDY_COACH_TEST_MODE", "1")
    from app.db import session as session_mod
    session_mod._engine = None
    session_mod._SessionLocal = None
    app = create_app()
    yield TestClient(app)


def test_get_mistakes_due_empty(client):
    resp = client.get("/api/mistakes/due", headers={"x-fingerprint": "fp-1"})
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_mistakes_due_returns_due_only(client):
    from app.db.session import get_session
    from app.db.repositories import (
        UserRepository, GoalRepository, TopicRepository,
        QuestionRepository, MistakeRepository,
    )
    s = next(get_session())
    user = UserRepository(s).get_or_create("fp-2")
    goal = GoalRepository(s).create(user_id=user.id, title="g", exam_date=None)
    topic = TopicRepository(s).create(goal_id=goal.id, name="HyDE", source_chunks=[])
    q = QuestionRepository(s).create(
        topic_id=topic.id,
        prompt="What is HyDE?",
        options_json=["A) X", "B) Y", "C) Z", "D) W"],
        answer="A",
        explanation="HyDE is...",
    )
    now = datetime.utcnow()
    MistakeRepository(s).create(
        user_id=user.id, question_id=q.id, user_answer="B",
        due_at=now - timedelta(hours=1), srs_interval_days=1, srs_ease=2.5,
    )
    # Future-due mistake — must NOT appear.
    MistakeRepository(s).create(
        user_id=user.id, question_id=q.id, user_answer="C",
        due_at=now + timedelta(days=7), srs_interval_days=7, srs_ease=2.5,
    )

    resp = client.get("/api/mistakes/due", headers={"x-fingerprint": "fp-2"})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    item = body[0]
    assert item["question"]["prompt"] == "What is HyDE?"
    assert item["question"]["answer"] == "A"
    assert item["topic_name"] == "HyDE"
    assert item["srs_interval_days"] == 1
    assert "mistake_id" in item and "due_at" in item
```

The exact `MistakeRepository.create` signature — verify by grepping `app/db/repositories.py` around line 240. The kwarg names above match the column names; if the constructor uses different names (e.g. `created_at`), adapt.

- [ ] **Step 2: Run test to verify RED**

`cd "study-coach/backend" && uv run pytest tests/api/test_routes_p3_mistakes.py -v` — FAIL with 404.

- [ ] **Step 3: Add handler to `backend/app/api/routes.py`**

```python
class MistakeQuestionOut(BaseModel):
    id: str
    prompt: str
    options: list[str]
    answer: str
    explanation: str


class MistakeDueOut(BaseModel):
    mistake_id: str
    question: MistakeQuestionOut
    due_at: str
    srs_interval_days: int
    srs_ease: float
    topic_name: str


@router.get("/mistakes/due", response_model=list[MistakeDueOut])
def get_mistakes_due(
    user_id: Annotated[str, Depends(get_user_id)],
    session: Annotated[Session, Depends(get_session)],
    limit: int = 20,
):
    from datetime import datetime
    from app.db.repositories import MistakeRepository
    rows = MistakeRepository(session).get_due_for_user(
        user_id=user_id, now=datetime.utcnow(), limit=limit,
    )
    # rows shape depends on repo — verify signature; adapt to whatever it returns.
    # Below assumes rows = list[Mistake] with .question and .question.topic relationships.
    out: list[MistakeDueOut] = []
    for m in rows:
        q = m.question
        out.append(MistakeDueOut(
            mistake_id=m.id,
            question=MistakeQuestionOut(
                id=q.id, prompt=q.prompt, options=q.options_json,
                answer=q.answer, explanation=q.explanation,
            ),
            due_at=m.due_at.isoformat(),
            srs_interval_days=m.srs_interval_days,
            srs_ease=m.srs_ease,
            topic_name=q.topic.name,
        ))
    return out
```

**Important**: Before writing this, grep `app/db/repositories.py` lines 264-280 to see the exact signature of `get_due_for_user` and adapt `now=` / `limit=` kwarg names. If it doesn't accept `now=`, default to "use datetime.utcnow() inside the repo" and skip that kwarg.

Also verify the SQLAlchemy `Mistake.question` relationship + `Question.topic` relationship exist in `app/db/models.py`. If not, the handler needs explicit joins or a more elaborate query — adapt accordingly (may bump test count by +1 if you add a unit test for the join).

- [ ] **Step 4: Run tests to verify GREEN**

Full suite → **208 passed**.

- [ ] **Step 5: Checkpoint**

---

## A6 — MistakeBank view + Redo flow

**Goal:** `/mistakes` route shows the due list; "Redo" button routes to `/quiz?mistake_id=X`. Quiz view (created in A7) reads this param to prefill topic.

**Files:**
- Create: `frontend/src/components/MistakeRow.vue`, `frontend/src/stores/mistakes.ts`
- Modify: `frontend/src/views/MistakeBank.vue` (replace placeholder)
- Modify: `frontend/src/lib/api.ts` (add `getMistakesDue`)

**Test count delta:** 0.

- [ ] **Step 1: Add typed wrapper to `frontend/src/lib/api.ts`**

```ts
export interface MistakeQuestionDto {
  id: string
  prompt: string
  options: string[]
  answer: string
  explanation: string
}

export interface MistakeDueDto {
  mistake_id: string
  question: MistakeQuestionDto
  due_at: string
  srs_interval_days: number
  srs_ease: number
  topic_name: string
}

export function getMistakesDue(limit = 20): Promise<MistakeDueDto[]> {
  return getJSON<MistakeDueDto[]>(`/api/mistakes/due?limit=${limit}`)
}
```

- [ ] **Step 2: Create `frontend/src/stores/mistakes.ts`**

```ts
import { defineStore } from 'pinia'
import { getMistakesDue, type MistakeDueDto } from '../lib/api'

interface MistakesState {
  due: MistakeDueDto[]
  loading: boolean
  error: string | null
}

export const useMistakes = defineStore('mistakes', {
  state: (): MistakesState => ({ due: [], loading: false, error: null }),
  actions: {
    async fetch() {
      this.loading = true
      this.error = null
      try {
        this.due = await getMistakesDue(50)
      } catch (e: any) {
        this.error = e?.message ?? 'failed'
      } finally {
        this.loading = false
      }
    },
  },
})
```

- [ ] **Step 3: Create `frontend/src/components/MistakeRow.vue`**

```vue
<script setup lang="ts">
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { RotateCcw } from 'lucide-vue-next'
import type { MistakeDueDto } from '../lib/api'

const props = defineProps<{ row: MistakeDueDto }>()
const router = useRouter()

const truncated = computed(() =>
  props.row.question.prompt.length > 120
    ? props.row.question.prompt.slice(0, 120) + '…'
    : props.row.question.prompt
)

const nextReview = computed(() => {
  const due = new Date(props.row.due_at)
  const diffMs = due.getTime() - Date.now()
  const days = Math.round(diffMs / 86_400_000)
  if (days < 0) return `overdue by ${-days}d`
  if (days === 0) return 'due today'
  return `due in ${days}d`
})

function redo() {
  router.push({ path: '/quiz', query: { mistake_id: props.row.mistake_id } })
}
</script>

<template>
  <div class="rounded-lg border border-border bg-surface p-4 flex items-start gap-4">
    <div class="flex-1 min-w-0">
      <p class="text-sm">{{ truncated }}</p>
      <div class="mt-2 flex gap-2 text-xs">
        <span class="font-mono px-2 py-0.5 rounded-md bg-primary-bg text-primary">
          {{ row.topic_name }}
        </span>
        <span class="font-mono text-fg-muted">{{ nextReview }}</span>
        <span class="font-mono text-fg-dim">interval {{ row.srs_interval_days }}d · ease {{ row.srs_ease.toFixed(2) }}</span>
      </div>
    </div>
    <button @click="redo"
            class="rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-white hover:bg-primary-2 inline-flex items-center gap-1.5 transition-colors">
      <RotateCcw class="w-4 h-4" /> Redo
    </button>
  </div>
</template>
```

- [ ] **Step 4: Replace `frontend/src/views/MistakeBank.vue`**

```vue
<script setup lang="ts">
import { onMounted, computed } from 'vue'
import { useMistakes } from '../stores/mistakes'
import MistakeRow from '../components/MistakeRow.vue'

const store = useMistakes()
onMounted(() => store.fetch())
const dueCount = computed(() => store.due.length)
</script>

<template>
  <div class="h-full overflow-y-auto p-8">
    <div class="max-w-4xl mx-auto">
      <header class="mb-6">
        <h1 class="text-2xl font-semibold">Mistake Bank</h1>
        <p class="text-sm text-fg-muted mt-1 font-mono">{{ dueCount }} due today</p>
      </header>

      <div v-if="store.loading" class="text-fg-muted text-sm">Loading…</div>
      <div v-else-if="store.error" class="text-sm text-danger">{{ store.error }}</div>
      <div v-else-if="dueCount === 0" class="rounded-lg border border-border bg-surface p-6 text-center">
        <p class="text-fg-muted">No mistakes due. Take a quiz to start tracking.</p>
      </div>
      <div v-else class="flex flex-col gap-3">
        <MistakeRow v-for="row in store.due" :key="row.mistake_id" :row="row" />
      </div>
    </div>
  </div>
</template>
```

- [ ] **Step 5: Run the standard frontend verification**

Route: `/mistakes`.

Interaction script:
1. Pre-seed: in `/chat`, send `quiz me on HyDE` → answer wrong (pick non-A) to generate a mistake.
2. Navigate to `/mistakes` — verify the row appears (or empty state if Ollama isn't running).
3. Click Redo — verify URL changes to `/quiz?mistake_id=...`. The /quiz view is still a placeholder until A7, but the navigation must work.

Save `cut-A6.png`.

- [ ] **Step 6: Checkpoint**

---

## A7 — QuizAdaptive view + DifficultySelector + MCQCard + GradeResult

**Goal:** `/quiz` route runs the full GENERATE→GRADE loop via existing `streamChat`. Difficulty selector prepends `[difficulty:hard]` to message. ModeChip toggles `x-quiz-mode`. `mistake_id` query routes to a re-quiz on the same topic.

**Files:**
- Create: `frontend/src/components/DifficultySelector.vue`, `MCQCard.vue`, `GradeResult.vue`, `frontend/src/stores/quiz.ts`
- Modify: `frontend/src/views/QuizAdaptive.vue` (replace placeholder)
- Modify: `frontend/src/stores/chat.ts` (uncomment the `useQuiz().setNeedsUpload(...)` line introduced in A3)

**Test count delta:** 0.

- [ ] **Step 1: Create `frontend/src/stores/quiz.ts`**

```ts
import { defineStore } from 'pinia'

export type Difficulty = 'easy' | 'med' | 'hard'

interface ParsedMCQ {
  prompt: string
  options: string[]
  // answer + explanation arrive on GRADE turn
}

interface QuizState {
  currentMCQ: ParsedMCQ | null
  lastGrade: { correct: boolean; correctAnswer: string; explanation: string } | null
  difficulty: Difficulty
  needsUpload: boolean
  streaming: boolean
  raw: string  // last assistant text — used by parser
}

export const useQuiz = defineStore('quiz', {
  state: (): QuizState => ({
    currentMCQ: null,
    lastGrade: null,
    difficulty: 'med',
    needsUpload: false,
    streaming: false,
    raw: '',
  }),
  actions: {
    setDifficulty(d: Difficulty) { this.difficulty = d },
    setNeedsUpload(v: boolean) { this.needsUpload = v },
    startStream() { this.streaming = true; this.raw = '' },
    appendRaw(t: string) { this.raw += t },
    finishStream() {
      this.streaming = false
      this.parse()
    },
    reset() {
      this.currentMCQ = null
      this.lastGrade = null
      this.raw = ''
    },
    parse() {
      // Heuristic: if text has "A)" "B)" "C)" "D)" — treat as MCQ GENERATE.
      // If text starts with "✓" or "✗" or has "Correct:" — treat as GRADE.
      const t = this.raw
      const mcqMatch = /[\s\S]*?(?<prompt>[^\n]{10,}\?)\s*\n+\s*A\)\s*([^\n]+)\n\s*B\)\s*([^\n]+)\n\s*C\)\s*([^\n]+)\n\s*D\)\s*([^\n]+)/u.exec(t)
      if (mcqMatch) {
        this.currentMCQ = {
          prompt: mcqMatch.groups!.prompt.trim(),
          options: [
            `A) ${mcqMatch[2].trim()}`,
            `B) ${mcqMatch[3].trim()}`,
            `C) ${mcqMatch[4].trim()}`,
            `D) ${mcqMatch[5].trim()}`,
          ],
        }
        this.lastGrade = null
        return
      }
      const correct = /\b(correct|✓)\b/i.test(t) && !/\bincorrect\b/i.test(t)
      const incorrect = /\b(incorrect|wrong|✗)\b/i.test(t)
      if (correct || incorrect) {
        const answerMatch = /correct answer.*?\b([A-D])\b/i.exec(t)
        const expMatch = /explanation[:\s]+([\s\S]*?)(?:$|\n\n)/i.exec(t)
        this.lastGrade = {
          correct,
          correctAnswer: answerMatch?.[1] ?? '',
          explanation: expMatch?.[1]?.trim() ?? t,
        }
      }
    },
  },
})
```

- [ ] **Step 2: Create `frontend/src/components/DifficultySelector.vue`**

```vue
<script setup lang="ts">
import type { Difficulty } from '../stores/quiz'

const props = defineProps<{ value: Difficulty }>()
const emit = defineEmits<{ (e: 'update:value', d: Difficulty): void }>()

const levels: Difficulty[] = ['easy', 'med', 'hard']
</script>

<template>
  <div role="radiogroup" aria-label="Difficulty"
       class="inline-flex rounded-md border border-border bg-surface p-0.5">
    <button v-for="level in levels" :key="level"
            role="radio" :aria-checked="props.value === level"
            @click="emit('update:value', level)"
            :class="[
              'px-3 py-1 text-xs font-mono rounded transition-colors',
              props.value === level
                ? 'bg-primary text-white'
                : 'text-fg-muted hover:text-fg hover:bg-white/5'
            ]">
      {{ level }}
    </button>
  </div>
</template>
```

- [ ] **Step 3: Create `frontend/src/components/MCQCard.vue`**

```vue
<script setup lang="ts">
import { ref } from 'vue'

const props = defineProps<{ prompt: string; options: string[] }>()
const emit = defineEmits<{ (e: 'submit', choice: string): void }>()

const selected = ref<string | null>(null)
const submitted = ref(false)

function submit() {
  if (!selected.value || submitted.value) return
  submitted.value = true
  emit('submit', selected.value)
}
</script>

<template>
  <div class="rounded-lg border border-border bg-surface p-6">
    <p class="text-base font-medium mb-4">{{ prompt }}</p>
    <div role="radiogroup" aria-label="Options" class="flex flex-col gap-2">
      <label v-for="opt in options" :key="opt"
             :class="[
               'flex items-start gap-3 rounded-md border p-3 cursor-pointer transition-colors',
               selected === opt[0]
                 ? 'border-primary-ring bg-primary-bg'
                 : 'border-border hover:bg-white/5'
             ]">
        <input type="radio" name="mcq" :value="opt[0]" v-model="selected" :disabled="submitted"
               class="mt-1 accent-primary" />
        <span class="text-sm">{{ opt }}</span>
      </label>
    </div>
    <button @click="submit" :disabled="!selected || submitted"
            class="mt-4 rounded-md bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary-2 disabled:opacity-40 transition-colors">
      Submit
    </button>
  </div>
</template>
```

- [ ] **Step 4: Create `frontend/src/components/GradeResult.vue`**

```vue
<script setup lang="ts">
import { CheckCircle2, XCircle } from 'lucide-vue-next'

defineProps<{ correct: boolean; correctAnswer: string; explanation: string }>()
defineEmits<{ (e: 'next'): void }>()
</script>

<template>
  <div :class="[
        'mt-4 rounded-lg border p-4',
        correct ? 'border-success/40 bg-success-bg' : 'border-danger/40 bg-danger-bg'
       ]">
    <div class="flex items-center gap-2 mb-2">
      <component :is="correct ? CheckCircle2 : XCircle"
                 :class="['w-5 h-5', correct ? 'text-success' : 'text-danger']" />
      <span class="text-sm font-semibold">{{ correct ? 'Correct' : 'Incorrect' }}</span>
      <span v-if="!correct && correctAnswer" class="font-mono text-xs text-fg-muted">
        correct answer: {{ correctAnswer }}
      </span>
    </div>
    <p class="text-sm whitespace-pre-wrap">{{ explanation }}</p>
    <button @click="$emit('next')"
            class="mt-3 rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-white hover:bg-primary-2 transition-colors">
      Next question
    </button>
  </div>
</template>
```

- [ ] **Step 5: Replace `frontend/src/views/QuizAdaptive.vue`**

```vue
<script setup lang="ts">
import { ref, onMounted, watch } from 'vue'
import { useRoute } from 'vue-router'
import { useQuiz } from '../stores/quiz'
import { useMistakes } from '../stores/mistakes'
import { useSettings, type Mode } from '../stores/settings'
import { streamChat } from '../lib/api'
import DifficultySelector from '../components/DifficultySelector.vue'
import MCQCard from '../components/MCQCard.vue'
import GradeResult from '../components/GradeResult.vue'
import ModeChip from '../components/ModeChip.vue'
// EmptyCorpusBanner imported in A8

const route = useRoute()
const quiz = useQuiz()
const mistakes = useMistakes()
const settings = useSettings()
const mode = ref<Mode>(settings.defaultQuizMode)
const topicHint = ref('')

onMounted(async () => {
  await mistakes.fetch()
  if (route.query.mistake_id) {
    const m = mistakes.due.find(d => d.mistake_id === route.query.mistake_id)
    topicHint.value = m?.topic_name ?? ''
  }
})

watch(() => route.query.mistake_id, (id) => {
  if (!id) { topicHint.value = ''; quiz.reset(); return }
  const m = mistakes.due.find(d => d.mistake_id === id)
  topicHint.value = m?.topic_name ?? ''
  quiz.reset()
})

function toggleMode() {
  mode.value = mode.value === 'agent_loop' ? 'deterministic' : 'agent_loop'
}

async function send(message: string) {
  quiz.startStream()
  await streamChat(
    message,
    settings.$state,
    {
      onToken: (t) => quiz.appendRaw(t),
      onDone: () => { quiz.finishStream(); mode.value = settings.defaultQuizMode },
      onError: (e) => { quiz.finishStream(); console.error(e); mode.value = settings.defaultQuizMode },
    },
    { quizMode: mode.value },
  )
}

function generate() {
  const topic = topicHint.value || 'HyDE'  // default — student usually has just-uploaded material
  send(`[difficulty:${quiz.difficulty}] quiz me on ${topic}`)
}

function submit(choice: string) {
  send(choice)
}

function nextQuestion() {
  quiz.reset()
  generate()
}
</script>

<template>
  <div class="h-full overflow-y-auto p-8">
    <div class="max-w-3xl mx-auto">
      <header class="flex items-center justify-between mb-6">
        <h1 class="text-2xl font-semibold">Quiz</h1>
        <div class="flex items-center gap-3">
          <DifficultySelector :value="quiz.difficulty" @update:value="quiz.setDifficulty" />
          <ModeChip :mode="mode" :default-mode="settings.defaultQuizMode" @toggle="toggleMode" />
        </div>
      </header>

      <!-- A8 will insert <EmptyCorpusBanner v-if="quiz.needsUpload"> here -->

      <div v-if="quiz.streaming" class="text-fg-muted text-sm">Generating…</div>

      <button v-else-if="!quiz.currentMCQ && !quiz.lastGrade" @click="generate"
              class="rounded-md bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary-2 transition-colors">
        Generate a question
      </button>

      <MCQCard v-else-if="quiz.currentMCQ && !quiz.lastGrade"
               :prompt="quiz.currentMCQ.prompt"
               :options="quiz.currentMCQ.options"
               @submit="submit" />

      <GradeResult v-if="quiz.lastGrade"
                   :correct="quiz.lastGrade.correct"
                   :correct-answer="quiz.lastGrade.correctAnswer"
                   :explanation="quiz.lastGrade.explanation"
                   @next="nextQuestion" />
    </div>
  </div>
</template>
```

- [ ] **Step 6: Uncomment the quiz refusal hook in `frontend/src/stores/chat.ts`**

Find the commented `useQuiz().setNeedsUpload(...)` line introduced in A3 and uncomment it. Add the import `import { useQuiz } from './quiz'` at the top if not already.

- [ ] **Step 7: Run the standard frontend verification**

Route: `/quiz`.

Interaction script:
1. Pre-seed: upload a PDF via `/library`.
2. Click "Generate a question" — verify MCQ renders with 4 options.
3. Click an option + Submit — verify GradeResult appears with explanation.
4. Click ModeChip — flips to deterministic for the next request.
5. Click "Next question" — verify a fresh MCQ.
6. Navigate to `/quiz?mistake_id=<id>` (use any from `/mistakes`) — verify topicHint prefills.

Save `cut-A7.png`.

- [ ] **Step 8: Checkpoint**

---

## A8 — EmptyCorpusBanner + dual-channel detection

**Goal:** When user has 0 chunks OR Quiz GENERATE token stream looks like a refusal, render `<EmptyCorpusBanner>` instead of an MCQ or error.

**Files:**
- Create: `frontend/src/components/EmptyCorpusBanner.vue`
- Modify: `frontend/src/stores/documents.ts` (create — also needed by A10)
- Modify: `frontend/src/lib/api.ts` (add `getDocuments` wrapper)
- Modify: `frontend/src/views/QuizAdaptive.vue` (mount banner)
- Modify: `frontend/src/stores/chat.ts` (already wired in A7 — verify `setNeedsUpload` actually fires)

**Test count delta:** 0.

- [ ] **Step 1: Add typed wrapper to `frontend/src/lib/api.ts`**

```ts
export interface DocumentDto {
  id: string
  filename: string
  chunks_count: number
}

export function getDocuments(): Promise<DocumentDto[]> {
  return getJSON<DocumentDto[]>('/api/documents')
}
```

- [ ] **Step 2: Create `frontend/src/stores/documents.ts`**

```ts
import { defineStore } from 'pinia'
import { getDocuments, type DocumentDto } from '../lib/api'

interface DocsState {
  docs: DocumentDto[]
  loading: boolean
  error: string | null
}

export const useDocuments = defineStore('documents', {
  state: (): DocsState => ({ docs: [], loading: false, error: null }),
  getters: {
    totalChunks: (s) => s.docs.reduce((sum, d) => sum + d.chunks_count, 0),
    isEmpty: (s) => s.docs.length === 0 || s.docs.every(d => d.chunks_count === 0),
  },
  actions: {
    async fetch() {
      this.loading = true
      this.error = null
      try { this.docs = await getDocuments() }
      catch (e: any) { this.error = e?.message ?? 'failed' }
      finally { this.loading = false }
    },
  },
})
```

- [ ] **Step 3: Create `frontend/src/components/EmptyCorpusBanner.vue`**

```vue
<script setup lang="ts">
import { useRouter } from 'vue-router'
import { BookOpen, Upload } from 'lucide-vue-next'

const router = useRouter()
function go() { router.push({ path: '/library', query: { return: '/quiz' } }) }
</script>

<template>
  <div class="rounded-lg border border-warning/30 bg-warning-bg p-6 flex items-start gap-4">
    <BookOpen class="w-6 h-6 text-warning mt-1 shrink-0" />
    <div class="flex-1">
      <h3 class="text-base font-semibold text-fg">Quiz needs your study materials</h3>
      <p class="text-sm text-fg-muted mt-1">
        Upload a PDF in Library to start generating questions grounded in your sources.
      </p>
      <button @click="go"
              class="mt-3 rounded-md bg-warning px-4 py-2 text-sm font-medium text-bg hover:opacity-90 inline-flex items-center gap-2 transition-opacity">
        <Upload class="w-4 h-4" /> Upload PDF
      </button>
    </div>
  </div>
</template>
```

- [ ] **Step 4: Mount in `QuizAdaptive.vue`**

In `<script setup>` add:
```ts
import { useDocuments } from '../stores/documents'
import EmptyCorpusBanner from '../components/EmptyCorpusBanner.vue'
import { computed } from 'vue'

const docs = useDocuments()
onMounted(() => docs.fetch())
const needsUpload = computed(() => docs.isEmpty || quiz.needsUpload)
```

In template, replace the comment `<!-- A8 will insert ... -->` with:
```vue
<EmptyCorpusBanner v-if="needsUpload" />
```

And gate the rest of the UI on `v-else`:
```vue
<div v-else>
  <div v-if="quiz.streaming">…</div>
  <button v-else-if="!quiz.currentMCQ && !quiz.lastGrade" …
</div>
```

- [ ] **Step 5: Run the standard frontend verification**

Route: `/quiz`.

Interaction script (two passes):

Pass 1 — pre-flight empty:
1. Open a fresh browser context (or clear IndexedDB) so no docs exist.
2. Navigate to `/quiz` — verify `<EmptyCorpusBanner>` renders with the upload CTA.
3. Click "Upload PDF" — URL becomes `/library?return=/quiz`.

Pass 2 — in-flight refusal:
1. With docs uploaded, navigate to `/quiz`.
2. Send a quiz request for an off-corpus topic (e.g. `quiz me on PyTorch` if the uploaded corpus is HKBU HyDE).
3. If the agent path refuses ("I'm unable to retrieve …"), verify banner appears even though docs exist (the in-flight detection fires via `chat.ts` → `setNeedsUpload(true)`).
4. To reset, click the upload CTA + re-fetch docs.

Save `cut-A8.png`.

- [ ] **Step 6: Checkpoint**

---

## A9 — GET /api/mastery + MasteryRepository.list_for_user_detailed

**Goal:** Return user's mastery scores (with last_reviewed), derived `weak_topics` (sorted asc, max 5, score < 0.5), and `overdue_milestones_count`.

**Files:**
- Modify: `backend/app/db/repositories.py` (append method)
- Modify: `backend/app/api/routes.py` (append handler)
- Create: `backend/tests/db/test_repositories_p3.py`, `backend/tests/api/test_routes_p3_mastery.py`

**Test count delta:** +4 (1 repo + 3 route: empty, with-scores, weak+overdue derivation). Baseline → 212.

- [ ] **Step 1: Write the failing repo test**

Create `backend/tests/db/test_repositories_p3.py`:

```python
"""Cut A9 — MasteryRepository.list_for_user_detailed."""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import Base
from app.db.repositories import (
    UserRepository, GoalRepository, TopicRepository, MasteryRepository,
)


@pytest.fixture
def session():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.close()


def test_list_for_user_detailed_returns_topic_and_mastery(session):
    user = UserRepository(session).get_or_create("fp-1")
    goal = GoalRepository(session).create(user_id=user.id, title="g", exam_date=None)
    t1 = TopicRepository(session).create(goal_id=goal.id, name="HyDE", source_chunks=[])
    t2 = TopicRepository(session).create(goal_id=goal.id, name="BM25", source_chunks=[])
    MasteryRepository(session).upsert(user_id=user.id, topic_id=t1.id, score=0.8)
    MasteryRepository(session).upsert(user_id=user.id, topic_id=t2.id, score=0.3)

    rows = MasteryRepository(session).list_for_user_detailed(user.id)
    assert len(rows) == 2
    by_name = {topic.name: mastery for topic, mastery in rows}
    assert by_name["HyDE"].score == 0.8
    assert by_name["BM25"].score == 0.3
    assert by_name["HyDE"].last_reviewed is not None
```

- [ ] **Step 2: Run RED**

`cd "study-coach/backend" && uv run pytest tests/db/test_repositories_p3.py -v` — FAIL with AttributeError on `list_for_user_detailed`.

- [ ] **Step 3: Add method to `MasteryRepository`**

At the end of class `MasteryRepository` in `backend/app/db/repositories.py`, append:

```python
    def list_for_user_detailed(self, user_id: str) -> list[tuple]:
        """Return (Topic, Mastery) pairs for the user.

        Used by P3 GET /api/mastery — it needs Topic.name + Mastery.score + last_reviewed.
        """
        stmt = (
            select(Topic, Mastery)
            .join(Mastery, Mastery.topic_id == Topic.id)
            .where(Mastery.user_id == user_id)
        )
        return [(topic, mastery) for topic, mastery in self.session.execute(stmt).all()]
```

Verify `Topic` and `Mastery` are imported at the top of `repositories.py` (they're used by other methods, should already be imported).

- [ ] **Step 4: Run GREEN**

`uv run pytest tests/db/test_repositories_p3.py -v` — PASS.

- [ ] **Step 5: Write the failing route tests**

Create `backend/tests/api/test_routes_p3_mastery.py`:

```python
"""Cut A9 — GET /api/mastery."""
from datetime import datetime, timedelta
import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/p3_mastery.db")
    monkeypatch.setenv("STUDY_COACH_TEST_MODE", "1")
    from app.db import session as session_mod
    session_mod._engine = None
    session_mod._SessionLocal = None
    app = create_app()
    yield TestClient(app)


def test_get_mastery_empty(client):
    resp = client.get("/api/mastery", headers={"x-fingerprint": "fp-1"})
    assert resp.status_code == 200
    assert resp.json() == {"scores": [], "weak_topics": [], "overdue_milestones_count": 0}


def test_get_mastery_returns_scores_and_weak_topics(client):
    from app.db.session import get_session
    from app.db.repositories import (
        UserRepository, GoalRepository, TopicRepository, MasteryRepository,
    )
    s = next(get_session())
    user = UserRepository(s).get_or_create("fp-2")
    goal = GoalRepository(s).create(user_id=user.id, title="g", exam_date=None)
    topics = {
        name: TopicRepository(s).create(goal_id=goal.id, name=name, source_chunks=[])
        for name in ["HyDE", "BM25", "RRF", "Reranker"]
    }
    MasteryRepository(s).upsert(user_id=user.id, topic_id=topics["HyDE"].id, score=0.85)
    MasteryRepository(s).upsert(user_id=user.id, topic_id=topics["BM25"].id, score=0.25)
    MasteryRepository(s).upsert(user_id=user.id, topic_id=topics["RRF"].id, score=0.4)
    MasteryRepository(s).upsert(user_id=user.id, topic_id=topics["Reranker"].id, score=0.7)

    resp = client.get("/api/mastery", headers={"x-fingerprint": "fp-2"})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["scores"]) == 4
    # weak_topics: score < 0.5, sorted asc, max 5
    assert body["weak_topics"] == ["BM25", "RRF"]
    assert body["overdue_milestones_count"] == 0


def test_get_mastery_overdue_count_from_active_plan(client):
    from app.db.session import get_session
    from app.db.repositories import (
        UserRepository, GoalRepository, PlanRepository,
    )
    s = next(get_session())
    user = UserRepository(s).get_or_create("fp-3")
    goal = GoalRepository(s).create(user_id=user.id, title="g", exam_date=None)
    past = (datetime.utcnow() - timedelta(days=1)).isoformat()
    future = (datetime.utcnow() + timedelta(days=7)).isoformat()
    PlanRepository(s).update_milestones(
        goal_id=goal.id,
        milestones=[
            {"title": "overdue done", "due_at": past, "done": True, "topic": None},   # NOT counted
            {"title": "overdue!", "due_at": past, "done": False, "topic": None},      # counted
            {"title": "future", "due_at": future, "done": False, "topic": None},      # NOT counted
        ],
    )
    resp = client.get("/api/mastery", headers={"x-fingerprint": "fp-3"})
    assert resp.json()["overdue_milestones_count"] == 1
```

- [ ] **Step 6: Run RED**

3 new tests FAIL (404).

- [ ] **Step 7: Add handler to `backend/app/api/routes.py`**

```python
class MasteryScoreOut(BaseModel):
    topic_id: str
    topic_name: str
    score: float
    last_reviewed: str


class MasteryOut(BaseModel):
    scores: list[MasteryScoreOut]
    weak_topics: list[str]
    overdue_milestones_count: int


@router.get("/mastery", response_model=MasteryOut)
def get_mastery(
    user_id: Annotated[str, Depends(get_user_id)],
    session: Annotated[Session, Depends(get_session)],
):
    from datetime import datetime
    from app.db.repositories import (
        MasteryRepository, GoalRepository, PlanRepository,
    )
    rows = MasteryRepository(session).list_for_user_detailed(user_id)
    scores = [
        MasteryScoreOut(
            topic_id=topic.id, topic_name=topic.name,
            score=m.score, last_reviewed=m.last_reviewed.isoformat(),
        )
        for topic, m in rows
    ]
    weak = sorted([s for s in scores if s.score < 0.5], key=lambda s: s.score)[:5]
    weak_names = [s.topic_name for s in weak]

    goals = GoalRepository(session).list_active_for_user(user_id)
    overdue = 0
    if goals:
        plan = PlanRepository(session).get_by_goal(goals[0].id)
        if plan is not None:
            now = datetime.utcnow().isoformat()
            for m in plan.milestones_json:
                if not m.get("done") and m.get("due_at") and m["due_at"] < now:
                    overdue += 1
    return MasteryOut(scores=scores, weak_topics=weak_names, overdue_milestones_count=overdue)
```

- [ ] **Step 8: Run GREEN**

`uv run pytest tests/api/test_routes_p3_mastery.py -v` — 3 PASS.

Full suite: `uv run pytest -q | tail -5` → **212 passed**.

- [ ] **Step 9: Checkpoint**

---

## A10 — Overview view + 4 widget cards + UploadGate + overview store + mastery store

**Goal:** `/` is the portfolio hero. 4 cards summarize state; `<UploadGate>` appears if no docs.

**Files:**
- Create: `frontend/src/stores/mastery.ts`, `frontend/src/stores/overview.ts`
- Create: `frontend/src/components/UploadGate.vue`, `MasteryCard.vue`, `PlanProgressCard.vue`, `MistakesDueCard.vue`, `WeakTopicsChips.vue`
- Modify: `frontend/src/views/Overview.vue` (replace placeholder)
- Modify: `frontend/src/lib/api.ts` (add `getMastery` wrapper)

**Test count delta:** 0.

- [ ] **Step 1: Add typed wrapper to `frontend/src/lib/api.ts`**

```ts
export interface MasteryScoreDto {
  topic_id: string
  topic_name: string
  score: number
  last_reviewed: string
}

export interface MasteryDto {
  scores: MasteryScoreDto[]
  weak_topics: string[]
  overdue_milestones_count: number
}

export function getMastery(): Promise<MasteryDto> {
  return getJSON<MasteryDto>('/api/mastery')
}
```

- [ ] **Step 2: Create `frontend/src/stores/mastery.ts`**

```ts
import { defineStore } from 'pinia'
import { getMastery, type MasteryDto } from '../lib/api'

interface MasteryState {
  data: MasteryDto
  loading: boolean
  error: string | null
}

export const useMastery = defineStore('mastery', {
  state: (): MasteryState => ({
    data: { scores: [], weak_topics: [], overdue_milestones_count: 0 },
    loading: false,
    error: null,
  }),
  actions: {
    async fetch() {
      this.loading = true
      this.error = null
      try { this.data = await getMastery() }
      catch (e: any) { this.error = e?.message ?? 'failed' }
      finally { this.loading = false }
    },
  },
})
```

- [ ] **Step 3: Create `frontend/src/stores/overview.ts` (derived)**

```ts
import { defineStore } from 'pinia'
import { computed } from 'vue'
import { useMastery } from './mastery'
import { usePlan } from './plan'
import { useMistakes } from './mistakes'
import { useDocuments } from './documents'

export const useOverview = defineStore('overview', () => {
  const m = useMastery()
  const p = usePlan()
  const x = useMistakes()
  const d = useDocuments()

  async function fetchAll() {
    await Promise.all([m.fetch(), x.fetch(), d.fetch(), p.fetch().catch(() => {})])
  }

  const topMastery = computed(() =>
    [...m.data.scores].sort((a, b) => b.score - a.score).slice(0, 5)
  )
  const nextMilestone = computed(() => {
    if (!p.plan) return null
    const pending = p.plan.milestones.filter(s => !s.done && s.due_at)
    pending.sort((a, b) => (a.due_at ?? '').localeCompare(b.due_at ?? ''))
    return pending[0] ?? null
  })

  return { fetchAll, topMastery, nextMilestone }
})
```

- [ ] **Step 4: Create the 5 widget components**

`UploadGate.vue`:

```vue
<script setup lang="ts">
import { useRouter } from 'vue-router'
import { Upload } from 'lucide-vue-next'
const router = useRouter()
</script>
<template>
  <div class="rounded-lg border border-warning/30 bg-warning-bg p-4 mb-6 flex items-center gap-3">
    <Upload class="w-5 h-5 text-warning" />
    <p class="text-sm flex-1">No sources uploaded yet. Upload a PDF to start tracking.</p>
    <button @click="router.push('/library')"
            class="rounded-md bg-warning px-3 py-1.5 text-xs font-medium text-bg hover:opacity-90">
      Upload PDF
    </button>
  </div>
</template>
```

`MasteryCard.vue`:

```vue
<script setup lang="ts">
import type { MasteryScoreDto } from '../lib/api'
defineProps<{ scores: MasteryScoreDto[] }>()
function barColor(score: number) {
  if (score >= 0.7) return 'bg-success'
  if (score >= 0.4) return 'bg-warning'
  return 'bg-danger'
}
</script>
<template>
  <section class="rounded-lg border border-border bg-surface p-6">
    <h2 class="text-sm font-semibold mb-4 text-fg-muted uppercase tracking-wider">Top mastery</h2>
    <div v-if="scores.length === 0" class="text-fg-dim text-sm">No quizzes taken yet.</div>
    <div v-else class="flex flex-col gap-3">
      <div v-for="s in scores" :key="s.topic_id">
        <div class="flex justify-between text-xs font-mono mb-1">
          <span>{{ s.topic_name }}</span>
          <span class="text-fg-muted">{{ (s.score * 100).toFixed(0) }}%</span>
        </div>
        <div class="h-2 bg-bg rounded-full overflow-hidden">
          <div :class="['h-full transition-all', barColor(s.score)]"
               :style="{ width: (s.score * 100) + '%' }"></div>
        </div>
      </div>
    </div>
  </section>
</template>
```

`PlanProgressCard.vue`:

```vue
<script setup lang="ts">
import { computed } from 'vue'
import { usePlan } from '../stores/plan'
import type { MilestoneDto } from '../lib/api'

const plan = usePlan()
const props = defineProps<{ nextMilestone: MilestoneDto | null }>()
const done = computed(() => plan.plan?.milestones.filter(m => m.done).length ?? 0)
const total = computed(() => plan.plan?.milestones.length ?? 0)
const pct = computed(() => (total.value === 0 ? 0 : Math.round(100 * done.value / total.value)))
</script>
<template>
  <section class="rounded-lg border border-border bg-surface p-6">
    <h2 class="text-sm font-semibold mb-4 text-fg-muted uppercase tracking-wider">Plan progress</h2>
    <div v-if="!plan.plan" class="text-fg-dim text-sm">No active plan.</div>
    <template v-else>
      <div class="flex items-baseline gap-2">
        <span class="text-2xl font-mono">{{ done }}</span>
        <span class="text-fg-muted">/ {{ total }} done</span>
        <span class="ml-auto text-xs font-mono text-fg-muted">{{ pct }}%</span>
      </div>
      <div class="h-2 bg-bg rounded-full overflow-hidden mt-2">
        <div class="h-full bg-primary transition-all" :style="{ width: pct + '%' }"></div>
      </div>
      <div v-if="props.nextMilestone" class="mt-4 text-xs text-fg-muted">
        Next: <span class="text-fg">{{ props.nextMilestone.title }}</span>
      </div>
    </template>
  </section>
</template>
```

`MistakesDueCard.vue`:

```vue
<script setup lang="ts">
import { useMistakes } from '../stores/mistakes'
const store = useMistakes()
</script>
<template>
  <section class="rounded-lg border border-border bg-surface p-6">
    <h2 class="text-sm font-semibold mb-4 text-fg-muted uppercase tracking-wider">Mistakes due</h2>
    <div v-if="store.due.length === 0" class="text-fg-dim text-sm">Nothing due. ✓</div>
    <template v-else>
      <div class="text-2xl font-mono mb-3">{{ store.due.length }}</div>
      <ul class="flex flex-col gap-2 text-xs text-fg-muted">
        <li v-for="m in store.due.slice(0, 3)" :key="m.mistake_id" class="truncate">
          {{ m.question.prompt }}
        </li>
      </ul>
      <RouterLink to="/mistakes" class="mt-3 inline-block text-xs text-primary hover:underline">
        Review all →
      </RouterLink>
    </template>
  </section>
</template>
```

`WeakTopicsChips.vue`:

```vue
<script setup lang="ts">
defineProps<{ topics: string[] }>()
</script>
<template>
  <section class="rounded-lg border border-border bg-surface p-6">
    <h2 class="text-sm font-semibold mb-4 text-fg-muted uppercase tracking-wider">Weak topics</h2>
    <div v-if="topics.length === 0" class="text-fg-dim text-sm">No weak topics yet.</div>
    <div v-else class="flex flex-wrap gap-2">
      <RouterLink v-for="t in topics" :key="t"
                  :to="{ path: '/quiz', query: { topic: t } }"
                  class="font-mono text-xs px-2 py-1 rounded-md bg-danger-bg text-danger border border-danger/30 hover:bg-danger/20 transition-colors">
        {{ t }}
      </RouterLink>
    </div>
  </section>
</template>
```

- [ ] **Step 5: Replace `frontend/src/views/Overview.vue`**

```vue
<script setup lang="ts">
import { onMounted } from 'vue'
import { useOverview } from '../stores/overview'
import { useDocuments } from '../stores/documents'
import { useMastery } from '../stores/mastery'
import UploadGate from '../components/UploadGate.vue'
import MasteryCard from '../components/MasteryCard.vue'
import PlanProgressCard from '../components/PlanProgressCard.vue'
import MistakesDueCard from '../components/MistakesDueCard.vue'
import WeakTopicsChips from '../components/WeakTopicsChips.vue'

const overview = useOverview()
const docs = useDocuments()
const mastery = useMastery()

onMounted(() => overview.fetchAll())
</script>

<template>
  <div class="h-full overflow-y-auto p-8">
    <div class="max-w-6xl mx-auto">
      <header class="mb-8">
        <h1 class="text-3xl font-bold tracking-tight">Overview</h1>
        <p class="text-sm text-fg-muted mt-1">
          {{ new Date().toDateString() }} · overdue
          <span class="font-mono text-warning">{{ mastery.data.overdue_milestones_count }}</span>
        </p>
      </header>

      <UploadGate v-if="docs.isEmpty" />

      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <MasteryCard :scores="overview.topMastery" />
        <PlanProgressCard :next-milestone="overview.nextMilestone" />
        <MistakesDueCard />
        <WeakTopicsChips :topics="mastery.data.weak_topics" />
      </div>

      <!-- RadarChart slot — A11 mounts here -->
    </div>
  </div>
</template>
```

- [ ] **Step 6: Run the standard frontend verification**

Route: `/`.

Verify all 4 cards render with empty-state messages if no data. With data (quiz once, generate plan once): cards populate. Save `cut-A10.png`.

- [ ] **Step 7: Checkpoint**

---

## A11 — RadarChart (chart.js integration)

**Goal:** Mount a 5-axis radar in Overview showing mastery / plan / quiz / streak / coverage. Streak + coverage are stubbed for now (real computation = P4); use placeholder values derived from existing data.

**Files:**
- Create: `frontend/src/components/RadarChart.vue`
- Modify: `frontend/src/views/Overview.vue` (mount)
- Modify: `frontend/package.json` (`pnpm add chart.js vue-chartjs`)

**Test count delta:** 0.

- [ ] **Step 1: Install**

`cd "study-coach/frontend" && pnpm add chart.js vue-chartjs`

Read `package.json` after install (Write-after-Read discipline).

- [ ] **Step 2: Create `frontend/src/components/RadarChart.vue`**

```vue
<script setup lang="ts">
import { computed } from 'vue'
import { Radar } from 'vue-chartjs'
import {
  Chart as ChartJS,
  RadialLinearScale, PointElement, LineElement, Filler, Tooltip, Legend,
} from 'chart.js'
import { useMastery } from '../stores/mastery'
import { useMistakes } from '../stores/mistakes'
import { usePlan } from '../stores/plan'
import { useDocuments } from '../stores/documents'

ChartJS.register(RadialLinearScale, PointElement, LineElement, Filler, Tooltip, Legend)

const mastery = useMastery()
const mistakes = useMistakes()
const plan = usePlan()
const docs = useDocuments()

// Simple normalized 0..1 vector across 5 dims.
const data = computed(() => {
  const avgMastery = mastery.data.scores.length === 0
    ? 0
    : mastery.data.scores.reduce((a, s) => a + s.score, 0) / mastery.data.scores.length
  const planProgress = !plan.plan || plan.plan.milestones.length === 0
    ? 0
    : plan.plan.milestones.filter(m => m.done).length / plan.plan.milestones.length
  // Quiz "accuracy" = 1 - normalizedMistakes; cap due count at 20 for normalization.
  const quizAccuracy = Math.max(0, 1 - Math.min(mistakes.due.length, 20) / 20)
  // Streak — P4. Placeholder = 0.5 if any activity, else 0.
  const streak = (mastery.data.scores.length || mistakes.due.length) ? 0.5 : 0
  // Coverage = doc count / 5 capped at 1.
  const coverage = Math.min(docs.docs.length / 5, 1)

  return {
    labels: ['Mastery', 'Plan progress', 'Quiz accuracy', 'Streak', 'Coverage'],
    datasets: [{
      label: 'You',
      data: [avgMastery, planProgress, quizAccuracy, streak, coverage],
      backgroundColor: 'rgba(99,102,241,0.25)',
      borderColor: '#6366f1',
      borderWidth: 2,
      pointBackgroundColor: '#6366f1',
    }],
  }
})

const options = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: { legend: { display: false } },
  scales: {
    r: {
      min: 0, max: 1,
      ticks: { display: false, stepSize: 0.25 },
      grid: { color: 'rgba(255,255,255,0.08)' },
      angleLines: { color: 'rgba(255,255,255,0.08)' },
      pointLabels: { color: '#b0b6c5', font: { size: 12, family: 'JetBrains Mono' } },
    },
  },
  animation: { duration: 250 },
}
</script>

<template>
  <section class="rounded-lg border border-border bg-surface p-6 mt-6">
    <h2 class="text-sm font-semibold mb-4 text-fg-muted uppercase tracking-wider">5-axis profile</h2>
    <div class="h-80">
      <Radar :data="data" :options="options" />
    </div>
  </section>
</template>
```

- [ ] **Step 3: Mount in `Overview.vue`**

After the `<div class="grid …">…</div>` block, add:

```vue
<RadarChart />
```

And import: `import RadarChart from '../components/RadarChart.vue'`.

- [ ] **Step 4: Run the standard frontend verification**

Route: `/`. Verify the radar renders, axes labeled, no console errors. Reduced-motion gate: with prefers-reduced-motion enabled in DevTools rendering panel, animation should be near-instant.

Save `cut-A11.png`.

- [ ] **Step 5: Checkpoint**

---

## A12 — Settings: default-mode selectors + judge_model input

**Goal:** Surface the new settings fields in the existing Settings view so the user can change defaults without DevTools / localStorage hacks.

**Files:**
- Modify: `frontend/src/views/Settings.vue`

**Test count delta:** 0.

- [ ] **Step 1: Read the current `Settings.vue`**

`Read frontend/src/views/Settings.vue` — note its current shape (form fields + persist call).

- [ ] **Step 2: Add the 3 new fields**

Append inside the existing `<form>` (or the equivalent block):

```vue
<fieldset class="rounded-lg border border-border bg-surface p-4 mt-4">
  <legend class="text-xs font-mono uppercase tracking-wider text-fg-muted px-2">P3 mode defaults</legend>

  <div class="mt-3">
    <label class="block text-sm mb-1">Plan view default mode</label>
    <select v-model="settings.defaultPlannerMode" @change="settings.persist"
            class="bg-bg border border-border rounded-md px-3 py-2 text-sm font-mono">
      <option value="agent_loop">agent_loop</option>
      <option value="deterministic">deterministic</option>
    </select>
  </div>

  <div class="mt-3">
    <label class="block text-sm mb-1">Quiz view default mode</label>
    <select v-model="settings.defaultQuizMode" @change="settings.persist"
            class="bg-bg border border-border rounded-md px-3 py-2 text-sm font-mono">
      <option value="agent_loop">agent_loop</option>
      <option value="deterministic">deterministic</option>
    </select>
  </div>

  <div class="mt-3">
    <label class="block text-sm mb-1">Judge model (x-judge-model)</label>
    <input v-model="settings.judgeModel" @blur="settings.persist"
           placeholder="e.g. qwen2.5:7b (empty = same as main)"
           class="bg-bg border border-border rounded-md px-3 py-2 text-sm font-mono w-full max-w-md" />
  </div>
</fieldset>
```

- [ ] **Step 3: Run the standard frontend verification**

Route: `/settings`. Change a value, reload — verify it persists.

Save `cut-A12.png`.

- [ ] **Step 4: Checkpoint**

---

## A13 — Full E2E + screenshots + docs + memory update

**Goal:** Final verification + ship docs.

**Files:**
- Modify: `study-coach/docs/EVAL.md` (append §P3 section)
- Modify: `study-coach/docs/ROADMAP.md` (§P3 marked done; §P4 refined)
- Create: `study-coach/docs/p3_frontend_productize.md` (sister blog)
- Modify: `/Users/lianghaozhe/.claude/projects/-Users-lianghaozhe-Downloads-Study-Compaion-and-JadeAI/memory/project_study_coach_refactor.md` (append P3 section)
- Save: `study-coach/docs/screenshots/p3/{overview,chat,plan,quiz,mistakes,library,settings}.png` (7 final view screenshots)

**Test count delta:** 0. Final backend baseline: **212 passed**. Final frontend build: clean.

- [ ] **Step 1: Run the full study loop end-to-end via chrome-devtools MCP**

Real Ollama running. Sequence:

1. Visit `/library` and upload `data/hkbu_corpus/Topic7.pdf` (or any test PDF).
2. Wait for chunks_count > 0; navigate to `/` — `<UploadGate>` should disappear.
3. Navigate to `/chat` — send `帮我做学习计划 on HyDE 画脑图`. Wait for done.
4. Navigate to `/plan` — verify milestones rendered + mindmap panel.
5. Navigate to `/quiz` — generate question, submit a wrong answer.
6. Navigate to `/mistakes` — verify the new mistake appears. Click Redo.
7. Submit the correct answer.
8. Navigate to `/` — verify mastery score moved, mistakes count decreased.

Take 7 final screenshots (one per view) saved as `docs/screenshots/p3/overview.png` etc.

If any step fails, file the issue inline in `docs/EVAL.md §P3 Known issues` and continue — don't block the cut.

- [ ] **Step 2: Append `docs/EVAL.md` § "P3 Frontend Productize"**

Write a focused 100-line section. Cover:
- TL;DR — 4 views + 4 endpoints shipped, alignment-safety banner operationalizes P2.3 §F3.
- Each view: 2-3 sentences + reference to screenshot.
- Mode dispatch UX validation — how the ModeChip + view-default pattern feels in practice (note any UX friction observed in E2E).
- Empty-corpus banner verification — both pre-flight and in-flight channels tested.
- Limitations (no mobile, no real streak/coverage, no token-cost dashboard).
- Per-cut test count: A1+A4+A5+A9 = +10 tests; final baseline 212.

- [ ] **Step 3: Update `docs/ROADMAP.md`**

In §P3 — change `- [ ]` items to `- [x]` and append the actual numbers:
- 7 views shipped + 4 backend GETs + 6 new Pinia stores
- Test baseline 212 (+10 over P2.3's 202)
- 14 cuts via subagent-driven-development (or executing-plans depending on chosen execution)

In §P4 — refine based on what surfaced (likely: real OAuth, real streak/coverage computation, mobile UI, drag-reorder milestones, batch mistake review).

- [ ] **Step 4: Write `docs/p3_frontend_productize.md` (sister blog ~800-1200 words)**

Section ideas:
- "What I built" — 7-view dashboard productize over P2.3 backend.
- "Mode dispatch UX as portfolio narrative" — how the `<ModeChip>` makes the dual-mode finding visible to anyone using the app, not just readable in EVAL.md.
- "Operationalizing the alignment-safety finding" — `<EmptyCorpusBanner>` is what makes P2.3 §F3 a product, not a data point.
- "Why no shadcn-vue / no Vitest / no mobile" — explicit scope discipline.
- "What this proves" — backend ablation + frontend product = full-stack portfolio artifact.

Sister to `agent_loop_vs_deterministic.md` + `quiz_ablation_followup.md` — keep the same voice/density.

- [ ] **Step 5: Update memory**

Append a new `P3 (shipped 2026-MM-DD)` block to `/Users/lianghaozhe/.claude/projects/-Users-lianghaozhe-Downloads-Study-Compaion-and-JadeAI/memory/project_study_coach_refactor.md`. Mirror the P2.3 entry shape: cuts list, key lessons learned, baseline counts.

Key lessons to record (refresh based on what actually happened):
- Tailwind 4 `@theme` block behavior (if any surprises).
- Whether Read-before-Write after `pnpm add` caught anything.
- Whether the dual-channel `<EmptyCorpusBanner>` triggered correctly in both pre-flight and in-flight paths.
- Whether the `quiz.ts` regex parser was robust enough or if it needed iteration after first real test.
- chrome-devtools MCP screenshot workflow notes (any flakiness in `take_screenshot` paths).

- [ ] **Step 6: Final checkpoint**

Run: `cd "study-coach/backend" && uv run pytest -q | tail -5` → **212 passed**.
Run: `cd "study-coach/frontend" && pnpm build` → clean.
Confirm: 14 screenshots in `docs/screenshots/p3/` (A0-A12 cut screenshots + 7 final view shots; A13 doesn't add a screenshot, the 7 are its deliverable).

P3 done.

---

## Self-Review Notes

**Spec coverage check** (against `docs/superpowers/specs/2026-05-25-p3-frontend-productize-design.md`):

| Spec section | Covered by |
|---|---|
| §1 Architecture overview | A0 (shell + tokens) |
| §2 Routing + Pinia stores | A0 (router) + per-view cuts (stores) |
| §3.1 Overview hero | A10 + A11 |
| §3.2 PlanTimeline | A2 + A3 |
| §3.3 QuizAdaptive | A7 + A8 |
| §3.4 MistakeBank | A6 |
| §4.1 GET /api/plans/current | A1 |
| §4.2 GET /api/mistakes/due | A5 |
| §4.3 GET /api/mastery | A9 |
| §4.5 GET /api/documents | A4 |
| §5 Mode dispatch flow | A0 (settings + llmHeaders) + A2/A7 (ModeChip wiring) |
| §6 P2.3 alignment-safety UX | A8 (banner + dual-channel detection in `parse.ts`) |
| §7 Cut sequencing | This entire plan |
| §10 Out of scope | Honoured throughout (no shadcn-vue, no Vitest, no mobile, no drag-reorder) |
| §11 cloud-adapt markers | Marker placements noted in `parse.ts` (A3) + can be sprinkled elsewhere as cuts land; minimum 5 across the project at A13 verification time |

**Placeholder scan:** Searched for "TBD"/"TODO"/"implement later" — none. Every code block is concrete. Repo method signatures (`get_due_for_user`, `list_active_for_user`, `get_by_goal`, `update_milestones`, `create`) are all verified existing per the grep at planning time.

**Type consistency:** `Mode` defined in `settings.ts` and used consistently across `streamChat`, `ModeChip`, `PlanTimeline`, `QuizAdaptive`. `MilestoneDto`/`MasteryScoreDto`/`MistakeDueDto`/`DocumentDto` defined once in `api.ts`, reused across stores and components.

**Known fragility / flag for executor**:
- `quiz.ts` MCQ-parsing regex is heuristic; if the LLM emits an unexpected MCQ format, parsing will return null and the user sees "Generating…" stuck. Adjust the regex iteratively in A7's verification cycle — if mid-cut you find it doesn't parse real output, fix the regex before declaring A7 done. Don't add backend changes to "fix" formatting.
- `MistakeRepository.get_due_for_user` signature must be verified at A5 implementation time (`now=` kwarg or not, `limit=` or not) — the test + handler shown assume kwargs that may differ; adapt to actual signature.
- `get_session` is a generator; `next(get_session())` works inline in tests but for production handlers we use `Depends(get_session)` (FastAPI handles the close). Don't confuse the two patterns.
- mermaid render in dark mode requires `theme: 'dark'` AND a parent surface with sufficient contrast — the `<MindmapPanel>` wraps in `bg-surface` which is fine.
- `Read frontend/package.json` before any `Edit` after `pnpm add` per memory `feedback_scaffold_write_readfirst`.

---

**Plan complete and saved to `study-coach/docs/superpowers/plans/2026-05-25-p3-frontend-productize.md`.**

Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task (one per cut A0-A13), review between cuts, two-stage code-quality review per CLAUDE.md discipline. Matches P2.1-③/④/⑤ + P2.2 + P2.3 workflow.

2. **Inline Execution** — Execute cuts in this session using `executing-plans`, batch with checkpoints. Faster turn-around per cut but less isolation.

Which approach?
