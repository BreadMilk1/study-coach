# P4b — Product Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Agent visibility panel, Goal Setup wizard, i18n, streak/coverage, heatmap, Gantt timeline, drag-reorder milestones, mark-understood, MCQ hardening.

**Architecture:** 9 independent feature cuts. Backend changes: new `GET /api/users/me/stats`, `PATCH /api/plans/{id}/milestones/reorder`, `POST /api/mistakes/{id}/mark-understood`, SSE `type:"trace"` events, streak/coverage in mastery response. Frontend changes: 5 new components (TracePanel, Onboarding, HeatmapCard, PlanTimeline, MobileNav stub), vue-i18n integration, drag handlers on MilestoneRow, settings store extended. All cuts are independent enough for parallel subagents except i18n which should run last (touches every component).

**Tech Stack:** vue-i18n, native HTML5 drag-and-drop, no new chart deps

**Current baseline:** 233 backend tests passing, frontend build passing.

---

## Cut 0: Settings store + API layer prep (shared foundation)

### Task 0.1: Extend settings store with debugMode + language

**Files:**
- Modify: `frontend/src/stores/settings.ts`

Add to `SettingsState` interface:

```typescript
interface SettingsState {
  // ... existing fields ...
  debugMode: boolean
  language: 'en' | 'zh-CN'
}
```

Add to `loadInitial()` default:

```typescript
debugMode: parsed.debugMode ?? false,
language: parsed.language ?? 'en',
```

The `persist()` action already saves `this.$state` — no change needed.

### Task 0.2: Add API functions for new endpoints

**Files:**
- Modify: `frontend/src/lib/api.ts`

Add at end of file:

```typescript
// --- P4b new endpoints ---

export interface ActivityDayDto {
  date: string
  count: number
}

export interface UserStatsDto {
  streak_days: number
  coverage: number
  total_sessions: number
  last_active_date: string | null
  activity_daily: ActivityDayDto[]
}

export function getUserStats(): Promise<UserStatsDto> {
  return getJSON<UserStatsDto>('/api/users/me/stats')
}

export async function reorderMilestones(
  planId: string,
  milestoneIds: string[],
): Promise<PlanCurrentDto> {
  const resp = await fetch(`/api/plans/${planId}/milestones/reorder`, {
    method: 'PATCH',
    headers: {
      'Content-Type': 'application/json',
      'x-fingerprint': getFingerprint(),
    },
    body: JSON.stringify({ milestone_ids: milestoneIds }),
  })
  if (!resp.ok) throw new Error(`reorder failed: ${resp.status}`)
  return resp.json() as Promise<PlanCurrentDto>
}

export async function markMistakeUnderstood(mistakeId: string): Promise<{
  mastery_score: number
  next_due_at: string | null
}> {
  const resp = await fetch(`/api/mistakes/${mistakeId}/mark-understood`, {
    method: 'POST',
    headers: { 'x-fingerprint': getFingerprint() },
  })
  if (!resp.ok) throw new Error(`mark-understood failed: ${resp.status}`)
  return resp.json()
}
```

Update the `getJSON` import: `getJSON` is already exported from this file.

**Commit:**

```bash
git add frontend/src/stores/settings.ts frontend/src/lib/api.ts
git commit -m "feat: add debugMode/language to settings store and P4b API functions"
```

---

## Cut 1: Agent Visibility (Debug Mode)

### Task 1.1: Backend — SSE trace events

**Files:**
- Modify: `backend/app/agent/graph.py`
- Modify: `backend/app/api/routes.py`

In `graph.py`, after each node completes, add trace emission. The simplest approach: inject a `trace` event at the router node and judge node.

In `router_node` (after intent classification):

```python
# After intent is determined, yield trace event if stream_writer available
if stream_writer := _safe_writer(config):
    stream_writer({"type": "trace", "step": "router", "intent": intent,
                   "active_quiz": state.get("active_quiz_question_id"),
                   "active_plan": state.get("active_plan_id")})
```

In `judge_node` (after score computed):

```python
if stream_writer := _safe_writer(config):
    stream_writer({"type": "trace", "step": "judge", "score": judge_score,
                   "weak_dims": weak_dims, "retry": retry_count})
```

In `tutor_node` (after generation, before judge):

```python
# After getting the LLM response
if stream_writer := _safe_writer(config):
    stream_writer({"type": "trace", "step": "tutor", "citations_count": len(citations)})
```

In `routes.py`, no change needed — `graph.astream(stream_mode="custom")` already forwards all `stream_writer` events. The frontend SSE parser just needs to add a case for `type: "trace"`.

### Task 1.2: Frontend — TracePanel component

**Files:**
- Create: `frontend/src/components/TracePanel.vue`
- Modify: `frontend/src/stores/chat.ts`
- Modify: `frontend/src/views/Chat.vue`
- Modify: `frontend/src/lib/api.ts` — add `onTrace` callback

In `chat.ts`, add trace state:

```typescript
interface TraceStep {
  step: string
  [key: string]: any
}

// In ChatState:
trace: TraceStep[]
```

In `chat.ts` `streamChat` call, add `onTrace: (step: TraceStep) => { this.trace.push(step) }`.

In `api.ts` `ChatStreamCallbacks`, add:
```typescript
onTrace?: (step: any) => void
```

In `api.ts` SSE parsing loop, add after `else if (event.type === 'done')`:
```typescript
else if (event.type === 'trace') cb.onTrace?.(event)
```

Create `TracePanel.vue`:

```vue
<script setup lang="ts">
import { useChatStore } from '../stores/chat'

const chat = useChatStore()
</script>

<template>
  <div v-if="chat.trace.length" class="rounded-lg border border-border bg-surface p-4 text-xs font-mono">
    <div class="flex items-center gap-2 mb-2 text-fg-muted">
      <span>Agent Trace</span>
    </div>
    <div v-for="(step, i) in chat.trace" :key="i" class="flex gap-3 py-1 border-b border-border/50 last:border-0">
      <span class="text-primary w-16 shrink-0">{{ step.step }}</span>
      <span class="text-fg">
        <template v-if="step.step === 'router'">
          intent={{ step.intent }}
          <span v-if="step.active_quiz">quiz={{ step.active_quiz }}</span>
          <span v-if="step.active_plan">plan active</span>
        </template>
        <template v-else-if="step.step === 'judge'">
          score={{ step.score?.toFixed(2) }}
          <span v-if="step.weak_dims?.length" class="text-warning">weak: {{ step.weak_dims.join(', ') }}</span>
          <span v-if="step.retry" class="text-fg-dim">retry#{{ step.retry }}</span>
        </template>
        <template v-else-if="step.step === 'tutor'">
          {{ step.citations_count }} citations
        </template>
      </span>
    </div>
  </div>
</template>
```

In `Chat.vue`, add below the chat input:

```html
<TracePanel v-if="settings.debugMode" />
```

```typescript
import { useSettings } from '../stores/settings'
const settings = useSettings()
```

**Commit:**

```bash
git add backend/app/agent/graph.py frontend/src/components/TracePanel.vue frontend/src/stores/chat.ts frontend/src/lib/api.ts frontend/src/views/Chat.vue
git commit -m "feat: add agent trace SSE events and Debug Mode TracePanel"
```

---

## Cut 2: Goal Setup Wizard

### Task 2.1: Backend — POST /api/goals endpoint already exists? 

Check: No explicit `POST /api/goals` route exists. The Planner already creates goals via `GoalRepository.create()`. Add the route.

**Files:**
- Modify: `backend/app/api/routes.py`

Add after existing routes:

```python
class GoalCreateIn(BaseModel):
    title: str
    exam_date: str | None = None


class GoalCreateOut(BaseModel):
    goal_id: str
    title: str


@router.post("/goals", response_model=GoalCreateOut)
def create_goal(
    body: GoalCreateIn,
    user_id: Annotated[str, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_session)],
):
    from datetime import datetime
    exam = datetime.fromisoformat(body.exam_date) if body.exam_date else None
    goal = GoalRepository(session).create(user_id=user_id, title=body.title, exam_date=exam)
    return GoalCreateOut(goal_id=goal.id, title=goal.title)
```

Add test in `backend/tests/api/test_auth_routes.py` (or new file `test_routes_p4b.py`):

```python
def test_create_goal(client: TestClient):
    token = issue_token("goal-user", "member")
    resp = client.post(
        "/api/goals",
        json={"title": "HKBU CS Final", "exam_date": "2026-06-15"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["goal_id"]
    assert data["title"] == "HKBU CS Final"
```

### Task 2.2: Frontend — Onboarding view + 3 step components

**Files:**
- Create: `frontend/src/views/Onboarding.vue`
- Create: `frontend/src/components/onboarding/StepName.vue`
- Create: `frontend/src/components/onboarding/StepDate.vue`
- Create: `frontend/src/components/onboarding/StepUpload.vue`
- Modify: `frontend/src/router.ts`

Add route:
```typescript
{ path: '/onboarding', name: 'onboarding', component: () => import('./views/Onboarding.vue') },
```

**Onboarding.vue** (wizard container with step indicator):

```vue
<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import StepName from '../components/onboarding/StepName.vue'
import StepDate from '../components/onboarding/StepDate.vue'
import StepUpload from '../components/onboarding/StepUpload.vue'

const router = useRouter()
const step = ref(1)
const goalTitle = ref('')
const examDate = ref('')

function onNameDone(title: string) { goalTitle.value = title; step.value = 2 }
function onDateDone(date: string) { examDate.value = date; step.value = 3 }
async function onUploadDone() {
  // Create goal via API
  const resp = await fetch('/api/goals', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title: goalTitle.value, exam_date: examDate.value || null }),
  })
  const { goal_id } = await resp.json()
  // Redirect to chat with pre-filled plan generation message
  router.push({ path: '/chat', query: { goal_id, auto: `帮我做学习计划 on ${goalTitle.value}` } })
}
</script>

<template>
  <div class="h-full flex items-center justify-center bg-bg">
    <div class="w-full max-w-lg">
      <div class="flex gap-2 mb-8 justify-center">
        <div v-for="s in 3" :key="s"
             class="w-8 h-8 rounded-full flex items-center justify-center text-xs font-mono"
             :class="s <= step ? 'bg-primary text-white' : 'bg-surface-2 text-fg-muted'">
          {{ s }}
        </div>
      </div>
      <StepName v-if="step === 1" @done="onNameDone" />
      <StepDate v-if="step === 2" @done="onDateDone" @skip="step = 3" />
      <StepUpload v-if="step === 3" @done="onUploadDone" @skip="onUploadDone" />
    </div>
  </div>
</template>
```

**StepName.vue:**

```vue
<script setup lang="ts">
import { ref } from 'vue'
const emit = defineEmits<{ done: [title: string] }>()
const title = ref('')
</script>
<template>
  <div class="space-y-4">
    <h2 class="text-xl font-semibold">What's your exam or goal?</h2>
    <input v-model="title" placeholder="e.g. HKBU CS Final"
           class="w-full rounded-md bg-surface-2 border border-border px-4 py-3 text-fg placeholder-fg-dim focus:border-primary-ring focus:outline-none" />
    <button @click="emit('done', title)" :disabled="!title.trim()"
            class="rounded-md bg-primary px-6 py-2 text-sm font-medium text-white hover:bg-primary-2 disabled:opacity-40">Next</button>
  </div>
</template>
```

**StepDate.vue** and **StepUpload.vue** follow same pattern. StepUpload reuses `useDocumentsStore().upload()` pattern.

**Commit:**

```bash
git add backend/app/api/routes.py backend/tests/api/test_routes_p4b.py frontend/src/views/Onboarding.vue frontend/src/components/onboarding/ frontend/src/router.ts
git commit -m "feat: add Goal Setup wizard (3-step onboarding)"
```

---

## Cut 3: i18n

### Task 3.1: Install vue-i18n and create locale files

**Files:**
- Create: `frontend/src/locales/en.json`
- Create: `frontend/src/locales/zh-CN.json`
- Create: `frontend/src/i18n.ts`
- Modify: `frontend/src/main.ts`

```bash
cd frontend && pnpm add vue-i18n
```

**en.json** — key navigation + UI strings:

```json
{
  "nav": { "overview": "Overview", "chat": "Chat", "plan": "Plan", "quiz": "Quiz", "mistakes": "Mistakes", "library": "Library", "settings": "Settings" },
  "chat": { "placeholder": "Ask a question...", "send": "Send" },
  "quiz": { "submit": "Submit Answer", "generate": "Generate Quiz" },
  "plan": { "dragHint": "Drag to reorder milestones" },
  "onboarding": { "step1Title": "What's your exam or goal?", "step2Title": "Exam date?", "step3Title": "Upload study materials", "skip": "Skip", "next": "Next", "start": "Start Learning" },
  "settings": { "debugMode": "Debug Mode", "language": "Language" }
}
```

**zh-CN.json** — Chinese translations for all above keys.

**i18n.ts:**

```typescript
import { createI18n } from 'vue-i18n'
import en from './locales/en.json'
import zhCN from './locales/zh-CN.json'

const saved = localStorage.getItem('study-coach:settings')
const locale = saved ? (JSON.parse(saved).language || 'en') : 'en'

export const i18n = createI18n({
  legacy: false,
  locale,
  fallbackLocale: 'en',
  messages: { en, 'zh-CN': zhCN },
})
```

**main.ts** — add `app.use(i18n)` after `app.use(pinia)`.

### Task 3.2: Batch-extract existing strings

Replace all hardcoded English strings in templates with `{{ $t('key') }}`. Cover:
- `App.vue` — nav labels
- `Chat.vue` — placeholder, send button
- `PlanTimeline.vue` — headings
- `QuizAdaptive.vue` — submit, generate
- `MistakeBank.vue` — headings
- `Library.vue` — upload text
- `Settings.vue` — field labels + add language dropdown + debug mode checkbox

Add language dropdown in Settings.vue:

```html
<div class="space-y-1">
  <label class="text-xs text-fg-muted">Language</label>
  <select v-model="settings.language" @change="settings.persist()"
          class="w-full rounded-md bg-surface-2 border border-border px-3 py-2 text-sm text-fg">
    <option value="en">English</option>
    <option value="zh-CN">中文</option>
  </select>
</div>
```

Add debug mode checkbox in Settings.vue:

```html
<label class="flex items-center gap-2 text-sm text-fg cursor-pointer">
  <input type="checkbox" v-model="settings.debugMode" @change="settings.persist()"
         class="rounded border-border bg-surface-2 text-primary focus:ring-primary-ring" />
  {{ $t('settings.debugMode') }}
</label>
```

**Commit:**

```bash
cd frontend && pnpm add vue-i18n
git add frontend/package.json frontend/pnpm-lock.yaml
git commit -m "chore: add vue-i18n dep"

git add frontend/src/locales/ frontend/src/i18n.ts frontend/src/main.ts frontend/src/views/ frontend/src/components/ frontend/src/App.vue
git commit -m "feat: add i18n with en/zh-CN and batch-extract UI strings"
```

---

## Cut 4: Streak & Coverage

### Task 4.1: Backend — SessionsRepository + stats endpoint

**Files:**
- Modify: `backend/app/db/repositories.py`
- Modify: `backend/app/api/routes.py`

Add `SessionsRepository.count_active_days`:

```python
class ChatSessionRepository:
    # ... existing methods ...

    def count_active_days(self, user_id: str, *, since_days: int = 30) -> int:
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(days=since_days)
        stmt = (
            select(ChatSession.started_at)
            .where(ChatSession.user_id == user_id, ChatSession.started_at >= cutoff)
            .order_by(ChatSession.started_at.desc())
        )
        dates = set()
        for row in self.session.execute(stmt).scalars():
            dates.add(row.date())
        # Count consecutive days ending today
        today = datetime.utcnow().date()
        streak = 0
        for d in sorted(dates, reverse=True):
            if d == today:
                streak += 1
                today = today - timedelta(days=1)  # advance the check backwards
            elif d == today - timedelta(days=1):
                # Next date is off by one — count it and jump check point
                streak += 1
                today = d
            else:
                break
        return streak

    def activity_daily(self, user_id: str, *, days: int = 30) -> list[dict]:
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(days=days)
        stmt = (
            select(ChatSession.started_at)
            .where(ChatSession.user_id == user_id, ChatSession.started_at >= cutoff)
        )
        counts: dict[str, int] = {}
        for row in self.session.execute(stmt).scalars():
            d = row.date().isoformat()
            counts[d] = counts.get(d, 0) + 1
        result = []
        for i in range(days):
            d = (datetime.utcnow() - timedelta(days=days - 1 - i)).date().isoformat()
            result.append({"date": d, "count": counts.get(d, 0)})
        return result
```

Add to `routes.py`:

```python
class UserStatsOut(BaseModel):
    streak_days: int
    coverage: float
    total_sessions: int
    last_active_date: str | None
    activity_daily: list[dict]


@router.get("/users/me/stats", response_model=UserStatsOut)
def get_user_stats(
    user_id: Annotated[str, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_session)],
):
    sessions_repo = ChatSessionRepository(session)
    streak = sessions_repo.count_active_days(user_id)
    activity = sessions_repo.activity_daily(user_id)

    # Coverage: grounded topics / total chunks
    docs = DocumentRepository(session).list_for_user(user_id)
    total_chunks = sum(d.chunks_count for d in docs)
    goals = GoalRepository(session).list_active_for_user(user_id)
    grounded_count = 0
    if goals:
        topics = session.execute(
            select(Topic).where(Topic.goal_id == goals[0].id)
        ).scalars().all()
        grounded_count = sum(1 for t in topics if t.source_chunks)
    coverage = grounded_count / max(total_chunks, 1)

    # Last active
    all_sessions = ChatSessionRepository(session)
    last = None
    # Use raw query for last active
    stmt = select(ChatSession.started_at).where(
        ChatSession.user_id == user_id
    ).order_by(ChatSession.started_at.desc()).limit(1)
    last_row = session.execute(stmt).scalar_one_or_none()
    last_active = last_row.isoformat() if last_row else None

    return UserStatsOut(
        streak_days=streak,
        coverage=round(coverage, 3),
        total_sessions=0,  # placeholder; ChatSession not created per-turn yet
        last_active_date=last_active,
        activity_daily=activity,
    )
```

Add test file `backend/tests/api/test_routes_p4b.py` with streak test.

### Task 4.2: Extend GET /api/mastery with streak + coverage

In `routes.py` `get_mastery()`, add to response:

```python
# After computing MasteryOut, add streak/coverage
stats = sessions_repo.count_active_days(user_id)
# ... compute coverage as above ...
return MasteryOut(
    scores=scores,
    weak_topics=weak_names,
    overdue_milestones_count=overdue,
    streak_days=streak,
    coverage=coverage,
)
```

Update `MasteryOut` model to include `streak_days: int = 0` and `coverage: float = 0.0`.

**Commit:**

```bash
git add backend/app/db/repositories.py backend/app/api/routes.py backend/tests/api/test_routes_p4b.py
git commit -m "feat: add streak/coverage computation and GET /api/users/me/stats"
```

---

## Cut 5: Activity Heatmap

### Task 5.1: Frontend — HeatmapCard component

**Files:**
- Create: `frontend/src/components/HeatmapCard.vue`
- Modify: `frontend/src/views/Overview.vue`

```vue
<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { getUserStats, type ActivityDayDto } from '../lib/api'

const days = ref<ActivityDayDto[]>([])

onMounted(async () => {
  try { const stats = await getUserStats(); days.value = stats.activity_daily }
  catch { /* no stats yet */ }
})

function color(count: number): string {
  if (count === 0) return '#11162a'
  if (count <= 2) return '#4338ca'
  if (count <= 5) return '#6366f1'
  return '#818cf8'
}
</script>

<template>
  <div class="rounded-lg border border-border bg-surface p-4">
    <h3 class="text-sm font-medium text-fg-muted mb-3">Activity (30 days)</h3>
    <div class="flex gap-1 flex-wrap">
      <div v-for="d in days" :key="d.date"
           class="w-3 h-3 rounded-sm"
           :style="{ background: color(d.count) }"
           :title="`${d.date}: ${d.count}`" />
    </div>
  </div>
</template>
```

In `Overview.vue`, add `<HeatmapCard />` in the widget grid (after radar, before MasteryCard).

**Commit:**

```bash
git add frontend/src/components/HeatmapCard.vue frontend/src/views/Overview.vue
git commit -m "feat: add 30-day activity heatmap card to Overview"
```

---

## Cut 6: Gantt Timeline

### Task 6.1: Frontend — PlanTimeline vertical timeline component

**Files:**
- Create: `frontend/src/components/PlanGantt.vue`
- Modify: `frontend/src/views/PlanTimeline.vue`

```vue
<script setup lang="ts">
import type { MilestoneDto } from '../lib/api'

defineProps<{ milestones: MilestoneDto[] }>()

function dotColor(m: MilestoneDto): string {
  if (m.done) return 'bg-success'
  if (m.due_at && new Date(m.due_at) <= new Date()) return 'bg-warning'
  return 'bg-border-strong'
}
</script>

<template>
  <div class="relative pl-6 border-l-2 border-border">
    <div v-for="m in milestones" :key="m.id ?? m.title" class="relative pb-6 last:pb-0">
      <div class="absolute -left-[11px] top-1 w-5 h-5 rounded-full border-2 border-surface"
           :class="dotColor(m)" />
      <div class="rounded-md bg-surface-2 border border-border p-3">
        <div class="flex items-center gap-2">
          <span v-if="m.done" class="text-success text-xs">DONE</span>
          <span class="text-sm font-medium">{{ m.title }}</span>
        </div>
        <div v-if="m.due_at" class="text-xs text-fg-muted mt-1">{{ m.due_at }}</div>
      </div>
    </div>
  </div>
</template>
```

In `PlanTimeline.vue`, add `<PlanGantt :milestones="plan.milestones" />` alongside or below `<MilestoneList>`.

**Commit:**

```bash
git add frontend/src/components/PlanGantt.vue frontend/src/views/PlanTimeline.vue
git commit -m "feat: add vertical Gantt timeline to Plan page"
```

---

## Cut 7: Drag-reorder Milestones

### Task 7.1: Backend — PATCH /api/plans/{id}/milestones/reorder

**Files:**
- Modify: `backend/app/api/routes.py`
- Modify: `backend/app/db/repositories.py`

In `repositories.py`, add to `PlanRepository`:

```python
    def reorder_milestones(self, plan_id: str, milestone_ids: list[str]) -> list[PlanMilestone]:
        for idx, mid in enumerate(milestone_ids):
            row = self.get_milestone(plan_id=plan_id, milestone_id=mid)
            if row is not None:
                row.sort_order = idx
                row.updated_at = datetime.utcnow()
        plan = self.session.get(Plan, plan_id)
        if plan:
            self._sync_milestones_json(plan)
        self.session.commit()
        return self.list_milestones(plan_id)
```

In `routes.py`:

```python
class ReorderIn(BaseModel):
    milestone_ids: list[str]


@router.patch("/plans/{plan_id}/milestones/reorder", response_model=PlanCurrentOut)
def reorder_plan_milestones(
    plan_id: str,
    body: ReorderIn,
    user_id: Annotated[str, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_session)],
):
    goal, plan = _plan_belongs_to_user(session, user_id=user_id, plan_id=plan_id)
    PlanRepository(session).reorder_milestones(plan.id, body.milestone_ids)
    refreshed = PlanRepository(session).get_by_goal(goal.id)
    return _plan_current_out(session, user_id=user_id, goal=goal, plan=refreshed)
```

### Task 7.2: Frontend — Drag handlers on MilestoneRow

**Files:**
- Modify: `frontend/src/components/MilestoneList.vue`

Add `draggable="true"` and event handlers to each milestone row:

```html
<div v-for="m in milestones" :key="m.id"
     draggable="true"
     @dragstart="onDragStart($event, m)"
     @dragover.prevent="onDragOver($event, m)"
     @drop="onDrop($event, m)"
     @dragend="onDragEnd"
     class="...">
```

```typescript
import { ref } from 'vue'
import { reorderMilestones, type MilestoneDto } from '../lib/api'

const dragged = ref<MilestoneDto | null>(null)

function onDragStart(e: DragEvent, m: MilestoneDto) { dragged.value = m }
function onDragOver(_e: DragEvent, _m: MilestoneDto) { /* visual feedback */ }
async function onDrop(_e: DragEvent, target: MilestoneDto) {
  if (!dragged.value || dragged.value.id === target.id) return
  const ids = milestones.value.map(m => m.id!)
  const from = ids.indexOf(dragged.value.id!)
  const to = ids.indexOf(target.id!)
  ids.splice(from, 1)
  ids.splice(to, 0, dragged.value.id!)
  // Optimistic update
  milestones.value = ids.map(id => milestones.value.find(m => m.id === id)!)
  try {
    await reorderMilestones(props.planId, ids)
  } catch {
    // Revert on failure — re-fetch plan
    emit('refresh')
  }
}
function onDragEnd() { dragged.value = null }
```

**Commit:**

```bash
git add backend/app/db/repositories.py backend/app/api/routes.py frontend/src/components/MilestoneList.vue
git commit -m "feat: add drag-reorder milestones with optimistic UI"
```

---

## Cut 8: Mistake "Mark as Understood"

### Task 8.1: Backend — POST /api/mistakes/{id}/mark-understood

**Files:**
- Modify: `backend/app/api/routes.py`

```python
class MarkUnderstoodOut(BaseModel):
    mastery_score: float
    next_due_at: str | None


@router.post("/mistakes/{mistake_id}/mark-understood", response_model=MarkUnderstoodOut)
def mark_mistake_understood(
    mistake_id: str,
    user_id: Annotated[str, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_session)],
):
    from app.db.repositories import MasteryRepository, MistakeRepository, QuestionRepository
    from app.srs.sm2 import next_schedule

    mistake = MistakeRepository(session).get_by_id(mistake_id)
    if mistake is None or mistake.user_id != user_id:
        raise HTTPException(status_code=404, detail="mistake not found")

    question = QuestionRepository(session).get_by_id(mistake.question_id)
    if question is None:
        raise HTTPException(status_code=404, detail="question not found")

    # SM-2 quality=5 = perfect recall from long-term memory
    sched = next_schedule(
        quality=5,
        previous_interval_days=mistake.srs_interval_days,
        previous_ease=mistake.srs_ease,
    )
    MistakeRepository(session).update_srs(
        mistake_id=mistake.id,
        interval_days=sched.interval_days,
        ease=sched.ease,
        due_at=sched.due_at,
    )
    new_score = MasteryRepository(session).apply_delta(
        user_id=user_id, topic_id=question.topic_id, delta=0.1,
    )
    return MarkUnderstoodOut(
        mastery_score=new_score,
        next_due_at=sched.due_at.isoformat(),
    )
```

### Task 8.2: Frontend — Mark understood button on MistakeRow

**Files:**
- Modify: `frontend/src/components/MistakeRow.vue`

Add a three-dot menu or button:

```html
<button @click="markUnderstood(mistake.mistake_id)"
        class="text-xs text-fg-muted hover:text-success transition-colors">
  Mark understood
</button>
```

```typescript
import { markMistakeUnderstood } from '../lib/api'
import { useMistakesStore } from '../stores/mistakes'
import { useMasteryStore } from '../stores/mastery'

async function markUnderstood(id: string) {
  await markMistakeUnderstood(id)
  useMistakesStore().fetch()
  useMasteryStore().fetch()
}
```

**Commit:**

```bash
git add backend/app/api/routes.py frontend/src/components/MistakeRow.vue
git commit -m "feat: add mark-as-understood for mistakes (SM-2 quality=5)"
```

---

## Cut 9: MCQ Format Hardening

### Task 9.1: Backend — Stricter generate_quiz prompt

**Files:**
- Modify: `backend/app/agent/tools/quiz.py`

In `generate_quiz()`, add to the prompt:

```python
prompt += (
    "\n\nCRITICAL FORMATTING: Each option MUST start with exactly "
    '"A) ", "B) ", "C) ", "D) " (letter, closing paren, space). '
    "Do NOT number with 1) 2) or use bullet points. "
    "Do NOT output bare option text without the prefix.\n"
    'Example correct format: "A) HyDEGenerator rewrites user queries into '
    'hypothetical answer documents before embedding"'
)
```

### Task 9.2: Frontend — Tighten parse.ts regex

**Files:**
- Modify: `frontend/src/lib/parse.ts`

In `parseQuizGenerate()`, add a stricter first-pass regex that expects `A)` prefix, with fallback:

```typescript
// Stricter: expect A) B) C) D) prefix
const PREFIX_RE = /([A-D])\)\s+(.+?)(?=\s*[A-D]\)\s+|\s*$)/gs
// Fallback: split on newlines for bare options
```

**Commit:**

```bash
git add backend/app/agent/tools/quiz.py frontend/src/lib/parse.ts
git commit -m "fix: harden MCQ format — stricter prompt constraints + A) prefix regex"
```

---

## P4b Verification

After all cuts complete:

```bash
# Backend tests
cd backend && uv run pytest tests/ -x -q
# Expected: ~250+ tests passing (233 + auth + P4b new)

# Frontend build
cd frontend && pnpm build
# Expected: no errors

# Visual smoke via chrome-devtools:
# - Debug Mode on → chat shows TracePanel
# - /onboarding → 3-step wizard renders
# - Settings → language switch toggles UI
# - /plan → Gantt timeline visible, drag-reorder works
# - /mistakes → "Mark understood" button
```
