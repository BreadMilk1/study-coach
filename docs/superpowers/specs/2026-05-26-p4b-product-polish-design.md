# P4b — Product Polish Design

> 2026-05-26 brainstorm. P4b = Agent visibility + Goal Setup wizard + i18n + streak/coverage + heatmap + Gantt + drag-reorder + mistake mark-understood + MCQ hardening.

## 1. Agent Visibility (Debug Mode)

### UI

Debug panel toggled via Settings → Debug Mode checkbox. When ON, a collapsible panel renders below the chat input in `/chat`:

```
┌──────────────────────────────────────────┐
│  🛠 Agent Trace                    [−]   │
│  ─────────────────────────────────────── │
│  Intent: quiz  →  Mode: deterministic    │
│  ┌──────────────────────────────────┐    │
│  │ Step 1: generate_quiz [1.2s]    │    │
│  │   in:  {topic:"HyDE", n:1}      │    │
│  │   out: QuizQuestionOut(...)      │    │
│  └──────────────────────────────────┘    │
│  Judge: 0.82 ✓  (weak: granularity)      │
│  Tokens: 342 in / 156 out               │
└──────────────────────────────────────────┘
```

### Data source

- **Agent loop path**: `agent_trace` already populated (P2.2/P2.3). Emit as SSE `type:"trace"` alongside existing `type:"token"` and `type:"done"`.
- **Deterministic path**: inject a simplified trace with intent + judge score + wall time. P2.2 known limitation ("deterministic path has no token cost data") closed here — add `{ type:"trace", judge_score, intent, wall_time_s }` after `done`.
- When Debug Mode OFF: backend does NOT emit `type:"trace"` events (saves SSE bandwidth). Frontend `chatStore.debugMode` defaults to `false`.

### Settings integration

- New `<fieldset>` in Settings: "Debug Mode" checkbox, persisted in `useSettingsStore.debugMode` (localStorage)
- Debug panel component: `TracePanel.vue`, mounted in `Chat.vue` below chat input

## 2. Goal Setup Wizard

### 3-step flow

```
Step 1: Name              Step 2: Date          Step 3: Upload
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ What's your   │    │ Exam date?   │    │ Upload study │
│ exam or goal? │ →  │ (optional)   │ →  │ materials    │   → /plan
│               │    │              │    │              │
│ [__________]  │    │ [date picker]│    │ [drop zone]  │
│  e.g. HKBU CS │    │              │    │              │
│     final     │    │    [Skip]    │    │   [Skip]     │
└──────────────┘    └──────────────┘    └──────────────┘
```

### Integration points (existing code coordination)

- Step 3 upload reuses `useDocumentsStore.upload()` from Library view (extract shared composable `useFileUpload` if needed)
- Completion: `POST /api/goals { title, exam_date? }` → get `goal_id` → auto-send `"帮我做学习计划 on {title}"` via Chat SSE → wait for plan generation → redirect to `/plan`
- If user already has active goal + plan: Overview shows "Continue your goal" CTA instead of wizard entry point. Guard: `useGoalStore.hasActiveGoal` computed.
- Route: `/onboarding` (new view), skipped if `hasActiveGoal`

### Components

- `views/Onboarding.vue` — wizard container with step indicator
- `components/onboarding/StepName.vue` — text input + examples
- `components/onboarding/StepDate.vue` — date picker (native `<input type="date">`) + skip
- `components/onboarding/StepUpload.vue` — file drop zone (shared logic with Library) + skip

## 3. i18n

### Implementation

- Dep: `vue-i18n` (new, Composition API mode)
- Locale files: `frontend/src/locales/en.json` + `zh-CN.json`
- Language switch: Settings dropdown, persisted to `localStorage`
- Template usage: `{{ $t('chat.placeholder') }}`
- Script usage: `const { t } = useI18n(); t('quiz.submit')`

### Extraction strategy

- P4b new components: write directly with `$t()` keys
- Existing components: one sub-agent batch sweep at end of P4b to convert all hardcoded strings
- Machine translate zh-CN → human verify key UI strings (nav labels, wizard steps, error messages)

## 4. Streak & Coverage

### API

```python
# GET /api/users/me/stats → UserStatsOut
class UserStatsOut(BaseModel):
    streak_days: int
    coverage: float           # 0.0..1.0
    total_sessions: int
    last_active_date: str | None
    activity_daily: list[ActivityDay]  # last 30 days

class ActivityDay(BaseModel):
    date: str
    count: int  # number of events (chat/quiz/plan)
```

### Computation

- **Streak**: `SessionsRepository.count_active_days(user_id, since=N days ago)` → count consecutive days ending today. Break at first gap.
- **Coverage**: `TopicRepository.count_grounded(user_id)` / `DocumentRepository.total_chunks(user_id)`. Grounded = topic has non-empty `source_chunks`. Floor total_chunks at 1 to avoid division by zero.

### Radar integration

- `GET /api/mastery` extended with `streak_days` and `coverage` fields
- Frontend `RadarChart` updated from 3 live + 2 placeholder → 5 live axes

## 5. Activity Heatmap (Overview)

- GitHub-style 7×? grid (7 days per row, up to 5 rows for 30 days)
- Color: indigo gradient (4 stops: `#11162a` empty → `#c7d2fe` light → `#6366f1` medium → `#818cf8` heavy)
- Pure Tailwind `div` implementation (no chart.js)
- Data from `GET /api/users/me/stats.activity_daily`
- Component: `HeatmapCard.vue`, placed in Overview grid below radar

## 6. Gantt Timeline (Plan page)

- Vertical timeline (roadmap style): vertical line + dot nodes for each milestone
- Dot colors: green (done), amber (due today), neutral gray (future)
- Click dot → expand milestone detail (same `MilestoneRow` data)
- Component: `PlanTimeline.vue`, placed alongside or replacing `MilestoneList.vue` (TBD at implementation based on which looks better side-by-side)
- Zero new deps — pure Tailwind

## 7. Drag-reorder Milestones

### Backend

```python
# PATCH /api/plans/{plan_id}/milestones/reorder
class ReorderIn(BaseModel):
    milestone_ids: list[str]  # new order

# → PlanRepository.reorder_milestones(plan_id, milestone_ids)
#   bulk UPDATE plan_milestones SET sort_order = index WHERE id = milestone_id
```

### Frontend

- Native HTML drag-and-drop: `draggable="true"`, `@dragstart`, `@dragover`, `@drop`
- Visual: `transition-transform` on drop, drag ghost via default browser behavior
- TOUCH: `@touchstart` / `@touchmove` / `@touchend` for mobile compatibility (P4c)
- After drop: `PATCH /api/plans/{id}/milestones/reorder` → refresh `usePlanStore`
- Component mod: add drag handlers to `MilestoneRow.vue`

## 8. Mistake "Mark as Understood"

### Backend

```python
# POST /api/mistakes/{id}/mark-understood
# → SM-2 quality=5, recalculates srs_interval/srs_ease/srs_due_at
# → MasteryRepository.apply_delta(topic_id, +0.1)
class MarkUnderstoodOut(BaseModel):
    mastery_score: float
    next_due_at: str | None  # None if interval becomes very long
```

### Frontend

- Context menu (three-dot button) on `MistakeRow` → "Mark as understood"
- On success: `useMistakesStore.fetch()` + `useMasteryStore.fetch()` refresh
- Confirmation: none (undo-able by re-quizzing or manual mastery adjustment in future)

## 9. MCQ Format Hardening

- `generate_quiz()` prompt: add `CRITICAL: Each option MUST start with "A) " / "B) " / "C) " / "D) " prefix.`
- `parse.ts` `parseQuizGenerate()`: tighten regex to expect `A)` prefix, keep tolerant fallback for bare options
- Backend `QuestionRepository.create()`: existing validation guard already rejects single-char options; add warning log on near-miss formats
- 2 new backend tests: formatted output → `parse()` roundtrip succeeds; LLM output with missing prefix → fallback still works

## 10. Cloud-adapt hooks

- `# cloud-adapt`: Debug panel latency breakdown may need CloudWatch/XRay integration for production
- `# cloud-adapt`: i18n locale detection from `Accept-Language` header (cloud BYOK users are more likely multilingual)
- `# cloud-adapt`: Heatmap activity_daily query may need materialized view at scale

## 11. Verification gates

- 4 new backend tests: streak computation, coverage edge cases, reorder endpoint, mark-understood
- 2 new backend tests: MCQ format roundtrip
- Frontend build passing + 5 new view/component smoke checks via chrome-devtools
- i18n: spot-check 3 views in zh-CN for key navigation labels
