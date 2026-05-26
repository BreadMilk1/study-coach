# Plan Milestone Progression — Design Spec

> Target: make Plan feel like a user-progressed study path, while keeping Mastery as quiz/mistake evidence. This is a P4-ready design; no implementation has started.

- **Date**: 2026-05-26
- **Baseline**: P3 shell shipped. Plan is generated from Chat intent, persisted to `plans.milestones_json`, and rendered read-only on `/plan`.
- **User finding**: Current Plan feels like "check-in refreshes the current plan snapshot", not "I progress through tasks one by one".
- **Decision**: Keep Chat-triggered Plan generation, add milestone-level identity and manual progression, and make AI check-in produce auditable adjustments.

---

## 1. Current Behavior

The existing product flow is sound and stays:

1. User asks Chat for a plan with explicit Plan intent, for example `帮我做学习计划 on HyDE`.
2. Backend routes to Planner, generates milestones, persists the current plan, and streams the response.
3. Plan page fetches `/api/plans/current` and shows the latest persisted milestones.
4. `Check-in progress` calls Chat with `进度怎么样了`; Planner reads current milestones, mastery, mistakes, and dates, then rewrites the plan snapshot.

Observed limitation:

- Milestones have no stable identity.
- The user cannot directly mark one milestone complete.
- AI changes are applied as a full replacement with no visible diff or reason.
- Completion, delay, insertion, and reordering are all flattened into "the plan changed".

---

## 2. Product Semantics

### 2.1 Definitions

`Mastery`
: System evidence that the user understands a topic. Current source of truth is Quiz/Mistake behavior: correct answers increase topic score, wrong answers create mistakes and decrease score.

`Plan`
: The intervention path generated from user goals, corpus context, weak topics, mistakes, and deadlines.

`Milestone completion`
: User declaration that a specific planned learning task is complete. It is useful progress evidence, but it is not the same as quiz-verified mastery.

### 2.2 Key Boundary

Manual milestone completion must not directly add or subtract mastery.

Reason:

- A completed reading/implementation/review task is self-reported progress.
- Mastery is evidence-based and should stay tied to quiz/mistake outcomes.
- Mixing them would make the Top mastery card and weak-topic logic less trustworthy.

Plan should still be strongly related to mastery:

- Milestones should link to topics.
- Plan generation should use weak mastery topics.
- Check-in should compare `done` milestones against topic mastery.
- A completed milestone with low mastery should trigger a validation prompt, not silently raise mastery.

---

## 3. Approaches Considered

### A. Extend `milestones_json`

Add `id`, `status`, and optional history fields to each JSON object.

Pros:

- Lowest migration cost.
- Existing Planner tool shape changes minimally.

Cons:

- Diffing and history become fragile JSON manipulation.
- Stable updates are hard once AI reorders or edits titles.
- JSON in-place mutation risks are easy to miss.

### B. Lightweight Normalization (Chosen)

Add first-class milestone rows and event/revision rows, while keeping `plans` as the parent aggregate.

Pros:

- Stable `milestone_id` enables safe manual completion and undo.
- AI changes can be represented as patches and events.
- Plan progress, history, and explanation become queryable.
- Keeps scope smaller than a full task manager.

Cons:

- Requires Alembic migration and repository/API updates.
- Planner tools need compatibility handling while migrating from `milestones_json`.

### C. Full Task Manager

Add status machine, drag reorder, subtasks, dependencies, activity logs, Gantt view, and batch operations.

Pros:

- Most complete product model.

Cons:

- Over-scopes Study Coach.
- Shifts the product from exam coach to task manager.
- Higher implementation and demo risk.

Decision: choose **B. Lightweight Normalization**.

---

## 4. Data Model

### 4.1 New Tables

`plan_milestones`

```text
id              string(36) primary key
plan_id         string(36) foreign key -> plans.id
topic_id        string(36) nullable foreign key -> topics.id
title           text
due_at          datetime nullable
done            bool default false
completed_at    datetime nullable
sort_order      integer
source          string(20)  # ai | user | migrated
created_at      datetime
updated_at      datetime
```

`plan_events`

```text
id              string(36) primary key
plan_id         string(36) foreign key -> plans.id
milestone_id    string(36) nullable foreign key -> plan_milestones.id
actor           string(20)  # user | ai | system
action          string(40)  # created | completed | reopened | postponed | reordered | retitled | suggested | applied | rejected
before_json     json nullable
after_json      json nullable
reason          text nullable
created_at      datetime
```

Optional later table, not required for the first implementation:

`plan_change_sets`

```text
id              string(36) primary key
plan_id         string(36) foreign key -> plans.id
status          string(20)  # pending | applied | rejected
summary         text
changes_json    json
created_at      datetime
applied_at      datetime nullable
```

### 4.2 Compatibility

`plans.milestones_json` remains during migration as a compatibility cache/source.

Implementation options:

- Migration creates `plan_milestones` rows from existing `plans.milestones_json`.
- New reads prefer `plan_milestones`; if none exist, fallback to JSON.
- New writes update normalized rows. Keeping JSON in sync is optional and should be avoided unless an existing eval path still requires it.

The implementation plan must explicitly decide whether to keep `milestones_json` as a read-only legacy field or a short-term denormalized cache.

---

## 5. API Design

### 5.1 Current Plan

`GET /api/plans/current`

Extend response:

```json
{
  "plan_id": "...",
  "goal_id": "...",
  "goal_title": "...",
  "milestones": [
    {
      "id": "...",
      "title": "Read HyDE source chunks and summarize the retrieval flow",
      "due_at": "2026-05-30T00:00:00",
      "done": false,
      "completed_at": null,
      "topic_id": "...",
      "topic": "HyDE",
      "mastery_score": 0.3,
      "sort_order": 1,
      "source": "ai"
    }
  ],
  "updated_at": "..."
}
```

### 5.2 Manual Progress

`PATCH /api/plans/{plan_id}/milestones/{milestone_id}`

Request:

```json
{
  "done": true
}
```

Response:

```json
{
  "plan": { "...": "same shape as /api/plans/current" },
  "event": {
    "action": "completed",
    "actor": "user",
    "reason": "User marked milestone complete"
  },
  "validation_hint": {
    "show_quick_quiz": true,
    "topic": "HyDE",
    "reason": "Topic mastery is still below the validation threshold"
  }
}
```

Rules:

- `done: true` sets `completed_at`.
- `done: false` clears `completed_at`.
- Each change writes `plan_events`.
- This endpoint never changes mastery.
- If related topic mastery is below threshold, backend may return `validation_hint`.

### 5.3 Events

`GET /api/plans/{plan_id}/events?limit=20`

Returns latest plan events for a compact "why did this change?" panel.

### 5.4 AI Check-in Suggestions

First implementation can keep the existing `Check-in progress` Chat flow, but should produce structured events.

Preferred target shape:

```json
{
  "summary": "Adjusted overdue HyDE work and added one quiz validation step.",
  "auto_applied": [
    {
      "action": "postponed",
      "milestone_id": "...",
      "reason": "Milestone was overdue and not done"
    }
  ],
  "requires_confirmation": [
    {
      "action": "created",
      "reason": "HyDE mastery is low after the reading milestone was completed",
      "after": { "title": "Take a quick quiz on HyDE", "topic": "HyDE" }
    }
  ]
}
```

---

## 6. AI Check-in Policy

Auto-apply low-risk changes:

- Mark complete only when the user explicitly says a specific item is done.
- Reopen only when the user explicitly says it is not done.
- Postpone overdue unfinished items.
- Minor date adjustment.

Require confirmation:

- Add new milestones.
- Delete or hide milestones. Deletion should be avoided in v1; use events instead.
- Major title rewrite.
- Reorder more than local date sorting.
- Any change that affects more than two milestones.

The first implementation may defer the confirmation UI and instead log AI reasons for applied changes. If so, it must not auto-apply high-risk changes.

---

## 7. Frontend UX

### 7.1 Plan Page

Milestone row:

- Checkbox / completion icon is clickable.
- Click once to complete.
- Click again to reopen.
- Row shows `completed_at` when complete.
- Topic chip shows related mastery score when available.
- If completed but mastery is below threshold, show a subtle "Validate with quiz" CTA.

Page additions:

- Compact progress summary: completed / total, overdue, low-mastery completed.
- "Recent changes" panel from `plan_events`.
- Existing `Check-in progress` button remains.

### 7.2 Quick Quiz Bridge

When validation is suggested:

- CTA links to `/quiz?topic=<topic_name>`.
- Quiz page pre-fills topic from query.
- It does not auto-generate without user action in v1.

This keeps completion fast and avoids surprising LLM calls.

### 7.3 Overview

Overview should distinguish:

- Plan progress: completion of milestones.
- Mastery progress: quiz/mistake evidence.
- Gap signal: completed milestones whose topics still have low mastery.

Do not merge plan progress and mastery into a single score.

---

## 8. Existing Risk Cleanup

These are related but can be separate implementation cuts:

1. Clamp `mastery.score` into `0..1` in `MasteryRepository.apply_delta`.
2. Align weak-topic threshold between Planner (`<0.4`) and `/api/mastery` (`<0.5`), or name them differently.
3. Move `overdue_milestones_count` out of `/api/mastery` eventually, because it is plan-derived, not mastery-derived.
4. Clarify whether `pending_mastery_delta` is still a future abstraction or dead code, because QuizMaster currently writes mastery directly.

These should not block milestone progression, but they should be documented in the implementation plan as follow-up or same-phase cleanup if touched.

---

## 9. Migration Risk, Recovery, Verification

### 9.1 Risks

- Existing local data has `plans.milestones_json`; migration must preserve it.
- Chroma/document data is independent and should not be touched.
- Planner deterministic and agent-loop variants currently expect JSON milestone lists.
- Eval scripts may still inspect `milestones_json`.

### 9.2 Recovery

- Before migration testing, copy `backend/study_coach.db` if local data matters.
- Alembic downgrade should drop normalized milestone/event tables only after preserving enough legacy JSON to keep old Plan reads possible.
- If migration fails locally, restore the copied SQLite file and restart backend.

### 9.3 Verification

Backend:

- Migration test: existing `milestones_json` becomes milestone rows with stable ids.
- Repository test: complete and reopen milestone writes events and updates timestamps.
- API test: `GET /api/plans/current` includes milestone ids and mastery hints.
- API test: `PATCH /api/plans/{id}/milestones/{id}` updates only that milestone.
- Planner test: check-in uses current normalized milestone state.
- Regression test: quiz mastery update still works and is not triggered by milestone completion.

Frontend:

- Pinia plan store can fetch, toggle, and replace current plan.
- Milestone row shows done/reopened states.
- Validate-with-quiz CTA appears only for low-mastery completed topic.
- `pnpm build` passes.

Manual:

- Clean incognito user: upload PDF -> Chat generates Plan -> Plan page shows clickable milestones.
- Complete one milestone -> Overview progress updates -> Mastery unchanged.
- Reopen it -> progress decreases -> event history shows reopen.
- Complete low-mastery topic -> quick quiz CTA appears -> Quiz can generate validation question.

---

## 10. Out of Scope

- Drag reorder.
- Gantt timeline.
- Subtasks.
- Dependencies between milestones.
- Multi-user shared plan editing.
- Auto-changing mastery from Plan completion.
- Auto-starting Quiz after completion.

---

## 11. References

- Existing P2.1 Plan design: `docs/superpowers/specs/2026-05-22-p2-1-5-plan-chain-design.md`
- Existing P3 frontend design: `docs/superpowers/specs/2026-05-25-p3-frontend-productize-design.md`
- SQLAlchemy JSON mutation behavior: https://docs.sqlalchemy.org/en/20/core/type_basics.html#sqlalchemy.types.JSON
- Alembic autogenerate / migration review: https://alembic.sqlalchemy.org/en/latest/autogenerate.html
- FastAPI request body and response models: https://fastapi.tiangolo.com/tutorial/body/ and https://fastapi.tiangolo.com/tutorial/response-model/
- Pinia actions: https://pinia.vuejs.org/core-concepts/actions.html
