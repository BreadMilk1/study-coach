import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { MilestoneDto, PlanCurrentDto, ValidationHintDto } from '../lib/api'

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })
  return { promise, resolve, reject }
}

const mocks = vi.hoisted(() => ({
  patchMilestoneDone: vi.fn(),
  getCurrentPlan: vi.fn(),
  getPlanEvents: vi.fn(),
}))

vi.mock('../lib/api', () => ({
  getCurrentPlan: mocks.getCurrentPlan,
  getPlanEvents: mocks.getPlanEvents,
  patchMilestoneDone: mocks.patchMilestoneDone,
}))

const baseMilestone: MilestoneDto = {
  id: 'milestone-new',
  title: 'New milestone',
  due_at: null,
  done: false,
  completed_at: null,
  topic_id: null,
  topic: null,
  mastery_score: null,
  validation_recommended: false,
  sort_order: null,
  source: null,
}

const basePlan: PlanCurrentDto = {
  plan_id: 'plan-1',
  goal_id: 'goal-1',
  goal_title: 'Exam',
  milestones: [baseMilestone],
  updated_at: '2026-01-01T00:00:00Z',
}

const staleHint: ValidationHintDto = {
  show_quick_quiz: true,
  topic: 'old-topic',
  reason: 'old reason',
}

describe('plan validation hint freshness', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    mocks.patchMilestoneDone.mockReset()
    mocks.getCurrentPlan.mockReset()
    mocks.getPlanEvents.mockReset()
  })

  it('clears a stale validation hint when a new milestone update starts', async () => {
    const pendingPatch = deferred<{
      plan: PlanCurrentDto
      event: {
        id: string
        plan_id: string
        milestone_id: string
        actor: string
        action: string
        before_json: null
        after_json: null
        reason: null
        created_at: string
      }
      validation_hint: ValidationHintDto
    }>()
    mocks.patchMilestoneDone.mockReturnValueOnce(pendingPatch.promise)

    const { usePlan } = await import('./plan')
    const store = usePlan()
    store.plan = basePlan
    store.lastValidationHint = { ...staleHint }

    const togglePromise = store.toggleMilestone('milestone-new', true)

    try {
      expect(mocks.patchMilestoneDone).toHaveBeenCalledOnce()
      expect(store.updatingMilestoneId).toBe('milestone-new')
      expect(store.lastValidationHint).toBeNull()
    } finally {
      pendingPatch.resolve({
        plan: {
          ...basePlan,
          milestones: [{ ...baseMilestone, done: true }],
          updated_at: '2026-01-02T00:00:00Z',
        },
        event: {
          id: 'evt-1',
          plan_id: 'plan-1',
          milestone_id: 'milestone-new',
          actor: 'user',
          action: 'completed',
          before_json: null,
          after_json: null,
          reason: null,
          created_at: '2026-01-02T00:00:00Z',
        },
        validation_hint: {
          show_quick_quiz: true,
          topic: 'new-topic',
          reason: 'new reason',
        },
      })
      await togglePromise
    }
  })

  it('clears a stale validation hint when a plan refresh starts', async () => {
    const pendingPlan = deferred<PlanCurrentDto>()
    mocks.getCurrentPlan.mockReturnValueOnce(pendingPlan.promise)
    mocks.getPlanEvents.mockResolvedValueOnce([])

    const { usePlan } = await import('./plan')
    const store = usePlan()
    store.plan = basePlan
    store.lastValidationHint = { ...staleHint }

    const fetchPromise = store.fetch()

    try {
      expect(mocks.getCurrentPlan).toHaveBeenCalledOnce()
      expect(store.loading).toBe(true)
      expect(store.lastValidationHint).toBeNull()
    } finally {
      pendingPlan.resolve(basePlan)
      await fetchPromise
    }
  })
})
