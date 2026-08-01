import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { invalidateLearningState } from '../lib/dataLifecycle'

const patchMilestoneDone = vi.fn()

vi.mock('../lib/api', () => ({
  getCurrentPlan: vi.fn(),
  getPlanEvents: vi.fn(),
  patchMilestoneDone,
}))

describe('plan mutation epoch guards', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    patchMilestoneDone.mockReset()
  })

  it('ignores a deferred milestone response after learning-state invalidation', async () => {
    let finish!: (value: unknown) => void
    patchMilestoneDone.mockReturnValueOnce(new Promise(resolve => { finish = resolve }))
    const { usePlan } = await import('./plan')
    const store = usePlan()
    store.plan = {
      plan_id: 'plan-1',
      goal_id: 'goal-1',
      goal_title: 'Exam',
      milestones: [],
      updated_at: '2026-01-01T00:00:00Z',
    }

    const pending = store.toggleMilestone('m1', true)
    invalidateLearningState()
    store.$reset()
    finish({
      plan: {
        plan_id: 'plan-1',
        goal_id: 'goal-1',
        goal_title: 'Exam',
        milestones: [{ id: 'm1', title: 'Done', done: true }],
        updated_at: '2026-01-02T00:00:00Z',
      },
      event: {
        id: 'evt-1',
        plan_id: 'plan-1',
        milestone_id: 'm1',
        actor: 'user',
        action: 'completed',
        before_json: null,
        after_json: null,
        reason: null,
        created_at: '2026-01-02T00:00:00Z',
      },
      validation_hint: { show_quick_quiz: false, topic: null, reason: null },
    })
    await pending

    expect(store.plan).toBeNull()
    expect(store.events).toEqual([])
    expect(store.lastValidationHint).toBeNull()
    expect(store.updatingMilestoneId).toBeNull()
  })
})
