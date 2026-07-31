import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { invalidateLearningState } from '../lib/dataLifecycle'

const reviewMistake = vi.fn()

vi.mock('../lib/api', () => ({
  reviewMistake,
}))

describe('quiz mistake review epoch guards', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    reviewMistake.mockReset()
  })

  it('ignores a deferred mistake-review response after learning-state invalidation', async () => {
    let finish!: (value: unknown) => void
    reviewMistake.mockReturnValueOnce(new Promise(resolve => { finish = resolve }))
    const { useQuiz } = await import('./quiz')
    const store = useQuiz()
    store.currentMistakeId = 'mistake-1'
    store.currentMCQ = { prompt: 'Q?', options: ['A) 1', 'B) 2', 'C) 3', 'D) 4'] }

    const pending = store.reviewCurrentMistake('A')
    invalidateLearningState()
    store.reset()
    finish({
      correct: true,
      correct_answer: 'A',
      explanation: 'ok',
      new_interval_days: 1,
      next_due_at: '2099-01-01T00:00:00Z',
    })
    await expect(pending).resolves.toBe(false)

    expect(store.currentMistakeId).toBeNull()
    expect(store.currentMCQ).toBeNull()
    expect(store.lastGrade).toBeNull()
  })
})
