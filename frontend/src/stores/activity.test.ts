import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('./settings', () => ({
  authHeaders: vi.fn(() => ({})),
  getAccessToken: vi.fn(() => Promise.resolve('test-token')),
  llmHeaders: vi.fn(() => ({})),
}))

import { invalidateLearningState } from '../lib/dataLifecycle'
import { useActivity } from './activity'

beforeEach(() => {
  setActivePinia(createPinia())
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('activity store', () => {
  it('loads activity days from user stats', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: vi.fn().mockResolvedValue({
        streak_days: 2,
        coverage: 0.5,
        total_sessions: 3,
        last_active_date: '2026-07-27',
        activity_daily: [{ date: '2026-07-27', count: 4 }],
      }),
    }))
    const activity = useActivity()

    await expect(activity.fetch()).resolves.toBe(true)

    expect(activity.days).toEqual([{ date: '2026-07-27', count: 4 }])
    expect(activity.loading).toBe(false)
    expect(activity.error).toBeNull()
  })

  it('clears owned state after learning data is cleared', () => {
    const activity = useActivity()
    activity.days = [{ date: '2026-07-26', count: 8 }]
    activity.loading = true
    activity.error = 'stale error'

    activity.resetAfterDataClear()

    expect(activity.days).toEqual([])
    expect(activity.loading).toBe(false)
    expect(activity.error).toBeNull()
  })

  it('does not let a pre-reset request restore stale activity', async () => {
    let resolveResponse!: (response: Response) => void
    vi.stubGlobal('fetch', vi.fn(() => new Promise<Response>((resolve) => {
      resolveResponse = resolve
    })))
    const activity = useActivity()

    const staleFetch = activity.fetch()
    expect(activity.loading).toBe(true)
    invalidateLearningState()
    activity.resetAfterDataClear()
    activity.days = [{ date: '2026-07-27', count: 1 }]
    activity.loading = true
    activity.error = 'new refresh pending'
    resolveResponse({
      ok: true,
      json: vi.fn().mockResolvedValue({
        streak_days: 9,
        coverage: 1,
        total_sessions: 99,
        last_active_date: '2026-07-26',
        activity_daily: [{ date: '2026-07-26', count: 99 }],
      }),
    } as unknown as Response)

    await expect(staleFetch).resolves.toBe(true)

    expect(activity.days).toEqual([{ date: '2026-07-27', count: 1 }])
    expect(activity.loading).toBe(true)
    expect(activity.error).toBe('new refresh pending')
  })

  it('reports refresh failures without overwriting activity', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 500 }))
    const activity = useActivity()
    activity.days = [{ date: '2026-07-26', count: 2 }]

    await expect(activity.fetch()).resolves.toBe(false)

    expect(activity.days).toEqual([{ date: '2026-07-26', count: 2 }])
    expect(activity.loading).toBe(false)
    expect(activity.error).toBe('/api/users/me/stats failed: 500')
  })
})
