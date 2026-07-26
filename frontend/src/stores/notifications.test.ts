import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useNotifications } from './notifications'

describe('notifications', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    setActivePinia(createPinia())
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('expires a toast after five seconds and supports manual dismissal', () => {
    const store = useNotifications()
    const first = store.push({ kind: 'success', message: 'Learning data cleared.' })
    const second = store.push({ kind: 'info', message: 'Settings saved.' })

    store.dismiss(second)

    expect(store.items.map(item => item.id)).toEqual([first])
    vi.advanceTimersByTime(4999)
    expect(store.items.map(item => item.id)).toEqual([first])
    vi.advanceTimersByTime(1)
    expect(store.items).toEqual([])
  })
})
