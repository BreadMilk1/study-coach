import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { memoryStorage } from '../test/memoryStorage'
import { useDataLifecycle, type LifecycleDependencies } from './dataLifecycle'

function summary(overrides: Partial<Awaited<ReturnType<LifecycleDependencies['summary']>>> = {}) {
  return {
    reset_enabled: true,
    has_learning_data: true,
    users: 1,
    documents: 1,
    source_chunks: 3,
    vectors: 3,
    chat_sessions: 0,
    messages: 0,
    citations: 0,
    goals: 0,
    topics: 0,
    plans: 0,
    plan_milestones: 0,
    plan_events: 0,
    questions: 0,
    mastery: 0,
    mistakes: 0,
    ...overrides,
  }
}

function dependencies(overrides: Partial<LifecycleDependencies> = {}): LifecycleDependencies {
  return {
    summary: async () => summary(),
    reset: async scope => ({ scope, status: 'completed', deleted: summary() }),
    resetClient: async () => undefined,
    markChoice: () => undefined,
    clearChoice: () => undefined,
    clearFactory: () => undefined,
    broadcast: () => undefined,
    reload: () => undefined,
    pause: async () => undefined,
    ...overrides,
  }
}

beforeEach(() => {
  setActivePinia(createPinia())
  vi.stubGlobal('sessionStorage', memoryStorage())
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('startup inspection', () => {
  it('requires an explicit choice when reset is enabled and learning data exists', async () => {
    const store = useDataLifecycle()
    store.initialize(dependencies())

    await store.inspect()

    expect(store.phase).toBe('choice_required')
    expect(store.canStartFresh).toBe(true)
  })

  it('keeps inspection failure blocking until explicitly continued without clearing', async () => {
    const calls: string[] = []
    const store = useDataLifecycle()
    store.initialize(dependencies({
      summary: async () => { throw new Error('offline') },
      markChoice: () => calls.push('choice'),
    }))

    await store.inspect()

    expect(store.phase).toBe('inspection_error')
    expect(store.canContinueWithoutClearing).toBe(true)
    expect(store.canStartFresh).toBe(false)
    store.continueWithoutClearing()
    expect(calls).toEqual(['choice'])
    expect(store.phase).toBe('ready')
  })

  it.each([
    { reset_enabled: false, has_learning_data: true },
    { reset_enabled: true, has_learning_data: false },
  ])('is ready without a gate for $reset_enabled/$has_learning_data', async overrides => {
    const store = useDataLifecycle()
    store.initialize(dependencies({ summary: async () => summary(overrides) }))

    await store.inspect()

    expect(store.phase).toBe('ready')
    expect(store.canStartFresh).toBe(false)
  })

  it('records an explicit decision to continue existing learning data', async () => {
    const calls: string[] = []
    const store = useDataLifecycle()
    store.initialize(dependencies({ markChoice: () => calls.push('choice') }))
    await store.inspect()

    store.continueExisting()

    expect(calls).toEqual(['choice'])
    expect(store.phase).toBe('ready')
  })

  it('does not let a stale inspection overwrite an external reset', async () => {
    let resolveSummary!: (value: ReturnType<typeof summary>) => void
    const pendingSummary = new Promise<ReturnType<typeof summary>>(resolve => {
      resolveSummary = resolve
    })
    const store = useDataLifecycle()
    store.initialize(dependencies({ summary: () => pendingSummary }))

    const inspection = store.inspect()
    await store.handleExternalReset('learning')
    resolveSummary(summary())
    await inspection

    expect(store.phase).toBe('external_reset')
    expect(store.summary).toBeNull()
  })
})

describe('reset confirmation', () => {
  it('returns learning confirmation to its originating ready or choice phase', async () => {
    const store = useDataLifecycle()
    store.initialize(dependencies())
    await store.inspect()

    store.requestLearningReset()
    expect(store.phase).toBe('confirming_learning')
    expect(store.pendingScope).toBe('learning')
    store.cancelReset()
    expect(store.phase).toBe('choice_required')
    expect(store.pendingScope).toBeNull()

    store.continueExisting()
    store.requestLearningReset()
    store.cancelReset()
    expect(store.phase).toBe('ready')
  })

  it('opens and cancels factory confirmation back to ready', () => {
    const store = useDataLifecycle()
    store.initialize(dependencies())
    store.phase = 'ready'

    store.requestFactoryReset()
    expect(store.phase).toBe('confirming_factory')
    expect(store.pendingScope).toBe('factory')
    store.cancelReset()
    expect(store.phase).toBe('ready')
    expect(store.pendingScope).toBeNull()
  })

  it('cannot cancel while reset work is in progress', () => {
    const store = useDataLifecycle()
    store.initialize(dependencies())
    store.phase = 'resetting'
    store.pendingScope = 'learning'

    store.cancelReset()

    expect(store.phase).toBe('resetting')
    expect(store.pendingScope).toBe('learning')
  })

  it('ignores lifecycle actions invoked from illegal source phases', async () => {
    const calls: string[] = []
    const store = useDataLifecycle()
    store.initialize(dependencies({
      reset: async scope => { calls.push(`reset:${scope}`); return { scope, status: 'completed', deleted: summary() } },
      resetClient: async () => { calls.push('client') },
      markChoice: () => calls.push('choice'),
      clearChoice: () => calls.push('clear-choice'),
      clearFactory: () => calls.push('clear-factory'),
      broadcast: scope => calls.push(`broadcast:${scope}`),
      reload: () => calls.push('reload'),
    }))
    store.phase = 'checking'

    store.continueExisting()
    store.continueWithoutClearing()
    store.requestLearningReset()
    store.requestFactoryReset()
    store.cancelReset()
    await store.confirmLearningReset()
    await store.confirmFactoryReset()
    await store.retryReset()
    await store.acknowledgeExternalReset()

    expect(calls).toEqual([])
    expect(store.phase).toBe('checking')
  })
})

describe('learning reset', () => {
  it('unlocks only after backend reset, client reset, choice, broadcast, and summary refresh', async () => {
    const calls: string[] = []
    const store = useDataLifecycle()
    store.initialize(dependencies({
      summary: async () => { calls.push('summary'); return summary({ has_learning_data: false }) },
      reset: async scope => {
        calls.push(`reset:${scope}`)
        return { scope, status: 'completed', deleted: summary() }
      },
      resetClient: async () => { calls.push('client') },
      markChoice: () => calls.push('choice'),
      broadcast: scope => calls.push(`broadcast:${scope}`),
    }))
    await store.inspect()
    calls.length = 0
    store.requestLearningReset()

    await store.confirmLearningReset()

    expect(calls).toEqual([
      'reset:learning',
      'client',
      'choice',
      'broadcast:learning',
      'summary',
    ])
    expect(store.lastResult?.scope).toBe('learning')
    expect(store.summary?.has_learning_data).toBe(false)
    expect(store.pendingScope).toBeNull()
    expect(store.phase).toBe('ready')
  })

  it('preserves browser state on backend failure and retries the same scope', async () => {
    const calls: string[] = []
    let attempts = 0
    const store = useDataLifecycle()
    store.initialize(dependencies({
      reset: async scope => {
        calls.push(`reset:${scope}`)
        attempts += 1
        if (attempts === 1) throw new Error('backend failed')
        return { scope, status: 'completed', deleted: summary() }
      },
      resetClient: async () => { calls.push('client') },
      markChoice: () => calls.push('choice'),
      broadcast: scope => calls.push(`broadcast:${scope}`),
    }))
    store.phase = 'ready'
    store.requestLearningReset()

    await store.confirmLearningReset()

    expect(calls).toEqual(['reset:learning'])
    expect(store.phase).toBe('reset_error')
    expect(store.pendingScope).toBe('learning')
    expect(store.error?.message).toBe('backend failed')

    await store.retryReset()

    expect(calls).toEqual([
      'reset:learning',
      'reset:learning',
      'client',
      'choice',
      'broadcast:learning',
    ])
    expect(store.phase).toBe('ready')
    expect(store.pendingScope).toBeNull()
  })

  it('stays in reset_error when client invalidation fails and does not announce completion', async () => {
    const calls: string[] = []
    const store = useDataLifecycle()
    store.initialize(dependencies({
      reset: async scope => {
        calls.push('reset')
        return { scope, status: 'completed', deleted: summary() }
      },
      resetClient: async () => { calls.push('client'); throw new Error('client failed') },
      markChoice: () => calls.push('choice'),
      broadcast: () => calls.push('broadcast'),
    }))
    store.phase = 'ready'
    store.requestLearningReset()

    await store.confirmLearningReset()

    expect(calls).toEqual(['reset', 'client'])
    expect(store.phase).toBe('reset_error')
    expect(store.pendingScope).toBe('learning')
  })

  it('does not let a stale local reset unlock after an external reset takes over', async () => {
    const calls: string[] = []
    let finishBackend!: (result: Awaited<ReturnType<LifecycleDependencies['reset']>>) => void
    const backend = new Promise<Awaited<ReturnType<LifecycleDependencies['reset']>>>(resolve => {
      finishBackend = resolve
    })
    const store = useDataLifecycle()
    store.initialize(dependencies({
      reset: scope => { calls.push(`reset:${scope}`); return backend },
      resetClient: async () => { calls.push('client') },
      clearChoice: () => calls.push('clear-choice'),
      markChoice: () => calls.push('choice'),
      broadcast: scope => calls.push(`broadcast:${scope}`),
      summary: async () => { calls.push('summary'); return summary() },
    }))
    store.phase = 'choice_required'
    store.requestLearningReset()
    const localReset = store.confirmLearningReset()
    expect(calls).toEqual(['reset:learning'])

    await store.handleExternalReset('learning')
    finishBackend({ scope: 'learning', status: 'completed', deleted: summary() })
    await localReset

    expect(calls).toEqual(['reset:learning', 'clear-choice', 'client'])
    expect(store.phase).toBe('external_reset')
  })

  it('starts only one backend reset when learning confirmation is submitted twice', async () => {
    let finishBackend!: (result: Awaited<ReturnType<LifecycleDependencies['reset']>>) => void
    const backend = new Promise<Awaited<ReturnType<LifecycleDependencies['reset']>>>(resolve => {
      finishBackend = resolve
    })
    const reset = vi.fn(() => backend)
    const store = useDataLifecycle()
    store.initialize(dependencies({ reset }))
    store.phase = 'ready'
    store.requestLearningReset()

    const first = store.confirmLearningReset()
    const second = store.confirmLearningReset()

    expect(reset).toHaveBeenCalledOnce()
    finishBackend({ scope: 'learning', status: 'completed', deleted: summary() })
    await Promise.all([first, second])
  })
})

describe('factory reset', () => {
  it('broadcasts completion, presents restart state, then clears browser data and reloads', async () => {
    const calls: string[] = []
    const store = useDataLifecycle()
    store.initialize(dependencies({
      reset: async scope => {
        calls.push(`reset:${scope}`)
        return { scope, status: 'completed', deleted: summary() }
      },
      broadcast: scope => calls.push(`broadcast:${scope}`),
      pause: async milliseconds => {
        calls.push(`pause:${milliseconds}:${store.phase}`)
        store.cancelReset()
        calls.push(`after-cancel:${store.phase}`)
      },
      clearFactory: () => calls.push('clear-factory'),
      reload: () => calls.push('reload'),
    }))
    store.phase = 'ready'
    store.requestFactoryReset()

    await store.confirmFactoryReset()

    expect(calls).toEqual([
      'reset:factory',
      'broadcast:factory',
      'pause:750:factory_restarting',
      'after-cancel:factory_restarting',
      'clear-factory',
      'reload',
    ])
    expect(store.lastResult?.scope).toBe('factory')
    expect(store.phase).toBe('factory_restarting')
  })

  it('does not clear or announce browser state when backend reset fails, then retries factory scope', async () => {
    const calls: string[] = []
    let attempts = 0
    const store = useDataLifecycle()
    store.initialize(dependencies({
      reset: async scope => {
        calls.push(`reset:${scope}`)
        attempts += 1
        if (attempts === 1) throw new Error('factory backend failed')
        return { scope, status: 'completed', deleted: summary() }
      },
      broadcast: scope => calls.push(`broadcast:${scope}`),
      pause: async () => { calls.push('pause') },
      clearFactory: () => calls.push('clear-factory'),
      reload: () => calls.push('reload'),
    }))
    store.phase = 'ready'
    store.requestFactoryReset()

    await store.confirmFactoryReset()

    expect(calls).toEqual(['reset:factory'])
    expect(store.phase).toBe('reset_error')
    expect(store.pendingScope).toBe('factory')

    await store.retryReset()

    expect(calls).toEqual([
      'reset:factory',
      'reset:factory',
      'broadcast:factory',
      'pause',
      'clear-factory',
      'reload',
    ])
  })

  it('does not let a stale local factory reset continue after an external reset takes over', async () => {
    const calls: string[] = []
    let finishBackend!: (result: Awaited<ReturnType<LifecycleDependencies['reset']>>) => void
    const backend = new Promise<Awaited<ReturnType<LifecycleDependencies['reset']>>>(resolve => {
      finishBackend = resolve
    })
    const store = useDataLifecycle()
    store.initialize(dependencies({
      reset: scope => { calls.push(`reset:${scope}`); return backend },
      broadcast: scope => calls.push(`broadcast:${scope}`),
      pause: async () => { calls.push('pause') },
      clearFactory: () => calls.push('clear-factory'),
      reload: () => calls.push('reload'),
      clearChoice: () => calls.push('clear-choice'),
      resetClient: async () => { calls.push('client') },
    }))
    store.phase = 'ready'
    store.requestFactoryReset()
    const localReset = store.confirmFactoryReset()

    await store.handleExternalReset('learning')
    finishBackend({ scope: 'factory', status: 'completed', deleted: summary() })
    await localReset

    expect(calls).toEqual(['reset:factory', 'clear-choice', 'client'])
    expect(store.phase).toBe('external_reset')
  })
})

describe('external reset', () => {
  it('blocks immediately and ignores acknowledgement while external client reset is pending', async () => {
    const calls: string[] = []
    let finishClient!: () => void
    const clientReset = new Promise<void>(resolve => { finishClient = resolve })
    const store = useDataLifecycle()
    store.initialize(dependencies({
      clearChoice: () => calls.push('clear-choice'),
      resetClient: async () => { calls.push('client'); await clientReset },
      markChoice: () => calls.push('choice'),
    }))
    store.phase = 'ready'

    const handling = store.handleExternalReset('learning')
    expect(calls).toEqual(['clear-choice', 'client'])
    expect(store.phase).toBe('external_reset')
    expect(store.externalClientReady).toBe(false)

    await store.acknowledgeExternalReset()
    expect(calls).toEqual(['clear-choice', 'client'])
    expect(store.phase).toBe('external_reset')

    finishClient()
    await handling

    expect(store.phase).toBe('external_reset')
    expect(store.externalClientReady).toBe(true)
    await store.acknowledgeExternalReset()
    expect(calls).toEqual(['clear-choice', 'client', 'choice'])
    expect(store.phase).toBe('ready')
  })

  it('clears every app browser key and reloads immediately after external factory reset', async () => {
    const calls: string[] = []
    const store = useDataLifecycle()
    store.initialize(dependencies({
      clearFactory: () => calls.push('clear-factory'),
      reload: () => calls.push('reload'),
    }))

    await store.handleExternalReset('factory')

    expect(calls).toEqual(['clear-factory', 'reload'])
    expect(store.phase).toBe('factory_restarting')
  })

  it('still reloads and remains blocked when external factory browser clearing fails', async () => {
    const calls: string[] = []
    const store = useDataLifecycle()
    store.initialize(dependencies({
      clearFactory: () => { calls.push('clear-factory'); throw new Error('storage blocked') },
      reload: () => calls.push('reload'),
    }))
    store.phase = 'ready'

    await store.handleExternalReset('factory')

    expect(calls).toEqual(['clear-factory', 'reload'])
    expect(store.phase).toBe('factory_restarting')
    expect(store.error?.message).toBe('storage blocked')
  })

  it('remains blocked for acknowledgement when external client refresh fails', async () => {
    const store = useDataLifecycle()
    store.initialize(dependencies({
      resetClient: async () => { throw new Error('refresh unavailable') },
    }))
    store.phase = 'ready'

    await store.handleExternalReset('learning')

    expect(store.phase).toBe('external_reset')
    expect(store.error?.message).toBe('refresh unavailable')
    expect(store.externalClientReady).toBe(false)
  })

  it('retries a failed external client reset on acknowledgement before unlocking', async () => {
    const calls: string[] = []
    let attempts = 0
    const store = useDataLifecycle()
    store.initialize(dependencies({
      resetClient: async () => {
        attempts += 1
        calls.push(`client:${attempts}`)
        if (attempts === 1) throw new Error('refresh unavailable')
      },
      markChoice: () => calls.push('choice'),
    }))

    await store.handleExternalReset('learning')
    await store.acknowledgeExternalReset()

    expect(calls).toEqual(['client:1', 'client:2', 'choice'])
    expect(store.externalClientReady).toBe(true)
    expect(store.phase).toBe('ready')
  })

  it('does not unlock when acknowledgement retry still cannot reset client state', async () => {
    const calls: string[] = []
    const store = useDataLifecycle()
    store.initialize(dependencies({
      resetClient: async () => { calls.push('client'); throw new Error('still unavailable') },
      markChoice: () => calls.push('choice'),
    }))

    await store.handleExternalReset('learning')
    await store.acknowledgeExternalReset()

    expect(calls).toEqual(['client', 'client'])
    expect(store.externalClientReady).toBe(false)
    expect(store.phase).toBe('external_reset')
    expect(store.error?.message).toBe('still unavailable')
  })

  it('does not let a stale acknowledgement retry override a newer external factory reset', async () => {
    const calls: string[] = []
    let attempts = 0
    let finishRetry!: () => void
    const retry = new Promise<void>(resolve => { finishRetry = resolve })
    const store = useDataLifecycle()
    store.initialize(dependencies({
      resetClient: async () => {
        attempts += 1
        calls.push(`client:${attempts}`)
        if (attempts === 1) throw new Error('first failed')
        await retry
      },
      markChoice: () => calls.push('choice'),
      clearFactory: () => calls.push('clear-factory'),
      reload: () => calls.push('reload'),
    }))
    await store.handleExternalReset('learning')

    const acknowledgement = store.acknowledgeExternalReset()
    await store.handleExternalReset('factory')
    finishRetry()
    await acknowledgement

    expect(calls).toEqual(['client:1', 'client:2', 'clear-factory', 'reload'])
    expect(store.phase).toBe('factory_restarting')
  })

  it('still resets client state and blocks when clearing the startup choice fails', async () => {
    const calls: string[] = []
    const store = useDataLifecycle()
    store.initialize(dependencies({
      clearChoice: () => { calls.push('clear-choice'); throw new Error('storage blocked') },
      resetClient: async () => { calls.push('client') },
    }))
    store.phase = 'ready'

    await store.handleExternalReset('learning')

    expect(calls).toEqual(['clear-choice', 'client'])
    expect(store.phase).toBe('external_reset')
    expect(store.error?.message).toBe('storage blocked')
  })
})
