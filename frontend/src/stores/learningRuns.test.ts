import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { memoryStorage } from '../test/memoryStorage'
import { EvalApiError } from '../lib/evalApi'
import type { EvalConnectionSnapshot, LearningRunEvent, LearningRunRequest } from '../lib/evalContracts'
import { HISTORICAL_FALLBACK_MS, resetLearningRunDependencies, useLearningRuns } from './learningRuns'

const REQUEST: LearningRunRequest = {
  experiment_id: 'tutor-prompt-regression-v1',
  task_case_id: 'tgqa-004',
  variant_id: 'tutor-v3',
  run_profile: 'evaluation',
}

const CONNECTION: EvalConnectionSnapshot = {
  provider: 'ollama',
  model: 'llama3.2',
}

function event(partial: LearningRunEvent): LearningRunEvent {
  return partial
}

function createStoreHarness() {
  const calls: string[] = []
  const emitters: Array<(value: LearningRunEvent) => void> = []
  const signals: AbortSignal[] = []
  let rejectStream: ((reason: unknown) => void) | null = null
  let resolveStream: (() => void) | null = null
  let now = 0
  const store = useLearningRuns()
  store.initialize({
    now: () => now,
    persistActiveRunId: () => undefined,
    streamRun: async (_request, _connection, onEvent, abortSignal) => {
      emitters.push(onEvent)
      signals.push(abortSignal)
      await new Promise<void>((resolve, reject) => {
        resolveStream = resolve
        rejectStream = reject
        abortSignal.addEventListener('abort', () => {
          calls.push('abort-stream')
          resolve()
        })
      })
    },
    cancelRun: async (_runId, options) => {
      calls.push(options?.keepalive ? 'cancel-keepalive' : 'cancel-endpoint')
      return {
        run_id: 'run-1',
        experiment_id: REQUEST.experiment_id,
        suite_execution_id: null,
        task_case_id: REQUEST.task_case_id,
        variant_id: REQUEST.variant_id,
        run_profile: 'evaluation',
        lifecycle: 'cancelled',
        outcome: null,
        latest_score_set: null,
        created_at: '2026-08-13T00:00:00Z',
        started_at: null,
        finished_at: null,
      }
    },
    getDetail: async () => {
      throw new Error('detail unused')
    },
  })
  return {
    store,
    calls,
    route: { unmount() {} },
    get controller() {
      return { signal: signals[signals.length - 1] as AbortSignal }
    },
    signalAt(index: number) {
      return signals[index] as AbortSignal
    },
    emit(value: LearningRunEvent, index = emitters.length - 1) {
      emitters[index]?.(value)
    },
    fail(reason: unknown) {
      rejectStream?.(reason)
    },
    end() {
      resolveStream?.()
    },
    advance(ms: number) {
      now += ms
    },
    async start() {
      const started = store.start(REQUEST, CONNECTION)
      await Promise.resolve()
      return started
    },
  }
}

beforeEach(() => {
  setActivePinia(createPinia())
  vi.stubGlobal('localStorage', memoryStorage())
})

afterEach(() => {
  resetLearningRunDependencies()
  vi.unstubAllGlobals()
})

describe('attached learning run store', () => {
  it('keeps an attached run alive across route changes', async () => {
    const harness = createStoreHarness()
    void harness.start()
    await Promise.resolve()
    harness.route.unmount()
    expect(harness.controller.signal.aborted).toBe(false)
  })

  it('calls cancel endpoint before aborting the stream', async () => {
    const harness = createStoreHarness()
    const started = harness.start()
    await Promise.resolve()
    harness.emit(event({
      schema_version: 'eval-api-v1',
      type: 'run_created',
      run_id: 'run-1',
    }))
    await harness.store.cancelActive()
    await started
    expect(harness.calls).toEqual(['cancel-endpoint', 'abort-stream'])
  })

  it('best-effort cancels on pagehide then aborts', async () => {
    const harness = createStoreHarness()
    void harness.start()
    await Promise.resolve()
    harness.emit(event({
      schema_version: 'eval-api-v1',
      type: 'run_created',
      run_id: 'run-1',
    }))
    harness.store.handlePageHide()
    expect(harness.calls).toEqual(['cancel-keepalive', 'abort-stream'])
  })

  it('records a network error without further mutations', async () => {
    const harness = createStoreHarness()
    const started = harness.start()
    await Promise.resolve()
    harness.fail(new Error('network down'))
    await started
    expect(harness.store.status).toBe('error')
    harness.emit(event({
      schema_version: 'eval-api-v1',
      type: 'run_created',
      run_id: 'late-run',
    }))
    expect(harness.store.activeRunId).toBeNull()
  })

  it('ignores events after the run is terminal', async () => {
    const harness = createStoreHarness()
    void harness.start()
    await Promise.resolve()
    harness.emit(event({
      schema_version: 'eval-api-v1',
      type: 'run_created',
      run_id: 'run-1',
    }))
    harness.emit(event({
      schema_version: 'eval-api-v1',
      type: 'run_finished',
      run_id: 'run-1',
      lifecycle: 'finished',
      outcome: 'success',
    }))
    harness.emit(event({
      schema_version: 'eval-api-v1',
      type: 'stage_started',
      run_id: 'run-1',
      stage: 'late',
    }))
    expect(harness.store.status).toBe('terminal')
    expect(harness.store.events.map(item => item.type)).toEqual(['run_created', 'run_finished'])
  })

  it('does not let an old stream write into a newer run', async () => {
    const harness = createStoreHarness()
    const firstGeneration = harness.store.generation
    void harness.start()
    await Promise.resolve()
    void harness.start()
    await Promise.resolve()
    harness.emit(event({
      schema_version: 'eval-api-v1',
      type: 'run_created',
      run_id: 'stale-run',
    }), 0)
    harness.emit(event({
      schema_version: 'eval-api-v1',
      type: 'run_created',
      run_id: 'fresh-run',
    }))
    expect(harness.store.generation).toBe(firstGeneration + 2)
    expect(harness.store.activeRunId).toBe('fresh-run')
    expect(harness.store.events.map(item => 'run_id' in item ? item.run_id : '')).toEqual(['fresh-run'])
  })

  it('surfaces a busy response without attaching to the other stream', async () => {
    const store = useLearningRuns()
    store.initialize({
      persistActiveRunId: () => undefined,
      streamRun: async () => {
        throw new EvalApiError(409, {
          code: 'evaluation_busy',
          message: 'another evaluation is already running',
          fields: [],
          active_entity_id: 'run-active-001',
          active_kind: 'run',
        })
      },
      cancelRun: async () => {
        throw new Error('unused')
      },
      getDetail: async () => {
        throw new Error('unused')
      },
    })

    await store.start(REQUEST, CONNECTION)

    expect(store.status).toBe('error')
    expect(store.busyTarget).toEqual({ id: 'run-active-001', kind: 'run' })
    expect(store.controller).not.toBeNull()
    expect(store.events).toEqual([])
  })

  it('only exposes historical fallback after 30s and never replaces the live run id', async () => {
    const harness = createStoreHarness()
    void harness.start()
    await Promise.resolve()
    harness.emit(event({
      schema_version: 'eval-api-v1',
      type: 'run_created',
      run_id: 'live-run',
    }))
    expect(harness.store.canOpenHistoricalFallback()).toBe(false)
    harness.advance(HISTORICAL_FALLBACK_MS)
    expect(harness.store.canOpenHistoricalFallback()).toBe(true)
    expect(harness.store.openHistoricalFallback('historical-run')).toBe('historical-run')
    expect(harness.store.activeRunId).toBe('live-run')
  })

  it('aborts the previous attached stream before a second start', async () => {
    const harness = createStoreHarness()
    void harness.start()
    await Promise.resolve()
    harness.emit(event({
      schema_version: 'eval-api-v1',
      type: 'run_created',
      run_id: 'run-1',
    }))
    const firstSignal = harness.signalAt(0)
    const second = harness.start()
    await Promise.resolve()
    await Promise.resolve()

    expect(firstSignal.aborted).toBe(true)
    expect(harness.controller.signal.aborted).toBe(false)
    expect(harness.calls[0]).toBe('cancel-endpoint')
    expect(harness.calls).toContain('abort-stream')
    await harness.store.cancelActive()
    await second
  })

  it('aborts the stream when the cancel endpoint fails', async () => {
    const harness = createStoreHarness()
    const started = harness.start()
    await Promise.resolve()
    harness.emit(event({
      schema_version: 'eval-api-v1',
      type: 'run_created',
      run_id: 'run-1',
    }))
    harness.store.initialize({
      cancelRun: async () => {
        harness.calls.push('cancel-endpoint')
        throw new EvalApiError(503, {
          code: 'evaluation_unavailable',
          message: 'evaluation storage is unavailable',
          fields: [],
          active_entity_id: null,
          active_kind: null,
        })
      },
    })

    await harness.store.cancelActive()
    await started

    expect(harness.calls).toEqual(['cancel-endpoint', 'abort-stream'])
    expect(harness.store.status).toBe('error')
    expect(harness.store.error?.code).toBe('evaluation_unavailable')
  })

  it('treats a stream close without run_finished as process_interrupted', async () => {
    const harness = createStoreHarness()
    const started = harness.start()
    await Promise.resolve()
    harness.emit(event({
      schema_version: 'eval-api-v1',
      type: 'run_created',
      run_id: 'run-1',
    }))
    harness.end()
    await started

    expect(harness.store.status).toBe('error')
    expect(harness.store.error?.code).toBe('process_interrupted')
    expect(harness.store.activeRunId).toBe('run-1')
  })
})
