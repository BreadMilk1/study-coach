import { defineStore } from 'pinia'

import { cancelLearningRun, getRunDetail, streamLearningRun } from '../lib/evalApi'
import { ACTIVE_LEARNING_RUN_KEY } from '../lib/dataLifecycle'
import type {
  EvalConnectionSnapshot,
  LearningRunEvent,
  LearningRunRequest,
  RunDetail,
} from '../lib/evalContracts'
import { EvalApiError } from '../lib/evalApi'

export const HISTORICAL_FALLBACK_MS = 30_000

export interface LearningRunDependencies {
  streamRun: typeof streamLearningRun
  cancelRun: typeof cancelLearningRun
  getDetail: typeof getRunDetail
  now?: () => number
  persistActiveRunId?: (runId: string | null) => void
}

export interface BusyTarget {
  id: string
  kind: 'run' | 'score_set'
}

type RunStatus = 'idle' | 'running' | 'cancelling' | 'terminal' | 'error'

let learningRunDependencies: LearningRunDependencies | null = null

function defaultPersist(runId: string | null): void {
  try {
    if (runId) localStorage.setItem(ACTIVE_LEARNING_RUN_KEY, runId)
    else localStorage.removeItem(ACTIVE_LEARNING_RUN_KEY)
  } catch { /* storage unavailable */ }
}

function dependencies(): LearningRunDependencies {
  return learningRunDependencies ?? {
    streamRun: streamLearningRun,
    cancelRun: cancelLearningRun,
    getDetail: getRunDetail,
    persistActiveRunId: defaultPersist,
  }
}

function persist(runId: string | null): void {
  (dependencies().persistActiveRunId ?? defaultPersist)(runId)
}

export const useLearningRuns = defineStore('learningRuns', {
  state: () => ({
    status: 'idle' as RunStatus,
    activeRunId: null as string | null,
    activeScoreSetId: null as string | null,
    events: [] as LearningRunEvent[],
    error: null as EvalApiError | null,
    busyTarget: null as BusyTarget | null,
    connection: null as EvalConnectionSnapshot | null,
    controller: null as AbortController | null,
    startedAt: null as number | null,
    generation: 0,
    detail: null as RunDetail | null,
  }),
  actions: {
    canOpenHistoricalFallback(): boolean {
      if (this.status !== 'running' || this.startedAt === null) return false
      const now = dependencies().now ?? Date.now
      return now() - this.startedAt >= HISTORICAL_FALLBACK_MS
    },
    initialize(overrides: Partial<LearningRunDependencies>): void {
      learningRunDependencies = { ...dependencies(), ...overrides }
    },

    async start(request: LearningRunRequest, connection: EvalConnectionSnapshot): Promise<void> {
      const generation = this.generation + 1
      const controller = new AbortController()
      this.generation = generation
      this.status = 'running'
      this.activeRunId = null
      this.activeScoreSetId = null
      this.events = []
      this.error = null
      this.busyTarget = null
      this.connection = connection
      this.controller = controller
      this.startedAt = (dependencies().now ?? Date.now)()
      this.detail = null

      try {
        await dependencies().streamRun(
          request,
          connection,
          event => this.applyEvent(generation, event),
          controller.signal,
        )
        if (this.generation === generation && this.status === 'running') {
          this.status = 'terminal'
        }
      } catch (reason) {
        if (controller.signal.aborted) {
          if (this.generation === generation && this.status !== 'terminal') {
            this.status = 'terminal'
          }
          return
        }
        if (this.generation !== generation) return
        const error = reason instanceof EvalApiError
          ? reason
          : new EvalApiError(0, {
            code: 'evaluation_unavailable',
            message: reason instanceof Error ? reason.message : 'evaluation stream failed',
            fields: [],
            active_entity_id: null,
            active_kind: null,
          })
        this.error = error
        this.status = 'error'
        if (error.code === 'evaluation_busy' && error.active_entity_id && error.active_kind) {
          this.busyTarget = { id: error.active_entity_id, kind: error.active_kind }
        }
      }
    },

    applyEvent(generation: number, event: LearningRunEvent): void {
      if (generation !== this.generation) return
      if (this.status === 'terminal' || this.status === 'error') return
      if (this.activeRunId && event.run_id !== this.activeRunId) return
      this.events.push(event)
      if (event.type === 'run_created') {
        this.activeRunId = event.run_id
        persist(event.run_id)
      }
      if ('score_set_id' in event) this.activeScoreSetId = event.score_set_id
      if (event.type === 'run_finished') this.status = 'terminal'
    },

    async cancelActive(): Promise<void> {
      const runId = this.activeRunId
      const controller = this.controller
      if (!runId && !controller) return
      this.status = 'cancelling'
      if (runId) await dependencies().cancelRun(runId)
      controller?.abort()
    },

    handlePageHide(): void {
      const runId = this.activeRunId
      if (runId) {
        void dependencies().cancelRun(runId, { keepalive: true })
      }
      this.controller?.abort()
    },

    openHistoricalFallback(historicalRunId: string): string {
      return historicalRunId
    },
  },
})

export function resetLearningRunDependencies(): void {
  learningRunDependencies = null
}
