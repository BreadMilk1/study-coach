import { defineStore } from 'pinia'

import type {
  DataLifecycleApiError,
  DataSummaryDto,
  ResetResultDto,
  ResetScope,
} from '../lib/api'
import { resolveStartupDecision } from '../lib/dataLifecycle'

export type LifecyclePhase =
  | 'checking'
  | 'ready'
  | 'choice_required'
  | 'inspection_error'
  | 'confirming_learning'
  | 'confirming_factory'
  | 'resetting'
  | 'reset_error'
  | 'external_reset'
  | 'factory_restarting'

export interface LifecycleDependencies {
  summary: () => Promise<DataSummaryDto>
  reset: (scope: ResetScope) => Promise<ResetResultDto>
  resetClient: () => Promise<void>
  markChoice: () => void
  clearChoice: () => void
  clearFactory: () => void
  broadcast: (scope: ResetScope) => void
  reload: () => void
  pause: (milliseconds: number) => Promise<void>
}

let lifecycleDependencies: LifecycleDependencies | null = null

function dependencies(): LifecycleDependencies {
  if (lifecycleDependencies === null) throw new Error('Lifecycle store is not initialized')
  return lifecycleDependencies
}

export const useDataLifecycle = defineStore('dataLifecycle', {
  state: () => ({
    phase: 'checking' as LifecyclePhase,
    summary: null as DataSummaryDto | null,
    lastResult: null as ResetResultDto | null,
    error: null as DataLifecycleApiError | Error | null,
    pendingScope: null as ResetScope | null,
    returnPhase: 'ready' as 'ready' | 'choice_required',
    externalClientReady: false,
    externalClientPending: false,
    operationGeneration: 0,
    summaryRefreshGeneration: 0,
    summaryRefreshing: false,
  }),
  getters: {
    canContinueWithoutClearing: state => state.phase === 'inspection_error',
    canStartFresh: state => state.phase === 'choice_required' && state.summary?.reset_enabled === true,
  },
  actions: {
    initialize(value: LifecycleDependencies) {
      lifecycleDependencies = value
    },
    async inspect() {
      const generation = ++this.operationGeneration
      this.phase = 'checking'
      this.error = null
      try {
        const summary = await dependencies().summary()
        if (generation !== this.operationGeneration) return
        this.summary = summary
        const decision = resolveStartupDecision({
          resetEnabled: this.summary.reset_enabled,
          hasLearningData: this.summary.has_learning_data,
        }, sessionStorage)
        this.phase = decision === 'ready' ? 'ready' : 'choice_required'
      } catch (error) {
        if (generation !== this.operationGeneration) return
        this.error = error instanceof Error ? error : new Error('Local data inspection failed.')
        this.phase = 'inspection_error'
      }
    },
    async refreshSummary() {
      if (this.phase !== 'ready') return
      const operationGeneration = this.operationGeneration
      const refreshGeneration = ++this.summaryRefreshGeneration
      this.summaryRefreshing = true
      this.error = null
      try {
        const summary = await dependencies().summary()
        if (
          operationGeneration !== this.operationGeneration
          || refreshGeneration !== this.summaryRefreshGeneration
          || this.phase !== 'ready'
        ) return
        this.summary = summary
      } catch (error) {
        if (
          operationGeneration !== this.operationGeneration
          || refreshGeneration !== this.summaryRefreshGeneration
          || this.phase !== 'ready'
        ) return
        this.error = error instanceof Error ? error : new Error('Local data summary refresh failed.')
      } finally {
        if (refreshGeneration === this.summaryRefreshGeneration) this.summaryRefreshing = false
      }
    },
    continueExisting() {
      if (this.phase !== 'choice_required') return
      dependencies().markChoice()
      this.phase = 'ready'
    },
    continueWithoutClearing() {
      if (this.phase !== 'inspection_error') return
      dependencies().markChoice()
      this.phase = 'ready'
    },
    requestLearningReset() {
      if (this.phase !== 'ready' && this.phase !== 'choice_required') return
      this.returnPhase = this.phase === 'choice_required' ? 'choice_required' : 'ready'
      this.pendingScope = 'learning'
      this.error = null
      this.phase = 'confirming_learning'
    },
    requestFactoryReset() {
      if (this.phase !== 'ready') return
      this.returnPhase = 'ready'
      this.pendingScope = 'factory'
      this.error = null
      this.phase = 'confirming_factory'
    },
    cancelReset() {
      if (
        this.phase !== 'confirming_learning'
        && this.phase !== 'confirming_factory'
        && this.phase !== 'reset_error'
      ) return
      this.pendingScope = null
      this.error = null
      this.phase = this.returnPhase
    },
    async confirmLearningReset() {
      if (this.phase !== 'confirming_learning') return
      const generation = ++this.operationGeneration
      this.phase = 'resetting'
      this.error = null
      try {
        const result = await dependencies().reset('learning')
        if (generation !== this.operationGeneration) return
        this.lastResult = result
        await dependencies().resetClient()
        if (generation !== this.operationGeneration) return
        dependencies().markChoice()
        dependencies().broadcast('learning')
        const summary = await dependencies().summary()
        if (generation !== this.operationGeneration) return
        this.summary = summary
        this.pendingScope = null
        this.phase = 'ready'
      } catch (error) {
        if (generation !== this.operationGeneration) return
        this.error = error instanceof Error ? error : new Error('Learning reset failed.')
        this.pendingScope = 'learning'
        this.phase = 'reset_error'
      }
    },
    async confirmFactoryReset() {
      if (this.phase !== 'confirming_factory') return
      const generation = ++this.operationGeneration
      this.phase = 'resetting'
      this.error = null
      try {
        const result = await dependencies().reset('factory')
        if (generation !== this.operationGeneration) return
        this.lastResult = result
        dependencies().broadcast('factory')
        this.phase = 'factory_restarting'
        await dependencies().pause(750)
        if (generation !== this.operationGeneration) return
        dependencies().clearFactory()
        dependencies().reload()
      } catch (error) {
        if (generation !== this.operationGeneration) return
        this.error = error instanceof Error ? error : new Error('Factory reset failed.')
        this.pendingScope = 'factory'
        this.phase = 'reset_error'
      }
    },
    async retryReset() {
      if (this.phase !== 'reset_error') return
      if (this.pendingScope === 'factory') {
        this.phase = 'confirming_factory'
        await this.confirmFactoryReset()
      } else if (this.pendingScope === 'learning') {
        this.phase = 'confirming_learning'
        await this.confirmLearningReset()
      }
    },
    async handleExternalReset(scope: ResetScope) {
      const generation = ++this.operationGeneration
      if (scope === 'factory') {
        this.phase = 'factory_restarting'
        this.externalClientReady = false
        this.externalClientPending = false
        let failure: Error | null = null
        try {
          dependencies().clearFactory()
        } catch (error) {
          failure = error instanceof Error ? error : new Error('Factory browser state clearing failed.')
        }
        try {
          dependencies().reload()
        } catch (error) {
          const reloadFailure = error instanceof Error ? error : new Error('Factory restart failed.')
          failure = failure === null
            ? reloadFailure
            : new Error('Factory browser restart failed.')
        }
        if (generation === this.operationGeneration) this.error = failure
        return
      }
      this.phase = 'external_reset'
      this.externalClientReady = false
      this.externalClientPending = true
      let failure: Error | null = null
      let clientReady = false
      try {
        dependencies().clearChoice()
      } catch (error) {
        failure = error instanceof Error ? error : new Error('External startup choice clearing failed.')
      }
      try {
        await dependencies().resetClient()
        clientReady = true
      } catch (error) {
        const clientFailure = error instanceof Error ? error : new Error('External client reset failed.')
        failure = failure === null
          ? clientFailure
          : new Error('External browser state reset failed.')
      } finally {
        if (generation !== this.operationGeneration) return
        this.error = failure
        this.externalClientReady = clientReady
        this.externalClientPending = false
        this.phase = 'external_reset'
      }
    },
    async acknowledgeExternalReset() {
      if (this.phase !== 'external_reset' || this.externalClientPending) return
      const generation = this.operationGeneration
      if (!this.externalClientReady) {
        this.externalClientPending = true
        try {
          await dependencies().resetClient()
          if (generation !== this.operationGeneration || this.phase !== 'external_reset') return
          this.externalClientReady = true
          this.error = null
        } catch (error) {
          if (generation !== this.operationGeneration || this.phase !== 'external_reset') return
          this.error = error instanceof Error ? error : new Error('External client reset failed.')
          return
        } finally {
          if (generation === this.operationGeneration) this.externalClientPending = false
        }
      }
      if (generation !== this.operationGeneration || this.phase !== 'external_reset') return
      try {
        dependencies().markChoice()
      } catch (error) {
        this.error = error instanceof Error ? error : new Error('Startup choice could not be saved.')
        return
      }
      this.error = null
      this.phase = 'ready'
    },
  },
})
