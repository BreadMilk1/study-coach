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
  clearFactorySession: () => void
  stageFactoryIdentity: (forceRotate?: boolean) => Promise<string>
  provisionFactoryIdentity: (fingerprint?: string) => Promise<string>
  invalidateProvisioning: () => void
  broadcast: (scope: ResetScope) => void
  reload: () => void
  pause: (milliseconds: number) => Promise<void>
}

const SAFE_RESET_REFUSAL_CODES = new Set([
  'data_operation_in_progress',
  'reset_in_progress',
  'reset_disabled',
  'invalid_confirmation',
])

let lifecycleDependencies: LifecycleDependencies | null = null

function dependencies(): LifecycleDependencies {
  if (lifecycleDependencies === null) throw new Error('Lifecycle store is not initialized')
  return lifecycleDependencies
}

function requiredRecoveryScope(error: unknown): ResetScope | null {
  if (!(error instanceof Error)) return null
  const lifecycleError = error as Error & {
    code?: unknown
    requiredScope?: unknown
  }
  if (lifecycleError.code !== 'reset_recovery_required') return null
  return lifecycleError.requiredScope === 'learning' || lifecycleError.requiredScope === 'factory'
    ? lifecycleError.requiredScope
    : null
}

function asLifecycleApiError(error: unknown): DataLifecycleApiError | null {
  if (!(error instanceof Error)) return null
  const candidate = error as DataLifecycleApiError
  if (typeof candidate.code !== 'string' || typeof candidate.status !== 'number') return null
  return candidate
}

function isSafeResetRefusal(error: unknown): boolean {
  const lifecycleError = asLifecycleApiError(error)
  if (lifecycleError === null) return false
  if (SAFE_RESET_REFUSAL_CODES.has(lifecycleError.code)) return true
  return lifecycleError.status === 401 || lifecycleError.status === 422
}

function shouldLatchRecovery(error: unknown, backendCompleted: boolean): boolean {
  if (backendCompleted) return true
  if (isSafeResetRefusal(error)) return false
  return true
}

export const useDataLifecycle = defineStore('dataLifecycle', {
  state: () => ({
    phase: 'checking' as LifecyclePhase,
    workspaceUnlocked: false,
    summary: null as DataSummaryDto | null,
    lastResult: null as ResetResultDto | null,
    error: null as DataLifecycleApiError | Error | null,
    pendingScope: null as ResetScope | null,
    recoveryScope: null as ResetScope | null,
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
      this.workspaceUnlocked = false
      this.error = null
      try {
        let summary = await dependencies().summary()
        if (generation !== this.operationGeneration) return

        // Factory reset may complete while the success response is lost. The
        // stale JWT can still call summary (row-less retry semantics) and see
        // empty learning data — but must not unlock until a live identity exists.
        // Only this explicit summary flag triggers re-provision (not arbitrary 401s).
        if (summary.current_user_exists === false) {
          const recoveryFingerprint = await dependencies().stageFactoryIdentity(false)
          if (generation !== this.operationGeneration) return
          dependencies().invalidateProvisioning()
          dependencies().clearFactory()
          await dependencies().provisionFactoryIdentity(recoveryFingerprint)
          if (generation !== this.operationGeneration) return
          summary = await dependencies().summary()
          if (generation !== this.operationGeneration) return
          if (summary.current_user_exists === false) {
            this.summary = summary
            this.error = new Error('Local identity could not be restored after factory reset.')
            this.phase = 'inspection_error'
            this.workspaceUnlocked = false
            return
          }
        }

        this.summary = summary
        const decision = resolveStartupDecision({
          resetEnabled: this.summary.reset_enabled,
          hasLearningData: this.summary.has_learning_data,
        }, sessionStorage)
        this.recoveryScope = null
        this.phase = decision === 'ready' ? 'ready' : 'choice_required'
        this.workspaceUnlocked = decision === 'ready'
      } catch (error) {
        if (generation !== this.operationGeneration) return
        this.error = error instanceof Error ? error : new Error('Local data inspection failed.')
        const recoveryScope = requiredRecoveryScope(error)
        if (recoveryScope !== null) {
          this.pendingScope = recoveryScope
          this.recoveryScope = recoveryScope
          this.phase = 'reset_error'
        } else {
          this.phase = 'inspection_error'
        }
        this.workspaceUnlocked = false
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
      this.workspaceUnlocked = true
      this.phase = 'ready'
    },
    continueWithoutClearing() {
      if (this.phase !== 'inspection_error') return
      dependencies().markChoice()
      this.workspaceUnlocked = true
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
      if (this.phase === 'reset_error' && this.recoveryScope !== null) return
      this.pendingScope = null
      this.error = null
      this.phase = this.returnPhase
    },
    async confirmLearningReset() {
      if (this.phase !== 'confirming_learning') return
      const generation = ++this.operationGeneration
      let backendCompleted = false
      this.phase = 'resetting'
      this.error = null
      try {
        const result = await dependencies().reset('learning')
        backendCompleted = true
        this.recoveryScope = null
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
        this.workspaceUnlocked = true
        this.phase = 'ready'
      } catch (error) {
        if (generation !== this.operationGeneration) return
        const requiredScope = requiredRecoveryScope(error)
        if (requiredScope !== null) {
          this.recoveryScope = requiredScope
          this.pendingScope = requiredScope
        } else if (shouldLatchRecovery(error, backendCompleted)) {
          this.recoveryScope = 'learning'
          this.pendingScope = 'learning'
        } else {
          this.recoveryScope = null
          this.pendingScope = 'learning'
        }
        this.error = error instanceof Error ? error : new Error('Learning reset failed.')
        this.phase = 'reset_error'
      }
    },
    async confirmFactoryReset() {
      if (this.phase !== 'confirming_factory') return
      const generation = ++this.operationGeneration
      let backendCompleted = false
      this.phase = 'resetting'
      this.error = null
      try {
        // Stage the replacement identity before the destructive request so a
        // lost success response and delayed tabs still share one fingerprint.
        const recoveryFingerprint = await dependencies().stageFactoryIdentity(true)
        if (generation !== this.operationGeneration) return
        const result = await dependencies().reset('factory')
        backendCompleted = true
        this.recoveryScope = null
        if (generation !== this.operationGeneration) return
        this.lastResult = result
        dependencies().invalidateProvisioning()
        dependencies().clearFactory()
        await dependencies().provisionFactoryIdentity(recoveryFingerprint)
        if (generation !== this.operationGeneration) return
        dependencies().broadcast('factory')
        this.phase = 'factory_restarting'
        await dependencies().pause(750)
        if (generation !== this.operationGeneration) return
        dependencies().reload()
      } catch (error) {
        if (generation !== this.operationGeneration) return
        const requiredScope = requiredRecoveryScope(error)
        if (requiredScope !== null) {
          this.recoveryScope = requiredScope
          this.pendingScope = requiredScope
        } else if (shouldLatchRecovery(error, backendCompleted)) {
          this.recoveryScope = 'factory'
          this.pendingScope = 'factory'
        } else {
          this.recoveryScope = null
          this.pendingScope = 'factory'
        }
        this.error = error instanceof Error ? error : new Error('Factory reset failed.')
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
      this.workspaceUnlocked = false
      this.recoveryScope = null
      if (scope === 'factory') {
        this.phase = 'factory_restarting'
        this.externalClientReady = false
        this.externalClientPending = false
        let failure: Error | null = null
        try {
          dependencies().invalidateProvisioning()
          dependencies().clearFactorySession()
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
      this.workspaceUnlocked = true
      this.phase = 'ready'
    },
  },
})
