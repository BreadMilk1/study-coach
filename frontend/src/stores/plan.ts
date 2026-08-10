import { defineStore } from 'pinia'
import {
  getCurrentPlan,
  getPlanEvents,
  patchMilestoneDone,
  type PlanCurrentDto,
  type PlanEventDto,
  type ValidationHintDto,
} from '../lib/api'
import { captureLearningStateEpoch, isLearningStateEpochCurrent } from '../lib/dataLifecycle'

interface PlanState {
  plan: PlanCurrentDto | null
  events: PlanEventDto[]
  lastValidationHint: ValidationHintDto | null
  loading: boolean
  updatingMilestoneId: string | null
  error: string | null
  noActive: boolean        // true if backend returned 404
  mindmapMermaid: string | null  // set by A3
}

export const usePlan = defineStore('plan', {
  state: (): PlanState => ({
    plan: null,
    events: [],
    lastValidationHint: null,
    loading: false,
    updatingMilestoneId: null,
    error: null,
    noActive: false,
    mindmapMermaid: null,
  }),
  actions: {
    resetAfterDataClear() {
      this.plan = null
      this.events = []
      this.lastValidationHint = null
      this.loading = false
      this.updatingMilestoneId = null
      this.error = null
      this.noActive = false
      this.mindmapMermaid = null
    },
    async fetch() {
      const epoch = captureLearningStateEpoch()
      this.loading = true
      this.error = null
      this.noActive = false
      this.lastValidationHint = null
      try {
        const plan = await getCurrentPlan()
        if (!isLearningStateEpochCurrent(epoch)) return true
        const events = await getPlanEvents(plan.plan_id)
        if (!isLearningStateEpochCurrent(epoch)) return true
        this.plan = plan
        this.events = events
        return true
      } catch (e: any) {
        if (e?.status === 404) {
          if (isLearningStateEpochCurrent(epoch)) {
            this.resetAfterDataClear()
            this.noActive = true
          }
          return true
        } else {
          if (isLearningStateEpochCurrent(epoch)) this.error = e?.message ?? 'failed'
          return false
        }
      } finally {
        if (isLearningStateEpochCurrent(epoch)) this.loading = false
      }
    },
    async fetchEvents() {
      const epoch = captureLearningStateEpoch()
      const planId = this.plan?.plan_id
      if (!planId) {
        if (isLearningStateEpochCurrent(epoch)) this.events = []
        return
      }
      const events = await getPlanEvents(planId)
      if (isLearningStateEpochCurrent(epoch)) this.events = events
    },
    async toggleMilestone(milestoneId: string, done: boolean) {
      if (!this.plan) return
      const epoch = captureLearningStateEpoch()
      this.updatingMilestoneId = milestoneId
      this.error = null
      this.lastValidationHint = null
      try {
        const result = await patchMilestoneDone(this.plan.plan_id, milestoneId, done)
        if (!isLearningStateEpochCurrent(epoch)) return
        this.plan = result.plan
        this.lastValidationHint = result.validation_hint
        this.events = [result.event, ...this.events.filter(e => e.id !== result.event.id)].slice(0, 20)
      } catch (e: any) {
        if (!isLearningStateEpochCurrent(epoch)) return
        this.error = e?.message ?? 'failed'
      } finally {
        if (isLearningStateEpochCurrent(epoch)) this.updatingMilestoneId = null
      }
    },
    setMindmap(mermaid: string) {
      this.mindmapMermaid = mermaid
    },
  },
})
