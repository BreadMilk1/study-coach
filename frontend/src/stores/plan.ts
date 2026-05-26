import { defineStore } from 'pinia'
import {
  getCurrentPlan,
  getPlanEvents,
  patchMilestoneDone,
  type PlanCurrentDto,
  type PlanEventDto,
  type ValidationHintDto,
} from '../lib/api'

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
    async fetch() {
      this.loading = true
      this.error = null
      this.noActive = false
      try {
        this.plan = await getCurrentPlan()
        await this.fetchEvents()
      } catch (e: any) {
        if (e?.status === 404) {
          this.noActive = true
          this.plan = null
          this.events = []
          this.lastValidationHint = null
        } else {
          this.error = e?.message ?? 'failed'
        }
      } finally {
        this.loading = false
      }
    },
    async fetchEvents() {
      if (!this.plan) {
        this.events = []
        return
      }
      this.events = await getPlanEvents(this.plan.plan_id)
    },
    async toggleMilestone(milestoneId: string, done: boolean) {
      if (!this.plan) return
      this.updatingMilestoneId = milestoneId
      this.error = null
      try {
        const result = await patchMilestoneDone(this.plan.plan_id, milestoneId, done)
        this.plan = result.plan
        this.lastValidationHint = result.validation_hint
        this.events = [result.event, ...this.events.filter(e => e.id !== result.event.id)].slice(0, 20)
      } catch (e: any) {
        this.error = e?.message ?? 'failed'
      } finally {
        this.updatingMilestoneId = null
      }
    },
    setMindmap(mermaid: string) {
      this.mindmapMermaid = mermaid
    },
  },
})
