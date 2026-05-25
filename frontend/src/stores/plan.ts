import { defineStore } from 'pinia'
import { getCurrentPlan, type PlanCurrentDto } from '../lib/api'

interface PlanState {
  plan: PlanCurrentDto | null
  loading: boolean
  error: string | null
  noActive: boolean        // true if backend returned 404
  mindmapMermaid: string | null  // set by A3
}

export const usePlan = defineStore('plan', {
  state: (): PlanState => ({
    plan: null,
    loading: false,
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
      } catch (e: any) {
        if (e?.status === 404) {
          this.noActive = true
          this.plan = null
        } else {
          this.error = e?.message ?? 'failed'
        }
      } finally {
        this.loading = false
      }
    },
    setMindmap(mermaid: string) {
      this.mindmapMermaid = mermaid
    },
  },
})
