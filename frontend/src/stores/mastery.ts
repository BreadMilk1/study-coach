import { defineStore } from 'pinia'
import { getMastery, type MasteryDto } from '../lib/api'

interface MasteryState {
  data: MasteryDto
  loading: boolean
  error: string | null
}

export const useMastery = defineStore('mastery', {
  state: (): MasteryState => ({
    data: { scores: [], weak_topics: [], overdue_milestones_count: 0 },
    loading: false,
    error: null,
  }),
  actions: {
    async fetch() {
      this.loading = true
      this.error = null
      try { this.data = await getMastery() }
      catch (e: any) { this.error = e?.message ?? 'failed' }
      finally { this.loading = false }
    },
  },
})
