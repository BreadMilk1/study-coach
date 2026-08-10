import { defineStore } from 'pinia'
import { getMastery, type MasteryDto } from '../lib/api'
import { captureLearningStateEpoch, isLearningStateEpochCurrent } from '../lib/dataLifecycle'

interface MasteryState {
  data: MasteryDto
  loading: boolean
  error: string | null
}

export const useMastery = defineStore('mastery', {
  state: (): MasteryState => ({
    data: { scores: [], weak_topics: [], overdue_milestones_count: 0, streak_days: 0, coverage: 0 },
    loading: false,
    error: null,
  }),
  actions: {
    resetAfterDataClear() {
      this.data = { scores: [], weak_topics: [], overdue_milestones_count: 0, streak_days: 0, coverage: 0 }
      this.loading = false
      this.error = null
    },
    async fetch() {
      const epoch = captureLearningStateEpoch()
      this.loading = true
      this.error = null
      try {
        const data = await getMastery()
        if (isLearningStateEpochCurrent(epoch)) this.data = data
        return true
      } catch (e: any) {
        if (isLearningStateEpochCurrent(epoch)) this.error = e?.message ?? 'failed'
        return false
      } finally {
        if (isLearningStateEpochCurrent(epoch)) this.loading = false
      }
    },
  },
})
