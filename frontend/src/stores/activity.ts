import { defineStore } from 'pinia'

import { getUserStats, type ActivityDayDto } from '../lib/api'
import { captureLearningStateEpoch, isLearningStateEpochCurrent } from '../lib/dataLifecycle'

interface ActivityState {
  days: ActivityDayDto[]
  loading: boolean
  error: string | null
}

export const useActivity = defineStore('activity', {
  state: (): ActivityState => ({ days: [], loading: false, error: null }),
  actions: {
    resetAfterDataClear() {
      this.days = []
      this.loading = false
      this.error = null
    },
    async fetch(): Promise<boolean> {
      const epoch = captureLearningStateEpoch()
      this.loading = true
      this.error = null
      try {
        const stats = await getUserStats()
        if (isLearningStateEpochCurrent(epoch)) this.days = stats.activity_daily
        return true
      } catch (error: any) {
        if (isLearningStateEpochCurrent(epoch)) this.error = error?.message ?? 'failed'
        return false
      } finally {
        if (isLearningStateEpochCurrent(epoch)) this.loading = false
      }
    },
  },
})
