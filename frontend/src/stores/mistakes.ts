import { defineStore } from 'pinia'
import { getMistakesDue, type MistakeDueDto } from '../lib/api'
import { captureLearningStateEpoch, isLearningStateEpochCurrent } from '../lib/dataLifecycle'

interface MistakesState {
  items: MistakeDueDto[]
  loading: boolean
  error: string | null
}

function isDue(row: MistakeDueDto): boolean {
  return new Date(row.due_at).getTime() <= Date.now()
}

export const useMistakes = defineStore('mistakes', {
  state: (): MistakesState => ({ items: [], loading: false, error: null }),
  getters: {
    due: (s) => s.items.filter(isDue),
  },
  actions: {
    resetAfterDataClear() {
      this.items = []
      this.loading = false
      this.error = null
    },
    async fetch(includeFuture = false) {
      const epoch = captureLearningStateEpoch()
      this.loading = true
      this.error = null
      try {
        const items = await getMistakesDue(50, includeFuture)
        if (isLearningStateEpochCurrent(epoch)) this.items = items
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
