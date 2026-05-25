import { defineStore } from 'pinia'
import { getMistakesDue, type MistakeDueDto } from '../lib/api'

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
    async fetch() {
      this.loading = true
      this.error = null
      try {
        this.items = await getMistakesDue(50, true)
      } catch (e: any) {
        this.error = e?.message ?? 'failed'
      } finally {
        this.loading = false
      }
    },
  },
})
