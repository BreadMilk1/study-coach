import { defineStore } from 'pinia'
import { getDocuments, type DocumentDto } from '../lib/api'
import { captureLearningStateEpoch, isLearningStateEpochCurrent } from '../lib/dataLifecycle'

interface DocsState {
  docs: DocumentDto[]
  loading: boolean
  error: string | null
}

export const useDocuments = defineStore('documents', {
  state: (): DocsState => ({ docs: [], loading: false, error: null }),
  getters: {
    totalChunks: (s) => s.docs.reduce((sum, d) => sum + d.chunks_count, 0),
    isEmpty: (s) => s.docs.length === 0 || s.docs.every(d => d.chunks_count === 0),
  },
  actions: {
    resetAfterDataClear() {
      this.docs = []
      this.loading = false
      this.error = null
    },
    async fetch() {
      const epoch = captureLearningStateEpoch()
      this.loading = true
      this.error = null
      try {
        const docs = await getDocuments()
        if (isLearningStateEpochCurrent(epoch)) this.docs = docs
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
