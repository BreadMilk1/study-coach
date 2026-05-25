import { defineStore } from 'pinia'
import { getDocuments, type DocumentDto } from '../lib/api'

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
    async fetch() {
      this.loading = true
      this.error = null
      try {
        this.docs = await getDocuments()
      } catch (e: any) {
        this.error = e?.message ?? 'failed'
      } finally {
        this.loading = false
      }
    },
  },
})
