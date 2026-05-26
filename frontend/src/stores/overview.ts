import { defineStore } from 'pinia'
import { computed } from 'vue'
import { useMastery } from './mastery'
import { usePlan } from './plan'
import { useMistakes } from './mistakes'
import { useDocuments } from './documents'

export const useOverview = defineStore('overview', () => {
  const m = useMastery()
  const p = usePlan()
  const x = useMistakes()
  const d = useDocuments()

  async function fetchAll() {
    await Promise.all([m.fetch(), x.fetch(true), d.fetch(), p.fetch().catch(() => {})])
  }

  const topMastery = computed(() =>
    [...m.data.scores].sort((a, b) => b.score - a.score).slice(0, 5)
  )
  const nextMilestone = computed(() => {
    if (!p.plan) return null
    const pending = p.plan.milestones.filter(s => !s.done && s.due_at)
    pending.sort((a, b) => (a.due_at ?? '').localeCompare(b.due_at ?? ''))
    return pending[0] ?? null
  })

  return { fetchAll, topMastery, nextMilestone }
})
