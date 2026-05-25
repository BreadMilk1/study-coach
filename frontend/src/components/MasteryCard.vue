<script setup lang="ts">
import type { MasteryScoreDto } from '../lib/api'
defineProps<{ scores: MasteryScoreDto[] }>()
function barColor(score: number) {
  if (score >= 0.7) return 'bg-success'
  if (score >= 0.4) return 'bg-warning'
  return 'bg-danger'
}
</script>
<template>
  <section class="rounded-lg border border-border bg-surface p-6">
    <h2 class="text-sm font-semibold mb-4 text-fg-muted uppercase tracking-wider">Top mastery</h2>
    <div v-if="scores.length === 0" class="text-fg-dim text-sm">No quizzes taken yet.</div>
    <div v-else class="flex flex-col gap-3">
      <div v-for="s in scores" :key="s.topic_id">
        <div class="flex justify-between text-xs font-mono mb-1">
          <span>{{ s.topic_name }}</span>
          <span class="text-fg-muted">{{ (s.score * 100).toFixed(0) }}%</span>
        </div>
        <div class="h-2 bg-bg rounded-full overflow-hidden">
          <div :class="['h-full transition-all', barColor(s.score)]"
               :style="{ width: (s.score * 100) + '%' }"></div>
        </div>
      </div>
    </div>
  </section>
</template>
