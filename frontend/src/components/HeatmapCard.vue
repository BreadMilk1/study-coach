<script setup lang="ts">
import { onMounted } from 'vue'

import { useActivity } from '../stores/activity'

const activity = useActivity()

onMounted(() => {
  void activity.fetch()
})

function color(count: number): string {
  if (count === 0) return '#11162a'
  if (count <= 2) return '#4338ca'
  if (count <= 5) return '#6366f1'
  return '#818cf8'
}
</script>

<template>
  <div class="rounded-lg border border-border bg-surface p-4">
    <h3 class="text-sm font-medium text-fg-muted mb-3">{{ $t('overview.activity') }}</h3>
    <div class="flex gap-1 flex-wrap">
      <div v-for="d in activity.days" :key="d.date"
           class="w-3 h-3 rounded-sm"
           :style="{ background: color(d.count) }"
           :title="`${d.date}: ${d.count}`" />
    </div>
  </div>
</template>
