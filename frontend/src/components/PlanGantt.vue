<script setup lang="ts">
import type { MilestoneDto } from '../lib/api'

defineProps<{ milestones: MilestoneDto[] }>()

function dotColor(m: MilestoneDto): string {
  if (m.done) return 'bg-success border-success'
  if (m.due_at && new Date(m.due_at) <= new Date()) return 'bg-warning border-warning'
  return 'bg-surface-2 border-border-strong'
}
</script>

<template>
  <div class="relative pl-6 border-l-2 border-border ml-2">
    <div v-for="m in milestones" :key="m.id ?? m.title" class="relative pb-5 last:pb-0">
      <div class="absolute -left-[13px] top-1.5 w-6 h-6 rounded-full border-2 border-surface"
           :class="dotColor(m)" />
      <div class="rounded-md bg-surface-2 border border-border p-3 hover:bg-surface transition-colors">
        <div class="flex items-center gap-2">
          <span v-if="m.done" class="text-[10px] font-mono uppercase tracking-wider text-success">Done</span>
          <span class="text-sm font-medium text-fg">{{ m.title }}</span>
        </div>
        <div v-if="m.due_at" class="text-xs text-fg-muted mt-1">{{ m.due_at }}</div>
      </div>
    </div>
    <div v-if="milestones.length === 0" class="text-sm text-fg-muted py-4">No milestones yet</div>
  </div>
</template>
