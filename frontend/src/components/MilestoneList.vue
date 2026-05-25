<script setup lang="ts">
import { computed } from 'vue'
import { CheckCircle2, AlertCircle, AlertTriangle, Circle } from 'lucide-vue-next'
import type { MilestoneDto } from '../lib/api'

const props = defineProps<{ milestones: MilestoneDto[] }>()

function statusOf(m: MilestoneDto): 'success' | 'warning' | 'danger' | 'neutral' {
  if (m.done) return 'success'
  if (!m.due_at) return 'neutral'
  const due = new Date(m.due_at).getTime()
  const now = Date.now()
  const dayMs = 86_400_000
  if (due < now - dayMs) return 'danger'           // overdue (yesterday or earlier)
  if (due < now + dayMs) return 'warning'          // due today
  return 'neutral'
}

const rows = computed(() =>
  props.milestones.map(m => ({ m, status: statusOf(m) })),
)

const iconFor = { success: CheckCircle2, warning: AlertCircle, danger: AlertTriangle, neutral: Circle }
const colorFor = {
  success: 'text-success',
  warning: 'text-warning',
  danger: 'text-danger',
  neutral: 'text-fg-muted',
}
</script>

<template>
  <ul class="flex flex-col gap-2">
    <li v-for="(r, i) in rows" :key="i"
        class="flex items-start gap-3 rounded-lg border border-border bg-surface p-3">
      <component :is="iconFor[r.status]" :class="['w-5 h-5 mt-0.5 shrink-0', colorFor[r.status]]" />
      <div class="flex-1 min-w-0">
        <div class="text-sm font-medium" :class="r.m.done ? 'line-through text-fg-muted' : ''">
          {{ r.m.title }}
        </div>
        <div class="mt-1 flex gap-2 text-xs text-fg-muted">
          <span v-if="r.m.topic"
                class="font-mono px-2 py-0.5 rounded-md bg-primary-bg text-primary">
            {{ r.m.topic }}
          </span>
          <span v-if="r.m.due_at" class="font-mono">due {{ new Date(r.m.due_at).toLocaleDateString() }}</span>
        </div>
      </div>
    </li>
  </ul>
</template>
