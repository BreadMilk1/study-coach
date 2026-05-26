<script setup lang="ts">
import { computed } from 'vue'
import { CheckCircle2, AlertCircle, AlertTriangle, Circle } from 'lucide-vue-next'
import type { MilestoneDto } from '../lib/api'

const props = defineProps<{
  milestones: MilestoneDto[]
  planId?: string
  updatingMilestoneId?: string | null
}>()

const emit = defineEmits<{
  toggle: [milestone: MilestoneDto]
  validate: [milestone: MilestoneDto]
  refresh: []
}>()

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

function masteryLabel(m: MilestoneDto): string | null {
  if (m.mastery_score === null || m.mastery_score === undefined) return null
  return `${Math.round(m.mastery_score * 100)}% mastery`
}

const rows = computed(() =>
  props.milestones.map(m => ({ m, status: statusOf(m), mastery: masteryLabel(m) })),
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
    <li v-for="r in rows" :key="r.m.id ?? r.m.title"
        class="flex items-start gap-3 rounded-lg border border-border bg-surface p-3">
      <button type="button"
              :disabled="!r.m.id || props.updatingMilestoneId === r.m.id"
              class="mt-0.5 shrink-0 disabled:opacity-40"
              :aria-label="r.m.done ? 'Reopen milestone' : 'Complete milestone'"
              @click="emit('toggle', r.m)">
        <component :is="iconFor[r.status]" :class="['w-5 h-5', colorFor[r.status]]" />
      </button>
      <div class="flex-1 min-w-0">
        <div class="text-sm font-medium" :class="r.m.done ? 'line-through text-fg-muted' : ''">
          {{ r.m.title }}
        </div>
        <div class="mt-1 flex flex-wrap gap-2 text-xs text-fg-muted">
          <span v-if="r.m.topic"
                class="font-mono px-2 py-0.5 rounded-md bg-primary-bg text-primary">
            {{ r.m.topic }}
          </span>
          <span v-if="r.mastery" class="font-mono px-2 py-0.5 rounded-md bg-bg text-fg-muted">
            {{ r.mastery }}
          </span>
          <span v-if="r.m.due_at" class="font-mono">due {{ new Date(r.m.due_at).toLocaleDateString() }}</span>
          <span v-if="r.m.completed_at" class="font-mono text-success">
            completed {{ new Date(r.m.completed_at).toLocaleDateString() }}
          </span>
          <button v-if="r.m.validation_recommended"
                  type="button"
                  class="text-xs text-warning underline"
                  @click="emit('validate', r.m)">
            Validate with quiz
          </button>
        </div>
      </div>
    </li>
  </ul>
</template>
