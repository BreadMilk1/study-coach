<script setup lang="ts">
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { RotateCcw } from 'lucide-vue-next'
import { markMistakeUnderstood, type MistakeDueDto } from '../lib/api'
import { useMistakes } from '../stores/mistakes'
import { useMastery } from '../stores/mastery'

const props = defineProps<{ row: MistakeDueDto }>()
const router = useRouter()

const truncated = computed(() =>
  props.row.question.prompt.length > 120
    ? props.row.question.prompt.slice(0, 120) + '…'
    : props.row.question.prompt
)

const nextReview = computed(() => {
  const due = new Date(props.row.due_at)
  const diffMs = due.getTime() - Date.now()
  const days = Math.round(diffMs / 86_400_000)
  if (days < 0) return `overdue by ${-days}d`
  if (days === 0) return 'due today'
  return `due in ${days}d`
})

function redo() {
  router.push({ path: '/quiz', query: { mistake_id: props.row.mistake_id } })
}

async function markUnderstood(id: string) {
  await markMistakeUnderstood(id)
  useMistakes().fetch()
  useMastery().fetch()
}
</script>

<template>
  <div class="rounded-lg border border-border bg-surface p-4 flex items-start gap-4">
    <div class="flex-1 min-w-0">
      <p class="text-sm">{{ truncated }}</p>
      <div class="mt-2 flex gap-2 text-xs">
        <span class="font-mono px-2 py-0.5 rounded-md bg-primary-bg text-primary">
          {{ row.topic_name }}
        </span>
        <span class="font-mono text-fg-muted">{{ nextReview }}</span>
        <span class="font-mono text-fg-dim">interval {{ row.srs_interval_days }}d · ease {{ row.srs_ease.toFixed(2) }}</span>
      </div>
    </div>
    <div class="flex items-center gap-2">
      <button @click="markUnderstood(row.mistake_id)"
              class="text-xs text-fg-muted hover:text-success transition-colors">
        {{ $t('mistakes.markUnderstood') }}
      </button>
      <button @click="redo"
              class="rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-white hover:bg-primary-2 inline-flex items-center gap-1.5 transition-colors">
        <RotateCcw class="w-4 h-4" /> {{ $t('mistakes.redo') }}
      </button>
    </div>
  </div>
</template>
