<script setup lang="ts">
import type { LearningRunEvent } from '../../lib/evalContracts'

defineProps<{
  events: LearningRunEvent[]
  answer?: string
  citations?: unknown[]
  evidence?: unknown[]
}>()
</script>

<template>
  <section class="min-w-0 overflow-y-auto p-4 md:p-6 border-b md:border-b-0 md:border-r border-border">
    <h2 class="text-xs uppercase tracking-wider text-fg-dim mb-3">{{ $t('runLab.evidence') }}</h2>
    <ol class="space-y-2 text-sm mb-6">
      <li v-for="(event, index) in events" :key="`${event.type}-${index}`" class="font-mono text-fg-muted">
        {{ event.type }}
        <span v-if="'stage' in event"> · {{ event.stage }}</span>
      </li>
    </ol>
    <div v-if="answer" class="mb-4">
      <h3 class="text-xs text-fg-dim mb-1">Answer</h3>
      <p class="text-sm whitespace-pre-wrap break-words">{{ answer }}</p>
    </div>
    <div v-if="citations?.length" class="mb-4">
      <h3 class="text-xs text-fg-dim mb-1">Citations</h3>
      <pre class="text-xs font-mono whitespace-pre-wrap break-words text-fg-muted">{{ JSON.stringify(citations, null, 2) }}</pre>
    </div>
    <div v-if="evidence?.length">
      <h3 class="text-xs text-fg-dim mb-1">Exact retrieved chunks</h3>
      <pre class="text-xs font-mono whitespace-pre-wrap break-words text-fg-muted">{{ JSON.stringify(evidence, null, 2) }}</pre>
    </div>
  </section>
</template>
