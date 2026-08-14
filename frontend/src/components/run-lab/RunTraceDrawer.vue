<script setup lang="ts">
import { ref } from 'vue'

import { usageLabel } from '../../lib/learningRunPresentation'

defineProps<{
  trace?: unknown
  usage?: unknown
}>()

const open = ref(false)
</script>

<template>
  <section class="border-t border-border bg-surface">
    <button
      type="button"
      class="w-full px-4 py-3 text-left text-sm text-fg-muted hover:bg-white/5"
      @click="open = !open"
    >
      {{ $t('runLab.trace') }} · {{ usageLabel(usage) }}
    </button>
    <div v-if="open" class="max-h-64 overflow-y-auto overflow-x-hidden px-4 pb-4">
      <p class="text-xs text-fg-dim mb-2">usage: {{ usageLabel(usage) }}</p>
      <pre class="text-xs font-mono whitespace-pre-wrap break-words text-fg-muted">{{ JSON.stringify(trace ?? null, null, 2) }}</pre>
    </div>
  </section>
</template>
