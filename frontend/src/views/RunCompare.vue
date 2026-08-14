<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute } from 'vue-router'

import { compareRuns } from '../lib/evalApi'
import type { CompareResponse } from '../lib/evalContracts'
import {
  compatibilityBadge,
  deltaCaption,
  shouldShowScoreDelta,
} from '../lib/learningRunPresentation'

const route = useRoute()
const result = ref<CompareResponse | null>(null)
const error = ref('')

const left = computed(() => String(route.query.left ?? ''))
const right = computed(() => String(route.query.right ?? ''))
const showDelta = computed(() => result.value ? shouldShowScoreDelta(result.value) : false)

async function load() {
  error.value = ''
  result.value = null
  if (!left.value || !right.value) return
  try {
    result.value = await compareRuns(left.value, right.value)
  } catch (reason) {
    error.value = reason instanceof Error ? reason.message : 'evaluation request failed'
  }
}

onMounted(() => {
  void load()
})

watch([left, right], () => {
  void load()
})
</script>

<template>
  <div class="h-full overflow-y-auto overflow-x-hidden p-6 md:p-8">
    <div class="max-w-4xl mx-auto min-w-0">
      <h1 class="text-3xl font-bold tracking-tight mb-6">{{ $t('runLab.compare') }}</h1>
      <p v-if="error" class="text-sm text-danger">{{ error }}</p>
      <section v-else-if="result" class="rounded-lg border border-border bg-surface p-4 space-y-4">
        <p class="text-sm">
          {{ compatibilityBadge(result.compatibility) }}
          · {{ result.left.variant_id }} vs {{ result.right.variant_id }}
        </p>
        <p class="text-sm text-fg-muted">{{ deltaCaption(result.scope) }}</p>
        <dl v-if="showDelta && result.delta" class="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
          <div v-for="(entry, key) in result.delta" :key="String(key)">
            <dt class="text-fg-dim">{{ key }}</dt>
            <dd class="font-mono">{{ entry.left }} → {{ entry.right }} ({{ entry.delta }})</dd>
          </div>
        </dl>
        <ul v-if="result.reasons.length" class="text-sm text-fg-muted space-y-1">
          <li v-for="reason in result.reasons" :key="reason">{{ reason }}</li>
        </ul>
      </section>
    </div>
  </div>
</template>
