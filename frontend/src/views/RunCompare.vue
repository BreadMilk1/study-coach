<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute } from 'vue-router'

import { compareRuns, getRunDetail } from '../lib/evalApi'
import type { CompareResponse, RunDetail, ScoreSetDetail } from '../lib/evalContracts'
import {
  compatibilityBadge,
  deltaCaption,
  findingDisplay,
  shouldShowScoreDelta,
  verdictLabel,
} from '../lib/learningRunPresentation'

const route = useRoute()
const result = ref<CompareResponse | null>(null)
const leftDetail = ref<RunDetail | null>(null)
const rightDetail = ref<RunDetail | null>(null)
const error = ref('')

const left = computed(() => String(route.query.left ?? ''))
const right = computed(() => String(route.query.right ?? ''))
const showDelta = computed(() => result.value ? shouldShowScoreDelta(result.value) : false)

function preferredScoreSet(detail: RunDetail | null): ScoreSetDetail | null {
  if (!detail?.score_sets.length) return null
  return detail.score_sets.find(item => item.scorer_version === 'hybrid-v1')
    ?? detail.score_sets[detail.score_sets.length - 1]
    ?? null
}

function answerText(detail: RunDetail | null): string {
  const artifact = detail?.candidate_artifact
  return artifact && typeof artifact.answer === 'string' ? artifact.answer : ''
}

const leftScore = computed(() => preferredScoreSet(leftDetail.value))
const rightScore = computed(() => preferredScoreSet(rightDetail.value))

async function load() {
  error.value = ''
  result.value = null
  leftDetail.value = null
  rightDetail.value = null
  if (!left.value || !right.value) return
  try {
    result.value = await compareRuns(left.value, right.value)
    const [leftRun, rightRun] = await Promise.all([
      getRunDetail(left.value),
      getRunDetail(right.value),
    ])
    leftDetail.value = leftRun
    rightDetail.value = rightRun
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
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <article class="min-w-0 space-y-2">
            <h2 class="text-sm font-semibold">{{ result.left.variant_id }}</h2>
            <p class="text-sm">{{ leftScore ? verdictLabel(leftScore.quality_verdict) : '—' }}</p>
            <p class="text-sm text-fg-muted whitespace-pre-wrap break-words">{{ answerText(leftDetail) || '—' }}</p>
            <ul v-if="Array.isArray(leftScore?.findings) && leftScore.findings.length" class="text-sm text-fg-muted space-y-1">
              <li v-for="(finding, index) in leftScore.findings" :key="index">{{ findingDisplay(finding) }}</li>
            </ul>
          </article>
          <article class="min-w-0 space-y-2">
            <h2 class="text-sm font-semibold">{{ result.right.variant_id }}</h2>
            <p class="text-sm">{{ rightScore ? verdictLabel(rightScore.quality_verdict) : '—' }}</p>
            <p class="text-sm text-fg-muted whitespace-pre-wrap break-words">{{ answerText(rightDetail) || '—' }}</p>
            <ul v-if="Array.isArray(rightScore?.findings) && rightScore.findings.length" class="text-sm text-fg-muted space-y-1">
              <li v-for="(finding, index) in rightScore.findings" :key="index">{{ findingDisplay(finding) }}</li>
            </ul>
          </article>
        </div>
      </section>
    </div>
  </div>
</template>
