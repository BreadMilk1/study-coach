<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { RouterLink, useRoute } from 'vue-router'

import MarkdownAnswer from '../components/run-lab/MarkdownAnswer.vue'

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
const sides = computed(() => result.value ? [
  { position: 'left', variant: result.value.left.variant_id, detail: leftDetail.value, score: leftScore.value },
  { position: 'right', variant: result.value.right.variant_id, detail: rightDetail.value, score: rightScore.value },
] : [])

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
  <div class="h-full overflow-y-auto overflow-x-hidden p-5 md:p-8">
    <div class="max-w-6xl mx-auto min-w-0">
      <header class="mb-7">
        <RouterLink to="/run-lab" class="text-sm text-fg-muted hover:text-fg underline underline-offset-4">
          ← {{ $t('runLab.backToLab') }}
        </RouterLink>
        <div class="mt-5 flex flex-wrap items-center gap-3">
          <h1 class="text-3xl font-bold tracking-tight">{{ $t('runLab.compare') }}</h1>
          <span v-if="result" class="border border-border-strong px-2.5 py-1 text-xs font-semibold rounded">
            {{ compatibilityBadge(result.compatibility) }}
          </span>
        </div>
        <p v-if="result" class="mt-3 text-sm text-fg-muted">{{ deltaCaption(result.scope) }}</p>
      </header>

      <p v-if="error" role="alert" class="mb-5 text-sm text-danger">{{ error }}</p>
      <template v-if="result">
        <section v-if="!showDelta || result.reasons.length" class="mb-7 border-y border-border-strong py-4 text-sm">
          <p v-if="!showDelta" class="text-fg-muted">{{ $t('runLab.deltaUnavailable') }}</p>
          <details v-if="result.reasons.length" :class="{ 'mt-3': !showDelta }">
            <summary class="cursor-pointer text-fg font-medium">{{ $t('runLab.comparisonReasons') }} ({{ result.reasons.length }})</summary>
            <ul class="mt-3 list-disc pl-5 space-y-2 text-fg-muted break-words">
              <li v-for="reason in result.reasons" :key="reason">{{ reason }}</li>
            </ul>
          </details>
        </section>

        <section v-if="showDelta && result.delta" class="mb-8 min-w-0">
          <h2 class="mb-3 text-sm font-semibold">{{ $t('runLab.scoreDelta') }}</h2>
          <div class="overflow-x-auto">
            <table class="w-full text-left text-sm border-collapse">
              <thead>
                <tr class="border-y border-border-strong text-fg-muted">
                  <th class="py-3 pr-4 font-medium">{{ $t('runLab.dimension') }}</th>
                  <th class="py-3 px-3 font-medium">{{ result.left.variant_id }}</th>
                  <th class="py-3 px-3 font-medium">{{ result.right.variant_id }}</th>
                  <th class="py-3 pl-3 font-medium">{{ $t('runLab.delta') }}</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(entry, key) in result.delta" :key="String(key)" class="border-b border-border-strong">
                  <th scope="row" class="py-3 pr-4 font-normal break-words">{{ key }}</th>
                  <td class="py-3 px-3 font-mono">{{ entry.left }}</td>
                  <td class="py-3 px-3 font-mono">{{ entry.right }}</td>
                  <td class="py-3 pl-3 font-mono">{{ entry.delta }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        <div class="compare-columns grid grid-cols-1 md:grid-cols-2 border-t border-border-strong">
          <article v-for="side in sides" :key="side.position" class="compare-column min-w-0">
            <header class="py-4 border-b border-border-strong">
              <p class="text-xs uppercase tracking-wider text-fg-dim mb-1">{{ $t(`runLab.${side.position}Variant`) }}</p>
              <h2 class="text-lg font-semibold break-words">{{ side.variant }}</h2>
            </header>
            <section class="py-4 border-b border-border-strong">
              <h3 class="text-xs uppercase tracking-wider text-fg-dim mb-2">{{ $t('runLab.verdict') }}</h3>
              <p class="text-lg font-semibold">{{ side.score ? verdictLabel(side.score.quality_verdict) : $t(side.detail ? 'runLab.scoreUnavailable' : 'runLab.detailsLoading') }}</p>
              <dl v-if="side.score" class="mt-4 space-y-2 text-sm">
                <div class="flex flex-wrap justify-between gap-x-4 gap-y-1">
                  <dt class="text-fg-dim">{{ $t('runLab.scorer') }}</dt>
                  <dd class="font-mono break-all">{{ side.score.scorer_id }}</dd>
                </div>
                <div class="flex flex-wrap justify-between gap-x-4 gap-y-1">
                  <dt class="text-fg-dim">{{ $t('runLab.scorerVersion') }}</dt>
                  <dd class="font-mono">{{ side.score.scorer_version }}</dd>
                </div>
                <div class="flex flex-wrap justify-between gap-x-4 gap-y-1">
                  <dt class="text-fg-dim">{{ $t('runLab.scoringStatus') }}</dt>
                  <dd class="font-mono">{{ side.score.status }}</dd>
                </div>
              </dl>
            </section>
            <section class="py-4 border-b border-border-strong">
              <h3 class="text-xs uppercase tracking-wider text-fg-dim mb-4">{{ $t('runLab.answer') }}</h3>
              <MarkdownAnswer v-if="answerText(side.detail)" :answer="answerText(side.detail)" class="text-sm text-fg-muted" />
              <p v-else class="text-sm text-fg-dim">{{ $t(side.detail ? 'runLab.answerUnavailable' : 'runLab.detailsLoading') }}</p>
            </section>
            <section class="py-4">
              <h3 class="text-xs uppercase tracking-wider text-fg-dim mb-3">{{ $t('runLab.findings') }}</h3>
              <ul v-if="Array.isArray(side.score?.findings) && side.score.findings.length" class="list-disc pl-5 space-y-3 text-sm text-fg-muted break-words">
                <li v-for="(finding, index) in side.score.findings" :key="index">{{ findingDisplay(finding) }}</li>
              </ul>
              <p v-else class="text-sm text-fg-dim">{{ $t(!side.detail ? 'runLab.detailsLoading' : Array.isArray(side.score?.findings) ? 'runLab.noFindings' : 'runLab.findingsUnavailable') }}</p>
            </section>
          </article>
        </div>
      </template>
      <p v-else-if="!error" class="text-sm text-fg-muted">{{ $t(left && right ? 'runLab.comparisonLoading' : 'runLab.selectComparison') }}</p>
    </div>
  </div>
</template>

<style scoped>
.compare-column + .compare-column { border-top: 1px solid var(--color-border-strong); }
@media (min-width: 768px) {
  .compare-column { display: grid; grid-row: span 4; grid-template-rows: subgrid; }
  .compare-column:first-child { padding-right: 1.75rem; }
  .compare-column + .compare-column { border-top: 0; border-left: 1px solid var(--color-border-strong); padding-left: 1.75rem; }
}
</style>
