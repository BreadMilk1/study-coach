<script setup lang="ts">
import { computed } from 'vue'

import type { QualityVerdict, ScoreSetDetail } from '../../lib/evalContracts'
import { displayVerdict, findingDisplay, verdictLabel } from '../../lib/learningRunPresentation'

const props = defineProps<{
  scoreSets: ScoreSetDetail[]
  baseline?: string
}>()

const latest = computed(() => props.scoreSets[props.scoreSets.length - 1] ?? null)

const citationHardGateFailed = computed(() => {
  const findings = latest.value?.findings
  if (!Array.isArray(findings)) return false
  return findings.some((item) => {
    if (!item || typeof item !== 'object') return false
    const record = item as Record<string, unknown>
    return record.code === 'citation_hard_gate' || record.gate === 'citation'
  })
})

const shownVerdict = computed<QualityVerdict>(() => displayVerdict({
  citationHardGateFailed: citationHardGateFailed.value,
  quality_verdict: latest.value?.quality_verdict,
}))
</script>

<template>
  <section class="min-w-0 overflow-y-auto p-4 md:p-6">
    <h2 class="text-xs uppercase tracking-wider text-fg-dim mb-3">{{ $t('runLab.verdict') }}</h2>
    <p class="text-lg font-semibold mb-4">{{ verdictLabel(shownVerdict) }}</p>
    <dl v-if="latest?.aggregate_scores" class="grid grid-cols-2 gap-2 text-sm mb-6">
      <div v-for="(value, key) in latest.aggregate_scores" :key="String(key)">
        <dt class="text-fg-dim">{{ key }}</dt>
        <dd class="font-mono">{{ value }}</dd>
      </div>
    </dl>
    <h3 class="text-xs uppercase tracking-wider text-fg-dim mb-2">{{ $t('runLab.findings') }}</h3>
    <div v-for="set in scoreSets" :key="set.score_set_id" class="mb-4">
      <p class="text-xs font-mono text-fg-dim mb-1">
        {{ set.scorer_version }} · {{ set.status }} · {{ set.quality_verdict }}
      </p>
      <ul v-if="Array.isArray(set.findings) && set.findings.length" class="space-y-1 text-sm text-fg-muted">
        <li v-for="(finding, index) in set.findings" :key="index">{{ findingDisplay(finding) }}</li>
      </ul>
      <p v-else class="text-sm text-fg-dim">—</p>
    </div>
    <p v-if="baseline" class="text-sm text-fg-muted">{{ baseline }}</p>
  </section>
</template>
