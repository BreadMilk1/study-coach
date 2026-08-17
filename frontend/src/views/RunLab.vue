<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { RouterLink } from 'vue-router'

import { EvalApiError, listExperiments, listRuns } from '../lib/evalApi'
import type { EvalConnectionSnapshot, ExperimentSummary, RunSummary } from '../lib/evalContracts'
import { useLearningRuns } from '../stores/learningRuns'
import { useSettings } from '../stores/settings'

const FROZEN_CASES = [
  'tgqa-001', 'tgqa-002', 'tgqa-003', 'tgqa-004',
  'tgqa-005', 'tgqa-006', 'tgqa-007', 'tgqa-008',
  'tgqa-009', 'tgqa-010', 'tgqa-011', 'tgqa-012',
]

const settings = useSettings()
const runsStore = useLearningRuns()
const experiments = ref<ExperimentSummary[]>([])
const runs = ref<RunSummary[]>([])
const disabled = ref(false)
const error = ref('')
const filter = ref<'all' | 'fail' | 'inconclusive'>('all')
const selectedCase = ref(FROZEN_CASES[0])
const selectedVariant = ref('tutor-v3')

const experiment = computed(() => experiments.value[0] ?? null)

const filteredRuns = computed(() => {
  if (filter.value === 'fail') {
    return runs.value.filter(run => run.latest_score_set?.quality_verdict === 'fail')
  }
  if (filter.value === 'inconclusive') {
    return runs.value.filter(run => run.latest_score_set?.quality_verdict === 'inconclusive')
  }
  return runs.value
})

const regressionCount = computed(() => (
  runs.value.filter(run => run.latest_score_set?.quality_verdict === 'fail').length
))
const inconclusiveCount = computed(() => (
  runs.value.filter(run => run.latest_score_set?.quality_verdict === 'inconclusive').length
))

function connection(): EvalConnectionSnapshot {
  return {
    provider: settings.provider,
    model: settings.model,
    apiKey: settings.apiKey || undefined,
    baseUrl: settings.baseUrl || undefined,
  }
}

async function refresh() {
  disabled.value = false
  error.value = ''
  try {
    experiments.value = await listExperiments()
    runs.value = await listRuns()
    if (experiment.value?.variants.length) {
      selectedVariant.value = experiment.value.variants[0].variant_id
    }
  } catch (reason) {
    if (reason instanceof EvalApiError && reason.code === 'evaluation_disabled') {
      disabled.value = true
      return
    }
    error.value = reason instanceof Error ? reason.message : 'evaluation request failed'
  }
}

async function startLive() {
  await runsStore.start({
    experiment_id: experiment.value?.experiment_id ?? 'tutor-prompt-regression-v1',
    task_case_id: selectedCase.value,
    variant_id: selectedVariant.value,
    run_profile: 'evaluation',
  }, connection())
}

onMounted(() => {
  void refresh()
})
</script>

<template>
  <div class="h-full overflow-y-auto overflow-x-hidden p-6 md:p-8">
    <div class="max-w-6xl mx-auto min-w-0">
      <header class="mb-6">
        <h1 class="text-3xl font-bold tracking-tight">{{ $t('runLab.title') }}</h1>
        <p class="text-sm text-fg-muted mt-1">{{ $t('runLab.subtitle') }}</p>
      </header>

      <div v-if="disabled" class="rounded-lg border border-warning/40 bg-warning-bg p-4 text-sm mb-6">
        {{ $t('runLab.disabled') }}
      </div>
      <p v-if="!disabled && error" class="text-sm text-danger mb-6">{{ error }}</p>
      <p v-if="!disabled && runsStore.error" class="text-sm text-danger mb-6">
        {{ runsStore.error.message }}
      </p>

      <section v-if="experiment" class="rounded-lg border border-border bg-surface p-4 mb-6">
        <p class="text-xs uppercase tracking-wider text-fg-dim">{{ $t('runLab.axis') }}</p>
        <p class="font-mono text-sm mb-3">{{ experiment.experiment_axes.join(', ') }}</p>
        <p class="text-xs uppercase tracking-wider text-fg-dim">{{ $t('runLab.suite') }}</p>
        <p class="text-sm text-fg-muted">
          {{ experiment.case_counts.answerable ?? 0 }} answerable ·
          {{ experiment.case_counts.multi_evidence ?? 0 }} multi-evidence ·
          {{ experiment.case_counts.expected_refusal ?? 0 }} expected-refusal
        </p>
        <p class="text-sm mt-3">
          {{ $t('runLab.regressions') }}:
          <span class="font-mono">{{ regressionCount }}</span>
          · {{ $t('runLab.inconclusive') }}:
          <span class="font-mono">{{ inconclusiveCount }}</span>
        </p>
      </section>

      <section v-if="!disabled" class="rounded-lg border border-border bg-surface p-4 mb-6">
        <div class="flex flex-wrap gap-3 items-end">
          <label class="text-sm">
            <span class="block text-fg-dim mb-1">{{ $t('runLab.liveCase') }}</span>
            <select v-model="selectedCase" class="bg-bg border border-border rounded-md px-2 py-1">
              <option v-for="caseId in FROZEN_CASES" :key="caseId" :value="caseId">{{ caseId }}</option>
            </select>
          </label>
          <label class="text-sm">
            <span class="block text-fg-dim mb-1">{{ $t('runLab.liveVariant') }}</span>
            <select v-model="selectedVariant" class="bg-bg border border-border rounded-md px-2 py-1">
              <option
                v-for="variant in experiment?.variants ?? [{ variant_id: 'tutor-v3' }]"
                :key="variant.variant_id"
                :value="variant.variant_id"
              >
                {{ variant.variant_id }}
              </option>
            </select>
          </label>
          <button
            type="button"
            class="rounded-md bg-primary px-4 py-2 text-sm text-white hover:bg-primary-2"
            :disabled="runsStore.status === 'running'"
            @click="startLive"
          >
            {{ $t('runLab.start') }}
          </button>
          <button
            v-if="runsStore.status === 'running'"
            type="button"
            class="rounded-md border border-border px-4 py-2 text-sm"
            @click="runsStore.cancelActive()"
          >
            {{ $t('runLab.cancel') }}
          </button>
        </div>
      </section>

      <section>
        <div class="flex items-center gap-2 mb-3 text-sm">
          <span class="text-fg-dim">{{ $t('runLab.history') }}</span>
          <button type="button" class="px-2 py-1 rounded-md" :class="filter === 'all' ? 'bg-primary-bg' : ''" @click="filter = 'all'">{{ $t('runLab.filterAll') }}</button>
          <button type="button" class="px-2 py-1 rounded-md" :class="filter === 'fail' ? 'bg-primary-bg' : ''" @click="filter = 'fail'">{{ $t('runLab.filterFail') }}</button>
          <button type="button" class="px-2 py-1 rounded-md" :class="filter === 'inconclusive' ? 'bg-primary-bg' : ''" @click="filter = 'inconclusive'">{{ $t('runLab.filterInconclusive') }}</button>
        </div>
        <p v-if="!filteredRuns.length" class="text-sm text-fg-dim">{{ $t('runLab.empty') }}</p>
        <ul v-else class="space-y-2">
          <li v-for="run in filteredRuns" :key="run.run_id" class="rounded-md border border-border px-3 py-2 text-sm flex flex-wrap gap-2 justify-between min-w-0">
            <RouterLink :to="`/run-lab/runs/${run.run_id}`" class="font-mono truncate hover:text-primary-2">
              {{ run.task_case_id }} · {{ run.variant_id }}
            </RouterLink>
            <span class="text-fg-muted">{{ run.lifecycle }} · {{ run.latest_score_set?.quality_verdict ?? '—' }}</span>
          </li>
        </ul>
      </section>
    </div>
  </div>
</template>
