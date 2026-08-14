<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'

import RunContractBar from '../components/run-lab/RunContractBar.vue'
import RunEvidenceTimeline from '../components/run-lab/RunEvidenceTimeline.vue'
import RunScorePanel from '../components/run-lab/RunScorePanel.vue'
import RunTraceDrawer from '../components/run-lab/RunTraceDrawer.vue'
import { getRunDetail, listRuns } from '../lib/evalApi'
import type { EvalConnectionSnapshot, RunDetail, RunSummary } from '../lib/evalContracts'
import { useLearningRuns } from '../stores/learningRuns'
import { useSettings } from '../stores/settings'

const route = useRoute()
const router = useRouter()
const settings = useSettings()
const runs = useLearningRuns()
const detail = ref<RunDetail | null>(null)
const history = ref<RunSummary[]>([])
const loadError = ref('')

const runId = computed(() => String(route.params.runId ?? ''))
const attached = computed(() => runs.activeRunId === runId.value)
const manifest = computed(() => detail.value?.manifest ?? {})
const artifact = computed(() => detail.value?.candidate_artifact ?? {})

const comparableHistoricalId = computed(() => {
  const current = detail.value?.summary
  if (!current) return ''
  const match = history.value.find(run => (
    run.run_id !== current.run_id
    && run.task_case_id === current.task_case_id
    && run.lifecycle === 'finished'
    && run.outcome === 'success'
  ))
  return match?.run_id ?? ''
})

function connection(): EvalConnectionSnapshot {
  return {
    provider: settings.provider,
    model: settings.model,
    apiKey: settings.apiKey || undefined,
    baseUrl: settings.baseUrl || undefined,
  }
}

async function load() {
  loadError.value = ''
  try {
    detail.value = await getRunDetail(runId.value)
    history.value = await listRuns()
  } catch (reason) {
    loadError.value = reason instanceof Error ? reason.message : 'evaluation request failed'
  }
}

function openHistorical() {
  const historicalId = runs.openHistoricalFallback(comparableHistoricalId.value)
  if (!historicalId) return
  void router.push({ name: 'run-detail', params: { runId: historicalId } })
}

async function rescore() {
  await runs.startRescore(runId.value, { scorer_version: 'hybrid-v2' }, connection())
}

onMounted(() => {
  void load()
})

watch(runId, () => {
  void load()
})
</script>

<template>
  <div class="h-full overflow-x-hidden overflow-y-auto md:overflow-hidden flex flex-col min-w-0">
    <p v-if="loadError" class="p-4 text-sm text-danger">{{ loadError }}</p>
    <RunContractBar
      :task="String(manifest.task_case_id ?? detail?.summary.task_case_id ?? '')"
      :corpus="String(manifest.corpus_snapshot_id ?? '')"
      :prompt="String(manifest.prompt_version ?? detail?.summary.variant_id ?? '')"
      :model="String(manifest.model ?? '')"
      :run-profile="detail?.summary.run_profile"
      :budget="manifest.budget ? JSON.stringify(manifest.budget) : ''"
    />
    <div class="flex-1 min-w-0 grid grid-cols-1 md:grid-cols-2 md:overflow-hidden">
      <RunScorePanel
        class="order-2 md:order-2"
        :score-sets="detail?.score_sets ?? []"
      />
      <RunEvidenceTimeline
        class="order-3 md:order-1"
        :events="attached ? runs.events : []"
        :answer="typeof artifact.answer === 'string' ? artifact.answer : undefined"
        :citations="Array.isArray(artifact.citations) ? artifact.citations : undefined"
        :evidence="Array.isArray(artifact.exact_evidence) ? artifact.exact_evidence : undefined"
      />
    </div>
    <RunTraceDrawer
      class="order-4"
      :trace="artifact.trace"
      :usage="artifact.usage"
    />
    <footer class="border-t border-border px-4 py-3 flex flex-wrap gap-2">
      <button
        v-if="runs.status === 'running' && attached"
        type="button"
        class="rounded-md border border-border px-3 py-2 text-sm"
        @click="runs.cancelActive()"
      >
        {{ $t('runLab.cancel') }}
      </button>
      <button
        type="button"
        class="rounded-md border border-border px-3 py-2 text-sm"
        :disabled="runs.status === 'running'"
        @click="rescore"
      >
        {{ $t('runLab.rescore') }}
      </button>
      <button
        v-if="runs.canOpenHistoricalFallback() && comparableHistoricalId"
        type="button"
        class="rounded-md bg-primary-bg px-3 py-2 text-sm"
        aria-label="Open comparable completed historical run"
        @click="openHistorical"
      >
        {{ $t('runLab.openCompletedHistoricalRun') }}
      </button>
    </footer>
  </div>
</template>
