<script setup lang="ts">
import { ref, watch } from 'vue'
import { AlertTriangle } from 'lucide-vue-next'

import { useSettings } from '../stores/settings'
import { useDataLifecycle } from '../stores/dataLifecycle'
import { checkToolCapable, pingModel, type DataCounts } from '../lib/api'
import { useI18n } from 'vue-i18n'

const { locale } = useI18n()
const lifecycle = useDataLifecycle()
const s = useSettings()
const toolTesting = ref(false)
const toolNote = ref('')
const pingTesting = ref(false)
const pingResult = ref<{ ok: boolean; note: string; latency_ms: number } | null>(null)

const countKeys: (keyof DataCounts)[] = [
  'documents',
  'source_chunks',
  'vectors',
  'chat_sessions',
  'messages',
  'citations',
  'goals',
  'topics',
  'questions',
  'mistakes',
  'mastery',
  'plans',
  'plan_milestones',
  'plan_events',
  'users',
]

watch(() => lifecycle.phase, (phase) => {
  if (phase === 'ready') void lifecycle.refreshSummary()
}, { immediate: true, flush: 'post' })

function save() {
  s.persist()
}

function onLanguageChange() {
  locale.value = s.language
  s.persist()
}

async function runPing() {
  pingTesting.value = true
  pingResult.value = null
  try {
    const result = await pingModel(s.$state)
    pingResult.value = result
  } catch (e) {
    pingResult.value = { ok: false, note: `Error: ${e}`, latency_ms: 0 }
  } finally {
    pingTesting.value = false
  }
}

async function runToolCheck() {
  toolTesting.value = true
  toolNote.value = ''
  try {
    const result = await checkToolCapable(s.$state)
    s.setToolCapable(result.tool_capable)
    toolNote.value = result.note
  } catch (e) {
    toolNote.value = `Error: ${e}`
  } finally {
    toolTesting.value = false
  }
}
</script>

<template>
  <div class="p-8 h-full overflow-y-auto">
    <h2 class="text-xl font-semibold mb-4">Settings</h2>
    <p class="text-white/60 text-sm mb-6">
      Bring-your-own-key. API key is stored in your browser localStorage; the server never sees nor persists it.
    </p>
    <p class="text-white/60 text-sm mb-6">{{ $t('settings.localFirst') }}</p>
    <div class="space-y-4">
      <label class="block">
        <span class="text-sm text-white/70">Provider</span>
        <select v-model="s.provider"
                class="mt-1 block w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 outline-none focus:border-indigo-400/40">
          <option value="ollama">Ollama (local)</option>
          <option value="openai">OpenAI</option>
          <option value="anthropic">Anthropic</option>
          <option value="gemini">Google Gemini</option>
        </select>
      </label>

      <label class="block">
        <span class="text-sm text-white/70">Model</span>
        <input v-model="s.model" placeholder="gemma3:4b / gpt-4o-mini / claude-haiku-4-5 / gemini-2.5-flash"
               class="mt-1 block w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 outline-none focus:border-indigo-400/40" />
      </label>

      <label class="block">
        <span class="text-sm text-white/70">API Key {{ s.provider === 'ollama' ? '(not needed for Ollama)' : '' }}</span>
        <input v-model="s.apiKey" type="password" placeholder="sk-…"
               class="mt-1 block w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 outline-none focus:border-indigo-400/40" />
      </label>

      <label class="block">
        <span class="text-sm text-white/70">Base URL (optional)</span>
        <input v-model="s.baseUrl" placeholder="http://localhost:11434 / https://api.openai.com/v1"
               class="mt-1 block w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 outline-none focus:border-indigo-400/40" />
      </label>

      <label class="block">
        <span class="text-sm text-white/70">Judge Model (optional, P2)</span>
        <input v-model="s.judgeModel" placeholder="leave empty to use same as Model"
               class="mt-1 block w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 outline-none focus:border-indigo-400/40" />
      </label>

      <fieldset class="rounded-lg border border-white/10 bg-white/5 p-4 mt-4">
        <legend class="text-xs font-mono uppercase tracking-wider text-white/60 px-2">Connection Test</legend>
        <p class="text-xs text-white/50 mb-3">Verify API key, base URL, and model name before using the app.</p>
        <div class="flex items-center gap-3">
          <button @click="runPing" :disabled="pingTesting"
                  class="px-3 py-1.5 rounded-md bg-indigo-500/20 border border-indigo-400/30 text-xs font-mono text-indigo-200 hover:bg-indigo-500/30 disabled:opacity-50 transition-colors">
            {{ pingTesting ? 'Testing…' : 'Test Connection' }}
          </button>
          <span v-if="pingResult?.ok" class="text-xs font-mono text-green-400">
            Connected ({{ pingResult.latency_ms }}ms)
          </span>
          <span v-else-if="pingResult && !pingResult.ok" class="text-xs font-mono text-red-400">
            Failed
          </span>
          <span v-else class="text-xs text-white/40">Not tested</span>
        </div>
        <p v-if="pingResult?.note" class="text-xs mt-2" :class="pingResult.ok ? 'text-white/50' : 'text-red-400'">{{ pingResult.note }}</p>
      </fieldset>

      <fieldset class="rounded-lg border border-white/10 bg-white/5 p-4 mt-4">
        <legend class="text-xs font-mono uppercase tracking-wider text-white/60 px-2">Tool Call Detection</legend>
        <p class="text-xs text-white/50 mb-3">Local models like gemma3:4b don't support tool calling — agent_loop mode won't work. Test your model to auto-lock unavailable modes.</p>
        <div class="flex items-center gap-3">
          <button @click="runToolCheck" :disabled="toolTesting"
                  class="px-3 py-1.5 rounded-md bg-indigo-500/20 border border-indigo-400/30 text-xs font-mono text-indigo-200 hover:bg-indigo-500/30 disabled:opacity-50 transition-colors">
            {{ toolTesting ? 'Testing…' : 'Test Tool Call' }}
          </button>
          <span v-if="s.toolCapable === true" class="text-xs font-mono text-green-400">Tool Call Supported</span>
          <span v-else-if="s.toolCapable === false" class="text-xs font-mono text-red-400">Tool Call Not Supported</span>
          <span v-else class="text-xs text-white/40">Untested</span>
        </div>
        <p v-if="toolNote" class="text-xs text-white/50 mt-2">{{ toolNote }}</p>
      </fieldset>

      <fieldset class="rounded-lg border border-white/10 bg-white/5 p-4 mt-4">
        <legend class="text-xs font-mono uppercase tracking-wider text-white/60 px-2">Preferences</legend>

        <label class="block mt-3">
          <span class="text-sm text-white/70">{{ $t('settings.language') }}</span>
          <select v-model="s.language" @change="onLanguageChange"
                  class="mt-1 block w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 outline-none focus:border-indigo-400/40 font-mono">
            <option value="en">English</option>
            <option value="zh-CN">中文</option>
          </select>
        </label>

        <label class="flex items-center gap-2 mt-3 cursor-pointer">
          <input type="checkbox" v-model="s.debugMode" @change="s.persist()"
                 class="rounded border-white/10 bg-white/5 text-indigo-500 focus:ring-indigo-400/40" />
          <span class="text-sm text-white/70">{{ $t('settings.debugMode') }}</span>
        </label>

      </fieldset>

      <fieldset class="rounded-lg border border-white/10 bg-white/5 p-4 mt-4">
        <legend class="text-xs font-mono uppercase tracking-wider text-white/60 px-2">P3 mode defaults</legend>

        <label class="block mt-3">
          <span class="text-sm text-white/70">Plan view default mode</span>
          <select v-model="s.defaultPlannerMode"
                  :disabled="s.toolCapable === false"
                  class="mt-1 block w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 outline-none focus:border-indigo-400/40 font-mono disabled:opacity-40">
            <option value="agent_loop">agent_loop</option>
            <option value="deterministic">deterministic</option>
          </select>
        </label>

        <label class="block mt-3">
          <span class="text-sm text-white/70">Quiz view default mode</span>
          <select v-model="s.defaultQuizMode"
                  :disabled="s.toolCapable === false"
                  class="mt-1 block w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 outline-none focus:border-indigo-400/40 font-mono disabled:opacity-40">
            <option value="agent_loop">agent_loop</option>
            <option value="deterministic">deterministic</option>
          </select>
        </label>
      </fieldset>

      <button @click="save"
              class="px-4 py-2 rounded-lg bg-indigo-500 hover:bg-indigo-400 text-sm">
        {{ $t('settings.save') }}
      </button>

      <section
        v-if="lifecycle.summary?.reset_enabled"
        class="mt-10 border-t border-danger/30 pt-8"
        aria-labelledby="danger-zone-title"
      >
        <p class="font-mono text-xs uppercase tracking-[0.16em] text-danger">
          {{ $t('settings.dangerZone.eyebrow') }}
        </p>
        <div class="mt-2 flex items-start gap-3">
          <AlertTriangle class="mt-0.5 h-5 w-5 shrink-0 text-danger" aria-hidden="true" />
          <div>
            <h3 id="danger-zone-title" class="text-lg font-semibold text-fg">
              {{ $t('settings.dangerZone.title') }}
            </h3>
            <p class="mt-1 max-w-3xl text-sm leading-6 text-fg-muted">
              {{ $t('settings.dangerZone.description') }}
            </p>
          </div>
        </div>

        <p class="mt-6 text-sm font-medium text-fg">
          {{ $t('settings.dangerZone.countsTitle') }}
        </p>
        <p
          v-if="lifecycle.summaryRefreshing"
          class="mt-2 text-sm text-primary-2"
          role="status"
          aria-live="polite"
        >
          {{ $t('settings.dangerZone.refreshingCounts') }}
        </p>
        <div
          v-else-if="lifecycle.phase === 'ready' && lifecycle.error"
          class="mt-3 flex flex-col gap-3 rounded-md bg-danger-bg px-3 py-3 sm:flex-row sm:items-center sm:justify-between"
          role="alert"
        >
          <p class="text-sm text-danger">
            {{ $t('settings.dangerZone.refreshError', { message: lifecycle.error.message }) }}
          </p>
          <button
            type="button"
            class="shrink-0 rounded-md border border-danger/50 px-3 py-1.5 text-sm font-medium text-danger transition-colors hover:bg-danger/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-danger/50"
            @click="lifecycle.refreshSummary()"
          >
            {{ $t('settings.dangerZone.retryCounts') }}
          </button>
        </div>
        <dl class="mt-3 grid grid-cols-2 gap-x-6 gap-y-3 sm:grid-cols-3 lg:grid-cols-4">
          <div
            v-for="key in countKeys"
            :key="key"
            class="border-b border-border pb-2"
          >
            <dt class="text-xs leading-5 text-fg-muted">
              {{ $t(`settings.dangerZone.counts.${key}`) }}
            </dt>
            <dd class="mt-1 font-mono text-base text-fg">
              {{ lifecycle.summary[key] }}
            </dd>
          </div>
        </dl>

        <div class="mt-8 divide-y divide-border-strong border-y border-border-strong">
          <div class="grid gap-4 py-5 sm:grid-cols-[minmax(0,1fr)_auto] sm:items-center">
            <div>
              <h4 class="text-sm font-semibold text-fg">
                {{ $t('settings.dangerZone.learningTitle') }}
              </h4>
              <p class="mt-1 max-w-3xl text-sm leading-6 text-fg-muted">
                {{ $t('settings.dangerZone.learningBody') }}
              </p>
            </div>
            <button
              type="button"
              class="rounded-md border border-danger/50 px-4 py-2 text-sm font-medium text-danger transition-colors hover:bg-danger-bg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-danger/50 disabled:cursor-not-allowed disabled:opacity-40"
              :disabled="lifecycle.phase !== 'ready' || lifecycle.summaryRefreshing"
              @click="lifecycle.requestLearningReset()"
            >
              {{ $t('dataLifecycle.actions.clearLearning') }}
            </button>
          </div>

          <div class="grid gap-4 py-5 sm:grid-cols-[minmax(0,1fr)_auto] sm:items-center">
            <div>
              <h4 class="text-sm font-semibold text-fg">
                {{ $t('settings.dangerZone.factoryTitle') }}
              </h4>
              <p class="mt-1 max-w-3xl text-sm leading-6 text-fg-muted">
                {{ $t('settings.dangerZone.factoryBody') }}
              </p>
            </div>
            <button
              type="button"
              class="rounded-md bg-danger px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-danger/85 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-danger/50 disabled:cursor-not-allowed disabled:opacity-40"
              :disabled="lifecycle.phase !== 'ready' || lifecycle.summaryRefreshing"
              @click="lifecycle.requestFactoryReset()"
            >
              {{ $t('dataLifecycle.actions.factoryReset') }}
            </button>
          </div>
        </div>
      </section>
    </div>
  </div>
</template>
