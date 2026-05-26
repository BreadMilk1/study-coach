<script setup lang="ts">
import { ref } from 'vue'
import { useSettings } from '../stores/settings'
import { checkToolCapable, pingModel } from '../lib/api'

const s = useSettings()
const toolTesting = ref(false)
const toolNote = ref('')
const pingTesting = ref(false)
const pingResult = ref<{ ok: boolean; note: string; latency_ms: number } | null>(null)

function save() {
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
  <div class="p-8 max-w-2xl">
    <h2 class="text-xl font-semibold mb-4">Settings</h2>
    <p class="text-white/60 text-sm mb-6">
      Bring-your-own-key. API key is stored in your browser localStorage; the server never sees nor persists it.
    </p>
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
        Save
      </button>
    </div>
  </div>
</template>
