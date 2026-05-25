<script setup lang="ts">
import { useSettings } from '../stores/settings'

const s = useSettings()

function save() {
  s.persist()
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

      <label v-if="s.provider !== 'ollama'" class="block">
        <span class="text-sm text-white/70">API Key</span>
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
        <legend class="text-xs font-mono uppercase tracking-wider text-white/60 px-2">P3 mode defaults</legend>

        <label class="block mt-3">
          <span class="text-sm text-white/70">Plan view default mode</span>
          <select v-model="s.defaultPlannerMode"
                  class="mt-1 block w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 outline-none focus:border-indigo-400/40 font-mono">
            <option value="agent_loop">agent_loop</option>
            <option value="deterministic">deterministic</option>
          </select>
        </label>

        <label class="block mt-3">
          <span class="text-sm text-white/70">Quiz view default mode</span>
          <select v-model="s.defaultQuizMode"
                  class="mt-1 block w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 outline-none focus:border-indigo-400/40 font-mono">
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
