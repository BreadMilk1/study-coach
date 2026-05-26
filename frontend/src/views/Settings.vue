<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useSettings, googleLogin, getAccessToken } from '../stores/settings'
import { checkToolCapable, pingModel } from '../lib/api'
import { useI18n } from 'vue-i18n'

const { locale } = useI18n()
const s = useSettings()
const toolTesting = ref(false)
const toolNote = ref('')
const pingTesting = ref(false)
const pingResult = ref<{ ok: boolean; note: string; latency_ms: number } | null>(null)

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

// Google Identity Services
const googleClientId = ref('')
const googleReady = ref(false)

onMounted(async () => {
  try {
    const resp = await fetch('/api/auth/config')
    const { google_client_id } = await resp.json()
    googleClientId.value = google_client_id
    if (google_client_id) {
      // Poll for GIS script (loaded async defer in index.html)
      await new Promise<void>(resolve => {
        const check = () => {
          if (typeof (window as any).google?.accounts?.id !== 'undefined') {
            resolve()
          } else {
            setTimeout(check, 300)
          }
        }
        check()
      })
      ;(window as any).google.accounts.id.initialize({
        client_id: google_client_id,
        callback: handleCredentialResponse,
        auto_select: false,
      })
      googleReady.value = true
    }
  } catch { /* not configured */ }
})

function triggerGoogleLogin() {
  ;(window as any).google?.accounts?.id?.prompt()
}

async function handleCredentialResponse(response: any) {
  try {
    await googleLogin(response.credential)
    s.tier = 'member'
    s.persist()
    location.reload()
  } catch (e: any) {
    alert(`Google sign-in failed: ${e.message}`)
  }
}

function signOut() {
  if (typeof (window as any).google !== 'undefined') {
    ;(window as any).google.accounts.id.disableAutoSelect()
  }
  s.accessToken = ''
  s.tier = 'guest'
  s.persist()
  getAccessToken().then(() => location.reload())
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

        <div class="mt-4 pt-3 border-t border-white/10">
          <span class="text-sm text-white/70">Account</span>
          <p class="text-xs text-white/40 mt-1">
            Tier: <span class="font-mono" :class="s.tier === 'member' ? 'text-green-400' : 'text-white/50'">{{ s.tier }}</span>
          </p>

          <div v-if="s.tier !== 'member'" class="mt-2 space-y-2">
            <button v-if="googleReady"
                    @click="triggerGoogleLogin"
                    class="px-4 py-2 rounded-lg bg-white text-gray-900 text-sm font-medium hover:bg-gray-100 transition-colors flex items-center gap-2">
              <svg class="w-4 h-4" viewBox="0 0 24 24"><path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 01-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z"/><path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/><path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/><path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/></svg>
              Sign in with Google
            </button>
            <p v-if="googleClientId && !googleReady" class="text-xs text-white/40">Loading Google Sign-In...</p>
            <p v-if="!googleClientId" class="text-xs text-amber-400">
              Google OAuth not configured. Set GOOGLE_CLIENT_ID on the server.
            </p>
          </div>

          <button v-if="s.tier === 'member'"
                  @click="signOut"
                  class="mt-2 px-3 py-1.5 rounded-md border border-white/10 text-xs text-white/50 hover:text-white/80 transition-colors">
            Sign out
          </button>
        </div>
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
    </div>
  </div>
</template>
