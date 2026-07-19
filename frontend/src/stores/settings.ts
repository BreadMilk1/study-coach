import { defineStore } from 'pinia'

export type Provider = 'ollama' | 'openai' | 'anthropic' | 'gemini'
export type Mode = 'agent_loop' | 'deterministic'

interface SettingsState {
  provider: Provider
  model: string
  apiKey: string
  baseUrl: string
  judgeModel: string
  defaultPlannerMode: Mode
  defaultQuizMode: Mode
  toolCapable: boolean | null  // null = untested, true/false = tested result
  debugMode: boolean
  language: 'en' | 'zh-CN'
  accessToken: string
  tier: 'guest' | 'member'
}

const STORAGE_KEY = 'study-coach:settings'

let _tokenPromise: Promise<string> | null = null

export async function getAccessToken(): Promise<string> {
  const raw = localStorage.getItem(STORAGE_KEY)
  if (raw) {
    try {
      const parsed = JSON.parse(raw)
      if (parsed.accessToken) return parsed.accessToken
    } catch { /* ignore */ }
  }
  // No token yet — provision anonymous
  if (!_tokenPromise) {
    _tokenPromise = (async () => {
      const fp = crypto.randomUUID()
      const stored = localStorage.getItem('study-coach:fingerprint')
      const fingerprint = stored || fp
      if (!stored) localStorage.setItem('study-coach:fingerprint', fingerprint)
      const resp = await fetch('/api/auth/anonymous', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fingerprint }),
      })
      if (!resp.ok) throw new Error('anonymous auth failed')
      const { access_token, tier } = await resp.json()
      // Persist token into settings
      const existingRaw = localStorage.getItem(STORAGE_KEY)
      const existing = existingRaw ? JSON.parse(existingRaw) : {}
      existing.accessToken = access_token
      existing.tier = tier || 'guest'
      localStorage.setItem(STORAGE_KEY, JSON.stringify(existing))
      return access_token as string
    })()
  }
  return _tokenPromise
}

export function authHeaders(): Record<string, string> {
  // Synchronous fallback: return bearer header if we have it in storage.
  // For first-ever call, the fetch will still work because backend
  // falls back to "default-user" when no token is present.
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw) {
      const parsed = JSON.parse(raw)
      if (parsed.accessToken) {
        return { Authorization: `Bearer ${parsed.accessToken}` }
      }
    }
  } catch { /* ignore */ }
  return {}
}

const TOOL_CAPABLE_KEY = 'study-coach:tool-capable'

function loadToolCapable(model: string): boolean | null {
  try {
    const raw = localStorage.getItem(`${TOOL_CAPABLE_KEY}:${model}`)
    if (raw === 'true') return true
    if (raw === 'false') return false
  } catch { /* empty */ }
  return null
}

function persistToolCapable(model: string, capable: boolean) {
  localStorage.setItem(`${TOOL_CAPABLE_KEY}:${model}`, String(capable))
}

function loadInitial(): SettingsState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw) {
      const parsed = JSON.parse(raw)
      const base = {
        defaultPlannerMode: 'agent_loop' as Mode,
        defaultQuizMode: 'agent_loop' as Mode,
        toolCapable: null as boolean | null,
        debugMode: parsed.debugMode ?? false,
        language: parsed.language ?? 'en',
        ...parsed,
      }
      // Restore per-model tool-capable cache (not stored in settings JSON)
      base.toolCapable = loadToolCapable(base.model)
      return base
    }
  } catch {
    /* empty */
  }
  return {
    provider: 'ollama',
    model: 'gemma3:4b',
    apiKey: '',
    baseUrl: '',
    judgeModel: '',
    defaultPlannerMode: 'agent_loop' as Mode,
    defaultQuizMode: 'agent_loop' as Mode,
    toolCapable: null,
    debugMode: false,
    language: 'en',
    accessToken: '',
    tier: 'guest' as const,
  }
}

export const useSettings = defineStore('settings', {
  state: () => loadInitial(),
  actions: {
    persist() {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(this.$state))
    },
    setToolCapable(capable: boolean) {
      this.toolCapable = capable
      persistToolCapable(this.model, capable)
    },
  },
})

export interface ModeOverrides {
  plannerMode?: Mode
  quizMode?: Mode
}

export function llmHeaders(s: SettingsState, overrides: ModeOverrides = {}): Record<string, string> {
  const h: Record<string, string> = {
    'x-provider': s.provider,
    'x-model': s.model,
  }
  if (s.apiKey) h['x-api-key'] = s.apiKey
  if (s.baseUrl) h['x-base-url'] = s.baseUrl
  if (s.judgeModel) h['x-judge-model'] = s.judgeModel
  h['x-planner-mode'] = overrides.plannerMode ?? (
    s.toolCapable === false ? 'deterministic' : s.defaultPlannerMode
  )
  h['x-quiz-mode'] = overrides.quizMode ?? (
    s.toolCapable === false ? 'deterministic' : s.defaultQuizMode
  )
  return h
}
