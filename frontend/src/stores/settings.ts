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

const DEFAULT_SETTINGS: SettingsState = {
  provider: 'ollama',
  model: 'gemma3:4b',
  apiKey: '',
  baseUrl: '',
  judgeModel: '',
  defaultPlannerMode: 'agent_loop',
  defaultQuizMode: 'agent_loop',
  toolCapable: null,
  debugMode: false,
  language: 'en',
  accessToken: '',
  tier: 'guest',
}

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
    const provisioning = (async () => {
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
    _tokenPromise = provisioning
    void provisioning.catch(() => {
      if (_tokenPromise === provisioning) _tokenPromise = null
    })
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

function normalizeSettings(value: unknown): SettingsState {
  const saved = typeof value === 'object' && value !== null
    ? value as Record<string, unknown>
    : {}
  const provider = saved.provider
  const model = saved.model
  const plannerMode = saved.defaultPlannerMode
  const quizMode = saved.defaultQuizMode

  return {
    provider: provider === 'ollama'
      || provider === 'openai'
      || provider === 'anthropic'
      || provider === 'gemini'
      ? provider
      : DEFAULT_SETTINGS.provider,
    model: typeof model === 'string' && model.trim() ? model : DEFAULT_SETTINGS.model,
    apiKey: typeof saved.apiKey === 'string' ? saved.apiKey : DEFAULT_SETTINGS.apiKey,
    baseUrl: typeof saved.baseUrl === 'string' ? saved.baseUrl : DEFAULT_SETTINGS.baseUrl,
    judgeModel: typeof saved.judgeModel === 'string' ? saved.judgeModel : DEFAULT_SETTINGS.judgeModel,
    defaultPlannerMode: plannerMode === 'agent_loop' || plannerMode === 'deterministic'
      ? plannerMode
      : DEFAULT_SETTINGS.defaultPlannerMode,
    defaultQuizMode: quizMode === 'agent_loop' || quizMode === 'deterministic'
      ? quizMode
      : DEFAULT_SETTINGS.defaultQuizMode,
    toolCapable: null,
    debugMode: typeof saved.debugMode === 'boolean' ? saved.debugMode : DEFAULT_SETTINGS.debugMode,
    language: saved.language === 'en' || saved.language === 'zh-CN'
      ? saved.language
      : DEFAULT_SETTINGS.language,
    accessToken: typeof saved.accessToken === 'string' ? saved.accessToken : DEFAULT_SETTINGS.accessToken,
    tier: saved.tier === 'guest' || saved.tier === 'member' ? saved.tier : DEFAULT_SETTINGS.tier,
  }
}

function loadInitial(): SettingsState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw) {
      const base = normalizeSettings(JSON.parse(raw))
      // Restore per-model tool-capable cache (not stored in settings JSON)
      base.toolCapable = loadToolCapable(base.model)
      return base
    }
  } catch {
    /* empty */
  }
  return { ...DEFAULT_SETTINGS }
}

export const useSettings = defineStore('settings', {
  state: () => loadInitial(),
  actions: {
    persist() {
      try {
        const raw = localStorage.getItem(STORAGE_KEY)
        const stored = raw ? normalizeSettings(JSON.parse(raw)) : null
        if (stored?.accessToken && stored.accessToken !== this.accessToken) {
          this.accessToken = stored.accessToken
          this.tier = stored.tier
        }
      } catch {
        // Persist the active state when no valid stored identity is available.
      }
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
