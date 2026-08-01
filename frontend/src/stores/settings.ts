import { defineStore } from 'pinia'

import { FACTORY_RECOVERY_FINGERPRINT_KEY } from '../lib/dataLifecycle'

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
const FINGERPRINT_KEY = 'study-coach:fingerprint'

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
let _identityGeneration = 0

function readStoredObject(raw: string | null): unknown {
  if (!raw) return {}
  try {
    return JSON.parse(raw)
  } catch {
    return {}
  }
}

export function invalidateAnonymousProvisioning(): void {
  _identityGeneration += 1
  _tokenPromise = null
}

async function requestAnonymousToken(fingerprint: string): Promise<{ access_token: string; tier: string }> {
  const resp = await fetch('/api/auth/anonymous', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ fingerprint }),
  })
  if (!resp.ok) throw new Error('anonymous auth failed')
  const body = await resp.json() as { access_token?: unknown; tier?: unknown }
  if (typeof body.access_token !== 'string' || !body.access_token.trim()) {
    throw new Error('anonymous auth failed')
  }
  return {
    access_token: body.access_token,
    tier: typeof body.tier === 'string' ? body.tier : 'guest',
  }
}

function persistIdentity(
  generation: number,
  capturedFingerprint: string,
  updater: (current: SettingsState) => SettingsState,
): string {
  if (generation !== _identityGeneration) {
    throw new Error('identity provisioning invalidated')
  }
  // Shared fingerprint/epoch must still match the request-time capture so a
  // stale tab cannot resurrect a deleted identity after another tab factory-reset.
  if (localStorage.getItem(FINGERPRINT_KEY) !== capturedFingerprint) {
    throw new Error('identity provisioning invalidated')
  }
  const next = updater(normalizeSettings(readStoredObject(localStorage.getItem(STORAGE_KEY))))
  localStorage.setItem(STORAGE_KEY, JSON.stringify(next))
  return next.accessToken
}

export async function getAccessToken(): Promise<string> {
  let raw: string | null = null
  try {
    raw = localStorage.getItem(STORAGE_KEY)
  } catch {
    raw = null
  }
  if (raw) {
    try {
      const parsed = normalizeSettings(JSON.parse(raw))
      if (parsed.accessToken) return parsed.accessToken
    } catch { /* ignore */ }
  }
  // No token yet — provision anonymous
  if (!_tokenPromise) {
    const generation = _identityGeneration
    const provisioning = (async () => {
      const fp = crypto.randomUUID()
      const stored = localStorage.getItem(FINGERPRINT_KEY)
      const fingerprint = stored || fp
      if (generation !== _identityGeneration) {
        throw new Error('identity provisioning invalidated')
      }
      if (!stored) localStorage.setItem(FINGERPRINT_KEY, fingerprint)
      const capturedFingerprint = fingerprint
      const { access_token, tier } = await requestAnonymousToken(fingerprint)
      return persistIdentity(generation, capturedFingerprint, current => ({
        ...current,
        accessToken: access_token,
        tier: tier === 'member' ? 'member' : 'guest',
      }))
    })()
    _tokenPromise = provisioning
    void provisioning.catch(() => {
      if (_tokenPromise === provisioning) _tokenPromise = null
    })
  }
  return _tokenPromise
}

async function derivedFactoryFingerprint(seed: string): Promise<string> {
  const bytes = new TextEncoder().encode(`study-coach:factory-recovery:${seed}`)
  const digest = await crypto.subtle.digest('SHA-256', bytes)
  const hex = Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('')
  return `factory-${hex.slice(0, 32)}`
}

/**
 * Stage one replacement fingerprint before a Factory reset can delete the
 * current identity. The value survives browser-state clearing so delayed tabs
 * and response-lost reloads converge on the same backend user.
 */
export async function stageFactoryRecoveryFingerprint(forceRotate = false): Promise<string> {
  const staged = localStorage.getItem(FACTORY_RECOVERY_FINGERPRINT_KEY)
  const current = localStorage.getItem(FINGERPRINT_KEY)
  if (staged && (!forceRotate || !current)) return staged

  let seed = current
  if (!seed) {
    const stored = normalizeSettings(readStoredObject(localStorage.getItem(STORAGE_KEY)))
    seed = stored.accessToken
  }
  const fingerprint = seed ? await derivedFactoryFingerprint(seed) : crypto.randomUUID()
  localStorage.setItem(FACTORY_RECOVERY_FINGERPRINT_KEY, fingerprint)
  return fingerprint
}

/** Establish a fresh local identity after factory browser clear. */
export async function provisionFactoryIdentity(
  fingerprint: string = crypto.randomUUID(),
): Promise<string> {
  invalidateAnonymousProvisioning()
  const generation = _identityGeneration
  localStorage.setItem(FINGERPRINT_KEY, fingerprint)
  const capturedFingerprint = fingerprint
  const { access_token, tier } = await requestAnonymousToken(fingerprint)
  return persistIdentity(generation, capturedFingerprint, () => ({
    ...DEFAULT_SETTINGS,
    accessToken: access_token,
    tier: tier === 'member' ? 'member' : 'guest',
  }))
}

export function authHeaders(): Record<string, string> {
  // Synchronous helper for callers that already persist a token. Learning
  // routes require a signed bearer whose user row still exists; they do not
  // fall back to a guest/default identity when this returns {}.
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw) {
      const parsed = normalizeSettings(JSON.parse(raw))
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
  const saved = typeof value === 'object' && value !== null && !Array.isArray(value)
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
