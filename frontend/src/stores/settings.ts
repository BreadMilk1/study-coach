import { defineStore } from 'pinia'

import { FACTORY_RECOVERY_FINGERPRINT_KEY } from '../lib/dataLifecycle'

export type Provider = 'ollama' | 'openai' | 'anthropic' | 'gemini'
export type Mode = 'agent_loop' | 'deterministic'

export type ConnectionField = 'provider' | 'model' | 'apiKey' | 'baseUrl'

export interface ConnectionSnapshot {
  revision: string
  provider: Provider
  model: string
  apiKey: string
  baseUrl: string
}

interface SettingsState {
  provider: Provider
  model: string
  apiKey: string
  baseUrl: string
  judgeModel: string
  defaultPlannerMode: Mode
  defaultQuizMode: Mode
  toolCapable: boolean | null  // null = unknown for the current connection
  debugMode: boolean
  language: 'en' | 'zh-CN'
  accessToken: string
  tier: 'guest' | 'member'
  // Random connection generation. Never derived from the API key; it gives
  // detection results and the capability cache an identity to bind to and is
  // rotated whenever a connection field really changes.
  connectionRevision: string
}

const STORAGE_KEY = 'study-coach:settings'
const FINGERPRINT_KEY = 'study-coach:fingerprint'
// Single v2 capability record. It stores the random revision and a boolean —
// never the API key, a key hash, the base URL or any provider response.
const CAPABILITY_CACHE_KEY = 'study-coach:connection-capability:v2'

const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i

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
  connectionRevision: '',
}

let _tokenPromise: Promise<string> | null = null
let _identityGeneration = 0
// In-memory result of a detection that is not (yet) bound to a saved
// configuration. Runtime-only: it is never serialized into settings JSON.
interface PendingCapability extends ConnectionSnapshot {
  toolCapable: boolean
}
let _pendingCapability: PendingCapability | null = null

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
  localStorage.setItem(STORAGE_KEY, JSON.stringify(serializeSettings(next)))
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

function sameConnection(
  a: Pick<SettingsState, ConnectionField>,
  b: Pick<SettingsState, ConnectionField>,
): boolean {
  return a.provider === b.provider && a.model === b.model
    && a.apiKey === b.apiKey && a.baseUrl === b.baseUrl
}

interface CapabilityRecordV2 {
  schemaVersion: 2
  connectionRevision: string
  toolCapable: boolean
}

function loadCapabilityRecord(): CapabilityRecordV2 | null {
  try {
    const raw = localStorage.getItem(CAPABILITY_CACHE_KEY)
    if (!raw) return null
    const parsed: unknown = JSON.parse(raw)
    if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) return null
    const record = parsed as Record<string, unknown>
    if (record.schemaVersion !== 2) return null
    if (typeof record.connectionRevision !== 'string' || !UUID_PATTERN.test(record.connectionRevision)) return null
    if (typeof record.toolCapable !== 'boolean') return null
    return {
      schemaVersion: 2,
      connectionRevision: record.connectionRevision,
      toolCapable: record.toolCapable,
    }
  } catch {
    return null
  }
}

function writeCapabilityRecord(record: CapabilityRecordV2): void {
  try {
    localStorage.setItem(CAPABILITY_CACHE_KEY, JSON.stringify(record))
  } catch { /* storage unavailable */ }
}

function clearCapabilityRecordFor(revision: string): void {
  const record = loadCapabilityRecord()
  if (record && record.connectionRevision === revision) {
    try { localStorage.removeItem(CAPABILITY_CACHE_KEY) } catch { /* ignore */ }
  }
}

function normalizeSettings(value: unknown): SettingsState {
  const saved = typeof value === 'object' && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {}
  const plannerMode = saved.defaultPlannerMode
  const quizMode = saved.defaultQuizMode

  const savedProvider: Provider | null = saved.provider === 'ollama'
    || saved.provider === 'openai'
    || saved.provider === 'anthropic'
    || saved.provider === 'gemini'
    ? saved.provider
    : null
  const provider = savedProvider ?? DEFAULT_SETTINGS.provider
  // A cloud provider with a missing/blank persisted model keeps the model
  // empty instead of silently borrowing the Ollama default, so the user fills
  // it in; Ollama and token-only records keep the default.
  const savedModel: string | null = typeof saved.model === 'string' && saved.model.trim() !== ''
    ? saved.model
    : null
  const model = savedModel ?? (provider === 'ollama' ? DEFAULT_SETTINGS.model : '')
  const savedApiKey: string | null = typeof saved.apiKey === 'string' ? saved.apiKey : null
  const savedBaseUrl: string | null = typeof saved.baseUrl === 'string' ? saved.baseUrl : null
  const apiKey = savedApiKey ?? DEFAULT_SETTINGS.apiKey
  const baseUrl = savedBaseUrl ?? DEFAULT_SETTINGS.baseUrl

  // A saved revision is only trusted when it is well-formed AND no connection
  // field needed correction — rewritten fields must never inherit capability
  // cached for the previous values.
  const connectionUnmodified = savedProvider !== null && savedModel !== null
    && savedApiKey !== null && savedBaseUrl !== null
  const savedRevision = typeof saved.connectionRevision === 'string'
    && UUID_PATTERN.test(saved.connectionRevision)
    ? saved.connectionRevision
    : ''

  return {
    provider,
    model,
    apiKey,
    baseUrl,
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
    connectionRevision: connectionUnmodified ? savedRevision : '',
  }
}

function serializeSettings(state: SettingsState): Record<string, unknown> {
  // Controlled serialization: toolCapable and detection runtime/pending
  // metadata never enter the settings JSON. Capability only lives in the v2
  // record, keyed by the connection revision.
  return {
    provider: state.provider,
    model: state.model,
    apiKey: state.apiKey,
    baseUrl: state.baseUrl,
    judgeModel: state.judgeModel,
    defaultPlannerMode: state.defaultPlannerMode,
    defaultQuizMode: state.defaultQuizMode,
    debugMode: state.debugMode,
    language: state.language,
    accessToken: state.accessToken,
    tier: state.tier,
    connectionRevision: state.connectionRevision,
  }
}

function readStoredSettings(): SettingsState | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return null
    return normalizeSettings(JSON.parse(raw))
  } catch {
    return null
  }
}

function loadInitial(): SettingsState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw) {
      const base = normalizeSettings(JSON.parse(raw))
      // Capability is only restored from a v2 record that matches the saved
      // connection revision; legacy per-model v1 cache keys are never read
      // or migrated.
      if (base.connectionRevision) {
        const record = loadCapabilityRecord()
        if (record && record.connectionRevision === base.connectionRevision) {
          base.toolCapable = record.toolCapable
        }
      }
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
        const stored = readStoredSettings()
        if (stored?.accessToken && stored.accessToken !== this.accessToken) {
          this.accessToken = stored.accessToken
          this.tier = stored.tier
        }
        // Conservative invalidation for direct four-field store assignments
        // that bypassed updateConnection: the fields changed while the
        // revision did not, so the old capability must not bind to them.
        if (
          stored
          && stored.connectionRevision === this.connectionRevision
          && !sameConnection(stored, this)
        ) {
          this.rotateConnectionRevision()
        }
      } catch {
        // Persist the active state when no valid stored identity is available.
      }
      localStorage.setItem(STORAGE_KEY, JSON.stringify(serializeSettings(this)))
      // A Save keeps the current revision, so a just-tested matching result
      // becomes cacheable here if the saved configuration now matches.
      this.writePendingCapabilityIfSavedMatches()
    },
    persistPreferences() {
      // Preference auto-save (language / debug toggles) writes ONLY the
      // non-connection preferences onto the already-stored settings snapshot.
      // The saved connection fields, judgeModel, connectionRevision and
      // identity stay untouched, no revision rotation/ensure runs, and no
      // in-memory detection result is promoted — adopting the active
      // connection and its capability is the explicit full Save's job.
      const stored = readStoredSettings()
      if (stored?.accessToken && stored.accessToken !== this.accessToken) {
        // Same identity adoption as the full Save: the newest stored token
        // wins and the active store stays in sync.
        this.accessToken = stored.accessToken
        this.tier = stored.tier
      }
      const base = stored ?? { ...DEFAULT_SETTINGS }
      const merged = {
        ...base,
        language: this.language,
        debugMode: this.debugMode,
        defaultPlannerMode: this.defaultPlannerMode,
        defaultQuizMode: this.defaultQuizMode,
        // Newest stored identity wins; without one keep the active token
        // (same as the full Save). No provisioning or identity UUID runs here.
        accessToken: base.accessToken || this.accessToken,
        tier: base.accessToken ? base.tier : this.tier,
      }
      localStorage.setItem(STORAGE_KEY, JSON.stringify(serializeSettings(merged)))
    },
    updateConnection(field: ConnectionField, value: string): boolean {
      const current = this[field]
      if (current === value) return false
      if (field === 'provider') this.provider = value as Provider
      else if (field === 'model') this.model = value
      else if (field === 'apiKey') this.apiKey = value
      else this.baseUrl = value
      this.rotateConnectionRevision()
      return true
    },
    ensureConnectionRevision(): boolean {
      // Bounded lazy creation for the first capability detection on a legacy
      // (revision-less) configuration: gives the check a generation to bind
      // to without touching normalizeSettings/hydration and without requiring
      // an unrelated connection edit. A valid revision is never re-rotated.
      if (UUID_PATTERN.test(this.connectionRevision)) return false
      this.rotateConnectionRevision()
      return true
    },
    rotateConnectionRevision() {
      this.connectionRevision = crypto.randomUUID()
      this.toolCapable = null
      _pendingCapability = null
    },
    beginToolRecheck() {
      // Starting a manual re-test drops the old capability for the current
      // revision immediately, so a failed re-test cannot leave the previous
      // result in place to masquerade as fresh after a refresh. Records for
      // other (saved) revisions are left alone.
      this.toolCapable = null
      _pendingCapability = null
      if (this.connectionRevision) clearCapabilityRecordFor(this.connectionRevision)
    },
    commitToolCheckResult(snapshot: ConnectionSnapshot, capable: boolean | null) {
      // Only the request that still matches the current revision and the full
      // four-field snapshot may write the capability state.
      if (snapshot.revision !== this.connectionRevision) return
      if (!sameConnection(snapshot, this)) return
      if (capable === null) {
        // An incomplete check stays unknown and is never cached as negative.
        this.toolCapable = null
        _pendingCapability = null
        return
      }
      this.toolCapable = capable
      _pendingCapability = {
        revision: snapshot.revision,
        provider: snapshot.provider,
        model: snapshot.model,
        apiKey: snapshot.apiKey,
        baseUrl: snapshot.baseUrl,
        toolCapable: capable,
      }
      this.writePendingCapabilityIfSavedMatches()
    },
    writePendingCapabilityIfSavedMatches() {
      const pending = _pendingCapability
      if (!pending || pending.revision !== this.connectionRevision) return
      if (!sameConnection(pending, this)) return
      if (!UUID_PATTERN.test(pending.revision)) return
      // Only a saved (persisted) configuration holds the cross-refresh
      // record; unsaved results stay in memory until a matching Save.
      const stored = readStoredSettings()
      if (!stored || stored.connectionRevision !== pending.revision) return
      if (!sameConnection(stored, pending)) return
      writeCapabilityRecord({
        schemaVersion: 2,
        connectionRevision: pending.revision,
        toolCapable: pending.toolCapable,
      })
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
