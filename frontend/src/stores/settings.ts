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
}

const STORAGE_KEY = 'study-coach:settings'

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
  if (overrides.plannerMode) h['x-planner-mode'] = overrides.plannerMode
  if (overrides.quizMode) h['x-quiz-mode'] = overrides.quizMode
  return h
}
