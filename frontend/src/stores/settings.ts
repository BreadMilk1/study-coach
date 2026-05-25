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
}

const STORAGE_KEY = 'study-coach:settings'

function loadInitial(): SettingsState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw) {
      const parsed = JSON.parse(raw)
      // back-fill new fields (existing localStorage may not have them)
      return {
        defaultPlannerMode: 'agent_loop',
        defaultQuizMode: 'agent_loop',
        ...parsed,
      }
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
    defaultPlannerMode: 'agent_loop',
    defaultQuizMode: 'agent_loop',
  }
}

export const useSettings = defineStore('settings', {
  state: () => loadInitial(),
  actions: {
    persist() {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(this.$state))
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
