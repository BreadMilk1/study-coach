import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { memoryStorage } from '../test/memoryStorage'
import { llmHeaders, useSettings } from './settings'

beforeEach(() => {
  setActivePinia(createPinia())
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('settings store', () => {
  it('hydrates a complete settings state from token-only storage', () => {
    vi.stubGlobal('localStorage', memoryStorage({
      'study-coach:settings': JSON.stringify({
        accessToken: 'anonymous-token',
        tier: 'guest',
      }),
    }))

    const settings = useSettings()

    expect(settings.$state).toMatchObject({
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
      accessToken: 'anonymous-token',
      tier: 'guest',
    })
    expect(llmHeaders(settings.$state)).toMatchObject({
      'x-provider': 'ollama',
      'x-model': 'gemma3:4b',
    })
  })

  it('falls back from unsupported persisted settings values', () => {
    vi.stubGlobal('localStorage', memoryStorage({
      'study-coach:settings': JSON.stringify({
        provider: 'undefined',
        model: '',
        apiKey: null,
        baseUrl: 123,
        judgeModel: null,
        defaultPlannerMode: 'automatic',
        defaultQuizMode: 'automatic',
        debugMode: 'yes',
        language: 'fr',
        accessToken: 123,
        tier: 'admin',
      }),
    }))

    const settings = useSettings()

    expect(settings.$state).toMatchObject({
      provider: 'ollama',
      model: 'gemma3:4b',
      apiKey: '',
      baseUrl: '',
      judgeModel: '',
      defaultPlannerMode: 'agent_loop',
      defaultQuizMode: 'agent_loop',
      debugMode: false,
      language: 'en',
      accessToken: '',
      tier: 'guest',
    })
  })

  it('restores user settings after persist and store recreation', () => {
    const storage = memoryStorage()
    vi.stubGlobal('localStorage', storage)
    const settings = useSettings()
    settings.provider = 'openai'
    settings.model = 'gpt-4o-mini'
    settings.apiKey = 'sk-test'
    settings.baseUrl = 'https://api.openai.test/v1'
    settings.judgeModel = 'qwen2.5:7b'
    settings.defaultPlannerMode = 'deterministic'
    settings.defaultQuizMode = 'deterministic'
    settings.language = 'zh-CN'

    settings.persist()
    setActivePinia(createPinia())
    const restored = useSettings()

    expect(restored.$state).toMatchObject({
      provider: 'openai',
      model: 'gpt-4o-mini',
      apiKey: 'sk-test',
      baseUrl: 'https://api.openai.test/v1',
      judgeModel: 'qwen2.5:7b',
      defaultPlannerMode: 'deterministic',
      defaultQuizMode: 'deterministic',
      language: 'zh-CN',
    })
    expect(llmHeaders(restored.$state)).toMatchObject({
      'x-provider': 'openai',
      'x-model': 'gpt-4o-mini',
      'x-api-key': 'sk-test',
      'x-base-url': 'https://api.openai.test/v1',
      'x-judge-model': 'qwen2.5:7b',
    })
  })
})
