import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { memoryStorage } from '../test/memoryStorage'
import {
  llmHeaders,
  useSettings,
  type ConnectionSnapshot,
} from './settings'

beforeEach(() => {
  setActivePinia(createPinia())
})

afterEach(() => {
  vi.unstubAllGlobals()
})

const REVISION_A = '00000000-0000-4000-8000-00000000000a'
const REVISION_B = '00000000-0000-4000-8000-00000000000b'

function stubSequentialUuids(): void {
  let counter = 0
  vi.stubGlobal('crypto', {
    randomUUID: vi.fn(() => {
      counter += 1
      return `00000000-0000-4000-8000-${String(counter).padStart(12, '0')}`
    }),
  })
}

function capabilityRecordValue(revision: string, toolCapable: boolean): string {
  return JSON.stringify({ schemaVersion: 2, connectionRevision: revision, toolCapable })
}

const CAPABILITY_CACHE_KEY = 'study-coach:connection-capability:v2'

function connectionSnapshot(
  revision: string,
  over: Partial<ConnectionSnapshot> = {},
): ConnectionSnapshot {
  return {
    revision,
    provider: 'openai',
    model: 'gpt-4o-mini',
    apiKey: 'sk-test',
    baseUrl: '',
    ...over,
  }
}

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

describe('settings connection revision', () => {
  it('keeps a well-formed saved revision when no connection field is corrected', () => {
    vi.stubGlobal('localStorage', memoryStorage({
      'study-coach:settings': JSON.stringify({
        provider: 'openai',
        model: 'gpt-4o-mini',
        apiKey: 'sk-test',
        baseUrl: '',
        connectionRevision: REVISION_A,
      }),
    }))

    const settings = useSettings()

    expect(settings.connectionRevision).toBe(REVISION_A)
    expect(settings.model).toBe('gpt-4o-mini')
  })

  it('drops the saved revision when any connection field needs correction', () => {
    const cases: Record<string, unknown>[] = [
      { provider: 'undefined', connectionRevision: REVISION_A },
      { provider: 'openai', model: '   ', connectionRevision: REVISION_A },
      { provider: 'openai', model: 'gpt-4o-mini', apiKey: 42, connectionRevision: REVISION_A },
      { provider: 'openai', model: 'gpt-4o-mini', baseUrl: true, connectionRevision: REVISION_A },
    ]
    for (const saved of cases) {
      setActivePinia(createPinia())
      vi.stubGlobal('localStorage', memoryStorage({
        'study-coach:settings': JSON.stringify(saved),
      }))

      const settings = useSettings()

      expect(settings.connectionRevision).toBe('')
    }
  })

  it('drops a malformed saved revision', () => {
    vi.stubGlobal('localStorage', memoryStorage({
      'study-coach:settings': JSON.stringify({
        provider: 'openai',
        model: 'gpt-4o-mini',
        apiKey: 'sk-test',
        baseUrl: '',
        connectionRevision: 'not-a-uuid',
      }),
    }))

    const settings = useSettings()

    expect(settings.connectionRevision).toBe('')
  })

  it('hydrates a cloud saved model that is missing or blank as empty with unknown capability', () => {
    for (const savedModel of [undefined, '   ']) {
      setActivePinia(createPinia())
      vi.stubGlobal('localStorage', memoryStorage({
        'study-coach:settings': JSON.stringify({
          provider: 'openai',
          apiKey: 'sk-test',
          model: savedModel,
        }),
      }))

      const settings = useSettings()

      expect(settings.model).toBe('')
      expect(settings.provider).toBe('openai')
      expect(settings.toolCapable).toBeNull()
    }
  })

  it('keeps the Ollama default model for ollama and token-only records', () => {
    setActivePinia(createPinia())
    vi.stubGlobal('localStorage', memoryStorage({
      'study-coach:settings': JSON.stringify({ provider: 'ollama', model: '' }),
    }))
    expect(useSettings().model).toBe('gemma3:4b')

    setActivePinia(createPinia())
    vi.stubGlobal('localStorage', memoryStorage({
      'study-coach:settings': JSON.stringify({ accessToken: 't', tier: 'guest' }),
    }))
    const tokenOnly = useSettings()
    expect(tokenOnly.provider).toBe('ollama')
    expect(tokenOnly.model).toBe('gemma3:4b')
    expect(tokenOnly.connectionRevision).toBe('')
    expect(tokenOnly.toolCapable).toBeNull()
  })

  it('performs no randomUUID calls while hydrating token-only settings', () => {
    const randomUUID = vi.fn(() => '00000000-0000-4000-8000-000000000001')
    vi.stubGlobal('crypto', { randomUUID })
    vi.stubGlobal('localStorage', memoryStorage({
      'study-coach:settings': JSON.stringify({ accessToken: 't', tier: 'guest' }),
    }))

    useSettings()

    expect(randomUUID).not.toHaveBeenCalled()
  })
})

describe('settings v2 capability cache', () => {
  it('restores capability only from a matching v2 record after refresh', () => {
    vi.stubGlobal('localStorage', memoryStorage({
      'study-coach:settings': JSON.stringify({
        provider: 'openai',
        model: 'gpt-4o-mini',
        apiKey: 'sk-test',
        baseUrl: '',
        connectionRevision: REVISION_A,
      }),
      [CAPABILITY_CACHE_KEY]: capabilityRecordValue(REVISION_A, true),
    }))

    const settings = useSettings()

    expect(settings.toolCapable).toBe(true)
  })

  it('ignores v2 records with a different revision, bad schema, or bad types', () => {
    const records = [
      capabilityRecordValue(REVISION_B, true),
      JSON.stringify({ schemaVersion: 1, connectionRevision: REVISION_A, toolCapable: true }),
      JSON.stringify({ schemaVersion: 2, connectionRevision: REVISION_A, toolCapable: 'yes' }),
      JSON.stringify({ schemaVersion: 2, connectionRevision: 'not-a-uuid', toolCapable: true }),
      'not-json',
    ]
    for (const record of records) {
      setActivePinia(createPinia())
      vi.stubGlobal('localStorage', memoryStorage({
        'study-coach:settings': JSON.stringify({
          provider: 'openai',
          model: 'gpt-4o-mini',
          apiKey: 'sk-test',
          baseUrl: '',
          connectionRevision: REVISION_A,
        }),
        [CAPABILITY_CACHE_KEY]: record,
      }))

      const settings = useSettings()

      expect(settings.toolCapable).toBeNull()
    }
  })

  it('never reads the legacy per-model v1 cache key', () => {
    vi.stubGlobal('localStorage', memoryStorage({
      'study-coach:settings': JSON.stringify({
        provider: 'ollama',
        model: 'gemma3:4b',
        connectionRevision: REVISION_A,
      }),
      'study-coach:tool-capable:gemma3:4b': 'true',
    }))

    const settings = useSettings()

    expect(settings.toolCapable).toBeNull()
  })

  it('omits toolCapable and pending metadata from the persisted settings JSON', () => {
    vi.stubGlobal('localStorage', memoryStorage())
    stubSequentialUuids()
    const settings = useSettings()
    settings.updateConnection('model', 'gpt-4o-mini')
    settings.toolCapable = true

    settings.persist()

    const persisted = JSON.parse(localStorage.getItem('study-coach:settings') ?? '{}')
    expect(persisted).not.toHaveProperty('toolCapable')
    expect(persisted).not.toHaveProperty('pendingCapability')
    expect(persisted.connectionRevision).toBe('00000000-0000-4000-8000-000000000001')
  })

  it('never writes the API key into the v2 cache key or value', () => {
    vi.stubGlobal('localStorage', memoryStorage())
    stubSequentialUuids()
    const settings = useSettings()
    settings.updateConnection('provider', 'openai')
    settings.updateConnection('model', 'gpt-4o-mini')
    settings.updateConnection('apiKey', 'sk-live-secret-marker')
    settings.persist()
    settings.commitToolCheckResult(
      connectionSnapshot(settings.connectionRevision, { apiKey: 'sk-live-secret-marker' }),
      true,
    )
    settings.persist()

    const rawValue = localStorage.getItem(CAPABILITY_CACHE_KEY) ?? ''
    expect(rawValue).not.toContain('sk-live-secret-marker')
    expect(JSON.parse(rawValue)).toMatchObject({
      schemaVersion: 2,
      connectionRevision: settings.connectionRevision,
      toolCapable: true,
    })
  })
})

describe('settings updateConnection and capability commits', () => {
  function seedSavedSettings(revision: string): void {
    vi.stubGlobal('localStorage', memoryStorage({
      'study-coach:settings': JSON.stringify({
        provider: 'openai',
        model: 'gpt-4o-mini',
        apiKey: 'sk-test',
        baseUrl: '',
        connectionRevision: revision,
      }),
      [CAPABILITY_CACHE_KEY]: capabilityRecordValue(revision, true),
    }))
  }

  it('rotates the revision and clears capability when a connection value really changes', () => {
    seedSavedSettings(REVISION_A)
    stubSequentialUuids()
    const settings = useSettings()
    expect(settings.toolCapable).toBe(true)

    const changed = settings.updateConnection('model', 'gpt-4o')

    expect(changed).toBe(true)
    expect(settings.connectionRevision).toBe('00000000-0000-4000-8000-000000000001')
    expect(settings.toolCapable).toBeNull()
  })

  it('does not rotate on duplicate assignment or non-connection fields', () => {
    seedSavedSettings(REVISION_A)
    stubSequentialUuids()
    const settings = useSettings()
    const before = settings.connectionRevision

    expect(settings.updateConnection('model', 'gpt-4o-mini')).toBe(false)
    settings.judgeModel = 'judge-x'
    settings.language = 'zh-CN'

    expect(settings.connectionRevision).toBe(before)
    expect(settings.toolCapable).toBe(true)
  })

  it('does not resurrect capability across A→B→A edits', () => {
    seedSavedSettings(REVISION_A)
    stubSequentialUuids()
    const settings = useSettings()

    settings.updateConnection('model', 'gpt-4o')       // → rev 1
    const pendingSnapshot = connectionSnapshot(settings.connectionRevision)
    settings.commitToolCheckResult(pendingSnapshot, true)
    settings.updateConnection('model', 'claude-x')     // → rev 2
    settings.updateConnection('model', 'gpt-4o')       // → rev 3

    expect(settings.connectionRevision).not.toBe(REVISION_A)
    expect(settings.connectionRevision).not.toBe(pendingSnapshot.revision)
    expect(settings.toolCapable).toBeNull()

    // The stale result from the first "A" generation must not commit anymore,
    // and the saved generation-A record is untouched by the unsaved edits.
    settings.commitToolCheckResult(pendingSnapshot, true)
    expect(settings.toolCapable).toBeNull()
    expect(localStorage.getItem(CAPABILITY_CACHE_KEY)).toBe(capabilityRecordValue(REVISION_A, true))
  })

  it('writes the v2 record immediately when the checked configuration is saved', () => {
    seedSavedSettings(REVISION_A)
    stubSequentialUuids()
    const settings = useSettings()
    settings.persist() // save so stored config matches the current connection
    settings.updateConnection('model', 'gpt-4o') // revision rotates; unsaved
    settings.persist()

    settings.commitToolCheckResult(
      connectionSnapshot(settings.connectionRevision, { model: 'gpt-4o' }),
      false,
    )

    expect(settings.toolCapable).toBe(false)
    expect(JSON.parse(localStorage.getItem(CAPABILITY_CACHE_KEY) ?? '{}')).toMatchObject({
      schemaVersion: 2,
      connectionRevision: settings.connectionRevision,
      toolCapable: false,
    })
  })

  it('keeps an unsaved check result in memory only until a matching Save', () => {
    seedSavedSettings(REVISION_A)
    stubSequentialUuids()
    const settings = useSettings()
    settings.updateConnection('model', 'gpt-4o') // unsaved change

    settings.commitToolCheckResult(
      connectionSnapshot(settings.connectionRevision, { model: 'gpt-4o' }),
      true,
    )
    expect(settings.toolCapable).toBe(true)
    expect(localStorage.getItem(CAPABILITY_CACHE_KEY)).toBe(capabilityRecordValue(REVISION_A, true))

    settings.persist()

    expect(JSON.parse(localStorage.getItem(CAPABILITY_CACHE_KEY) ?? '{}')).toMatchObject({
      connectionRevision: settings.connectionRevision,
      toolCapable: true,
    })
  })

  it('ignores a null tool-check result for caching', () => {
    seedSavedSettings(REVISION_A)
    stubSequentialUuids()
    const settings = useSettings()
    settings.persist()

    settings.commitToolCheckResult(connectionSnapshot(settings.connectionRevision), null)

    expect(settings.toolCapable).toBeNull()
    expect(JSON.parse(localStorage.getItem(CAPABILITY_CACHE_KEY) ?? '{}'))
      .toMatchObject({ connectionRevision: REVISION_A, toolCapable: true })
  })

  it('rejects commits whose revision or connection snapshot no longer match', () => {
    vi.stubGlobal('localStorage', memoryStorage({
      'study-coach:settings': JSON.stringify({
        provider: 'openai',
        model: 'gpt-4o-mini',
        apiKey: 'sk-test',
        baseUrl: '',
        connectionRevision: REVISION_A,
      }),
    }))
    stubSequentialUuids()
    const settings = useSettings()
    expect(settings.toolCapable).toBeNull()

    // Snapshot field differs from the current connection: not adopted.
    settings.commitToolCheckResult(
      connectionSnapshot(REVISION_A, { apiKey: 'sk-other' }),
      true,
    )
    expect(settings.toolCapable).toBeNull()

    // Snapshot revision is not the current generation: not adopted.
    settings.commitToolCheckResult(
      connectionSnapshot('00000000-0000-4000-8000-000000000099'),
      true,
    )
    expect(settings.toolCapable).toBeNull()
  })

  it('clears the current revision record when a manual re-test starts', () => {
    seedSavedSettings(REVISION_A)
    const settings = useSettings()

    settings.beginToolRecheck()

    expect(settings.toolCapable).toBeNull()
    expect(localStorage.getItem(CAPABILITY_CACHE_KEY)).toBeNull()
  })

  it('keeps a different-revision record when re-testing an edited unsaved config', () => {
    seedSavedSettings(REVISION_A)
    stubSequentialUuids()
    const settings = useSettings()
    settings.updateConnection('model', 'gpt-4o')

    settings.beginToolRecheck()

    expect(localStorage.getItem(CAPABILITY_CACHE_KEY)).toBe(capabilityRecordValue(REVISION_A, true))
  })
})

describe('settings persist invalidation', () => {
  it('conservatively rotates the revision for direct four-field store edits', () => {
    vi.stubGlobal('localStorage', memoryStorage())
    stubSequentialUuids()
    const settings = useSettings()
    settings.provider = 'openai'
    settings.model = 'gpt-4o-mini'
    settings.apiKey = 'sk-test'
    settings.persist() // saved as generation 1
    const savedRevision = settings.connectionRevision
    settings.commitToolCheckResult(connectionSnapshot(settings.connectionRevision), true)
    settings.persist()
    expect(settings.toolCapable).toBe(true)

    settings.model = 'claude-x' // direct bypass of updateConnection
    settings.persist()

    expect(settings.connectionRevision).not.toBe(savedRevision)
    expect(settings.toolCapable).toBeNull()
  })

  it('does not rotate the revision when persisting unchanged settings', () => {
    vi.stubGlobal('localStorage', memoryStorage())
    stubSequentialUuids()
    const settings = useSettings()
    settings.updateConnection('model', 'gpt-4o-mini')
    settings.persist()
    const savedRevision = settings.connectionRevision

    settings.persist()

    expect(settings.connectionRevision).toBe(savedRevision)
  })

  it('keeps llmHeaders deterministic fallback semantics for false', () => {
    vi.stubGlobal('localStorage', memoryStorage())
    const settings = useSettings()
    settings.toolCapable = false

    expect(llmHeaders(settings.$state)['x-planner-mode']).toBe('deterministic')
    expect(llmHeaders(settings.$state, { plannerMode: 'agent_loop' })['x-planner-mode'])
      .toBe('agent_loop')
  })
})

describe('settings ensureConnectionRevision', () => {
  it('creates a valid revision lazily for empty generations without rotating valid ones', () => {
    vi.stubGlobal('localStorage', memoryStorage({
      'study-coach:settings': JSON.stringify({ provider: 'ollama', model: 'gemma3:4b' }),
    }))
    stubSequentialUuids()
    const settings = useSettings()
    expect(settings.connectionRevision).toBe('')

    expect(settings.ensureConnectionRevision()).toBe(true)
    expect(settings.connectionRevision).toBe('00000000-0000-4000-8000-000000000001')

    expect(settings.ensureConnectionRevision()).toBe(false)
    expect(settings.connectionRevision).toBe('00000000-0000-4000-8000-000000000001')
  })

  it('lets a first detection on a revision-less saved config persist after Save', () => {
    vi.stubGlobal('localStorage', memoryStorage({
      'study-coach:settings': JSON.stringify({ provider: 'ollama', model: 'gemma3:4b' }),
    }))
    stubSequentialUuids()
    const settings = useSettings()

    settings.ensureConnectionRevision()
    settings.commitToolCheckResult(
      connectionSnapshot(settings.connectionRevision, {
        provider: 'ollama', model: 'gemma3:4b', apiKey: '', baseUrl: '',
      }),
      true,
    )
    // Memory only until the explicit Save of the same configuration.
    expect(localStorage.getItem(CAPABILITY_CACHE_KEY)).toBeNull()

    settings.persist()

    expect(JSON.parse(localStorage.getItem(CAPABILITY_CACHE_KEY) ?? '{}')).toMatchObject({
      schemaVersion: 2,
      connectionRevision: settings.connectionRevision,
      toolCapable: true,
    })
    expect(JSON.parse(localStorage.getItem('study-coach:settings') ?? '{}').connectionRevision)
      .toBe(settings.connectionRevision)
  })
})
