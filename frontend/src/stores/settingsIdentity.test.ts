import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { memoryStorage } from '../test/memoryStorage'

const nativeSubtle = globalThis.crypto.subtle

function authResponse(token: string, tier = 'guest', userId = 'user-1'): Response {
  return {
    ok: true,
    json: vi.fn().mockResolvedValue({ access_token: token, user_id: userId, tier }),
  } as unknown as Response
}

describe('anonymous identity provisioning', () => {
  beforeEach(() => {
    vi.resetModules()
    setActivePinia(createPinia())
    vi.stubGlobal('localStorage', memoryStorage())
    vi.stubGlobal('crypto', { randomUUID: vi.fn(() => 'test-fingerprint') })
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('starts a new request after a transient provisioning failure', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce({ ok: false } as Response)
      .mockResolvedValueOnce(authResponse('recovered-token'))
    vi.stubGlobal('fetch', fetchMock)
    const { getAccessToken } = await import('./settings')

    await expect(getAccessToken()).rejects.toThrow('anonymous auth failed')
    await expect(getAccessToken()).resolves.toBe('recovered-token')

    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it('does not overwrite a token that arrives after the settings store was created', async () => {
    let finishProvisioning!: (response: Response) => void
    vi.stubGlobal('fetch', vi.fn(() => new Promise<Response>((resolve) => {
      finishProvisioning = resolve
    })))
    const { getAccessToken, useSettings } = await import('./settings')

    const provisioning = getAccessToken()
    const settings = useSettings()
    expect(settings.accessToken).toBe('')

    finishProvisioning(authResponse('new-token'))
    await expect(provisioning).resolves.toBe('new-token')
    settings.persist()

    expect(settings.accessToken).toBe('new-token')
    expect(JSON.parse(localStorage.getItem('study-coach:settings') ?? '{}')).toMatchObject({
      accessToken: 'new-token',
      tier: 'guest',
    })
  })

  it('converges multi-tab factory recovery without resurrecting a deleted identity', async () => {
    const sharedStorage = memoryStorage()
    sharedStorage.setItem('study-coach:fingerprint', 'pre-factory-fp')
    sharedStorage.setItem('study-coach:settings', JSON.stringify({
      accessToken: 'stale-deleted-user-token',
      provider: 'ollama',
      model: 'gemma3:4b',
    }))

    // Tab B starts provisioning against the pre-factory fingerprint.
    vi.resetModules()
    vi.stubGlobal('localStorage', sharedStorage)
    vi.stubGlobal('crypto', { randomUUID: vi.fn(() => 'tab-b-fp') })
    let finishTabB!: (response: Response) => void
    vi.stubGlobal('fetch', vi.fn(() => new Promise<Response>((resolve) => {
      finishTabB = resolve
    })))
    const tabB = await import('./settings')
    // Clear token so Tab B actually enters anonymous provisioning.
    sharedStorage.setItem('study-coach:settings', JSON.stringify({
      provider: 'ollama',
      model: 'gemma3:4b',
    }))
    const staleB = tabB.getAccessToken()

    // Tab A recovers first via factory identity provision (new fingerprint + token).
    vi.resetModules()
    vi.stubGlobal('localStorage', sharedStorage)
    vi.stubGlobal('crypto', { randomUUID: vi.fn(() => 'tab-a-recovery-fp') })
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(authResponse('tab-a-recovery-token')))
    const tabA = await import('./settings')
    await expect(tabA.provisionFactoryIdentity()).resolves.toBe('tab-a-recovery-token')

    finishTabB(authResponse('tab-b-would-resurrect-old'))
    await expect(staleB).rejects.toThrow('identity provisioning invalidated')

    expect(sharedStorage.getItem('study-coach:fingerprint')).toBe('tab-a-recovery-fp')
    expect(JSON.parse(sharedStorage.getItem('study-coach:settings') ?? '{}')).toMatchObject({
      accessToken: 'tab-a-recovery-token',
    })
    expect(JSON.parse(sharedStorage.getItem('study-coach:settings') ?? '{}').accessToken)
      .not.toBe('stale-deleted-user-token')
    expect(JSON.parse(sharedStorage.getItem('study-coach:settings') ?? '{}').accessToken)
      .not.toBe('tab-b-would-resurrect-old')
  })

  it('stages one recovery fingerprint before concurrent tabs create backend users', async () => {
    const sharedStorage = memoryStorage({
      'study-coach:fingerprint': 'pre-factory-fp',
      'study-coach:settings': JSON.stringify({
        accessToken: 'stale-deleted-user-token',
        provider: 'ollama',
        model: 'gemma3:4b',
      }),
    })
    const backendUsers = new Map<string, string>()
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      const body = JSON.parse(String(init?.body ?? '{}')) as { fingerprint?: string }
      const fingerprint = body.fingerprint ?? ''
      let userId = backendUsers.get(fingerprint)
      if (!userId) {
        userId = `user-${backendUsers.size + 1}`
        backendUsers.set(fingerprint, userId)
      }
      return authResponse(`token-${userId}`, 'guest', userId)
    })
    vi.stubGlobal('fetch', fetchMock)
    vi.stubGlobal('localStorage', sharedStorage)
    vi.stubGlobal('crypto', {
      randomUUID: vi.fn()
        .mockReturnValueOnce('tab-a-random-fingerprint')
        .mockReturnValueOnce('tab-b-random-fingerprint'),
      subtle: nativeSubtle,
    })

    vi.resetModules()
    const tabA = await import('./settings')
    vi.resetModules()
    const tabB = await import('./settings')

    const recoveryA = await tabA.stageFactoryRecoveryFingerprint()

    // Factory clear removes the stale identity, but the staged target survives
    // for a second tab whose stale summary resolves later.
    sharedStorage.removeItem('study-coach:settings')
    sharedStorage.removeItem('study-coach:fingerprint')
    const recoveryB = await tabB.stageFactoryRecoveryFingerprint()
    expect(recoveryA).toBe(recoveryB)

    const [tokenA, tokenB] = await Promise.all([
      tabA.provisionFactoryIdentity(recoveryA),
      tabB.provisionFactoryIdentity(recoveryB),
    ])

    expect(backendUsers.size).toBe(1)
    expect(tokenA).toBe(tokenB)
    expect(sharedStorage.getItem('study-coach:fingerprint')).toBe(recoveryA)

    const nextResetFingerprint = await tabA.stageFactoryRecoveryFingerprint(true)
    expect(nextResetFingerprint).not.toBe(recoveryA)
    expect(sharedStorage.getItem('study-coach:factory-recovery-fingerprint'))
      .toBe(nextResetFingerprint)
  })

  it('does not let a stale tab write old token after shared fingerprint changes', async () => {
    const sharedStorage = memoryStorage()

    // Tab B: start anonymous provisioning against empty shared storage.
    vi.resetModules()
    vi.stubGlobal('localStorage', sharedStorage)
    vi.stubGlobal('crypto', { randomUUID: vi.fn(() => 'tab-b-fingerprint') })
    let finishTabB!: (response: Response) => void
    vi.stubGlobal('fetch', vi.fn(() => new Promise<Response>((resolve) => {
      finishTabB = resolve
    })))
    const tabB = await import('./settings')
    const staleB = tabB.getAccessToken()

    // Tab A: factory reset replaces shared fingerprint/token before Tab B resumes.
    vi.resetModules()
    vi.stubGlobal('localStorage', sharedStorage)
    vi.stubGlobal('crypto', { randomUUID: vi.fn(() => 'tab-a-new-fingerprint') })
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(authResponse('tab-a-new-token')))
    const tabA = await import('./settings')
    await expect(tabA.provisionFactoryIdentity()).resolves.toBe('tab-a-new-token')

    finishTabB(authResponse('tab-b-old-token'))
    await expect(staleB).rejects.toThrow('identity provisioning invalidated')

    expect(sharedStorage.getItem('study-coach:fingerprint')).toBe('tab-a-new-fingerprint')
    expect(JSON.parse(sharedStorage.getItem('study-coach:settings') ?? '{}')).toMatchObject({
      accessToken: 'tab-a-new-token',
      provider: 'ollama',
      model: 'gemma3:4b',
      apiKey: '',
    })
  })

  it('recovers from malformed settings JSON when persisting a new token', async () => {
    localStorage.setItem('study-coach:settings', 'null')
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(authResponse('repaired-token')))
    const { getAccessToken } = await import('./settings')

    await expect(getAccessToken()).resolves.toBe('repaired-token')
    expect(JSON.parse(localStorage.getItem('study-coach:settings') ?? '{}')).toMatchObject({
      accessToken: 'repaired-token',
      provider: 'ollama',
      model: 'gemma3:4b',
    })
  })

  it('preserves valid provider preferences during ordinary anonymous provisioning', async () => {
    localStorage.setItem('study-coach:settings', JSON.stringify({
      provider: 'openai',
      model: 'gpt-4o-mini',
      apiKey: 'sk-test',
      language: 'zh-CN',
    }))
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(authResponse('kept-token')))
    const { getAccessToken } = await import('./settings')

    await expect(getAccessToken()).resolves.toBe('kept-token')
    expect(JSON.parse(localStorage.getItem('study-coach:settings') ?? '{}')).toMatchObject({
      accessToken: 'kept-token',
      provider: 'openai',
      model: 'gpt-4o-mini',
      apiKey: 'sk-test',
      language: 'zh-CN',
    })
  })

  it('rejects non-string access_token payloads without writing storage', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: vi.fn().mockResolvedValue({ access_token: 123, tier: 'guest' }),
    } as unknown as Response))
    const { getAccessToken } = await import('./settings')

    await expect(getAccessToken()).rejects.toThrow('anonymous auth failed')
    expect(localStorage.getItem('study-coach:settings')).toBeNull()
  })

  it('recovers when settings value is a primitive or array', async () => {
    for (const bad of ['"oops"', '123', '[1,2]', 'true']) {
      vi.resetModules()
      setActivePinia(createPinia())
      const storage = memoryStorage()
      storage.setItem('study-coach:settings', bad)
      vi.stubGlobal('localStorage', storage)
      vi.stubGlobal('crypto', { randomUUID: vi.fn(() => 'fp-repair') })
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue(authResponse('fixed-token')))
      const { getAccessToken } = await import('./settings')

      await expect(getAccessToken()).resolves.toBe('fixed-token')
      expect(JSON.parse(storage.getItem('study-coach:settings') ?? '{}')).toMatchObject({
        accessToken: 'fixed-token',
        provider: 'ollama',
      })
    }
  })
})
