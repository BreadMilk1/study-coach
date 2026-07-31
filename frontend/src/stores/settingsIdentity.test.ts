import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { memoryStorage } from '../test/memoryStorage'

function authResponse(token: string, tier = 'guest'): Response {
  return {
    ok: true,
    json: vi.fn().mockResolvedValue({ access_token: token, tier }),
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
