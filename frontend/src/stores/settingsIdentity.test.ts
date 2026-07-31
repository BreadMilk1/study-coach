import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { memoryStorage } from '../test/memoryStorage'

function authResponse(token: string): Response {
  return {
    ok: true,
    json: vi.fn().mockResolvedValue({ access_token: token, tier: 'guest' }),
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
})
