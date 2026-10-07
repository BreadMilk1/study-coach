import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { memoryStorage } from '../test/memoryStorage'

type ConnectionCheckInput = import('./api').ConnectionCheckInput
type ModelCheckErrorInstance = InstanceType<typeof import('./api').ModelCheckError>

let checkToolCapable: typeof import('./api').checkToolCapable
let pingModel: typeof import('./api').pingModel
let ModelCheckError: typeof import('./api').ModelCheckError

// Import the real api module only after a token is present in storage, so the
// module-level getAccessToken() short-circuits instead of provisioning.
beforeAll(async () => {
  vi.stubGlobal('localStorage', memoryStorage({
    'study-coach:settings': JSON.stringify({ accessToken: 'preseeded-token', tier: 'guest' }),
  }))
  ;({ checkToolCapable, pingModel, ModelCheckError } = await import('./api'))
})

afterAll(() => {
  vi.unstubAllGlobals()
})

afterEach(() => {
  vi.restoreAllMocks()
})

const CONNECTION: ConnectionCheckInput = {
  provider: 'openai',
  model: 'gpt-4o-mini',
  apiKey: 'sk-test',
  baseUrl: 'https://api.openai.test/v1',
}

function jsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: vi.fn(async () => body),
  } as unknown as Response
}

function stubFetch(handler: (url: string, init?: RequestInit) => Response): ReturnType<typeof vi.fn> {
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input)
    if (!url.includes('/api/models/')) {
      throw new Error(`unexpected fetch target: ${url}`)
    }
    return handler(url, init)
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

describe('model check API helpers', () => {
  it('sends the connection headers and never auto-initiates anything else', async () => {
    const fetchMock = stubFetch(() => jsonResponse({
      tool_capable: true, model: 'gpt-4o-mini', note: 'ok',
    }))

    await checkToolCapable(CONNECTION)

    expect(fetchMock).toHaveBeenCalledTimes(1)
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit]
    expect(init.method).toBeUndefined()
    expect(init.headers).toMatchObject({
      'x-provider': 'openai',
      'x-model': 'gpt-4o-mini',
      'x-api-key': 'sk-test',
      'x-base-url': 'https://api.openai.test/v1',
    })
  })

  it('omits empty key and base-url headers', async () => {
    const fetchMock = stubFetch(() => jsonResponse({ ok: true, model: 'm', latency_ms: 1, note: '' }))
    await pingModel({ provider: 'ollama', model: 'gemma3:4b', apiKey: '', baseUrl: '' })

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit]
    expect(init.headers).toMatchObject({ 'x-provider': 'ollama', 'x-model': 'gemma3:4b' })
    expect(init.headers).not.toHaveProperty('x-api-key')
    expect(init.headers).not.toHaveProperty('x-base-url')
  })

  it('keeps a nullable tool_capable in the parsed DTO', async () => {
    stubFetch(() => jsonResponse({ tool_capable: null, model: 'gemma3:4b', note: 'incomplete' }))

    const dto = await checkToolCapable(CONNECTION)

    expect(dto.tool_capable).toBeNull()
    expect(dto.note).toBe('incomplete')
  })

  it('maps a 400 invalid_llm_config body to fixed safe field messages', async () => {
    const cases = [
      { field: 'provider', expected: ModelCheckErrorFieldMessage.provider },
      { field: 'model', expected: ModelCheckErrorFieldMessage.model },
      { field: 'api_key', expected: ModelCheckErrorFieldMessage.api_key },
    ]
    for (const { field, expected } of cases) {
      stubFetch(() => jsonResponse(
        { detail: { code: 'invalid_llm_config', field, message: 'backend sentence' } },
        400,
      ))

      await expect(pingModel(CONNECTION)).rejects.toSatisfy((error: unknown) => {
        expect(error).toBeInstanceOf(ModelCheckError)
        const checkError = error as ModelCheckErrorInstance
        expect(checkError.message).toBe(expected)
        expect(checkError.field).toBe(field)
        // The raw backend sentence and any request input are not echoed.
        expect(checkError.message).not.toContain('backend sentence')
        expect(checkError.message).not.toContain('sk-test')
        return true
      })
    }
  })

  it('maps a malformed 400 body to the fixed generic message', async () => {
    stubFetch(() => jsonResponse({ detail: 'totally unexpected' }, 400))

    await expect(pingModel(CONNECTION)).rejects.toSatisfy((error: unknown) => {
      expect(error).toBeInstanceOf(ModelCheckError)
      expect((error as ModelCheckErrorInstance).message).toBe(FixedGenericCheckMessage)
      expect((error as ModelCheckErrorInstance).field).toBeNull()
      return true
    })
  })

  it('treats inherited or unknown 400 field selectors as unknown fields', async () => {
    // Only the three own keys of the fixed field table may select a field
    // message; inherited Object properties (constructor/toString/__proto__)
    // must fall back to the fixed generic message with field=null and never
    // surface the raw body values.
    for (const field of ['constructor', 'toString', '__proto__', 'field-unknown']) {
      stubFetch(() => jsonResponse(
        { detail: { code: 'invalid_llm_config', field, message: 'backend sentence' } },
        400,
      ))

      await expect(pingModel(CONNECTION)).rejects.toSatisfy((error: unknown) => {
        expect(error).toBeInstanceOf(ModelCheckError)
        const checkError = error as ModelCheckErrorInstance
        expect(checkError.message).toBe(FixedGenericCheckMessage)
        expect(checkError.field).toBeNull()
        expect(checkError.message).not.toContain('backend sentence')
        expect(checkError.message).not.toContain(field)
        return true
      })
    }
  })

  it('never surfaces raw error bodies for non-400 failures', async () => {
    stubFetch(() => jsonResponse({ detail: 'RELEASE-SECRET-MARKER-9f3a' }, 500))

    await expect(checkToolCapable(CONNECTION)).rejects.toSatisfy((error: unknown) => {
      expect(error).toBeInstanceOf(ModelCheckError)
      const message = (error as ModelCheckErrorInstance).message
      expect(message).toBe(FixedGenericCheckMessage)
      expect(message).not.toContain('SECRET')
      return true
    })
  })
})

// Fixed safe texts asserted above; defined here so the tests fail loudly
// while api.ts has not implemented them yet.
const ModelCheckErrorFieldMessage = {
  provider: 'The provider is not supported. Choose ollama, openai, anthropic, or gemini in Settings.',
  model: 'A model name is required for the selected provider.',
  api_key: 'An API key is required for the selected provider.',
}

const FixedGenericCheckMessage = 'The connection check could not be completed.'
