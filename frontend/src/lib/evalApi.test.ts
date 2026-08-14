import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const { getAccessTokenMock } = vi.hoisted(() => ({
  getAccessTokenMock: vi.fn(() => Promise.resolve('module-token')),
}))

vi.mock('../stores/settings', () => ({
  getAccessToken: getAccessTokenMock,
}))

import { EvalApiError, evalHeaders, parseSseFrame, streamLearningRun } from './evalApi'
import type { EvalConnectionSnapshot, LearningRunRequest } from './evalContracts'

const REQUEST: LearningRunRequest = {
  experiment_id: 'tutor-prompt-regression-v1',
  task_case_id: 'tgqa-004',
  variant_id: 'tutor-v3',
  run_profile: 'evaluation',
}

const CONNECTION: EvalConnectionSnapshot = {
  provider: 'ollama',
  model: 'llama3.2',
  apiKey: 'secret-value',
  baseUrl: 'https://secret.example/v1',
}

function jsonResponse(body: unknown, status: number): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: vi.fn().mockResolvedValue(body),
    body: null,
  } as unknown as Response
}

function streamResponse(chunks: string[]): Response {
  const encoder = new TextEncoder()
  let index = 0
  return {
    ok: true,
    status: 200,
    body: {
      getReader: () => ({
        read: async () => {
          if (index >= chunks.length) return { done: true, value: undefined }
          const value = encoder.encode(chunks[index])
          index += 1
          return { done: false, value }
        },
        cancel: vi.fn().mockResolvedValue(undefined),
      }),
    },
  } as unknown as Response
}

describe('evalHeaders', () => {
  it('sends bearer and registry-matching connection headers only', () => {
    expect(evalHeaders('signed-token', CONNECTION)).toEqual({
      Authorization: 'Bearer signed-token',
      'Content-Type': 'application/json',
      'x-provider': 'ollama',
      'x-model': 'llama3.2',
      'x-api-key': 'secret-value',
      'x-base-url': 'https://secret.example/v1',
    })
  })
})

describe('streamLearningRun', () => {
  beforeEach(() => {
    getAccessTokenMock.mockReset()
    getAccessTokenMock.mockResolvedValue('signed-token')
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('awaits a signed token before posting the first stream request', async () => {
    let resolveToken!: (token: string) => void
    getAccessTokenMock.mockReturnValueOnce(new Promise(resolve => {
      resolveToken = resolve
    }))
    const fetchMock = vi.fn().mockResolvedValue(streamResponse([]))
    vi.stubGlobal('fetch', fetchMock)

    const pending = streamLearningRun(REQUEST, CONNECTION, () => undefined, new AbortController().signal)
    await Promise.resolve()
    expect(fetchMock).not.toHaveBeenCalled()

    resolveToken('signed-token')
    await pending
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(fetchMock.mock.calls[0]?.[0]).toBe('/api/eval/runs/stream')
    expect(fetchMock.mock.calls[0]?.[1]).toMatchObject({
      method: 'POST',
      headers: evalHeaders('signed-token', CONNECTION),
      body: JSON.stringify(REQUEST),
    })
    const body = JSON.parse(String(fetchMock.mock.calls[0]?.[1]?.body))
    expect(body).toEqual(REQUEST)
    expect(body).not.toHaveProperty('prompt')
    expect(body).not.toHaveProperty('corpus_path')
    expect(body).not.toHaveProperty('expected_answer')
    expect(body).not.toHaveProperty('api_key')
    expect(body).not.toHaveProperty('judge')
  })

  it('parses split frames and multiple frames in one chunk', async () => {
    const events: string[] = []
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(streamResponse([
      'data: {"schema_version":"eval-api-v1","type":"run_created","run_id":"run-1"}\n\ndata: {"schema_version":"eval-api-v1","type":"stage_started"',
      ',"run_id":"run-1","stage":"tutor"}\n\n',
    ])))

    await streamLearningRun(REQUEST, CONNECTION, event => {
      events.push(event.type)
    }, new AbortController().signal)

    expect(events).toEqual(['run_created', 'stage_started'])
  })

  it('fails closed on a malformed frame', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(streamResponse([
      'data: {"schema_version":"eval-api-v1","type":"run_created","run_id":"run-1"}\n\ndata: {not-json}\n\n',
    ])))

    await expect(streamLearningRun(
      REQUEST,
      CONNECTION,
      () => undefined,
      new AbortController().signal,
    )).rejects.toMatchObject({ code: 'malformed_event' })
  })

  it('emits a terminal run_finished frame', async () => {
    const events: string[] = []
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(streamResponse([
      'data: {"schema_version":"eval-api-v1","type":"run_finished","run_id":"run-1","lifecycle":"finished","outcome":"success"}\n\n',
    ])))

    await streamLearningRun(REQUEST, CONNECTION, event => {
      events.push(event.type)
    }, new AbortController().signal)

    expect(events).toEqual(['run_finished'])
  })

  it('keeps evaluation_busy active id and kind instead of flattening to failed: 409', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({
      detail: {
        code: 'evaluation_busy',
        message: 'another evaluation is already running',
        active_entity_id: 'run-active-001',
        active_kind: 'run',
      },
    }, 409)))

    const error = await streamLearningRun(
      REQUEST,
      CONNECTION,
      () => undefined,
      new AbortController().signal,
    ).catch((reason: unknown) => reason)

    expect(error).toBeInstanceOf(EvalApiError)
    expect(error).toMatchObject({
      status: 409,
      code: 'evaluation_busy',
      active_entity_id: 'run-active-001',
      active_kind: 'run',
      message: 'another evaluation is already running',
    })
    expect(String(error)).not.toContain('failed: 409')
  })
})

describe('parseSseFrame', () => {
  it('ignores blank frames', () => {
    expect(parseSseFrame('')).toBeNull()
    expect(parseSseFrame('\n')).toBeNull()
  })
})
