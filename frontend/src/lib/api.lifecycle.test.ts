import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const { getAccessTokenMock } = vi.hoisted(() => ({
  getAccessTokenMock: vi.fn(() => Promise.resolve('module-token')),
}))

vi.mock('../stores/settings', () => ({
  authHeaders: vi.fn(() => ({})),
  getAccessToken: getAccessTokenMock,
  llmHeaders: vi.fn(() => ({})),
}))

import {
  DataLifecycleApiError,
  getDataSummary,
  resetData,
  type DataCounts,
  type DataSummaryDto,
  type ResetResultDto,
  type ResetScope,
} from './api'

const EMPTY_COUNTS: DataCounts = {
  users: 0,
  documents: 0,
  source_chunks: 0,
  vectors: 0,
  chat_sessions: 0,
  messages: 0,
  citations: 0,
  goals: 0,
  topics: 0,
  plans: 0,
  plan_milestones: 0,
  plan_events: 0,
  questions: 0,
  mastery: 0,
  mistakes: 0,
}

const EMPTY_EVAL = {
  runs: 0,
  score_sets: 0,
  scorer_executions: 0,
  estimated_bytes: 0,
}

const EMPTY_SUMMARY: DataSummaryDto = {
  reset_enabled: false,
  has_learning_data: false,
  current_user_exists: true,
  eval: EMPTY_EVAL,
  ...EMPTY_COUNTS,
}

function jsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: vi.fn().mockResolvedValue(body),
  } as unknown as Response
}

describe('strict lifecycle API', () => {
  beforeEach(() => {
    getAccessTokenMock.mockReset()
    getAccessTokenMock.mockResolvedValue('signed-token')
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('awaits a signed token before requesting the complete summary', async () => {
    let resolveToken!: (token: string) => void
    getAccessTokenMock.mockReturnValueOnce(new Promise((resolve) => {
      resolveToken = resolve
    }))
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(EMPTY_SUMMARY))
    vi.stubGlobal('fetch', fetchMock)

    const request = getDataSummary()
    await Promise.resolve()
    expect(fetchMock).not.toHaveBeenCalled()

    resolveToken('signed-token')
    await expect(request).resolves.toEqual(EMPTY_SUMMARY)
    expect(fetchMock).toHaveBeenCalledWith('/api/data/summary', {
      headers: { Authorization: 'Bearer signed-token' },
    })
  })

  it('requests a fresh access token for every lifecycle call', async () => {
    getAccessTokenMock
      .mockResolvedValueOnce('first-token')
      .mockResolvedValueOnce('second-token')
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse(EMPTY_SUMMARY))
      .mockResolvedValueOnce(jsonResponse(EMPTY_SUMMARY))
    vi.stubGlobal('fetch', fetchMock)

    await getDataSummary()
    await getDataSummary()

    expect(getAccessTokenMock).toHaveBeenCalledTimes(2)
    expect(fetchMock.mock.calls[0]?.[1]).toEqual({
      headers: { Authorization: 'Bearer first-token' },
    })
    expect(fetchMock.mock.calls[1]?.[1]).toEqual({
      headers: { Authorization: 'Bearer second-token' },
    })
  })

  it.each<[ResetScope, string]>([
    ['learning', 'CLEAR_LEARNING_DATA'],
    ['factory', 'FACTORY_RESET'],
  ])('sends the exact %s reset confirmation', async (scope, confirmation) => {
    const result: ResetResultDto = {
      scope,
      status: 'completed',
      deleted: EMPTY_COUNTS,
      deleted_eval: EMPTY_EVAL,
    }
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(result))
    vi.stubGlobal('fetch', fetchMock)

    await expect(resetData(scope)).resolves.toEqual(result)
    expect(fetchMock).toHaveBeenCalledWith('/api/data/reset', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: 'Bearer signed-token',
      },
      body: JSON.stringify({ scope, confirmation }),
    })
  })

  it('does not send a request when token provisioning fails', async () => {
    getAccessTokenMock.mockRejectedValueOnce(new Error('token unavailable'))
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)

    await expect(getDataSummary()).rejects.toThrow('token unavailable')
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('parses structured FastAPI lifecycle errors into safe fields', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({
      detail: {
        code: 'reset_stage_failed',
        failed_stage: 'sqlite',
        retryable: true,
        message: 'Reset failed. Retry is safe.',
      },
    }, 500)))

    const error = await getDataSummary().catch((reason: unknown) => reason)

    expect(error).toBeInstanceOf(DataLifecycleApiError)
    expect(error).toMatchObject({
      status: 500,
      code: 'reset_stage_failed',
      failedStage: 'sqlite',
      retryable: true,
      requiredScope: null,
      message: 'Reset failed. Retry is safe.',
    })
  })

  it('parses the required reset scope for interrupted-reset recovery', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({
      detail: {
        code: 'reset_recovery_required',
        required_scope: 'learning',
        message: 'Retry the incomplete reset.',
      },
    }, 409)))

    await expect(getDataSummary()).rejects.toMatchObject({
      status: 409,
      code: 'reset_recovery_required',
      requiredScope: 'learning',
      message: 'Retry the incomplete reset.',
    })
  })

  it('uses safe defaults for non-JSON error bodies', async () => {
    const response = {
      ok: false,
      status: 502,
      json: vi.fn().mockRejectedValue(new Error('not JSON')),
    } as unknown as Response
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(response))

    await expect(getDataSummary()).rejects.toMatchObject({
      status: 502,
      code: 'data_lifecycle_failed',
      failedStage: null,
      retryable: false,
      requiredScope: null,
      message: 'Data lifecycle request failed.',
    })
  })

  it('does not expose irregular error body values', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({
      detail: ['database-password-should-not-leak'],
      code: { nested: 'secret-code' },
      message: 1234,
    }, 500)))

    const error = await getDataSummary().catch((reason: unknown) => reason) as Error

    expect(error).toMatchObject({
      status: 500,
      code: 'data_lifecycle_failed',
      failedStage: null,
      retryable: false,
      requiredScope: null,
      message: 'Data lifecycle request failed.',
    })
    expect(error.message).not.toContain('password')
    expect(error.message).not.toContain('secret')
  })
})
