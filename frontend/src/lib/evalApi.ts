import { getAccessToken } from '../stores/settings'
import {
  learningRunRequest,
  parseCompareResponse,
  parseEvalErrorDetail,
  parseEvalEvent,
  parseRunDetail,
  type CompareResponse,
  type EvalConnectionSnapshot,
  type EvalErrorDetail,
  type ExperimentSummary,
  type LearningRunEvent,
  type LearningRunRequest,
  type RescoreRequest,
  type RunDetail,
  type RunSummary,
  type ScoreSetSummary,
} from './evalContracts'

export class EvalApiError extends Error {
  readonly status: number
  readonly code: string
  readonly fields: string[]
  readonly active_entity_id: string | null
  readonly active_kind: EvalErrorDetail['active_kind']

  constructor(status: number, detail: EvalErrorDetail) {
    super(detail.message)
    this.name = 'EvalApiError'
    this.status = status
    this.code = detail.code
    this.fields = detail.fields
    this.active_entity_id = detail.active_entity_id
    this.active_kind = detail.active_kind
  }

  static fromDetail(status: number, value: unknown): EvalApiError {
    try {
      return new EvalApiError(status, parseEvalErrorDetail(value))
    } catch {
      return new EvalApiError(status, {
        code: 'evaluation_request_failed',
        message: `evaluation request failed: ${status}`,
        fields: [],
        active_entity_id: null,
        active_kind: null,
      })
    }
  }
}

export function evalHeaders(
  token: string,
  connection?: EvalConnectionSnapshot,
): Record<string, string> {
  const headers: Record<string, string> = {
    Authorization: `Bearer ${token}`,
  }
  if (!connection) return headers
  headers['Content-Type'] = 'application/json'
  headers['x-provider'] = connection.provider
  headers['x-model'] = connection.model
  if (connection.apiKey) headers['x-api-key'] = connection.apiKey
  if (connection.baseUrl) headers['x-base-url'] = connection.baseUrl
  return headers
}

async function authorizedHeaders(connection?: EvalConnectionSnapshot): Promise<Record<string, string>> {
  const token = await getAccessToken()
  return evalHeaders(token, connection)
}

async function throwIfNotOk(response: Response): Promise<void> {
  if (response.ok) return
  const body = await response.json().catch(() => null)
  throw EvalApiError.fromDetail(response.status, body)
}

export async function streamLearningRun(
  request: LearningRunRequest,
  connection: EvalConnectionSnapshot,
  onEvent: (event: LearningRunEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  await readEvalStream(
    '/api/eval/runs/stream',
    learningRunRequest(request),
    connection,
    onEvent,
    signal,
  )
}

export async function streamRescore(
  runId: string,
  request: RescoreRequest,
  connection: EvalConnectionSnapshot,
  onEvent: (event: LearningRunEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  await readEvalStream(
    `/api/eval/runs/${runId}/rescore/stream`,
    { scorer_version: request.scorer_version, run_profile: 'evaluation' },
    connection,
    onEvent,
    signal,
  )
}

async function readEvalStream(
  path: string,
  body: object,
  connection: EvalConnectionSnapshot,
  onEvent: (event: LearningRunEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  const headers = await authorizedHeaders(connection)
  const response = await fetch(path, {
    method: 'POST',
    headers,
    body: JSON.stringify(body),
    signal,
  })
  if (!response.ok) {
    const payload = await response.json().catch(() => null)
    throw EvalApiError.fromDetail(response.status, payload)
  }
  if (!response.body) {
    throw new EvalApiError(500, {
      code: 'evaluation_unavailable',
      message: 'evaluation stream is unavailable',
      fields: [],
      active_entity_id: null,
      active_kind: null,
    })
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  try {
    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const frames = buffer.split('\n\n')
      buffer = frames.pop() ?? ''
      for (const frame of frames) {
        const event = parseSseFrame(frame)
        if (event) onEvent(event)
      }
    }
    const trailing = parseSseFrame(buffer)
    if (trailing) onEvent(trailing)
  } finally {
    try { await reader.cancel() }
    catch { /* already closed */ }
  }
}

export function parseSseFrame(frame: string): LearningRunEvent | null {
  const trimmed = frame.trim()
  if (!trimmed) return null
  if (!trimmed.startsWith('data: ')) {
    throw new EvalApiError(500, {
      code: 'malformed_event',
      message: 'evaluation stream frame is malformed',
      fields: [],
      active_entity_id: null,
      active_kind: null,
    })
  }
  try {
    return parseEvalEvent(JSON.parse(trimmed.slice(6)))
  } catch (error) {
    if (error instanceof EvalApiError) throw error
    throw new EvalApiError(500, {
      code: 'malformed_event',
      message: error instanceof Error ? error.message : 'evaluation stream frame is malformed',
      fields: [],
      active_entity_id: null,
      active_kind: null,
    })
  }
}

export async function listExperiments(): Promise<ExperimentSummary[]> {
  const headers = await authorizedHeaders()
  const response = await fetch('/api/eval/experiments', { headers })
  await throwIfNotOk(response)
  return await response.json() as ExperimentSummary[]
}

export async function listRuns(): Promise<RunSummary[]> {
  const headers = await authorizedHeaders()
  const response = await fetch('/api/eval/runs', { headers })
  await throwIfNotOk(response)
  return await response.json() as RunSummary[]
}

export async function cancelScoreSet(
  scoreSetId: string,
  options: { keepalive?: boolean } = {},
): Promise<ScoreSetSummary> {
  const headers = await authorizedHeaders()
  const response = await fetch(`/api/eval/score-sets/${scoreSetId}/cancel`, {
    method: 'POST',
    headers,
    keepalive: options.keepalive === true,
  })
  await throwIfNotOk(response)
  return await response.json() as ScoreSetSummary
}

export async function cancelLearningRun(
  runId: string,
  options: { keepalive?: boolean } = {},
): Promise<RunSummary> {
  const headers = await authorizedHeaders()
  const response = await fetch(`/api/eval/runs/${runId}/cancel`, {
    method: 'POST',
    headers,
    keepalive: options.keepalive === true,
  })
  await throwIfNotOk(response)
  return await response.json() as RunSummary
}

export async function getRunDetail(runId: string): Promise<RunDetail> {
  const headers = await authorizedHeaders()
  const response = await fetch(`/api/eval/runs/${runId}`, { headers })
  await throwIfNotOk(response)
  return parseRunDetail(await response.json())
}

export async function compareRuns(left: string, right: string): Promise<CompareResponse> {
  const headers = await authorizedHeaders()
  const response = await fetch(`/api/eval/compare?left=${encodeURIComponent(left)}&right=${encodeURIComponent(right)}`, {
    headers,
  })
  await throwIfNotOk(response)
  return parseCompareResponse(await response.json())
}
