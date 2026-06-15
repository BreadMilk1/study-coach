import { authHeaders, getAccessToken, llmHeaders, type ModeOverrides } from '../stores/settings'

// Auto-provision anonymous token on module load
getAccessToken()
import type { Citation } from '../stores/chat'

interface ChatStreamCallbacks {
  onSession?: (sessionId: string) => void
  onCitations?: (cs: Citation[]) => void
  onToken?: (text: string) => void
  onTrace?: (step: any) => void
  onDone?: () => void
  onError?: (err: unknown) => void
}

const CHAT_SESSION_KEY = 'study-coach:current-chat-session-id'

export function getStoredChatSessionId(): string {
  try { return localStorage.getItem(CHAT_SESSION_KEY) || '' }
  catch { return '' }
}

export function setStoredChatSessionId(sessionId: string): void {
  try { localStorage.setItem(CHAT_SESSION_KEY, sessionId) }
  catch { /* ignore */ }
}

export async function streamChat(
  message: string,
  settings: any,
  cb: ChatStreamCallbacks,
  overrides: ModeOverrides = {},
): Promise<void> {
  try {
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...authHeaders(),
        ...llmHeaders(settings, overrides),
      },
      body: JSON.stringify({
        message,
        session_id: getStoredChatSessionId() || undefined,
      }),
    })
    if (!resp.ok || !resp.body) throw new Error(`chat failed: ${resp.status}`)
    const reader = resp.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n\n')
      buffer = lines.pop() ?? ''
      for (const line of lines) {
        const trimmed = line.trim()
        if (!trimmed.startsWith('data: ')) continue
        const json = trimmed.slice(6)
        try {
          const event = JSON.parse(json)
          if (event.type === 'session') {
            setStoredChatSessionId(event.session_id)
            cb.onSession?.(event.session_id)
          } else if (event.type === 'token') cb.onToken?.(event.text)
          else if (event.type === 'citations') cb.onCitations?.(event.citations)
          else if (event.type === 'trace') cb.onTrace?.(event)
          else if (event.type === 'done') cb.onDone?.()
        } catch { /* ignore malformed */ }
      }
    }
  } catch (e) {
    cb.onError?.(e)
  }
}

export async function uploadDocument(file: File): Promise<{
  document_id: string
  filename: string
  chunks_count: number
}> {
  const form = new FormData()
  form.append('file', file)
  const resp = await fetch('/api/documents', {
    method: 'POST',
    headers: { ...authHeaders() },
    body: form,
  })
  if (!resp.ok) throw new Error(`upload failed: ${resp.status}`)
  return resp.json()
}

// P3 — typed GET helper. Backend resolves user via x-fingerprint header.
export async function getJSON<T>(path: string): Promise<T> {
  const resp = await fetch(path, {
    headers: { ...authHeaders() },
  })
  if (resp.status === 404) {
    // Caller decides whether 404 is data-empty or hard error.
    const err: any = new Error(`${path} returned 404`)
    err.status = 404
    throw err
  }
  if (!resp.ok) throw new Error(`${path} failed: ${resp.status}`)
  return resp.json() as Promise<T>
}

export interface MilestoneDto {
  id: string | null
  title: string
  due_at: string | null
  done: boolean
  completed_at: string | null
  topic_id: string | null
  topic: string | null
  mastery_score: number | null
  validation_recommended: boolean
  sort_order: number | null
  source: string | null
}

export interface PlanCurrentDto {
  plan_id: string
  goal_id: string
  goal_title: string
  milestones: MilestoneDto[]
  updated_at: string
}

export function getCurrentPlan(): Promise<PlanCurrentDto> {
  return getJSON<PlanCurrentDto>('/api/plans/current')
}

export interface PlanEventDto {
  id: string
  plan_id: string
  milestone_id: string | null
  actor: string
  action: string
  before_json: Record<string, unknown> | null
  after_json: Record<string, unknown> | null
  reason: string | null
  created_at: string
}

export interface ValidationHintDto {
  show_quick_quiz: boolean
  topic: string | null
  reason: string | null
}

export interface MilestonePatchDto {
  plan: PlanCurrentDto
  event: PlanEventDto
  validation_hint: ValidationHintDto
}

export async function patchMilestoneDone(
  planId: string,
  milestoneId: string,
  done: boolean,
): Promise<MilestonePatchDto> {
  const resp = await fetch(`/api/plans/${planId}/milestones/${milestoneId}`, {
    method: 'PATCH',
    headers: {
      'Content-Type': 'application/json',
      ...authHeaders(),
    },
    body: JSON.stringify({ done }),
  })
  if (!resp.ok) throw new Error(`milestone update failed: ${resp.status}`)
  return resp.json() as Promise<MilestonePatchDto>
}

export function getPlanEvents(planId: string, limit = 20): Promise<PlanEventDto[]> {
  return getJSON<PlanEventDto[]>(`/api/plans/${planId}/events?limit=${limit}`)
}

export interface MistakeQuestionDto {
  id: string
  prompt: string
  options: string[]
  answer: string
  explanation: string
}

export interface MistakeDueDto {
  mistake_id: string
  question: MistakeQuestionDto
  due_at: string
  srs_interval_days: number
  srs_ease: number
  topic_name: string
}

export function getMistakesDue(limit = 20, includeFuture = false): Promise<MistakeDueDto[]> {
  const params = new URLSearchParams({ limit: String(limit) })
  if (includeFuture) params.set('include_future', 'true')
  return getJSON<MistakeDueDto[]>(`/api/mistakes/due?${params.toString()}`)
}

export interface DocumentDto {
  id: string
  filename: string
  chunks_count: number
}

export function getDocuments(): Promise<DocumentDto[]> {
  return getJSON<DocumentDto[]>('/api/documents')
}

export interface MasteryScoreDto {
  topic_id: string
  topic_name: string
  score: number
  last_reviewed: string
}

export interface MasteryDto {
  scores: MasteryScoreDto[]
  weak_topics: string[]
  overdue_milestones_count: number
}

export function getMastery(): Promise<MasteryDto> {
  return getJSON<MasteryDto>('/api/mastery')
}

export interface ToolCheckDto {
  tool_capable: boolean
  model: string
  note: string
}

export async function checkToolCapable(s: any): Promise<ToolCheckDto> {
  const headers: Record<string, string> = {
    'x-provider': s.provider,
    'x-model': s.model,
  }
  if (s.apiKey) headers['x-api-key'] = s.apiKey
  if (s.baseUrl) headers['x-base-url'] = s.baseUrl
  const resp = await fetch('/api/models/tool-check', { headers })
  if (!resp.ok) throw new Error(`tool-check failed: ${resp.status}`)
  return resp.json() as Promise<ToolCheckDto>
}

export interface MistakeReviewOut {
  correct: boolean
  correct_answer: string
  explanation: string
  new_interval_days: number
  next_due_at: string
}

export async function reviewMistake(
  mistakeId: string,
  answer: string,
): Promise<MistakeReviewOut> {
  const resp = await fetch(`/api/mistakes/${mistakeId}/review`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...authHeaders(),
    },
    body: JSON.stringify({ answer }),
  })
  if (!resp.ok) throw new Error(`review failed: ${resp.status}`)
  return resp.json() as Promise<MistakeReviewOut>
}

export interface PingDto {
  ok: boolean
  model: string
  latency_ms: number
  note: string
}

export async function pingModel(s: any): Promise<PingDto> {
  const headers: Record<string, string> = {
    'x-provider': s.provider,
    'x-model': s.model,
  }
  if (s.apiKey) headers['x-api-key'] = s.apiKey
  if (s.baseUrl) headers['x-base-url'] = s.baseUrl
  const resp = await fetch('/api/models/ping', { headers })
  if (!resp.ok) throw new Error(`ping failed: ${resp.status}`)
  return resp.json() as Promise<PingDto>
}

// --- P4b new endpoints ---

export interface ActivityDayDto {
  date: string
  count: number
}

export interface UserStatsDto {
  streak_days: number
  coverage: number
  total_sessions: number
  last_active_date: string | null
  activity_daily: ActivityDayDto[]
}

export function getUserStats(): Promise<UserStatsDto> {
  return getJSON<UserStatsDto>('/api/users/me/stats')
}

export interface ChatSessionDto {
  session_id: string
  started_at: string
  summary: string | null
}

export interface ChatMessageCitationDto {
  chunk_id: string
  page: number
  span_start: number
  span_end: number
  source: string | null
}

export interface ChatMessageDto {
  id: string
  role: 'user' | 'assistant'
  content: string
  created_at: string
  citations: ChatMessageCitationDto[]
}

export interface ChatMessagesDto {
  session_id: string
  messages: ChatMessageDto[]
}

export function getCurrentChatSession(): Promise<ChatSessionDto> {
  return getJSON<ChatSessionDto>('/api/chat/sessions/current')
}

export function getChatSessionMessages(sessionId: string): Promise<ChatMessagesDto> {
  return getJSON<ChatMessagesDto>(`/api/chat/sessions/${sessionId}/messages`)
}

export async function reorderMilestones(
  planId: string,
  milestoneIds: string[],
): Promise<PlanCurrentDto> {
  const resp = await fetch(`/api/plans/${planId}/milestones/reorder`, {
    method: 'PATCH',
    headers: {
      'Content-Type': 'application/json',
      ...authHeaders(),
    },
    body: JSON.stringify({ milestone_ids: milestoneIds }),
  })
  if (!resp.ok) throw new Error(`reorder failed: ${resp.status}`)
  return resp.json() as Promise<PlanCurrentDto>
}

export async function markMistakeUnderstood(mistakeId: string): Promise<{
  mastery_score: number
  next_due_at: string | null
}> {
  const resp = await fetch(`/api/mistakes/${mistakeId}/mark-understood`, {
    method: 'POST',
    headers: { ...authHeaders() },
  })
  if (!resp.ok) throw new Error(`mark-understood failed: ${resp.status}`)
  return resp.json()
}
