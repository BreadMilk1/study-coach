import { getFingerprint } from './fingerprint'
import { llmHeaders, type ModeOverrides } from '../stores/settings'
import type { Citation } from '../stores/chat'

interface ChatStreamCallbacks {
  onCitations?: (cs: Citation[]) => void
  onToken?: (text: string) => void
  onDone?: () => void
  onError?: (err: unknown) => void
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
        'x-fingerprint': getFingerprint(),
        ...llmHeaders(settings, overrides),
      },
      body: JSON.stringify({ message }),
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
          if (event.type === 'token') cb.onToken?.(event.text)
          else if (event.type === 'citations') cb.onCitations?.(event.citations)
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
    headers: { 'x-fingerprint': getFingerprint() },
    body: form,
  })
  if (!resp.ok) throw new Error(`upload failed: ${resp.status}`)
  return resp.json()
}

// P3 — typed GET helper. Backend resolves user via x-fingerprint header.
export async function getJSON<T>(path: string): Promise<T> {
  const resp = await fetch(path, {
    headers: { 'x-fingerprint': getFingerprint() },
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
      'x-fingerprint': getFingerprint(),
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
