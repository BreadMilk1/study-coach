import { createPinia, setActivePinia } from 'pinia'
import { markRaw } from 'vue'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('../stores/settings', () => ({
  authHeaders: vi.fn(() => ({})),
  getAccessToken: vi.fn(() => Promise.resolve('test-token')),
  llmHeaders: vi.fn(() => ({})),
}))

import { useChat } from './chat'
import { memoryStorage } from '../test/memoryStorage'

beforeEach(() => {
  setActivePinia(createPinia())
  vi.stubGlobal('localStorage', memoryStorage())
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('chat quiz signal restore', () => {
  it('restores a persisted quiz question id onto the assistant message', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: vi.fn().mockResolvedValue({
        session_id: 'session-quiz-signal',
        messages: [{
          id: 'assistant-quiz-1',
          role: 'assistant',
          content: '📝 Quiz on HyDE:\n\nWhat does HyDE rewrite?',
          created_at: '2026-08-06T12:00:00Z',
          citations: [],
          agent_run: null,
          quiz_question_id: 'question-123',
        }],
      }),
    })
    vi.stubGlobal('fetch', fetchMock)

    const chat = useChat()
    chat.sessionId = 'session-quiz-signal'

    await chat.restoreCurrentSession()

    expect(fetchMock).toHaveBeenCalledOnce()
    expect(chat.messages).toHaveLength(1)
    expect(chat.messages[0]).toMatchObject({
      id: 'assistant-quiz-1',
      role: 'assistant',
      quizQuestionId: 'question-123',
    })
  })
})

describe('chat quiz signal live mutation', () => {
  it('attaches a live persisted quiz id to the in-flight assistant message', () => {
    const chat = useChat()
    // markRaw: keep Object.is identity under Pinia reactive storage so the
    // public action mutates the same in-flight assistant instance.
    const assistant = markRaw({
      id: 'assistant-live-1',
      role: 'assistant' as const,
      content: '',
      citations: [] as [],
    })
    chat.messages.push(assistant)

    expect(chat.messages).toHaveLength(1)
    expect(chat.messages[0]).toBe(assistant)

    chat.setQuizQuestionId(assistant, 'question-live-123')

    expect(assistant).toMatchObject({
      quizQuestionId: 'question-live-123',
    })
  })
})
