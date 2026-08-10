import { afterEach, describe, expect, it, vi } from 'vitest'

vi.mock('../stores/settings', () => ({
  authHeaders: vi.fn(() => ({})),
  getAccessToken: vi.fn(() => Promise.resolve('test-token')),
  llmHeaders: vi.fn(() => ({})),
}))

import { streamChat } from './api'

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('streamChat quiz_question transport', () => {
  it('forwards a persisted quiz question id from the chat SSE stream', async () => {
    const encoder = new TextEncoder()
    // First chunk carries the new quiz_question signal; include type=done so
    // the existing SSE path invokes onDone (stream end alone does not).
    const sseChunk = [
      'data: {"type":"quiz_question","question_id":"question-123"}',
      'data: {"type":"done"}',
      '',
    ].join('\n\n')
    const read = vi.fn()
      .mockResolvedValueOnce({
        value: encoder.encode(sseChunk),
        done: false,
      })
      .mockResolvedValueOnce({ value: undefined, done: true })
    const cancel = vi.fn().mockResolvedValue(undefined)
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        body: { getReader: () => ({ read, cancel }) },
      }),
    )

    const onQuizQuestion = vi.fn()
    const onDone = vi.fn()
    const callbacks = { onDone, onQuizQuestion }

    await streamChat('quiz me on HyDE', {}, callbacks)

    expect(onDone).toHaveBeenCalledOnce()
    expect(onQuizQuestion).toHaveBeenCalledTimes(1)
    expect(onQuizQuestion).toHaveBeenCalledWith('question-123')
  })
})
