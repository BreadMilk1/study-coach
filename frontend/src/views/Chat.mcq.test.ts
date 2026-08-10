import { createRenderer, nextTick, ssrContextKey } from 'vue'
import { describe, expect, it, vi } from 'vitest'
import type { Message } from '../stores/chat'

type StreamChatCallbacks = {
  onQuizQuestion?: (questionId: string) => void
  onDone?: () => void
  onSession?: (sessionId: string) => void
  onCitations?: (cs: unknown) => void
  onAgentRun?: (run: unknown) => void
  onToken?: (text: string) => void
  onTrace?: (step: unknown) => void
  onError?: (err: unknown) => void
}

const mocks = vi.hoisted(() => {
  const mcqAssistantContent = [
    '📝 Quiz on HyDE:',
    '',
    'What does HyDE rewrite?',
    '',
    'A) Queries',
    'B) Documents',
    'C) Embeddings',
    'D) Answers',
    '',
    'Reply with A, B, C, or D.',
  ].join('\n')
  return {
    mcqAssistantContent,
    chat: {
      messages: [
        {
          id: 'assistant-1',
          role: 'assistant' as const,
          content: mcqAssistantContent,
          citations: [],
        },
        {
          id: 'user-1',
          role: 'user' as const,
          content: 'A',
        },
      ] as Message[],
      streaming: false,
      trace: [],
      restoreCurrentSession: vi.fn(async () => undefined),
      pushUser: vi.fn(),
      startAssistant: vi.fn(),
      setSessionId: vi.fn(),
      setCitations: vi.fn(),
      setAgentRun: vi.fn(),
      setQuizQuestionId: vi.fn(),
      appendToken: vi.fn(),
      finish: vi.fn(),
    },
    streamChat: vi.fn(async (
      _message: string,
      _settings: unknown,
      _callbacks: StreamChatCallbacks,
    ) => undefined),
    replace: vi.fn(),
  }
})

vi.mock('../stores/chat', () => ({ useChat: () => mocks.chat }))
vi.mock('../stores/settings', () => ({
  useSettings: () => ({ $state: {}, debugMode: false }),
}))
vi.mock('../lib/api', () => ({ streamChat: mocks.streamChat }))
vi.mock('vue-router', () => ({
  useRoute: () => ({ query: {} }),
  useRouter: () => ({ replace: mocks.replace }),
  RouterLink: { template: '<span><slot /></span>' },
}))

import Chat from './Chat.vue'

// Vitest node env compiles SFCs with ssrRender only. Lifecycle tests stub
// client render the same way; setup/computed must still run for real.
Object.assign(Chat, { render: () => null })

type TestNode = { parent?: TestNode; children: TestNode[]; text?: string }

function mountChat() {
  const root: TestNode = { children: [] }
  const renderer = createRenderer<TestNode, TestNode>({
    patchProp: () => undefined,
    insert: (child, parent) => {
      child.parent = parent
      parent.children.push(child)
    },
    remove: (child) => {
      const parent = child.parent
      if (!parent) return
      parent.children = parent.children.filter(node => node !== child)
    },
    createElement: () => ({ children: [] }),
    createText: text => ({ children: [], text }),
    createComment: text => ({ children: [], text }),
    setText: (node, text) => { node.text = text },
    setElementText: (node, text) => { node.text = text },
    parentNode: node => node.parent ?? null,
    nextSibling: () => null,
  })
  const app = renderer.createApp(Chat)
  app.provide(ssrContextKey, { modules: new Set<string>() })
  app.config.globalProperties.$t = (key: string) => key
  app.mount(root)
  return app
}

function readLiveMcq(app: { _instance?: { setupState?: { liveMcq?: unknown } } }) {
  const live = app._instance?.setupState?.liveMcq as
    | { value?: unknown }
    | unknown
    | undefined
  if (live && typeof live === 'object' && live !== null && 'value' in live) {
    return (live as { value: unknown }).value
  }
  return live ?? null
}

function readSend(app: { _instance?: { setupState?: Record<string, unknown> } }) {
  const send = app._instance?.setupState?.send
  if (typeof send !== 'function') throw new Error('send missing from Chat setupState')
  return send as (textOverride?: string) => Promise<void>
}

async function flushMounted() {
  await nextTick()
  await Promise.resolve()
  await nextTick()
}

describe('Chat restored-session MCQ', () => {
  it('does not expose an earlier MCQ as live when the restored session ends with a user answer', async () => {
    const app = mountChat()
    await flushMounted()

    expect(mocks.chat.restoreCurrentSession).toHaveBeenCalledOnce()
    expect(mocks.chat.messages.at(-1)?.role).toBe('user')
    expect(mocks.chat.messages.at(-1)?.content).toBe('A')

    const liveMcq = readLiveMcq(app as { _instance?: { setupState?: { liveMcq?: unknown } } })
    expect(liveMcq).toBeNull()

    app.unmount()
  })
})

describe('Chat live quiz signal wiring', () => {
  it('writes the live persisted quiz signal onto the in-flight assistant message', async () => {
    vi.clearAllMocks()
    mocks.chat.messages = []
    mocks.chat.streaming = false

    const assistant = {
      id: 'assistant-live-1',
      role: 'assistant' as const,
      content: '',
      citations: [] as [],
    }
    mocks.chat.startAssistant.mockReturnValueOnce(assistant)
    mocks.streamChat.mockImplementationOnce(
      async (_message, _settings, callbacks: StreamChatCallbacks) => {
        callbacks.onQuizQuestion?.('question-live-123')
        callbacks.onDone?.()
      },
    )

    const app = mountChat()
    try {
      await flushMounted()
      const send = readSend(app as { _instance?: { setupState?: Record<string, unknown> } })
      await send('quiz me on HyDE')

      expect(mocks.streamChat).toHaveBeenCalledOnce()
      expect(mocks.chat.startAssistant).toHaveBeenCalledOnce()
      expect(mocks.chat.finish).toHaveBeenCalledOnce()
      expect(mocks.chat.setQuizQuestionId).toHaveBeenCalledTimes(1)
      expect(mocks.chat.setQuizQuestionId).toHaveBeenCalledWith(assistant, 'question-live-123')
    } finally {
      app.unmount()
    }
  })

  it('does not commit a quiz signal when the stream fails before done', async () => {
    vi.clearAllMocks()
    mocks.chat.messages = []
    mocks.chat.streaming = false

    const assistant = {
      id: 'assistant-error-1',
      role: 'assistant' as const,
      content: '',
      citations: [] as [],
    }
    mocks.chat.startAssistant.mockReturnValueOnce(assistant)
    mocks.streamChat.mockImplementationOnce(
      async (_message, _settings, callbacks: StreamChatCallbacks) => {
        callbacks.onQuizQuestion?.('question-error-123')
        callbacks.onError?.(new Error('stream interrupted'))
      },
    )

    const app = mountChat()
    try {
      await flushMounted()
      const send = readSend(app as { _instance?: { setupState?: Record<string, unknown> } })
      await send('quiz me on HyDE')

      expect(mocks.streamChat).toHaveBeenCalledOnce()
      expect(mocks.chat.startAssistant).toHaveBeenCalledOnce()
      expect(mocks.chat.appendToken).toHaveBeenCalledOnce()
      expect(mocks.chat.finish).toHaveBeenCalledOnce()
      expect(mocks.chat.setQuizQuestionId).not.toHaveBeenCalled()
    } finally {
      app.unmount()
    }
  })
})

describe('Chat live MCQ render gate', () => {
  it('only exposes an A-D assistant response as live when it carries a persisted quiz signal', async () => {
    vi.clearAllMocks()
    mocks.chat.streaming = false
    mocks.chat.messages = [{
      id: 'assistant-render-1',
      role: 'assistant' as const,
      content: mocks.mcqAssistantContent,
      citations: [],
    }]

    const appWithoutSignal = mountChat()
    try {
      await flushMounted()
      const liveMcqWithoutSignal = readLiveMcq(
        appWithoutSignal as { _instance?: { setupState?: { liveMcq?: unknown } } },
      )
      expect(liveMcqWithoutSignal).toBeNull()
    } finally {
      appWithoutSignal.unmount()
    }

    mocks.chat.streaming = false
    mocks.chat.messages = [{
      id: 'assistant-render-1',
      role: 'assistant' as const,
      content: mocks.mcqAssistantContent,
      citations: [],
      quizQuestionId: 'question-render-123',
    }]

    const appWithSignal = mountChat()
    try {
      await flushMounted()
      const liveMcqWithSignal = readLiveMcq(
        appWithSignal as { _instance?: { setupState?: { liveMcq?: unknown } } },
      )
      expect(liveMcqWithSignal).toMatchObject({
        prompt: 'What does HyDE rewrite?',
        options: [
          'A) Queries',
          'B) Documents',
          'C) Embeddings',
          'D) Answers',
        ],
      })
    } finally {
      appWithSignal.unmount()
    }
  })
})
