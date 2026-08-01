import { createRenderer, nextTick, ssrContextKey } from 'vue'
import { describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => {
  let resolveFetch: (() => void) | undefined
  const fetchPending = new Promise<void>((resolve) => {
    resolveFetch = resolve
  })
  return {
    resolveFetch: () => resolveFetch?.(),
    quiz: {
      currentMCQ: null,
      lastGrade: null,
      difficulty: 'med',
      needsUpload: false,
      streaming: false,
      errorMsg: '',
      currentMistakeId: null,
      reset: vi.fn(),
      startStream: vi.fn(),
      appendRaw: vi.fn(),
      finishStream: vi.fn(),
      setDifficulty: vi.fn(),
      reviewCurrentMistake: vi.fn(),
    },
    mistakes: {
      items: [],
      due: [],
      fetch: vi.fn(() => fetchPending),
    },
    documents: {
      isEmpty: false,
      fetch: vi.fn(async () => true),
    },
    mastery: {
      fetch: vi.fn(),
    },
    streamChat: vi.fn(async () => undefined),
  }
})

vi.mock('../stores/quiz', () => ({ useQuiz: () => mocks.quiz }))
vi.mock('../stores/mistakes', () => ({ useMistakes: () => mocks.mistakes }))
vi.mock('../stores/mastery', () => ({ useMastery: () => mocks.mastery }))
vi.mock('../stores/documents', () => ({ useDocuments: () => mocks.documents }))
vi.mock('../stores/settings', () => ({
  useSettings: () => ({
    $state: {},
    toolCapable: true,
    defaultQuizMode: 'agent_loop',
  }),
}))
vi.mock('../lib/api', () => ({ streamChat: mocks.streamChat }))
vi.mock('vue-router', () => ({
  useRoute: () => ({ query: { topic: 'ChatGPT' } }),
}))

import QuizAdaptive from './QuizAdaptive.vue'

Object.assign(QuizAdaptive, { render: () => null })

function mountQuiz() {
  type TestNode = { parent?: TestNode; children: TestNode[]; text?: string }
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
  const app = renderer.createApp(QuizAdaptive)
  app.provide(ssrContextKey, { modules: new Set<string>() })
  app.config.globalProperties.$t = (key: string) => key
  app.mount({ children: [] })
  return app
}

describe('Quiz route lifecycle', () => {
  it('does not generate a topic quiz after the route unmounts during startup fetches', async () => {
    const app = mountQuiz()
    await nextTick()
    expect(mocks.mistakes.fetch).toHaveBeenCalledOnce()
    expect(mocks.documents.fetch).toHaveBeenCalledOnce()

    app.unmount()
    mocks.resolveFetch()
    await Promise.resolve()
    await nextTick()
    await Promise.resolve()

    expect(mocks.quiz.startStream).not.toHaveBeenCalled()
    expect(mocks.streamChat).not.toHaveBeenCalled()
  })
})
