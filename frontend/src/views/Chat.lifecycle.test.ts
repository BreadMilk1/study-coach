import { createRenderer, nextTick, ssrContextKey } from 'vue'
import { describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => {
  let resolveRestore: (() => void) | undefined
  const restorePending = new Promise<void>((resolve) => {
    resolveRestore = resolve
  })
  const assistant = { id: 'assistant', role: 'assistant', content: '', citations: [] }
  return {
    resolveRestore: () => resolveRestore?.(),
    chat: {
      messages: [],
      streaming: false,
      trace: [],
      restoreCurrentSession: vi.fn(() => restorePending),
      pushUser: vi.fn(),
      startAssistant: vi.fn(() => assistant),
      setSessionId: vi.fn(),
      setCitations: vi.fn(),
      setAgentRun: vi.fn(),
      appendToken: vi.fn(),
      finish: vi.fn(),
    },
    streamChat: vi.fn(async () => undefined),
    replace: vi.fn(),
  }
})

vi.mock('../stores/chat', () => ({ useChat: () => mocks.chat }))
vi.mock('../stores/settings', () => ({
  useSettings: () => ({ $state: {}, debugMode: false }),
}))
vi.mock('../lib/api', () => ({ streamChat: mocks.streamChat }))
vi.mock('vue-router', () => ({
  useRoute: () => ({ query: { auto: 'make a plan', goal_id: 'goal-1' } }),
  useRouter: () => ({ replace: mocks.replace }),
  RouterLink: { template: '<span><slot /></span>' },
}))

import Chat from './Chat.vue'

Object.assign(Chat, { render: () => null })

function mountChat() {
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
  const app = renderer.createApp(Chat)
  app.provide(ssrContextKey, { modules: new Set<string>() })
  app.config.globalProperties.$t = (key: string) => key
  app.mount({ children: [] })
  return app
}

describe('Chat route lifecycle', () => {
  it('does not send an auto prompt after the route unmounts during session restore', async () => {
    const app = mountChat()
    await nextTick()
    expect(mocks.chat.restoreCurrentSession).toHaveBeenCalledOnce()

    app.unmount()
    mocks.resolveRestore()
    await Promise.resolve()
    await nextTick()
    await Promise.resolve()

    expect(mocks.chat.pushUser).not.toHaveBeenCalled()
    expect(mocks.chat.startAssistant).not.toHaveBeenCalled()
    expect(mocks.streamChat).not.toHaveBeenCalled()
    expect(mocks.replace).not.toHaveBeenCalled()
  })
})
