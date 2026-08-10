import { createRenderer, nextTick, ssrContextKey } from 'vue'
import { afterEach, describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({
  uploadDocument: vi.fn(async () => ({
    filename: 'notes.pdf',
    chunks_count: 2,
  })),
  docs: {
    docs: [] as unknown[],
    loading: false,
    error: null as string | null,
    fetch: vi.fn(async () => undefined),
  },
  push: vi.fn(async () => undefined),
}))

vi.mock('../lib/api', () => ({
  uploadDocument: mocks.uploadDocument,
}))
vi.mock('../stores/documents', () => ({
  useDocuments: () => mocks.docs,
}))
vi.mock('vue-router', () => ({
  useRoute: () => ({
    query: {
      return: '/quiz',
      topic: 'HyDE rewrite',
    },
  }),
  useRouter: () => ({ push: mocks.push }),
  RouterLink: { template: '<span><slot /></span>' },
}))

import Library from './Library.vue'

// Vitest node env compiles SFCs with ssrRender only. Lifecycle tests stub
// client render the same way; setup/onMounted/onFile/returnPath must still run.
Object.assign(Library, { render: () => null })

type TestNode = { parent?: TestNode; children: TestNode[]; text?: string }

function mountLibrary() {
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
  const app = renderer.createApp(Library)
  app.provide(ssrContextKey, { modules: new Set<string>() })
  app.config.globalProperties.$t = (key: string) => key
  app.mount(root)
  return app
}

function setupStateOf(app: { _instance?: { setupState?: Record<string, unknown> } }) {
  const state = app._instance?.setupState
  if (!state) throw new Error('Library setupState missing')
  return state
}

function readOnFile(app: { _instance?: { setupState?: Record<string, unknown> } }) {
  const onFile = setupStateOf(app).onFile
  if (typeof onFile !== 'function') throw new Error('onFile missing from Library setupState')
  return onFile as (e: Event) => Promise<void>
}

describe('Library quiz topic return after upload', () => {
  afterEach(() => {
    mocks.uploadDocument.mockClear()
    mocks.docs.fetch.mockClear()
    mocks.push.mockClear()
  })

  it('preserves the quiz topic when returning after a successful upload', async () => {
    const app = mountLibrary()
    try {
      await nextTick()
      expect(mocks.docs.fetch).toHaveBeenCalledOnce()

      const onFile = readOnFile(app as { _instance?: { setupState?: Record<string, unknown> } })
      const file = new File(['%PDF-fake'], 'notes.pdf', { type: 'application/pdf' })
      const input = { files: [file] } as unknown as HTMLInputElement
      const changeEvent = { target: input } as unknown as Event

      await onFile(changeEvent)
      await Promise.resolve()
      await nextTick()

      expect(mocks.uploadDocument).toHaveBeenCalledOnce()
      expect(mocks.uploadDocument).toHaveBeenCalledWith(file)
      expect(mocks.docs.fetch).toHaveBeenCalledTimes(2)
      expect(mocks.push).toHaveBeenCalledOnce()
      expect(mocks.push).toHaveBeenCalledWith({
        path: '/quiz',
        query: {
          topic: 'HyDE rewrite',
        },
      })
    } finally {
      app.unmount()
    }
  })
})
