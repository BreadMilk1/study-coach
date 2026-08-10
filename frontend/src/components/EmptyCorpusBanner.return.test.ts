import { createRenderer, nextTick, ssrContextKey } from 'vue'
import { describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({
  push: vi.fn(),
}))

vi.mock('vue-router', () => ({
  useRoute: () => ({ query: { topic: 'HyDE rewrite' } }),
  useRouter: () => ({ push: mocks.push }),
}))

import EmptyCorpusBanner from './EmptyCorpusBanner.vue'

// Vitest node env compiles SFCs with ssrRender only. Lifecycle tests stub
// client render the same way; setup/go must still run for real.
Object.assign(EmptyCorpusBanner, { render: () => null })

type TestNode = { parent?: TestNode; children: TestNode[]; text?: string }

function mountBanner() {
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
  const app = renderer.createApp(EmptyCorpusBanner)
  app.provide(ssrContextKey, { modules: new Set<string>() })
  app.config.globalProperties.$t = (key: string) => key
  app.mount(root)
  return app
}

function readGo(app: { _instance?: { setupState?: Record<string, unknown> } }) {
  const go = app._instance?.setupState?.go
  if (typeof go !== 'function') throw new Error('go missing from EmptyCorpusBanner setupState')
  return go as () => void
}

describe('EmptyCorpusBanner quiz topic return handoff', () => {
  it('preserves the quiz topic when sending an empty corpus flow to Library', async () => {
    const app = mountBanner()
    await nextTick()

    readGo(app as { _instance?: { setupState?: Record<string, unknown> } })()

    expect(mocks.push).toHaveBeenCalledOnce()
    expect(mocks.push).toHaveBeenCalledWith({
      path: '/library',
      query: {
        return: '/quiz',
        topic: 'HyDE rewrite',
      },
    })

    app.unmount()
  })
})
