import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

import { createRenderer, nextTick, ssrContextKey } from 'vue'
import { createMemoryHistory, createRouter } from 'vue-router'
import { describe, expect, it } from 'vitest'

import MobileNav from './MobileNav.vue'

// Vitest node env compiles SFCs with ssrRender only. Lifecycle tests stub
// client render the same way; setup/ref must still run for real.
Object.assign(MobileNav, { render: () => null })

type TestNode = { parent?: TestNode; children: TestNode[]; text?: string }

type SetupState = {
  moreOpen: boolean | { value: boolean }
}

function mountMobileNav(router: ReturnType<typeof createRouter>) {
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

  const app = renderer.createApp(MobileNav)
  app.use(router)
  app.provide(ssrContextKey, { modules: new Set<string>() })
  app.config.globalProperties.$t = (key: string) => key
  app.mount(root)
  return app
}

function setupStateOf(app: { _instance?: { setupState?: SetupState } }): SetupState {
  const state = app._instance?.setupState
  if (!state) throw new Error('MobileNav setupState missing')
  return state
}

function readMoreOpen(state: SetupState): boolean {
  const more = state.moreOpen
  if (more && typeof more === 'object' && 'value' in more) return more.value
  return Boolean(more)
}

function writeMoreOpen(state: SetupState, value: boolean) {
  const more = state.moreOpen
  if (more && typeof more === 'object' && 'value' in more) {
    more.value = value
    return
  }
  state.moreOpen = value
}

describe('MobileNav More sheet route change', () => {
  it('closes the More sheet when navigation changes to a primary tab', async () => {
    const router = createRouter({
      history: createMemoryHistory(),
      routes: [
        { path: '/library', component: { render: () => null } },
        { path: '/chat', component: { render: () => null } },
      ],
    })
    await router.push('/library')
    await router.isReady()

    const app = mountMobileNav(router)
    await nextTick()

    const state = setupStateOf(app as { _instance?: { setupState?: SetupState } })
    writeMoreOpen(state, true)
    await nextTick()
    expect(readMoreOpen(state)).toBe(true)

    await router.push('/chat')
    await nextTick()
    await Promise.resolve()
    await nextTick()

    expect(router.currentRoute.value.path).toBe('/chat')
    expect(readMoreOpen(state)).toBe(false)

    app.unmount()
  })
})

describe('MobileNav Run Lab entry', () => {
  it('includes a /run-lab destination in the More sheet', () => {
    const source = readFileSync(fileURLToPath(new URL('./MobileNav.vue', import.meta.url)), 'utf8')
    expect(source).toMatch(/to:\s*'\/run-lab'/)
  })
})
