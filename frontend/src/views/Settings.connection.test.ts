import { createRenderer, nextTick, ssrContextKey } from 'vue'
import * as VueRuntime from 'vue'
import { compileScript, compileTemplate, parse } from 'vue/compiler-sfc'
import { createPinia, setActivePinia } from 'pinia'
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { memoryStorage } from '../test/memoryStorage'
import { useSettings } from '../stores/settings'

// The real Settings.vue, real settings store and real api helpers run against
// a stubbed fetch with deferred promises; vue-i18n is the only mocked module
// (its runtime needs a full plugin install this renderer harness skips).
vi.mock('vue-i18n', () => ({
  useI18n: () => ({ locale: { value: 'en' }, t: (key: string) => key }),
}))

type TestNode = {
  parent?: TestNode
  children: TestNode[]
  tag?: string
  text?: string
  props: Record<string, unknown>
  handlers: Record<string, (event?: unknown) => void>
  // Listeners registered via el.addEventListener (the v-model directives).
  // In the real DOM they run before the template's @change prop handler
  // because the directive's created hook patches before props.
  listeners: Record<string, Array<(event?: unknown) => void>>
  addEventListener?: (type: string, handler: (event?: unknown) => void) => void
  removeEventListener?: () => void
  [key: string]: unknown
}

const REVISION_A = '00000000-0000-4000-8000-00000000000a'
const CAPABILITY_KEY = 'study-coach:connection-capability:v2'

function capabilityRecord(revision: string, toolCapable: boolean): string {
  return JSON.stringify({ schemaVersion: 2, connectionRevision: revision, toolCapable })
}

function okResponse(body: unknown): Response {
  return { ok: true, status: 200, json: async () => body } as unknown as Response
}

function statusResponse(status: number, body: unknown): Response {
  return { ok: false, status, json: async () => body } as unknown as Response
}

type Deferred = {
  promise: Promise<Response>
  resolve: (response: Response) => void
  reject: (error: unknown) => void
}

function deferred(): Deferred {
  let resolve!: (response: Response) => void
  let reject!: (error: unknown) => void
  const promise = new Promise<Response>((res, rej) => {
    resolve = res
    reject = rej
  })
  return { promise, resolve, reject }
}

let fetchMock: ReturnType<typeof vi.fn>
const pending: { ping: Deferred[]; tool: Deferred[] } = { ping: [], tool: [] }
let storage: Storage

function installCompiledRender(component: object, source: string, filename: string, id: string) {
  const { descriptor } = parse(source, { filename })
  if (!descriptor.template) throw new Error(`${filename} is missing a template`)

  const compiledScript = compileScript(descriptor, { id })
  const compiledTemplate = compileTemplate({
    source: descriptor.template.content,
    filename,
    id,
    compilerOptions: { mode: 'function', bindingMetadata: compiledScript.bindings },
  })
  if (compiledTemplate.errors.length > 0) {
    throw new Error(compiledTemplate.errors.map(String).join('\n'))
  }
  const render = new Function('Vue', compiledTemplate.code)(VueRuntime)
  Object.assign(component, { render })
}

let Settings: typeof import('./Settings.vue').default

beforeAll(async () => {
  // Seed a token before importing: api.ts provisions anonymously at module
  // load, and with a token present that path stays quiet.
  vi.stubGlobal('localStorage', memoryStorage({
    'study-coach:settings': JSON.stringify({ accessToken: 'preseeded-token', tier: 'guest' }),
  }))
  const [componentModule, sourceModule] = await Promise.all([
    import('./Settings.vue'),
    import('./Settings.vue?raw'),
  ])
  Settings = componentModule.default
  installCompiledRender(
    Settings,
    (sourceModule as { default: string }).default,
    'Settings.vue',
    'settings-connection',
  )
})

afterAll(() => {
  vi.unstubAllGlobals()
})

function mountSettings(): { app: { mount: (root: TestNode) => void; unmount: () => void }; root: TestNode } {
  const root: TestNode = { children: [], props: {}, handlers: {}, listeners: {} }
  const renderer = createRenderer<TestNode, TestNode>({
    patchProp: (el, key, _prev, next) => {
      if (key.startsWith('on') && typeof next === 'function') {
        el.handlers[key.slice(2).toLowerCase()] = next as (event?: unknown) => void
      } else {
        el.props[key] = next
      }
    },
    insert: (child, parent) => {
      child.parent = parent
      parent.children.push(child)
    },
    remove: (child) => {
      const parent = child.parent
      if (!parent) return
      parent.children = parent.children.filter(node => node !== child)
    },
    createElement: (tag) => {
      const node: TestNode = {
        children: [],
        tag,
        props: {},
        handlers: {},
        listeners: {},
        removeEventListener: () => undefined,
      }
      // Real addEventListener semantics: capture v-model directive listeners
      // so a dispatched change event runs the same listener chain a browser
      // would (directive listener first, then the template @change handler).
      node.addEventListener = (type, handler) => {
        if (typeof handler !== 'function') return
        const list = node.listeners[type] ?? (node.listeners[type] = [])
        if (!list.includes(handler)) list.push(handler)
      }
      // v-model directives in the real SFC inspect these DOM-ish fields.
      if (tag === 'select') Object.assign(node, { options: [], selectedIndex: -1 })
      return node
    },
    createText: text => ({ children: [], tag: '#text', text, props: {}, handlers: {}, listeners: {} }),
    createComment: text => ({ children: [], tag: '#comment', text, props: {}, handlers: {}, listeners: {} }),
    setText: (node, text) => { node.text = text },
    setElementText: (node, text) => { node.text = text },
    parentNode: node => node.parent ?? null,
    nextSibling: () => null,
  })
  const app = renderer.createApp(Settings)
  const pinia = createPinia()
  setActivePinia(pinia)
  app.use(pinia)
  app.provide(ssrContextKey, { modules: new Set<string>() })
  app.config.globalProperties.$t = (key: string) => key
  app.mount(root)
  return { app, root }
}

// Simulates a real change event on a v-model bound control: the select gets
// its options rebuilt from the rendered <option> children with the chosen
// value marked selected (what a browser would do), and the checkbox flips
// its checked flag. Every registered change listener — the v-model directive
// first, then the template handler — then runs in registration order.
function dispatchChange(node: TestNode, value: string | boolean): void {
  if (node.tag === 'select') {
    node.options = node.children
      .filter(child => child.tag === 'option')
      .map(child => ({
        value: String(child.props.value ?? ''),
        selected: child.props.value === value,
      }))
  } else {
    node.checked = value
  }
  const event = { target: { value, checked: value } }
  for (const handler of [
    ...(node.listeners.change ?? []),
    ...(node.handlers.change ? [node.handlers.change] : []),
  ]) {
    handler(event)
  }
}

async function flushUi() {
  // Bounded microtask draining: each await lets the deferred response chain
  // (fetch → json → catch/finally → scheduler) advance by one turn, so drain
  // enough turns for it to finish. Deterministic — no timed sleep anywhere.
  for (let i = 0; i < 50; i += 1) await Promise.resolve()
  await nextTick()
  await Promise.resolve()
  await nextTick()
}

function nodeText(node: TestNode): string {
  return `${node.text ?? ''}${node.children.map(nodeText).join('')}`
}

function findNode(root: TestNode, predicate: (node: TestNode) => boolean): TestNode | undefined {
  for (const child of root.children) {
    if (predicate(child)) return child
    const found = findNode(child, predicate)
    if (found) return found
  }
  return undefined
}

function requireNode(root: TestNode, predicate: (node: TestNode) => boolean, label: string): TestNode {
  const node = findNode(root, predicate)
  if (!node) throw new Error(`node not found: ${label}`)
  return node
}

function buttonByText(root: TestNode, label: string): TestNode {
  return requireNode(root, node => node.tag === 'button' && nodeText(node).includes(label), `button ${label}`)
}

function click(node: TestNode): void {
  const handler = node.handlers.click
  if (!handler) throw new Error('no click handler on node')
  handler({})
}

function editConnection(node: TestNode, event: 'input' | 'change', value: string): void {
  const handler = node.handlers[event]
  if (!handler) throw new Error(`no ${event} handler on node`)
  handler({ target: { value } })
}

const modelInput = (root: TestNode) =>
  requireNode(
    root,
    node => node.tag === 'input' && String(node.props.placeholder ?? '').includes('gpt-4o-mini'),
    'model input',
  )
const providerSelect = (root: TestNode) =>
  requireNode(
    root,
    node => node.tag === 'select' && node.children.some(o => o.props?.value === 'ollama'),
    'provider select',
  )

beforeEach(() => {
  storage = memoryStorage({
    'study-coach:settings': JSON.stringify({
      provider: 'openai',
      model: 'gpt-4o-mini',
      apiKey: 'sk-test',
      baseUrl: '',
      accessToken: 'preseeded-token',
      tier: 'guest',
      connectionRevision: REVISION_A,
    }),
    [CAPABILITY_KEY]: capabilityRecord(REVISION_A, true),
  })
  vi.stubGlobal('localStorage', storage)
  let counter = 0
  vi.stubGlobal('crypto', {
    randomUUID: vi.fn(() => {
      counter += 1
      return `00000000-0000-4000-8000-${String(counter).padStart(12, '0')}`
    }),
  })
  pending.ping = []
  pending.tool = []
  fetchMock = vi.fn((input: RequestInfo | URL): Promise<Response> => {
    const url = String(input)
    const d = deferred()
    if (url.includes('/api/models/ping')) pending.ping.push(d)
    else if (url.includes('/api/models/tool-check')) pending.tool.push(d)
    else d.reject(new Error(`unexpected fetch target: ${url}`))
    return d.promise
  })
  vi.stubGlobal('fetch', fetchMock)
  // Drain any capability result left over by an earlier test in this file
  // (rotating a throwaway store clears the module-level pending result and
  // leaves storage untouched).
  setActivePinia(createPinia())
  useSettings().rotateConnectionRevision()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('Settings connection checks', () => {
  it('does not adopt a stale ping success that resolves after a connection edit', async () => {
    const { app, root } = mountSettings()
    try {
      await flushUi()
      click(buttonByText(root, 'Test Connection'))
      await flushUi()

      const modelField = modelInput(root)
      editConnection(modelField, 'input', 'gpt-4o')
      // Resolve in the same block as the edit: the revision watcher is sync,
      // so the response continuation must already see an invalidated request.
      pending.ping[0].resolve(
        okResponse({ ok: true, model: 'gpt-4o-mini', latency_ms: 5, note: 'Connected — responded in 5ms' }),
      )
      await flushUi()

      const { useSettings } = await import('../stores/settings')
      const settings = useSettings()
      expect(settings.connectionRevision).not.toBe(REVISION_A)
      expect(settings.toolCapable).toBeNull()
      expect(nodeText(root)).not.toContain('Connected')
      expect(nodeText(root)).not.toContain('Testing…')
      expect(fetchMock).toHaveBeenCalledTimes(1)
    } finally {
      app.unmount()
    }
  })

  it('keeps a newer pending ping while an older error and finally settle', async () => {
    const { app, root } = mountSettings()
    try {
      await flushUi()
      click(buttonByText(root, 'Test Connection'))
      await flushUi()
      editConnection(modelInput(root), 'input', 'gpt-4o')
      await flushUi() // re-render restores the button label after the sync clear
      click(buttonByText(root, 'Test Connection'))
      await flushUi()
      expect(pending.ping).toHaveLength(2)

      pending.ping[0].reject(new TypeError('network dropped: RAW-DETAIL'))
      await flushUi()
      // The stale failure must neither write its note nor clear the newer
      // request's loading flag.
      expect(nodeText(root)).not.toContain('network dropped')
      expect(nodeText(root)).toContain('Testing…')

      pending.ping[1].resolve(
        okResponse({ ok: false, model: 'gpt-4o', latency_ms: 3, note: 'Failed: could not connect' }),
      )
      await flushUi()
      expect(nodeText(root)).toContain('Failed: could not connect')
      expect(nodeText(root)).not.toContain('Testing…')
    } finally {
      app.unmount()
    }
  })

  it('clears check state synchronously when a connection field is edited', async () => {
    const { app, root } = mountSettings()
    try {
      await flushUi()
      click(buttonByText(root, 'Test Connection'))
      await flushUi()
      expect(nodeText(root)).toContain('Testing…')

      editConnection(modelInput(root), 'input', 'gpt-4o')
      await flushUi()

      expect(nodeText(root)).not.toContain('Testing…')
      expect(nodeText(root)).toContain('Not tested')
      const { useSettings } = await import('../stores/settings')
      expect(useSettings().toolCapable).toBeNull()
    } finally {
      app.unmount()
    }
  })

  it('does not commit a stale tool-check result after a connection edit', async () => {
    const { app, root } = mountSettings()
    try {
      await flushUi()
      const settingsModule = await import('../stores/settings')
      const settings = settingsModule.useSettings()
      expect(settings.toolCapable).toBe(true)

      click(buttonByText(root, 'Test Tool Call'))
      await flushUi()
      // The re-test itself drops the cached capability before the request.
      expect(settings.toolCapable).toBeNull()
      expect(storage.getItem(CAPABILITY_KEY)).toBeNull()

      editConnection(modelInput(root), 'input', 'gpt-4o')
      pending.tool[0].resolve(
        okResponse({ tool_capable: false, model: 'gpt-4o-mini', note: 'No tool calls observed' }),
      )
      await flushUi()

      expect(settings.toolCapable).toBeNull()
      expect(storage.getItem(CAPABILITY_KEY)).toBeNull()
      expect(nodeText(root)).not.toContain('No Tool Calls Observed')
      expect(nodeText(root)).not.toContain('No tool calls observed')
    } finally {
      app.unmount()
    }
  })

  it('writes the v2 record when a check finishes for the saved configuration', async () => {
    const { app, root } = mountSettings()
    try {
      await flushUi()
      click(buttonByText(root, 'Test Tool Call'))
      pending.tool[0].resolve(
        okResponse({ tool_capable: false, model: 'gpt-4o-mini', note: 'No tool calls observed in this probe' }),
      )
      await flushUi()

      const { useSettings } = await import('../stores/settings')
      const settings = useSettings()
      expect(settings.toolCapable).toBe(false)
      expect(JSON.parse(storage.getItem(CAPABILITY_KEY) ?? '{}')).toEqual({
        schemaVersion: 2,
        connectionRevision: REVISION_A,
        toolCapable: false,
      })
      expect(nodeText(root)).toContain('No Tool Calls Observed')
      expect(nodeText(root)).toContain('No tool calls observed in this probe')
      // agent_loop defaults stay locked to deterministic after a false probe.
      const modeSelect = requireNode(
        root,
        node => node.tag === 'select' && node.children.some(o => o.props?.value === 'agent_loop'),
        'mode select',
      )
      expect(modeSelect.props.disabled).toBe(true)
    } finally {
      app.unmount()
    }
  })

  it('does not write responses that arrive after the view unmounted', async () => {
    const { app, root } = mountSettings()
    await flushUi()
    click(buttonByText(root, 'Test Tool Call'))
    click(buttonByText(root, 'Test Connection'))
    await flushUi()
    app.unmount()

    pending.tool[0].resolve(okResponse({ tool_capable: true, model: 'm', note: 'ok' }))
    pending.ping[0].resolve(okResponse({ ok: true, model: 'm', latency_ms: 1, note: 'Connected' }))
    await flushUi()

    const { useSettings } = await import('../stores/settings')
    const settings = useSettings()
    expect(settings.toolCapable).toBeNull()
    expect(storage.getItem(CAPABILITY_KEY)).toBeNull()
  })

  it('explains a stable 400 with fixed safe text and no echo of the body', async () => {
    const { app, root } = mountSettings()
    try {
      await flushUi()
      click(buttonByText(root, 'Test Connection'))

      pending.ping[0].resolve(
        statusResponse(400, {
          detail: {
            code: 'invalid_llm_config',
            field: 'provider',
            message: 'backend-fixed-sentence-SECRET',
          },
        }),
      )
      await flushUi()


      expect(nodeText(root)).toContain(
        'The provider is not supported. Choose ollama, openai, anthropic, or gemini in Settings.',
      )
      expect(nodeText(root)).not.toContain('backend-fixed-sentence-SECRET')
      expect(nodeText(root)).toContain('Failed')
    } finally {
      app.unmount()
    }
  })

  it('shows the fixed generic message for unknown check failures', async () => {
    const { app, root } = mountSettings()
    try {
      await flushUi()
      click(buttonByText(root, 'Test Tool Call'))
      pending.tool[0].reject(new TypeError('RAW-FETCH-DETAIL'))
      await flushUi()

      expect(nodeText(root)).toContain('The connection check failed. Please try again.')
      expect(nodeText(root)).not.toContain('RAW-FETCH-DETAIL')
      expect(nodeText(root)).not.toContain('TypeError')
    } finally {
      app.unmount()
    }
  })

  it('edits the four connection fields through the store action only', async () => {
    const { app, root } = mountSettings()
    try {
      await flushUi()
      const { useSettings } = await import('../stores/settings')
      const settings = useSettings()

      editConnection(providerSelect(root), 'change', 'anthropic')
      editConnection(modelInput(root), 'input', 'claude-haiku-4-5')
      editConnection(
        requireNode(root, node => node.tag === 'input' && node.props.type === 'password', 'api key input'),
        'input',
        'sk-new',
      )
      editConnection(
        requireNode(
          root,
          node => node.tag === 'input' && String(node.props.placeholder ?? '').includes('api.openai.com'),
          'base url input',
        ),
        'input',
        'https://proxy.test/v1',
      )

      expect(settings.provider).toBe('anthropic')
      expect(settings.model).toBe('claude-haiku-4-5')
      expect(settings.apiKey).toBe('sk-new')
      expect(settings.baseUrl).toBe('https://proxy.test/v1')
      // Every real change rotated the revision and cleared the capability.
      expect(settings.connectionRevision).not.toBe(REVISION_A)
      expect(settings.toolCapable).toBeNull()
      // Editing back to the original values must not revive the generation.
      editConnection(modelInput(root), 'input', 'gpt-4o-mini')
      expect(settings.connectionRevision).not.toBe(REVISION_A)
    } finally {
      app.unmount()
    }
  })
})

describe('Settings first-detection connection revision', () => {
  const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i

  function seedWithoutRevision(saved: Record<string, unknown>): void {
    storage = memoryStorage({
      'study-coach:settings': JSON.stringify({
        accessToken: 'preseeded-token',
        tier: 'guest',
        ...saved,
      }),
    })
    vi.stubGlobal('localStorage', storage)
  }

  async function runFirstToolCheck(root: TestNode, capable: boolean): Promise<void> {
    click(buttonByText(root, 'Test Tool Call'))
    pending.tool[0].resolve(
      okResponse({ tool_capable: capable, model: 'gemma3:4b', note: 'Tool call round trip completed.' }),
    )
    await flushUi()
  }

  it('establishes a revision on the first detection so Save makes it restorable: saved ollama defaults', async () => {
    seedWithoutRevision({ provider: 'ollama', model: 'gemma3:4b' })
    const { app, root } = mountSettings()
    try {
      await flushUi()
      await runFirstToolCheck(root, true)

      const settings = useSettings()
      // The ensure step did not invalidate its own request: the commit landed.
      expect(settings.toolCapable).toBe(true)
      const detectedRevision = settings.connectionRevision
      expect(detectedRevision).toMatch(UUID_PATTERN)
      // Not cached before a Save.
      expect(storage.getItem(CAPABILITY_KEY)).toBeNull()

      click(buttonByText(root, 'settings.save'))
      await flushUi()

      expect(JSON.parse(storage.getItem('study-coach:settings') ?? '{}').connectionRevision)
        .toBe(detectedRevision)
      expect(JSON.parse(storage.getItem(CAPABILITY_KEY) ?? '{}')).toMatchObject({
        schemaVersion: 2,
        connectionRevision: detectedRevision,
        toolCapable: true,
      })

      setActivePinia(createPinia())
      const restored = useSettings()
      expect(restored.toolCapable).toBe(true)
      expect(restored.connectionRevision).toBe(detectedRevision)
    } finally {
      app.unmount()
    }
  })

  it('establishes a revision on the first detection so Save makes it restorable: token-only defaults', async () => {
    seedWithoutRevision({})
    const { app, root } = mountSettings()
    try {
      await flushUi()
      await runFirstToolCheck(root, true)

      const settings = useSettings()
      expect(settings.toolCapable).toBe(true)
      const detectedRevision = settings.connectionRevision
      expect(detectedRevision).toMatch(UUID_PATTERN)

      click(buttonByText(root, 'settings.save'))
      await flushUi()

      expect(JSON.parse(storage.getItem(CAPABILITY_KEY) ?? '{}')).toMatchObject({
        connectionRevision: detectedRevision,
        toolCapable: true,
      })

      setActivePinia(createPinia())
      const restored = useSettings()
      expect(restored.provider).toBe('ollama')
      expect(restored.model).toBe('gemma3:4b')
      expect(restored.toolCapable).toBe(true)
    } finally {
      app.unmount()
    }
  })

  it('restores a legitimate negative first detection for a legacy saved cloud config', async () => {
    seedWithoutRevision({ provider: 'openai', model: 'gpt-4o-mini', apiKey: 'sk-test', baseUrl: '' })
    const { app, root } = mountSettings()
    try {
      await flushUi()
      await runFirstToolCheck(root, false)

      const settings = useSettings()
      expect(settings.toolCapable).toBe(false)
      const detectedRevision = settings.connectionRevision
      expect(detectedRevision).toMatch(UUID_PATTERN)

      click(buttonByText(root, 'settings.save'))
      await flushUi()

      expect(JSON.parse(storage.getItem(CAPABILITY_KEY) ?? '{}')).toMatchObject({
        connectionRevision: detectedRevision,
        toolCapable: false,
      })

      setActivePinia(createPinia())
      const restored = useSettings()
      expect(restored.toolCapable).toBe(false)
      expect(restored.connectionRevision).toBe(detectedRevision)
    } finally {
      app.unmount()
    }
  })

  it('never auto-saves an unsaved edited configuration and keeps its result in memory', async () => {
    // Saved A (with revision); the user edits to unsaved B and checks.
    const { app, root } = mountSettings()
    try {
      await flushUi()
      editConnection(modelInput(root), 'input', 'gpt-4o')
      await flushUi()
      await runFirstToolCheck(root, true)

      const settings = useSettings()
      expect(settings.toolCapable).toBe(true)
      // Detection must not persist the unsaved B fields nor cache the result
      // for the unsaved generation; a different-revision record for saved A
      // legitimately survives.
      const storedSettings = JSON.parse(storage.getItem('study-coach:settings') ?? '{}')
      expect(storedSettings.model).toBe('gpt-4o-mini')
      expect(storedSettings.connectionRevision).toBe(REVISION_A)
      const recordDuringDetect = JSON.parse(storage.getItem(CAPABILITY_KEY) ?? '{}')
      expect(recordDuringDetect.connectionRevision ?? REVISION_A).toBe(REVISION_A)
      expect(recordDuringDetect.connectionRevision ?? REVISION_A)
        .not.toBe(settings.connectionRevision)

      // Only an explicit Save of the same B adopts the result.
      click(buttonByText(root, 'settings.save'))
      await flushUi()
      expect(JSON.parse(storage.getItem(CAPABILITY_KEY) ?? '{}')).toMatchObject({
        connectionRevision: settings.connectionRevision,
        toolCapable: true,
      })
    } finally {
      app.unmount()
    }
  })

  it('keeps a failed first detection unknown and caches nothing', async () => {
    seedWithoutRevision({ provider: 'ollama', model: 'gemma3:4b' })
    const { app, root } = mountSettings()
    try {
      await flushUi()
      click(buttonByText(root, 'Test Tool Call'))
      pending.tool[0].resolve(
        okResponse({ tool_capable: null, model: 'gemma3:4b', note: 'Tool check did not complete a valid round trip.' }),
      )
      await flushUi()

      const settings = useSettings()
      expect(settings.toolCapable).toBeNull()

      click(buttonByText(root, 'settings.save'))
      await flushUi()

      expect(storage.getItem(CAPABILITY_KEY)).toBeNull()
      setActivePinia(createPinia())
      expect(useSettings().toolCapable).toBeNull()
    } finally {
      app.unmount()
    }
  })
})

describe('Settings preference auto-save boundary', () => {
  const languageSelect = (root: TestNode) =>
    requireNode(
      root,
      node => node.tag === 'select' && node.children.some(o => o.props?.value === 'en'),
      'language select',
    )
  const debugCheckbox = (root: TestNode) =>
    requireNode(root, node => node.tag === 'input' && node.props.type === 'checkbox', 'debug checkbox')

  function saveButton(root: TestNode): TestNode {
    return buttonByText(root, 'settings.save')
  }

  it('does not persist the unsaved connection or promote its result: language entry', async () => {
    // beforeEach seed: saved A (revision R_A) + A capability record (true).
    const { app, root } = mountSettings()
    try {
      await flushUi()
      // Unsaved B through the real connection bindings, then a full check.
      editConnection(providerSelect(root), 'change', 'anthropic')
      editConnection(modelInput(root), 'input', 'claude-haiku-4-5')
      editConnection(
        requireNode(root, node => node.tag === 'input' && node.props.type === 'password', 'api key input'),
        'input',
        'sk-b',
      )
      editConnection(
        requireNode(
          root,
          node => node.tag === 'input' && String(node.props.placeholder ?? '').includes('api.openai.com'),
          'base url input',
        ),
        'input',
        'https://b.test/v1',
      )
      await flushUi()
      click(buttonByText(root, 'Test Tool Call'))
      pending.tool[0].resolve(
        okResponse({ tool_capable: true, model: 'claude-haiku-4-5', note: 'Tool call round trip completed.' }),
      )
      await flushUi()
      const settings = useSettings()
      expect(settings.toolCapable).toBe(true)

      // Pre-control: storage still holds A and the A capability record.
      expect(JSON.parse(storage.getItem('study-coach:settings') ?? '{}').model).toBe('gpt-4o-mini')
      expect(JSON.parse(storage.getItem(CAPABILITY_KEY) ?? '{}').connectionRevision).toBe(REVISION_A)

      // Preference auto-save WITHOUT clicking Save.
      dispatchChange(languageSelect(root), 'zh-CN')
      await flushUi()

      const storedAfter = JSON.parse(storage.getItem('study-coach:settings') ?? '{}')
      expect(storedAfter.language).toBe('zh-CN')
      expect(storedAfter.provider).toBe('openai')
      expect(storedAfter.model).toBe('gpt-4o-mini')
      expect(storedAfter.apiKey).toBe('sk-test')
      expect(storedAfter.connectionRevision).toBe(REVISION_A)
      // The capability record is untouched: B's true result was not promoted.
      expect(JSON.parse(storage.getItem(CAPABILITY_KEY) ?? '{}')).toMatchObject({
        connectionRevision: REVISION_A,
        toolCapable: true,
      })

      // Active memory keeps B, its revision and its result.
      expect(settings.provider).toBe('anthropic')
      expect(settings.model).toBe('claude-haiku-4-5')
      expect(settings.toolCapable).toBe(true)
      expect(settings.connectionRevision).not.toBe(REVISION_A)

      // Refresh equivalent: a recreated store restores saved A + A's cache.
      setActivePinia(createPinia())
      const restored = useSettings()
      expect(restored.model).toBe('gpt-4o-mini')
      expect(restored.toolCapable).toBe(true)
      expect(restored.language).toBe('zh-CN')

      // Only the explicit Save adopts B and its matched result.
      click(saveButton(root))
      await flushUi()
      expect(JSON.parse(storage.getItem('study-coach:settings') ?? '{}').model).toBe('claude-haiku-4-5')
      expect(JSON.parse(storage.getItem(CAPABILITY_KEY) ?? '{}')).toMatchObject({
        connectionRevision: settings.connectionRevision,
        toolCapable: true,
      })
      setActivePinia(createPinia())
      const restoredB = useSettings()
      expect(restoredB.model).toBe('claude-haiku-4-5')
      expect(restoredB.toolCapable).toBe(true)
    } finally {
      app.unmount()
    }
  })

  it('does not persist the unsaved connection or promote its result: debug entry', async () => {
    const { app, root } = mountSettings()
    try {
      await flushUi()
      editConnection(modelInput(root), 'input', 'gpt-4o')
      await flushUi()
      click(buttonByText(root, 'Test Tool Call'))
      pending.tool[0].resolve(
        okResponse({ tool_capable: true, model: 'gpt-4o', note: 'Tool call round trip completed.' }),
      )
      await flushUi()

      dispatchChange(debugCheckbox(root), true)
      await flushUi()

      const storedAfter = JSON.parse(storage.getItem('study-coach:settings') ?? '{}')
      expect(storedAfter.debugMode).toBe(true)
      expect(storedAfter.model).toBe('gpt-4o-mini')
      expect(JSON.parse(storage.getItem(CAPABILITY_KEY) ?? '{}').connectionRevision).toBe(REVISION_A)

      const settings = useSettings()
      expect(settings.model).toBe('gpt-4o')
      expect(settings.toolCapable).toBe(true)

      click(saveButton(root))
      await flushUi()
      expect(JSON.parse(storage.getItem('study-coach:settings') ?? '{}').model).toBe('gpt-4o')
      expect(JSON.parse(storage.getItem(CAPABILITY_KEY) ?? '{}')).toMatchObject({
        connectionRevision: settings.connectionRevision,
        toolCapable: true,
      })
    } finally {
      app.unmount()
    }
  })

  it('keeps a failed detection unknown and uncached across preference auto-save', async () => {
    const { app, root } = mountSettings()
    try {
      await flushUi()
      editConnection(modelInput(root), 'input', 'gpt-4o')
      await flushUi()
      click(buttonByText(root, 'Test Tool Call'))
      pending.tool[0].resolve(
        okResponse({
          tool_capable: null,
          model: 'gpt-4o',
          note: 'Tool check did not complete a valid round trip.',
        }),
      )
      await flushUi()
      expect(useSettings().toolCapable).toBeNull()

      dispatchChange(languageSelect(root), 'zh-CN')
      await flushUi()

      const storedAfter = JSON.parse(storage.getItem('study-coach:settings') ?? '{}')
      expect(storedAfter.language).toBe('zh-CN')
      expect(storedAfter.model).toBe('gpt-4o-mini')
      expect(JSON.parse(storage.getItem(CAPABILITY_KEY) ?? '{}').connectionRevision).toBe(REVISION_A)

      // Even the explicit Save must not cache the failed result.
      click(saveButton(root))
      await flushUi()
      expect(JSON.parse(storage.getItem('study-coach:settings') ?? '{}').model).toBe('gpt-4o')
      expect(JSON.parse(storage.getItem(CAPABILITY_KEY) ?? '{}').connectionRevision).toBe(REVISION_A)
    } finally {
      app.unmount()
    }
  })

  it('never writes an undetected unsaved connection through preference auto-save', async () => {
    const { app, root } = mountSettings()
    try {
      await flushUi()
      editConnection(modelInput(root), 'input', 'gpt-4o')
      await flushUi()

      dispatchChange(debugCheckbox(root), true)
      await flushUi()

      const storedAfter = JSON.parse(storage.getItem('study-coach:settings') ?? '{}')
      expect(storedAfter.debugMode).toBe(true)
      expect(storedAfter.model).toBe('gpt-4o-mini')
      expect(JSON.parse(storage.getItem(CAPABILITY_KEY) ?? '{}').connectionRevision).toBe(REVISION_A)

      // Explicit Save of the undetected B persists B but still no B cache.
      click(saveButton(root))
      await flushUi()
      expect(JSON.parse(storage.getItem('study-coach:settings') ?? '{}').model).toBe('gpt-4o')
      expect(JSON.parse(storage.getItem(CAPABILITY_KEY) ?? '{}').connectionRevision).toBe(REVISION_A)
    } finally {
      app.unmount()
    }
  })

  it('does not invalidate an in-flight check when preferences auto-save', async () => {
    const { app, root } = mountSettings()
    try {
      await flushUi()
      editConnection(modelInput(root), 'input', 'gpt-4o')
      await flushUi()
      click(buttonByText(root, 'Test Tool Call'))

      // Preference auto-save while the request is in flight.
      dispatchChange(languageSelect(root), 'zh-CN')
      await flushUi()

      pending.tool[0].resolve(
        okResponse({ tool_capable: true, model: 'gpt-4o', note: 'Tool call round trip completed.' }),
      )
      await flushUi()

      const settings = useSettings()
      // The preference change did not kill the request; the result stays in
      // memory only and the stored configuration is still A.
      expect(settings.toolCapable).toBe(true)
      const storedAfter = JSON.parse(storage.getItem('study-coach:settings') ?? '{}')
      expect(storedAfter.model).toBe('gpt-4o-mini')
      expect(storedAfter.language).toBe('zh-CN')
      expect(JSON.parse(storage.getItem(CAPABILITY_KEY) ?? '{}').connectionRevision).toBe(REVISION_A)

      click(saveButton(root))
      await flushUi()
      expect(JSON.parse(storage.getItem(CAPABILITY_KEY) ?? '{}')).toMatchObject({
        connectionRevision: settings.connectionRevision,
        toolCapable: true,
      })
    } finally {
      app.unmount()
    }
  })

  it('never fills a partial stored record with the active unsaved connection', async () => {
    storage = memoryStorage({
      'study-coach:settings': JSON.stringify({ accessToken: 'preseeded-token', tier: 'guest' }),
    })
    vi.stubGlobal('localStorage', storage)
    const { app, root } = mountSettings()
    try {
      await flushUi()
      editConnection(modelInput(root), 'input', 'gpt-4o')
      await flushUi()

      dispatchChange(languageSelect(root), 'zh-CN')
      await flushUi()

      const storedAfter = JSON.parse(storage.getItem('study-coach:settings') ?? '{}')
      expect(storedAfter.language).toBe('zh-CN')
      // The partial record keeps its normalized defaults — not the active edit.
      expect(storedAfter.provider).toBe('ollama')
      expect(storedAfter.model).toBe('gemma3:4b')
      expect(storedAfter.apiKey).toBe('')
    } finally {
      app.unmount()
    }
  })

  it('keeps plain preference auto-save working for an already saved configuration', async () => {
    const { app, root } = mountSettings()
    try {
      await flushUi()
      dispatchChange(languageSelect(root), 'zh-CN')
      dispatchChange(debugCheckbox(root), true)
      await flushUi()

      const storedAfter = JSON.parse(storage.getItem('study-coach:settings') ?? '{}')
      expect(storedAfter.language).toBe('zh-CN')
      expect(storedAfter.debugMode).toBe(true)
      expect(storedAfter.model).toBe('gpt-4o-mini')
      expect(storedAfter.connectionRevision).toBe(REVISION_A)
      expect(JSON.parse(storage.getItem(CAPABILITY_KEY) ?? '{}')).toMatchObject({
        connectionRevision: REVISION_A,
        toolCapable: true,
      })
      const settings = useSettings()
      expect(settings.connectionRevision).toBe(REVISION_A)
      expect(settings.toolCapable).toBe(true)
    } finally {
      app.unmount()
    }
  })
})
