import { createRenderer, h, nextTick, reactive, ssrContextKey } from 'vue'
import * as VueRuntime from 'vue'
import { compileScript, compileTemplate, parse } from 'vue/compiler-sfc'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { Citation } from '../stores/chat'
import type { ChunkDto } from '../lib/api'

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })
  return { promise, resolve, reject }
}

const mocks = vi.hoisted(() => {
  const pending = new Map<string, ReturnType<typeof deferred<ChunkDto>>>()
  return {
    pending,
    getChunk: vi.fn((chunkId: string) => {
      const entry = pending.get(chunkId)
      if (!entry) {
        throw new Error(`No deferred promise registered for chunk ${chunkId}`)
      }
      return entry.promise
    }),
  }
})

vi.mock('../lib/api', () => ({
  getChunk: mocks.getChunk,
}))

import CitationPreviewModal from './CitationPreviewModal.vue'
import citationPreviewModalSource from './CitationPreviewModal.vue?raw'

const { descriptor } = parse(citationPreviewModalSource, { filename: 'CitationPreviewModal.vue' })
if (!descriptor.template) {
  throw new Error('CitationPreviewModal.vue is missing a template')
}
const compiledScript = compileScript(descriptor, { id: 'citation-preview-modal-test' })
const compiledTemplate = compileTemplate({
  source: descriptor.template.content,
  filename: 'CitationPreviewModal.vue',
  id: 'citation-preview-modal-test',
  compilerOptions: {
    mode: 'function',
    bindingMetadata: compiledScript.bindings,
  },
})
if (compiledTemplate.errors.length > 0) {
  throw new Error(compiledTemplate.errors.map(String).join('\n'))
}
const render = new Function('Vue', compiledTemplate.code)(VueRuntime)
Object.assign(CitationPreviewModal, { render })

type TestNode = { parent?: TestNode; children: TestNode[]; tag?: string; text?: string }

type ModalSetupState = {
  chunk: ChunkDto | null | { value: ChunkDto | null }
}

async function flushUi() {
  await Promise.resolve()
  await nextTick()
  await Promise.resolve()
  await nextTick()
}

function readChunkContent(chunk: ModalSetupState['chunk']): string {
  if (!chunk) return ''
  if (typeof chunk === 'object' && 'value' in chunk) {
    return chunk.value?.content ?? ''
  }
  return chunk.content ?? ''
}

function findUniqueHeading(root: TestNode): TestNode {
  const matches: TestNode[] = []
  function visit(node: TestNode) {
    if (node.tag === 'h3') matches.push(node)
    node.children.forEach(visit)
  }
  visit(root)
  if (matches.length !== 1) {
    throw new Error(`Expected one h3 heading, found ${matches.length}`)
  }
  return matches[0]
}

function readNodeText(node: TestNode): string {
  return `${node.text ?? ''}${node.children.map(readNodeText).join('')}`
}

function mountModal(initialCitation: Citation) {
  const root: TestNode = { children: [] }
  const state = reactive<{ citation: Citation | null }>({ citation: initialCitation })
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
    createElement: tag => ({ children: [], tag }),
    createText: text => ({ children: [], text }),
    createComment: text => ({ children: [], text }),
    setText: (node, text) => { node.text = text },
    setElementText: (node, text) => { node.text = text },
    parentNode: node => node.parent ?? null,
    nextSibling: () => null,
  })

  const Wrapper = {
    setup() {
      return () => h(CitationPreviewModal, { citation: state.citation })
    },
  }

  const app = renderer.createApp(Wrapper)
  app.provide(ssrContextKey, { modules: new Set<string>() })
  app.mount(root)

  function displayedContent(): string {
    const rootInstance = (app as unknown as {
      _instance: { subTree: { component: { setupState: ModalSetupState } } }
    })._instance
    return readChunkContent(rootInstance.subTree.component.setupState.chunk)
  }

  function headingText(): string {
    return readNodeText(findUniqueHeading(root))
  }

  return {
    displayedContent,
    headingText,
    setCitation(citation: Citation | null) {
      state.citation = citation
    },
    unmount: () => app.unmount(),
  }
}

beforeEach(() => {
  mocks.pending.clear()
  mocks.getChunk.mockClear()
})

describe('CitationPreviewModal stale response', () => {
  it('keeps citation B after a late response for previously requested citation A arrives', async () => {
    const citationA: Citation = {
      chunk_id: 'chunk-a',
      source: 'notes.pdf',
      page: 1,
      span_start: 0,
      span_end: 0,
    }
    const citationB: Citation = {
      chunk_id: 'chunk-b',
      source: 'notes.pdf',
      page: 2,
      span_start: 0,
      span_end: 0,
    }
    const chunkA: ChunkDto = {
      chunk_id: 'chunk-a',
      content: 'CONTENT_FROM_CITATION_A',
      source: 'notes.pdf',
      page: 1,
    }
    const chunkB: ChunkDto = {
      chunk_id: 'chunk-b',
      content: 'CONTENT_FROM_CITATION_B',
      source: 'notes.pdf',
      page: 2,
    }

    const deferredA = deferred<ChunkDto>()
    const deferredB = deferred<ChunkDto>()
    mocks.pending.set('chunk-a', deferredA)
    mocks.pending.set('chunk-b', deferredB)

    const modal = mountModal(citationA)
    await flushUi()
    expect(mocks.getChunk).toHaveBeenCalledWith('chunk-a')

    modal.setCitation(citationB)
    await flushUi()
    expect(mocks.getChunk).toHaveBeenCalledWith('chunk-b')

    deferredB.resolve(chunkB)
    await flushUi()
    expect(modal.displayedContent()).toContain('CONTENT_FROM_CITATION_B')
    expect(modal.displayedContent()).not.toContain('CONTENT_FROM_CITATION_A')

    deferredA.resolve(chunkA)
    await flushUi()

    // Stale A must not overwrite the active B preview.
    expect(modal.displayedContent()).toContain('CONTENT_FROM_CITATION_B')
    expect(modal.displayedContent()).not.toContain('CONTENT_FROM_CITATION_A')

    modal.unmount()
  })
})

describe('CitationPreviewModal source precedence', () => {
  it('prefers the fetched chunk source over the citation fallback in the modal heading', async () => {
    const citation: Citation = {
      chunk_id: 'hash:1:0',
      source: 'hash:1:0',
      page: 1,
      span_start: 0,
      span_end: 0,
    }
    const fetchedChunk: ChunkDto = {
      chunk_id: 'hash:1:0',
      content: 'Fetched chunk content',
      source: 'lecture-notes.pdf',
      page: 1,
    }
    const deferredChunk = deferred<ChunkDto>()
    mocks.pending.set(citation.chunk_id, deferredChunk)

    let modal: ReturnType<typeof mountModal> | undefined
    try {
      modal = mountModal(citation)
      await flushUi()
      expect(mocks.getChunk).toHaveBeenCalledWith(citation.chunk_id)

      deferredChunk.resolve(fetchedChunk)
      await flushUi()

      expect(modal.headingText()).toBe('lecture-notes.pdf')
    } finally {
      modal?.unmount()
      mocks.pending.delete(citation.chunk_id)
    }
  })
})
