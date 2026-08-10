import { createRenderer, h, nextTick, ssrContextKey } from 'vue'
import { describe, expect, it, vi } from 'vitest'
import type { Citation } from '../stores/chat'
import type { ChunkDto } from '../lib/api'

const mocks = vi.hoisted(() => ({
  getChunk: vi.fn(async (_chunkId: string): Promise<ChunkDto> => ({
    chunk_id: 'chunk-escape',
    content: 'preview body for escape test',
    source: 'notes.pdf',
    page: 1,
  })),
}))

vi.mock('../lib/api', () => ({
  getChunk: mocks.getChunk,
}))

import CitationPreviewModal from './CitationPreviewModal.vue'

// Vitest node env compiles SFCs with ssrRender only. Lifecycle tests in this
// repo stub client render the same way; setup/watch still run for real.
Object.assign(CitationPreviewModal, { render: () => null })

type TestNode = { parent?: TestNode; children: TestNode[]; text?: string }

function createWindowStub(): EventTarget {
  return new EventTarget()
}

function mountModal(citation: Citation, onClose: () => void) {
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

  const Wrapper = {
    setup() {
      return () => h(CitationPreviewModal, { citation, onClose })
    },
  }

  const app = renderer.createApp(Wrapper)
  app.provide(ssrContextKey, { modules: new Set<string>() })
  app.mount(root)
  return () => app.unmount()
}

function dispatchEscape(target: EventTarget) {
  // Prefer KeyboardEvent when available (Node web APIs / browser); fall back
  // to a plain Event with a key property so the production check runs.
  let event: Event
  if (typeof KeyboardEvent === 'function') {
    event = new KeyboardEvent('keydown', { key: 'Escape', bubbles: true })
  } else {
    event = new Event('keydown', { bubbles: true })
    Object.defineProperty(event, 'key', { value: 'Escape' })
  }
  target.dispatchEvent(event)
}

async function flushUi() {
  await Promise.resolve()
  await nextTick()
  await Promise.resolve()
  await nextTick()
}

describe('CitationPreviewModal Escape key', () => {
  it('closes the citation preview when Escape is pressed outside the modal focus', async () => {
    const citation: Citation = {
      chunk_id: 'chunk-escape',
      source: 'notes.pdf',
      page: 1,
      span_start: 0,
      span_end: 8,
    }
    const onClose = vi.fn()

    const windowStub = createWindowStub()
    const originalWindowDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'window')
    Object.defineProperty(globalThis, 'window', {
      configurable: true,
      writable: true,
      value: windowStub,
    })

    let unmount: (() => void) | undefined
    try {
      unmount = mountModal(citation, onClose)
      await flushUi()

      // Real setup/watch must have started and asked for the chunk.
      expect(mocks.getChunk).toHaveBeenCalledOnce()
      expect(mocks.getChunk).toHaveBeenCalledWith('chunk-escape')

      // Do not focus the overlay, call onKey, or fake template keydown —
      // dispatch Escape on window as if focus were elsewhere on the page.
      dispatchEscape(windowStub)
      await flushUi()

      expect(onClose).toHaveBeenCalledTimes(1)
    } finally {
      unmount?.()
      if (originalWindowDescriptor) {
        Object.defineProperty(globalThis, 'window', originalWindowDescriptor)
      } else {
        // @ts-expect-error restoring absent window in node
        delete globalThis.window
      }
    }
  })
})
