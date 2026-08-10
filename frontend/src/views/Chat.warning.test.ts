import { createRenderer, nextTick, ssrContextKey } from 'vue'
import * as VueRuntime from 'vue'
import { compileScript, compileTemplate, parse } from 'vue/compiler-sfc'
import { describe, expect, it, vi } from 'vitest'
import type { Message } from '../stores/chat'

type TestNode = {
  parent?: TestNode
  children: TestNode[]
  tag?: string
  text?: string
  addEventListener?: () => void
  removeEventListener?: () => void
}

const mocks = vi.hoisted(() => {
  const assistantContent = [
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
    '',
    '---',
    '⚠️ Self-check note: this answer scored low (0.42; weak on groundedness).',
  ].join('\n')

  return {
    assistantContent,
    chat: {
      messages: [{
        id: 'assistant-warning-1',
        role: 'assistant' as const,
        content: assistantContent,
        citations: [],
        quizQuestionId: 'question-persisted-1',
      }] as Message[],
      streaming: false,
      trace: [],
      restoreCurrentSession: vi.fn(async () => undefined),
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
  useRoute: () => ({ query: {} }),
  useRouter: () => ({ replace: mocks.replace }),
}))

import MCQCard from '../components/MCQCard.vue'
import mcqCardSource from '../components/MCQCard.vue?raw'
import CitationPreviewModal from '../components/CitationPreviewModal.vue'
import citationPreviewModalSource from '../components/CitationPreviewModal.vue?raw'
import Chat from './Chat.vue'
import chatSource from './Chat.vue?raw'

function installCompiledRender(
  component: object,
  source: string,
  filename: string,
  id: string,
) {
  const { descriptor } = parse(source, { filename })
  if (!descriptor.template) throw new Error(`${filename} is missing a template`)

  const compiledScript = compileScript(descriptor, { id })
  const compiledTemplate = compileTemplate({
    source: descriptor.template.content,
    filename,
    id,
    compilerOptions: {
      mode: 'function',
      bindingMetadata: compiledScript.bindings,
    },
  })
  if (compiledTemplate.errors.length > 0) {
    throw new Error(compiledTemplate.errors.map(String).join('\n'))
  }

  const render = new Function('Vue', compiledTemplate.code)(VueRuntime)
  Object.assign(component, { render })
}

installCompiledRender(MCQCard, mcqCardSource, 'MCQCard.vue', 'chat-warning-mcq-card')
installCompiledRender(
  CitationPreviewModal,
  citationPreviewModalSource,
  'CitationPreviewModal.vue',
  'chat-warning-citation-preview',
)
installCompiledRender(Chat, chatSource, 'Chat.vue', 'chat-warning-chat')

function readNodeText(node: TestNode): string {
  return `${node.text ?? ''}${node.children.map(readNodeText).join('')}`
}

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
    createElement: tag => ({
      children: [],
      tag,
      addEventListener: () => undefined,
      removeEventListener: () => undefined,
    }),
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
  app.component('RouterLink', { render: () => null })
  app.mount(root)
  return { app, root }
}

async function flushUi() {
  await nextTick()
  await Promise.resolve()
  await nextTick()
}

describe('Chat MCQ warning visibility', () => {
  it('keeps the backend Judge warning visible when a persisted MCQ is rendered', async () => {
    const { app, root } = mountChat()
    try {
      await flushUi()
      const rendered = readNodeText(root)

      expect(rendered).toContain('What does HyDE rewrite?')
      expect(rendered).toContain('A) Queries')
      expect(rendered).toContain('⚠️ Self-check note')
    } finally {
      app.unmount()
    }
  })
})
