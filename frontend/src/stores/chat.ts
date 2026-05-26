import { defineStore } from 'pinia'
import { extractMermaid, looksLikeEmptyCorpusRefusal } from '../lib/parse'
import { usePlan } from './plan'
import { useQuiz } from './quiz'

export interface Citation {
  chunk_id: string
  source: string
  page: number
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
}

interface ChatState {
  messages: Message[]
  streaming: boolean
  trace: any[]
}

export const useChat = defineStore('chat', {
  state: (): ChatState => ({
    messages: [],
    streaming: false,
    trace: [],
  }),
  actions: {
    pushUser(content: string) {
      this.messages.push({ id: crypto.randomUUID(), role: 'user', content })
    },
    startAssistant(): Message {
      const m: Message = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: '',
        citations: [],
      }
      this.messages.push(m)
      this.streaming = true
      return m
    },
    appendToken(m: Message, text: string) {
      m.content += text
    },
    setCitations(m: Message, cs: Citation[]) {
      m.citations = cs
    },
    finish() {
      this.streaming = false
      const last = this.messages[this.messages.length - 1]
      if (last && last.role === 'assistant' && last.content) {
        const mm = extractMermaid(last.content)
        if (mm) usePlan().setMindmap(mm)
        useQuiz().setNeedsUpload(looksLikeEmptyCorpusRefusal(last.content))
      }
    },
  },
})
