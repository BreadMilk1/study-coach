import { defineStore } from 'pinia'
import {
  getChatSessionMessages,
  getCurrentChatSession,
  getStoredChatSessionId,
  setStoredChatSessionId,
} from '../lib/api'
import { extractMermaid, looksLikeEmptyCorpusRefusal } from '../lib/parse'
import { usePlan } from './plan'
import { useQuiz } from './quiz'

export interface Citation {
  chunk_id: string
  source: string
  page: number
}

export interface AgentRunToolCall {
  name: string
  error: boolean
  args_preview: string
  output_preview: string
}

export interface AgentRun {
  node: string
  mode: string
  total_iterations: number
  total_tool_calls: number
  tool_call_breakdown: Record<string, number>
  tool_errors: number
  input_tokens: number
  output_tokens: number
  wall_time_s: number
  exit_reason: string
  llm_error: string | null
  tool_calls: AgentRunToolCall[]
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
  agentRun?: AgentRun | null
}

interface ChatState {
  messages: Message[]
  streaming: boolean
  trace: any[]
  sessionId: string
  restoring: boolean
}

export const useChat = defineStore('chat', {
  state: (): ChatState => ({
    messages: [],
    streaming: false,
    trace: [],
    sessionId: getStoredChatSessionId(),
    restoring: false,
  }),
  actions: {
    setSessionId(sessionId: string) {
      this.sessionId = sessionId
      setStoredChatSessionId(sessionId)
    },
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
    setAgentRun(m: Message, run: AgentRun) {
      m.agentRun = run
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
    async restoreCurrentSession() {
      if (this.streaming || this.restoring || this.messages.length > 0) return
      this.restoring = true
      try {
        let sessionId = this.sessionId || getStoredChatSessionId()
        if (!sessionId) {
          const current = await getCurrentChatSession()
          sessionId = current.session_id
        }
        if (!sessionId) return
        const history = await getChatSessionMessages(sessionId)
        this.setSessionId(history.session_id)
        this.messages = history.messages.map((m) => ({
          id: m.id,
          role: m.role,
          content: m.content,
          citations: m.citations.map((c) => ({
            chunk_id: c.chunk_id,
            source: c.source || c.chunk_id,
            page: c.page,
          })),
          agentRun: m.agent_run,
        }))
      } catch {
        // No prior session yet, or it belongs to a different user/token.
      } finally {
        this.restoring = false
      }
    },
  },
})
