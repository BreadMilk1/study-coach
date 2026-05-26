import { defineStore } from 'pinia'
import { looksLikeEmptyCorpusRefusal } from '../lib/parse'
import { parseQuizAssistantText } from '../lib/quiz'

export type Difficulty = 'easy' | 'med' | 'hard'

interface ParsedMCQ {
  prompt: string
  options: string[]
}

const ERROR_DETECT_RE = /(?:⚠|could\s+not\s+reach|budget\s+exhausted|try\s+a\s+different\s+topic)/i

interface QuizState {
  currentMCQ: ParsedMCQ | null
  lastGrade: { correct: boolean; correctAnswer: string; explanation: string } | null
  difficulty: Difficulty
  needsUpload: boolean
  streaming: boolean
  raw: string
  errorMsg: string
  /** Non-null when redoing a specific mistake — POST to /api/mistakes/{id}/review */
  currentMistakeId: string | null
}

export const useQuiz = defineStore('quiz', {
  state: (): QuizState => ({
    currentMCQ: null,
    lastGrade: null,
    difficulty: 'med',
    needsUpload: false,
    streaming: false,
    raw: '',
    errorMsg: '',
    currentMistakeId: null,
  }),
  actions: {
    setDifficulty(d: Difficulty) { this.difficulty = d },
    setNeedsUpload(v: boolean) { this.needsUpload = v },
    startStream() { this.streaming = true; this.raw = ''; this.errorMsg = '' },
    appendRaw(t: string) { this.raw += t },
    finishStream() {
      this.streaming = false
      this.parse()
    },
    reset() {
      this.currentMCQ = null
      this.lastGrade = null
      this.raw = ''
      this.streaming = false
      this.needsUpload = false
      this.errorMsg = ''
      this.currentMistakeId = null
    },
    parse() {
      this.needsUpload = looksLikeEmptyCorpusRefusal(this.raw)
      const parsed = parseQuizAssistantText(this.raw)

      if (parsed.kind === 'mcq') {
        this.currentMCQ = parsed.currentMCQ
        this.lastGrade = null
        return
      }

      if (parsed.kind === 'grade') {
        this.currentMCQ = null
        this.lastGrade = parsed.lastGrade
        return
      }

      // kind === 'none' — may be a backend error surfaced via SSE token
      if (ERROR_DETECT_RE.test(this.raw)) {
        this.errorMsg = this.raw.trim()
      }
    },
  },
})
