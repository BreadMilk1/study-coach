import { defineStore } from 'pinia'
import { looksLikeEmptyCorpusRefusal } from '../lib/parse'
import { parseQuizAssistantText } from '../lib/quiz'

export type Difficulty = 'easy' | 'med' | 'hard'

interface ParsedMCQ {
  prompt: string
  options: string[]
}

interface QuizState {
  currentMCQ: ParsedMCQ | null
  lastGrade: { correct: boolean; correctAnswer: string; explanation: string } | null
  difficulty: Difficulty
  needsUpload: boolean
  streaming: boolean
  raw: string  // last assistant text — used by parser
}

export const useQuiz = defineStore('quiz', {
  state: (): QuizState => ({
    currentMCQ: null,
    lastGrade: null,
    difficulty: 'med',
    needsUpload: false,
    streaming: false,
    raw: '',
  }),
  actions: {
    setDifficulty(d: Difficulty) { this.difficulty = d },
    setNeedsUpload(v: boolean) { this.needsUpload = v },
    startStream() { this.streaming = true; this.raw = '' },
    appendRaw(t: string) { this.raw += t },
    finishStream() {
      this.streaming = false
      this.parse()
    },
    reset() {
      this.currentMCQ = null
      this.lastGrade = null
      this.raw = ''
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
      }
    },
  },
})
