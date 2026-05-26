export interface ParsedMCQ {
  prompt: string
  options: string[]
}

export interface ParsedGrade {
  correct: boolean
  correctAnswer: string
  explanation: string
}

export type ParsedQuizAssistantText =
  | { kind: 'mcq'; currentMCQ: ParsedMCQ }
  | { kind: 'grade'; lastGrade: ParsedGrade }
  | { kind: 'none' }

const OPTION_BLOCK_RE =
  /^\s*A\)\s*(.+)\n\s*B\)\s*(.+)\n\s*C\)\s*(.+)\n\s*D\)\s*(.+)\s*$/m
const GRADE_START_RE = /^\s*(?:[✓✔📝]|✗|Correct\b|Incorrect\b|Wrong\b)/i

function stripMarkdown(text: string): string {
  return text
    .replace(/\*\*/g, '')
    .replace(/^#+\s*/gm, '')
    .replace(/^\s*[-*>]\s*/gm, '')
    .trim()
}

function extractPrompt(prefix: string): string {
  const lines = stripMarkdown(prefix)
    .split('\n')
    .map(line => line.trim())
    .filter(Boolean)

  for (let i = lines.length - 1; i >= 0; i -= 1) {
    if (lines[i].includes('?')) return lines[i]
  }
  // Fall back to last non-trivial line (skip "Quiz on ...:" header lines)
  for (let i = lines.length - 1; i >= 0; i -= 1) {
    if (!lines[i].endsWith(':')) return lines[i]
  }
  return lines.at(-1) ?? ''
}

function parseMCQ(text: string): ParsedQuizAssistantText | null {
  const optionMatch = OPTION_BLOCK_RE.exec(text)
  if (!optionMatch) return null

  const prompt = extractPrompt(text.slice(0, optionMatch.index))
  if (!prompt) return null

  return {
    kind: 'mcq',
    currentMCQ: {
      prompt,
      options: [
        `A) ${optionMatch[1].trim()}`,
        `B) ${optionMatch[2].trim()}`,
        `C) ${optionMatch[3].trim()}`,
        `D) ${optionMatch[4].trim()}`,
      ],
    },
  }
}

function parseGrade(text: string): ParsedQuizAssistantText | null {
  const normalized = stripMarkdown(text)
  if (!GRADE_START_RE.test(normalized)) return null

  const answerMatch = /correct answer[:\s]*([A-D])\b/i.exec(normalized)
  const incorrect = /(?:\bincorrect\b|\bwrong\b|✗)/i.test(normalized)
  const correct = !incorrect

  // Extract explanation: everything after the verdict prefix
  let explanation = normalized
  explanation = explanation
    .replace(/^.*?(?:[✓✔]\s*Correct!?\s*|✗\s*Incorrect\.?\s*|Wrong\.?\s*)/i, '')
    .replace(/Correct answer:\s*[A-D]\.?\s*/i, '')
    .trim()

  return {
    kind: 'grade',
    lastGrade: {
      correct,
      correctAnswer: answerMatch?.[1] ?? '',
      explanation,
    },
  }
}

export function parseQuizAssistantText(text: string): ParsedQuizAssistantText {
  return parseMCQ(text) ?? parseGrade(text) ?? { kind: 'none' }
}
