import test from 'node:test'
import assert from 'node:assert/strict'

import { parseQuizAssistantText } from './quiz.ts'

test('parses deterministic quiz question into MCQ state', () => {
  const parsed = parseQuizAssistantText(
    [
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
    ].join('\n'),
  )

  assert.deepEqual(parsed, {
    kind: 'mcq',
    currentMCQ: {
      prompt: 'What does HyDE rewrite?',
      options: ['A) Queries', 'B) Documents', 'C) Embeddings', 'D) Answers'],
    },
  })
})

test('parses agent-loop markdown quiz without misclassifying it as grade', () => {
  const parsed = parseQuizAssistantText(
    [
      'Here is your quiz question on Reciprocal Rank Fusion:',
      '',
      '**What is the core mechanism and primary advantage of using Reciprocal Rank Fusion (RRF) in information retrieval?**',
      '',
      'A) It is faster than traditional weighted averaging methods.',
      'B) It requires knowing the precise weight assigned to every retrieved document.',
      'C) It combines ranking scores from multiple sources by considering the inverse of the rank.',
      'D) It can only be used when all search sources provide identical scoring metrics.',
      '',
      '***',
      '',
      '**Correct Answer: C**',
      '',
      '**Explanation:** RRF combines rankings from multiple retrieval methods by using rank positions rather than raw scores.',
    ].join('\n'),
  )

  assert.deepEqual(parsed, {
    kind: 'mcq',
    currentMCQ: {
      prompt: 'What is the core mechanism and primary advantage of using Reciprocal Rank Fusion (RRF) in information retrieval?',
      options: [
        'A) It is faster than traditional weighted averaging methods.',
        'B) It requires knowing the precise weight assigned to every retrieved document.',
        'C) It combines ranking scores from multiple sources by considering the inverse of the rank.',
        'D) It can only be used when all search sources provide identical scoring metrics.',
      ],
    },
  })
})

test('parses incorrect grade feedback into grade state', () => {
  const parsed = parseQuizAssistantText(
    '✗ Incorrect. Correct answer: A. Because A.',
  )

  assert.deepEqual(parsed, {
    kind: 'grade',
    lastGrade: {
      correct: false,
      correctAnswer: 'A',
      explanation: 'Because A.',
    },
  })
})
