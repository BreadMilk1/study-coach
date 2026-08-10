import { expect, test } from 'vitest'

/** Mirror of CitationPreviewModal span slicing for unit coverage. */
function slicePreview(text: string, spanStart: number, spanEnd: number) {
  if (!text || spanEnd <= spanStart || spanStart < 0 || spanEnd > text.length) {
    return { before: text, highlight: '', after: '' }
  }
  return {
    before: text.slice(0, spanStart),
    highlight: text.slice(spanStart, spanEnd),
    after: text.slice(spanEnd),
  }
}

test('highlights span when bounds are valid', () => {
  expect(slicePreview('Hello world', 6, 11)).toEqual({
    before: 'Hello ',
    highlight: 'world',
    after: '',
  })
})

test('falls back to full text when span is empty or out of range', () => {
  expect(slicePreview('Hello', 0, 0)).toEqual({
    before: 'Hello',
    highlight: '',
    after: '',
  })
  expect(slicePreview('Hello', 0, 99)).toEqual({
    before: 'Hello',
    highlight: '',
    after: '',
  })
})
