import { describe, expect, it } from 'vitest'

import { parseAnswerMarkdown, safeAnswerLink } from './answerMarkdown'

describe('Compare answer Markdown', () => {
  it('renders emphasis, paragraphs, lists, tables, and code as structured content', () => {
    const html = parseAnswerMarkdown('**Evidence**\n\n- first\n- second\n\n| Claim | Source |\n| --- | --- |\n| A | B |\n\n```js\nconst x = "<tag>"\n```')
    expect(html).toContain('<strong>Evidence</strong>')
    expect(html).toContain('<ul>')
    expect(html).toContain('<li>second</li>')
    expect(html).toContain('<table>')
    expect(html).toContain('<th>Claim</th>')
    expect(html).toContain('<pre><code')
    expect(html).toContain('&lt;tag&gt;')
  })

  it('displays raw HTML as text and does not render external images', () => {
    const html = parseAnswerMarkdown('<img src="https://tracker.test/x" onerror="alert(1)">\n\n<script>alert(1)</script>\n\n![diagram](https://tracker.test/image.png)')
    expect(html).not.toMatch(/<(?:img|script)\b/i)
    expect(html).toContain('&lt;img')
    expect(html).toContain('&lt;script&gt;')
    expect(html).toContain('diagram')
    expect(html).not.toContain('src="https://tracker.test')
  })

  it.each(['javascript:alert(1)', 'JaVaScRiPt:alert(1)', 'java\tscript:alert(1)', 'data:text/html,hello', 'vbscript:msgbox(1)', '//tracker.test/x', '/local', '&#x6a;avascript:alert(1)'])('rejects unsafe or non-http link %s', href => {
    expect(safeAnswerLink(href)).toBeNull()
  })

  it('retains allowed links and formats unsafe link labels without anchors', () => {
    expect(safeAnswerLink('HTTPS://example.test/path')).toBe('https://example.test/path')
    expect(parseAnswerMarkdown('[**safe**](https://example.test/)')).toContain('<a href="https://example.test/"><strong>safe</strong></a>')
    expect(parseAnswerMarkdown('[**blocked**](javascript:alert%281%29)')).toContain('<strong>blocked</strong>')
    expect(parseAnswerMarkdown('[blocked](javascript:alert%281%29)')).not.toContain('<a')
  })
})
