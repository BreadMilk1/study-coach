import { Marked } from 'marked'

function escapeHtml(value: string): string {
  return value.replace(/[&<>"']/g, character => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  })[character]!)
}

export function safeAnswerLink(href: string): string | null {
  if (!/^https?:\/\//i.test(href)) return null
  try {
    const url = new URL(href)
    return url.protocol === 'http:' || url.protocol === 'https:' ? url.href : null
  } catch {
    return null
  }
}

// Keep this renderer isolated from Chat and other Markdown consumers.
const markdown = new Marked({
  gfm: true,
  breaks: false,
  renderer: {
    html({ text }) { return escapeHtml(text) },
    image({ text }) { return escapeHtml(text) },
    link({ href, tokens }) {
      const text = this.parser.parseInline(tokens)
      const safeHref = safeAnswerLink(href)
      return safeHref ? `<a href="${escapeHtml(safeHref)}">${text}</a>` : text
    },
  },
})

// This intermediate HTML must always be sanitized before entering the DOM.
export function parseAnswerMarkdown(source: string): string {
  return markdown.parse(source, { async: false })
}
