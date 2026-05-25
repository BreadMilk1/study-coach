// Pure utilities — no Vue imports. Easy to test by inspection.

const MERMAID_RE = /```mermaid\n([\s\S]*?)```/

export function extractMermaid(text: string): string | null {
  const m = MERMAID_RE.exec(text)
  return m ? m[1].trim() : null
}

// P2.3 §Finding 3 — agent-loop refusal phrases.
// cloud-adapt: cloud GPT/DeepSeek may use different phrasing — extend list.
const REFUSAL_PATTERNS = [
  /i'?m\s+unable\s+to\s+retrieve/i,
  /no\s+(information|content|context)\s+(was\s+)?(found|available|retrieved)/i,
  /cannot\s+(quiz|generate)\s+(you\s+)?without/i,
  /please\s+(provide|upload)\s+(more\s+)?(context|source|pdf)/i,
  /retriever\s+returned\s+(no|empty)/i,
]

export function looksLikeEmptyCorpusRefusal(text: string): boolean {
  return REFUSAL_PATTERNS.some(re => re.test(text))
}
