// Pure utilities — no Vue imports. Easy to test by inspection.

const MERMAID_RE = /```mermaid\n([\s\S]*?)```/

export function extractMermaid(text: string): string | null {
  const m = MERMAID_RE.exec(text)
  return m ? m[1].trim() : null
}

// P2.3 §Finding 3 — agent-loop refusal phrases.
// cloud-adapt: cloud models use different phrasing — extend list as needed.
const REFUSAL_PATTERNS = [
  /i'?m\s+unable\s+to\s+retrieve/i,
  /no\s+(information|content|context|source\s+material)\s+(was\s+)?(found|available|retrieved)/i,
  /cannot\s+(quiz|generate)\s+(you\s+)?without/i,
  /please\s+(provide|upload)\s+(more\s+)?(context|source|pdf|a\s+PDF)/i,
  /retriever\s+returned\s+(no|empty)/i,
  /wasn'?t\s+able\s+to\s+find/i,
  /came\s+up\s+empty/i,
  /your\s+(PDF|document|source)\s+(corpus|material)/i,
]

export function looksLikeEmptyCorpusRefusal(text: string): boolean {
  return REFUSAL_PATTERNS.some(re => re.test(text))
}
