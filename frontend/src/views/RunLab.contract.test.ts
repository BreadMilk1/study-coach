import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

import { describe, expect, it } from 'vitest'

import { displayVerdict, shouldShowScoreDelta } from '../lib/learningRunPresentation'

function readSource(relativePath: string): string {
  return readFileSync(fileURLToPath(new URL(relativePath, import.meta.url)), 'utf8')
}

describe('Run Lab source contracts', () => {
  it('registers the three run-lab routes exactly', () => {
    const router = readSource('../router.ts')
    expect(router).toMatch(/path:\s*'\/run-lab'/)
    expect(router).toMatch(/path:\s*'\/run-lab\/runs\/:runId'/)
    expect(router).toMatch(/path:\s*'\/run-lab\/compare'/)
  })

  it('exposes Run Lab in desktop and mobile navigation', () => {
    const app = readSource('../App.vue')
    const mobile = readSource('../components/MobileNav.vue')
    expect(app).toMatch(/to:\s*'\/run-lab'/)
    expect(mobile).toMatch(/to:\s*'\/run-lab'/)
  })

  it('composes contract, evidence, score, and trace drawer on the detail page', () => {
    const detail = readSource('./RunDetail.vue')
    expect(detail).toContain('RunContractBar')
    expect(detail).toContain('RunEvidenceTimeline')
    expect(detail).toContain('RunScorePanel')
    expect(detail).toContain('RunTraceDrawer')
  })

  it('labels historical fallback as a completed historical run and does not replace the live id', () => {
    const detail = readSource('./RunDetail.vue')
    expect(detail).toMatch(/completed historical run/i)
    expect(detail).toContain('openHistoricalFallback')
    expect(detail).not.toMatch(/activeRunId\s*=(?!=)/)
  })

  it('does not offer a browser suite batch button', () => {
    const lab = readSource('./RunLab.vue')
    expect(lab).not.toMatch(/batch/i)
  })

  it('hides score delta for scorer mismatch and incompatible compares', () => {
    expect(shouldShowScoreDelta({
      compatibility: 'controlled',
      rescore_required: true,
      delta: null,
    })).toBe(false)
    expect(shouldShowScoreDelta({
      compatibility: 'incompatible',
      rescore_required: false,
      delta: null,
    })).toBe(false)
  })

  it('cannot display Pass when the citation hard gate failed', () => {
    expect(displayVerdict({
      citationHardGateFailed: true,
      quality_verdict: 'pass',
    })).toBe('fail')
  })
})
