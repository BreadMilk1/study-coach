import { describe, expect, it } from 'vitest'

import {
  canClaimSuiteQuality,
  compatibilityBadge,
  deltaCaption,
  displayVerdict,
  shouldShowScoreDelta,
  usageLabel,
  verdictLabel,
} from './learningRunPresentation'

describe('learning run presentation policy', () => {
  it('labels verdicts without inventing a pass', () => {
    expect(verdictLabel('pass')).toBe('Pass')
    expect(verdictLabel('fail')).toBe('Fail')
    expect(displayVerdict({ citationHardGateFailed: true, quality_verdict: 'pass' })).toBe('fail')
  })

  it('hides score delta unless the compare is controlled and same scorer', () => {
    expect(shouldShowScoreDelta({
      compatibility: 'controlled',
      rescore_required: false,
      delta: { groundedness: { left: 4, right: 3, delta: -1 } },
    })).toBe(true)
    expect(shouldShowScoreDelta({
      compatibility: 'controlled',
      rescore_required: true,
      delta: null,
    })).toBe(false)
    expect(shouldShowScoreDelta({
      compatibility: 'informational',
      rescore_required: false,
      delta: { groundedness: { left: 4, right: 3, delta: -1 } },
    })).toBe(false)
    expect(shouldShowScoreDelta({
      compatibility: 'incompatible',
      rescore_required: false,
      delta: null,
    })).toBe(false)
  })

  it('keeps case and suite claim language from generalizing', () => {
    expect(deltaCaption('case')).toBe('case delta')
    expect(deltaCaption('suite')).toBe('frozen 12-case suite delta')
    expect(canClaimSuiteQuality('case')).toBe(false)
    expect(canClaimSuiteQuality('suite')).toBe(true)
  })

  it('exposes compatibility badges and unavailable usage', () => {
    expect(compatibilityBadge('controlled')).toBe('Controlled')
    expect(compatibilityBadge('informational')).toBe('Informational')
    expect(compatibilityBadge('incompatible')).toBe('Incompatible')
    expect(usageLabel('unavailable')).toBe('unavailable')
    expect(usageLabel(null)).toBe('unavailable')
  })
})
