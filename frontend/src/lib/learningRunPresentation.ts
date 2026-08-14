import type { CompareCompatibility, CompareResponse, CompareScope, QualityVerdict } from './evalContracts'

export function verdictLabel(verdict: QualityVerdict): string {
  switch (verdict) {
    case 'pass':
      return 'Pass'
    case 'fail':
      return 'Fail'
    case 'inconclusive':
      return 'Inconclusive'
    case 'not_evaluated':
      return 'Not evaluated'
  }
}

export function findingLabel(severity: 'critical' | 'noncritical'): string {
  return severity === 'critical' ? 'Critical finding' : 'Noncritical finding'
}

export function operationalErrorMessage(code: string | null | undefined): string {
  switch (code) {
    case 'process_interrupted':
      return 'The evaluation process was interrupted.'
    case 'cancelled':
      return 'The evaluation was cancelled.'
    case 'scorer_parse_error':
      return 'A scorer result could not be parsed.'
    case 'scorer_timeout':
      return 'Scoring timed out.'
    case 'generation_timeout':
      return 'Tutor generation timed out.'
    case 'model_unavailable':
      return 'The evaluation model is unavailable.'
    case 'budget_exceeded':
      return 'The evaluation budget was exceeded.'
    case 'corpus_unavailable':
    case 'corpus_mismatch':
      return 'The isolated evaluation corpus is unavailable.'
    default:
      return 'The evaluation reported an operational error.'
  }
}

export function compatibilityBadge(compatibility: CompareCompatibility): string {
  switch (compatibility) {
    case 'controlled':
      return 'Controlled'
    case 'informational':
      return 'Informational'
    case 'incompatible':
      return 'Incompatible'
  }
}

export function shouldShowScoreDelta(compare: Pick<CompareResponse, 'compatibility' | 'rescore_required' | 'delta'>): boolean {
  return compare.compatibility === 'controlled'
    && compare.rescore_required !== true
    && compare.delta !== null
}

export function deltaCaption(scope: CompareScope): string {
  return scope === 'suite' ? 'frozen 12-case suite delta' : 'case delta'
}

export function canClaimSuiteQuality(scope: CompareScope): boolean {
  return scope === 'suite'
}

export function usageLabel(usage: unknown): string {
  if (usage === 'unavailable' || usage == null) return 'unavailable'
  return 'available'
}

export function displayVerdict(input: {
  citationHardGateFailed?: boolean
  quality_verdict?: QualityVerdict | null
}): QualityVerdict {
  if (input.citationHardGateFailed) return 'fail'
  return input.quality_verdict ?? 'not_evaluated'
}
