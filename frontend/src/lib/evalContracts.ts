export const EVAL_SCHEMA_VERSION = 'eval-api-v1' as const

export type EvalSchemaVersion = typeof EVAL_SCHEMA_VERSION
export type RunProfile = 'evaluation'
export type ActiveKind = 'run' | 'score_set'
export type CompareCompatibility = 'controlled' | 'informational' | 'incompatible'
export type CompareScope = 'case' | 'suite'
export type QualityVerdict = 'pass' | 'fail' | 'inconclusive' | 'not_evaluated'
export type RunLifecycle = 'queued' | 'running' | 'finished' | 'cancelled'
export type RunOutcome = 'success' | 'system_failed' | 'timed_out' | 'budget_exceeded'
export type ScoreSetStatus = 'pending' | 'running' | 'completed' | 'partial' | 'failed' | 'cancelled'
export type ScorerExecutionStatus = 'success' | 'failed' | 'skipped'

export interface LearningRunRequest {
  experiment_id: string
  task_case_id: string
  variant_id: string
  run_profile: RunProfile
}

export interface RescoreRequest {
  scorer_version: string
  run_profile?: RunProfile
}

export interface EvalConnectionSnapshot {
  provider: string
  model: string
  apiKey?: string
  baseUrl?: string
}

export interface EvalErrorDetail {
  code: string
  message: string
  fields: string[]
  active_entity_id: string | null
  active_kind: ActiveKind | null
}

export class EvalContractError extends Error {
  readonly code: string

  constructor(code: string, message: string) {
    super(message)
    this.name = 'EvalContractError'
    this.code = code
  }
}

interface EvalEventBase {
  schema_version: EvalSchemaVersion
  run_id: string
}

export interface RunCreatedEvent extends EvalEventBase {
  type: 'run_created'
}

export interface StageStartedEvent extends EvalEventBase {
  type: 'stage_started'
  stage: string
}

export interface StageCompletedEvent extends EvalEventBase {
  type: 'stage_completed'
  stage: string
}

export interface ScoreSetCreatedEvent extends EvalEventBase {
  type: 'score_set_created'
  score_set_id: string
}

export interface ScorerCompletedEvent extends EvalEventBase {
  type: 'scorer_completed'
  score_set_id: string
  scorer_id: string
  status: 'success' | 'skipped'
}

export interface ScorerFailedEvent extends EvalEventBase {
  type: 'scorer_failed'
  score_set_id: string
  scorer_id: string
  status: 'failed'
  error_code?: string | null
}

export interface ScoreSetFinishedEvent extends EvalEventBase {
  type: 'score_set_finished'
  score_set_id: string
  status: 'completed' | 'partial' | 'failed' | 'cancelled'
  quality_verdict: QualityVerdict
}

export interface RunFinishedEvent extends EvalEventBase {
  type: 'run_finished'
  lifecycle: 'finished' | 'cancelled'
  outcome?: RunOutcome | null
  error_code?: string | null
}

export type LearningRunEvent =
  | RunCreatedEvent
  | StageStartedEvent
  | StageCompletedEvent
  | ScoreSetCreatedEvent
  | ScorerCompletedEvent
  | ScorerFailedEvent
  | ScoreSetFinishedEvent
  | RunFinishedEvent

export interface VariantSummary {
  variant_id: string
  prompt_version: string
}

export interface ExperimentSummary {
  experiment_id: string
  task_family: string
  experiment_axes: string[]
  variants: VariantSummary[]
  case_counts: Record<string, number>
  run_profile: string
  budgets: Record<string, number>
  regression_count: number
}

export interface ScoreSetSummary {
  score_set_id: string
  scorer_id: string
  scorer_version: string
  scorer_definition_hash: string | null
  status: ScoreSetStatus
  quality_verdict: QualityVerdict
  aggregate_scores: Record<string, unknown> | null
  created_at: string
  finished_at: string | null
}

export interface RunSummary {
  run_id: string
  experiment_id: string
  suite_execution_id: string | null
  task_case_id: string
  variant_id: string
  run_profile: string
  lifecycle: RunLifecycle
  outcome: RunOutcome | null
  latest_score_set: ScoreSetSummary | null
  created_at: string
  started_at: string | null
  finished_at: string | null
}

export interface ScoreSetDetail extends ScoreSetSummary {
  artifact_input_hash: string
  scorer_snapshot: Record<string, unknown> | null
  operational_error_code: string | null
  operational_error_message: string | null
  findings: unknown
}

export interface ScorerExecutionDetail {
  execution_id: string
  score_set_id: string
  scorer_id: string
  scorer_version: string
  status: ScorerExecutionStatus
  input_hash: string
  output: Record<string, unknown> | unknown[] | null
  operational_error_code: string | null
  operational_error_message: string | null
  latency_ms: number | null
  usage: Record<string, unknown> | null
  created_at: string
}

export interface RunDetail {
  summary: RunSummary
  manifest: Record<string, unknown>
  candidate_artifact: Record<string, unknown> | null
  score_sets: ScoreSetDetail[]
  scorer_executions: ScorerExecutionDetail[]
  operational_error: Record<string, unknown> | null
}

export interface CompareRunRef {
  run_id: string
  variant_id: string
}

export interface ScoreDelta {
  left: number
  right: number
  delta: number
}

export interface CompareResponse {
  compatibility: CompareCompatibility
  reasons: string[]
  left: CompareRunRef
  right: CompareRunRef
  scorer_bundle: { scorer_id: string; version: string }
  delta: Record<string, ScoreDelta> | null
  scope: CompareScope
  rescore_required: boolean
  caption: string
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null
}

function requiredString(record: Record<string, unknown>, key: string): string {
  const value = record[key]
  if (typeof value !== 'string' || !value) {
    throw new EvalContractError('malformed_event', `missing ${key}`)
  }
  return value
}

export function parseEvalEvent(value: unknown): LearningRunEvent {
  const record = asRecord(value)
  if (!record) {
    throw new EvalContractError('malformed_event', 'event is not an object')
  }
  if (record.schema_version !== EVAL_SCHEMA_VERSION) {
    throw new EvalContractError('unsupported_schema', 'unknown eval schema version')
  }
  const run_id = requiredString(record, 'run_id')
  const type = record.type
  switch (type) {
    case 'run_created':
      return { schema_version: EVAL_SCHEMA_VERSION, type, run_id }
    case 'stage_started':
    case 'stage_completed':
      return {
        schema_version: EVAL_SCHEMA_VERSION,
        type,
        run_id,
        stage: requiredString(record, 'stage'),
      }
    case 'score_set_created':
      return {
        schema_version: EVAL_SCHEMA_VERSION,
        type,
        run_id,
        score_set_id: requiredString(record, 'score_set_id'),
      }
    case 'scorer_completed': {
      const status = record.status
      if (status !== 'success' && status !== 'skipped') {
        throw new EvalContractError('malformed_event', 'invalid scorer status')
      }
      return {
        schema_version: EVAL_SCHEMA_VERSION,
        type,
        run_id,
        score_set_id: requiredString(record, 'score_set_id'),
        scorer_id: requiredString(record, 'scorer_id'),
        status,
      }
    }
    case 'scorer_failed':
      return {
        schema_version: EVAL_SCHEMA_VERSION,
        type,
        run_id,
        score_set_id: requiredString(record, 'score_set_id'),
        scorer_id: requiredString(record, 'scorer_id'),
        status: 'failed',
        error_code: typeof record.error_code === 'string' ? record.error_code : null,
      }
    case 'score_set_finished': {
      const status = record.status
      const quality_verdict = record.quality_verdict
      if (
        status !== 'completed'
        && status !== 'partial'
        && status !== 'failed'
        && status !== 'cancelled'
      ) {
        throw new EvalContractError('malformed_event', 'invalid score set status')
      }
      if (
        quality_verdict !== 'pass'
        && quality_verdict !== 'fail'
        && quality_verdict !== 'inconclusive'
        && quality_verdict !== 'not_evaluated'
      ) {
        throw new EvalContractError('malformed_event', 'invalid quality verdict')
      }
      return {
        schema_version: EVAL_SCHEMA_VERSION,
        type,
        run_id,
        score_set_id: requiredString(record, 'score_set_id'),
        status,
        quality_verdict,
      }
    }
    case 'run_finished': {
      const lifecycle = record.lifecycle
      if (lifecycle !== 'finished' && lifecycle !== 'cancelled') {
        throw new EvalContractError('malformed_event', 'invalid run lifecycle')
      }
      const outcome = record.outcome
      if (
        outcome != null
        && outcome !== 'success'
        && outcome !== 'system_failed'
        && outcome !== 'timed_out'
        && outcome !== 'budget_exceeded'
      ) {
        throw new EvalContractError('malformed_event', 'invalid run outcome')
      }
      return {
        schema_version: EVAL_SCHEMA_VERSION,
        type,
        run_id,
        lifecycle,
        outcome: outcome ?? null,
        error_code: typeof record.error_code === 'string' ? record.error_code : null,
      }
    }
    default:
      throw new EvalContractError('unsupported_event', 'unknown eval event type')
  }
}

export function isTerminalRunEvent(event: LearningRunEvent): boolean {
  return event.type === 'run_finished'
}

export function parseEvalErrorDetail(value: unknown): EvalErrorDetail {
  const record = asRecord(value)
  const nested = record ? asRecord(record.detail) : null
  const raw = nested ?? record
  if (!raw || typeof raw.code !== 'string') {
    throw new EvalContractError('malformed_error', 'eval error is not structured')
  }
  const activeKind = raw.active_kind
  return {
    code: raw.code,
    message: typeof raw.message === 'string' ? raw.message : 'evaluation request failed',
    fields: Array.isArray(raw.fields)
      ? raw.fields.filter((field): field is string => typeof field === 'string')
      : [],
    active_entity_id: typeof raw.active_entity_id === 'string' ? raw.active_entity_id : null,
    active_kind: activeKind === 'run' || activeKind === 'score_set' ? activeKind : null,
  }
}

export function parseRunDetail(value: unknown): RunDetail {
  const record = asRecord(value)
  const summary = record ? asRecord(record.summary) : null
  if (!record || !summary || typeof summary.run_id !== 'string') {
    throw new EvalContractError('malformed_detail', 'run detail is invalid')
  }
  return value as RunDetail
}

export function parseCompareResponse(value: unknown): CompareResponse {
  const record = asRecord(value)
  if (!record || (record.compatibility !== 'controlled' && record.compatibility !== 'informational' && record.compatibility !== 'incompatible')) {
    throw new EvalContractError('malformed_compare', 'compare response is invalid')
  }
  return {
    compatibility: record.compatibility,
    reasons: Array.isArray(record.reasons) ? record.reasons.filter((item): item is string => typeof item === 'string') : [],
    left: record.left as CompareRunRef,
    right: record.right as CompareRunRef,
    scorer_bundle: record.scorer_bundle as CompareResponse['scorer_bundle'],
    delta: (record.delta ?? null) as CompareResponse['delta'],
    scope: record.scope === 'suite' ? 'suite' : 'case',
    rescore_required: record.rescore_required === true,
    caption: typeof record.caption === 'string' ? record.caption : 'case delta',
  }
}

export function learningRunRequest(input: LearningRunRequest): LearningRunRequest {
  return {
    experiment_id: input.experiment_id,
    task_case_id: input.task_case_id,
    variant_id: input.variant_id,
    run_profile: 'evaluation',
  }
}
